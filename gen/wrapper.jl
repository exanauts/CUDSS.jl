using Clang
using Clang.Generators

using JuliaFormatter

using CUDA_SDK_jll, CUDSS_jll

function rewriter!(ctx, options)
    for node in get_nodes(ctx.dag)
        # remove aliases for function names
        #
        # when NVIDIA changes the behavior of an API, they version the function
        # (`cuFunction_v2`), and sometimes even change function names. To maintain backwards
        # compatibility, they ship aliases with their headers such that compiled binaries
        # will keep using the old version, and newly-compiled ones will use the developer's
        # CUDA version. remove those, since we target multiple CUDA versions.
        #
        # remove this if we ever decide to support a single supported version of CUDA.
        if node isa ExprNode{<:AbstractMacroNodeType}
            isempty(node.exprs) && continue
            expr = node.exprs[1]
            if Meta.isexpr(expr, :const)
                expr = expr.args[1]
            end
            if Meta.isexpr(expr, :(=))
                lhs, rhs = expr.args
                if rhs isa Expr && rhs.head == :call
                    name = string(rhs.args[1])
                    if endswith(name, "STRUCT_SIZE")
                        rhs.head = :macrocall
                        rhs.args[1] = Symbol("@", name)
                        insert!(rhs.args, 2, nothing)
                    end
                end
                isa(lhs, Symbol) || continue
                if Meta.isexpr(rhs, :call) && rhs.args[1] in (:__CUDA_API_PTDS, :__CUDA_API_PTSZ)
                    rhs = rhs.args[2]
                end
                isa(rhs, Symbol) || continue
                lhs, rhs = String(lhs), String(rhs)
                function get_prefix(str)
                    # cuFooBar -> cu
                    isempty(str) && return nothing
                    islowercase(str[1]) || return nothing
                    for i in 2:length(str)
                        if isuppercase(str[i])
                            return str[1:i-1]
                        end
                    end
                    return nothing
                end
                lhs_prefix = get_prefix(lhs)
                lhs_prefix === nothing && continue
                rhs_prefix = get_prefix(rhs)
                if lhs_prefix == rhs_prefix
                    @debug "Removing function alias: `$expr`"
                    empty!(node.exprs)
                end
            end
        end

        if Generators.is_function(node) && !Generators.is_variadic_function(node)
            expr = node.exprs[1]
            call_expr = expr.args[2].args[1].args[3]    # assumes `use_ccall_macro` is true

            # replace `@ccall` with `@gcsafe_ccall`
            expr.args[2].args[1].args[1] = Symbol("@gcsafe_ccall")

            target_expr = call_expr.args[1].args[1]
            fn = String(target_expr.args[2].value)

            # look up API options for this function
            fn_options = Dict{String,Any}()
            templates = Dict{String,Any}()
            template_types = nothing
            if haskey(options, "api")
                names = [fn]

                # _64 aliases are used by CUBLAS with Int64 arguments. they otherwise have
                # an idential signature, so we can reuse the same type rewrites.
                if endswith(fn, "_64")
                    push!(names, fn[1:end-3])
                end

                # look for a template rewrite: many libraries have very similar functions,
                # e.g., `cublas[SDHCZ]gemm`, for which we can use the same type rewrites
                # registered as `cublas𝕏gemm` template with `T` and `S` placeholders.
                for name in copy(names), (typcode,(T,S)) in ["S"=>("Cfloat","Cfloat"),
                                                              "D"=>("Cdouble","Cdouble"),
                                                              "H"=>("Float16","Float16"),
                                                              "C"=>("cuComplex","Cfloat"),
                                                              "Z"=>("cuDoubleComplex","Cdouble")]
                    idx = findfirst(typcode, name)
                    while idx !== nothing
                        template_name = name[1:idx.start-1] * "𝕏" * name[idx.stop+1:end]
                        if haskey(options["api"], template_name)
                            templates[template_name] = ["T" => T, "S" => S]
                            push!(names, template_name)
                        end
                        idx = findnext(typcode, name, idx.stop+1)
                    end
                end

                # the exact name is always checked first, so it's always possible to
                # override the type rewrites for a specific function
                # (e.g. if a _64 function ever passes a `Ptr{Cint}` index).
                for name in names
                    template_types = get(templates, name, nothing)
                    if haskey(options["api"], name)
                        fn_options = options["api"][name]
                        break
                    end
                end
            end

            # rewrite pointer argument types
            arg_exprs = call_expr.args[1].args[2:end]
            argtypes = get(fn_options, "argtypes", Dict())
            for (arg, typ) in argtypes
                i = parse(Int, arg)
                i in 1:length(arg_exprs) || error("invalid argtypes for $fn: index $arg is out of bounds")

                # _64 aliases should use Int64 instead of Int32/Cint
                if endswith(fn, "_64")
                    typ = replace(typ, "Cint" => "Int64", "Int32" => "Int64")
                end

                # expand type templates
                if template_types !== nothing
                    typ = replace(typ, template_types...)
                end

                arg_exprs[i].args[2] = Meta.parse(typ)
            end

            # insert `initialize_context()` before each function with a `ccall`
            if get(fn_options, "needs_context", true)
                pushfirst!(expr.args[2].args, :(initialize_context()))
            end

            # insert `@checked` before each function with a `ccall` returning a checked type`
            rettyp = call_expr.args[2]
            checked_types = if haskey(options, "api")
                get(options["api"], "checked_rettypes", Dict())
            else
                String[]
            end
            if rettyp isa Symbol && String(rettyp) in checked_types
                node.exprs[1] = Expr(:macrocall, Symbol("@checked"), nothing, expr)
            end
        end
    end
end

function main()
    cuda = joinpath(CUDA_SDK_jll.artifact_dir, "cuda", "include")
    @assert CUDA_SDK_jll.is_available()

    cudss = joinpath(CUDSS_jll.artifact_dir, "include")
    @assert CUDSS_jll.is_available()

    args = get_default_args()
    push!(args, "-I$cudss", "-I$cuda")

    # Clang.jl stumbles without stubbing out this attribute
    push!(args, "-D__device_builtin__=__attribute__((unused))")

    options = load_options(joinpath(@__DIR__, "cudss.toml"))

    # create context
    headers = ["$cudss/cudss.h"]
    targets = ["$cudss/cudss.h"]  # ["$cudss/cudss_distributed_interface.h", "$cudss/cudss_threading_interface.h"]
    ctx = create_context(headers, args, options)

    # run generator
    build!(ctx, BUILDSTAGE_NO_PRINTING)

    # Only keep the wrapped headers
    replace!(get_nodes(ctx.dag)) do node
        path = normpath(Clang.get_filename(node.cursor))
        should_wrap = any(targets) do target
            occursin(target, path)
        end
        if !should_wrap
            return ExprNode(node.id, Generators.Skip(), node.cursor, Expr[], node.adj)
        end
        return node
    end

    rewriter!(ctx, options)
    build!(ctx, BUILDSTAGE_PRINTING_ONLY)
    output_file = options["general"]["output_file_path"]
    format_file(output_file, YASStyle())

    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
