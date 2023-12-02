struct CUDSSError <: Exception
    code::cudssStatus_t
end

Base.convert(::Type{cudssStatus_t}, err::CUDSSError) = err.code

Base.showerror(io::IO, err::CUDSSError) =
    print(io, "CUDSSError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

name(err::CUDSSError) = string(err.code)

## COV_EXCL_START
function description(err)
    if err.code == CUDSS_STATUS_SUCCESS
        return "the operation completed successfully"
    elseif err.code == CUDSS_STATUS_NOT_INITIALIZED
        return "the library was not initialized"
    elseif err.code == CUDSS_STATUS_ALLOC_FAILED
        return "the resource allocation failed"
    elseif err.code == CUDSS_STATUS_INVALID_VALUE
        return "an invalid value was used as an argument"
    elseif err.code == CUDSS_STATUS_NOT_SUPPORTED
        return "a parameter is not supported"
    elseif err.code == CUDSS_STATUS_ARCH_MISMATCH
        return "an absent device architectural feature is required"
    elseif err.code == CUDSS_STATUS_EXECUTION_FAILED
        return "the GPU program failed to execute"
    elseif err.code == CUDSS_STATUS_INTERNAL_ERROR
        return "an internal operation failed"
    elseif err.code == CUDSS_STATUS_ZERO_PIVOT
        return "zero pivots were encountered during the numerical factorization"
    else
        return "no description for this error"
    end
end

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUDSS_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUDSSError(res))
    end
end

function check(f, errs...)
    res = retry_reclaim(in((CUDSS_STATUS_ALLOC_FAILED, errs...))) do
        return f()
    end

    if res != CUDSS_STATUS_SUCCESS
        throw_api_error(res)
    end

    return
end
