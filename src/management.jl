# cuDSS functions for managing the library
#
# This module follows the same handle management pattern as CUDA.jl's built-in libraries
# (CUBLAS, CUSPARSE, CUSOLVER, etc.) to ensure seamless integration with CUDA.jl's
# task-based concurrency model.
#
# Key design principles:
# - Each Julia Task gets its own CUDSS handle (via task_local_storage)
# - Handles are cached per CuContext and reused across tasks
# - Stream changes are automatically detected and propagated to cuDSS
# - This enables safe concurrent usage of CUDSS from multiple tasks

function cudssCreate()
    handle = Ref{cudssHandle_t}()
    cudssCreate(handle)
    handle[]
end

function cudssCreateMg(device_count, device_indices)
    handle = Ref{cudssHandle_t}()
    cudssCreateMg(handle, device_count, device_indices)
    handle[]
end

function cudssGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cudssGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cudssGetProperty(CUDA.MAJOR_VERSION),
                          cudssGetProperty(CUDA.MINOR_VERSION),
                          cudssGetProperty(CUDA.PATCH_LEVEL))

## handles
#
# Handle lifecycle:
# 1. Handles are created on-demand when a task first uses CUDSS
# 2. Each handle is associated with a specific CuContext
# 3. When a task completes, its handle is returned to the cache for reuse
# 4. Handles are destroyed when the context is destroyed or under memory pressure

# Constructor: creates a new cuDSS handle within the given CUDA context
function handle_ctor(ctx)
    context!(ctx) do
        cudssCreate()
    end
end

# Destructor: destroys a cuDSS handle within its associated context
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cudssDestroy(handle)
    end
end

# Cache of idle handles, keyed by CuContext. Handles are reused across tasks
# to avoid the overhead of creating/destroying handles frequently.
const idle_handles = HandleCache{CuContext,cudssHandle_t}(handle_ctor, handle_dtor)

"""
    handle()

Get the cuDSS handle for the current task and CUDA context.

This function implements CUDA.jl-compatible task-based concurrency:
- Each Julia Task maintains its own library state via `task_local_storage()`
- The handle is automatically configured to use the current task's CUDA stream
- Stream changes (e.g., via `CUDA.stream!()`) are detected and applied automatically

This follows the same pattern as CUBLAS.handle(), CUSPARSE.handle(), etc.,
ensuring consistent behavior across all CUDA libraries.
"""
function handle()
    # Get the current CUDA state for this task (device, context, stream, math_mode)
    # CUDA.jl creates one stream per task per device automatically
    cuda = CUDA.active_state()

    # Task-local storage: each task maintains its own Dict of context -> library state
    # This ensures that concurrent tasks don't interfere with each other
    LibraryState = @NamedTuple{handle::cudssHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUDSS) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # Create new state for this context if needed
    @noinline function new_state(cuda)
        # Get a handle from the cache (or create a new one)
        new_handle = pop!(idle_handles, cuda.context)

        # When this task finishes, return the handle to the cache for reuse
        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        # Configure the handle to use this task's stream
        cudssSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # Detect stream changes (e.g., user called CUDA.stream!() or we're in a different task)
    # and update the handle accordingly
    @noinline function update_stream(cuda, state)
        cudssSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

# Create a batch of pointers in device memory from a batch of dense vectors or matrices
@inline function unsafe_cudss_batch(batch::Vector{<:CuArray{T}}) where T <: BlasFloat
    cuPtrs = Base.unsafe_convert.(CuPtr{Cvoid}, batch) |> CuVector
    return cuPtrs
end

# Create a batch of pointers in device memory from a batch of sparse matrices
@inline function unsafe_cudss_batch(batch::Vector{CuSparseMatrixCSR{T,INT}}) where {T <: BlasFloat, INT <: CudssInt}
    rowPtrs = [Base.unsafe_convert(CuPtr{Cvoid}, Aᵢ.rowPtr) for Aᵢ in batch] |> CuVector
    colVals = [Base.unsafe_convert(CuPtr{Cvoid}, Aᵢ.colVal) for Aᵢ in batch] |> CuVector
    nzVals = [Base.unsafe_convert(CuPtr{Cvoid}, Aᵢ.nzVal) for Aᵢ in batch] |> CuVector
    return rowPtrs, colVals, nzVals
end
