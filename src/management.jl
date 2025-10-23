# cuDSS functions for managing the library

function cudssCreate()
    handle = Ref{cudssHandle_t}()
    cudssCreate(handle)
    handle[]
end

function cudssCreateMg(device_indices::Vector{Cint})
    handle = Ref{cudssHandle_t}()
    device_count = length(device_indices)
    cudssCreateMg(handle, device_count |> Cint, device_indices)
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

## single-gpu handles

function handle_ctor(ctx)
    context!(ctx) do
        cudssCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cudssDestroy(handle)
    end
end

const idle_handles = HandleCache{CuContext,cudssHandle_t}(handle_ctor, handle_dtor)

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cudssHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUDSS) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle)
        end

        cudssSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cudssSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end

## multi-gpu handles

"""
    devices!(devs::Vector{Cint})

Set the device indices to use for multi-GPU operations in the current task.
The devices are stored in task-local storage and will be used by subsequent
calls to `mg_handle()`.

# Example
```julia
CUDSS.devices!([Cint(0), Cint(1), Cint(2)])  # Use GPUs 0, 1, 2
```
"""
function devices!(devs::Vector{Cint})
    task_local_storage(:CUDSS_MG_devices, sort(devs))
    return
end

"""
    devices()

Get the device indices configured for multi-GPU operations in the current task.
Returns a vector of device indices (as `Cint`). If not configured, defaults to
using only the first device (device 0).
"""
devices() = get!(task_local_storage(), :CUDSS_MG_devices) do
    # Default: use only first device
    [Cint(0)]
end::Vector{Cint}

"""
    ndevices()

Get the number of devices configured for multi-GPU operations in the current task.
"""
ndevices() = length(devices())

function mg_handle()
    cuda = CUDA.active_state()

    # every task maintains library state per set of devices
    # Note: Unlike cuSOLVER, cuDSS multi-GPU handles DO support stream setting.
    # The stream is set on the "primary" device (the current device), and
    # cuDSS manages internal streams on other devices for data transfers.
    LibraryState = @NamedTuple{handle::cudssHandle_t, stream::CuStream, devices::Vector{Cint}}
    states = get!(task_local_storage(), :CUDSS_MG) do
        Dict{UInt,LibraryState}()
    end::Dict{UInt,LibraryState}

    # derive a key from the active context and selected devices
    key = hash(cuda.context)
    for dev_idx in devices()
        # hash the device indices to distinguish different MG configurations
        key = hash(dev_idx, key)
    end

    # get library state
    @noinline function new_state(cuda)
        dev_indices = devices()

        # multi-GPU handles can't be reused for different device sets
        # so we create fresh handles rather than pooling
        new_handle = cudssCreateMg(dev_indices)

        finalizer(current_task()) do task
            context!(cuda.context; skip_destroyed=true) do
                cudssDestroy(new_handle)
            end
        end

        cudssSetStream(new_handle, cuda.stream)

        (; handle=new_handle, cuda.stream, devices=dev_indices)
    end
    state = get!(states, key) do
        new_state(cuda)
    end

    # update stream if changed
    @noinline function update_stream(cuda, state)
        cudssSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream, state.devices)
    end
    if state.stream != cuda.stream
        states[key] = state = update_stream(cuda, state)
    end

    # validate devices haven't changed (can't reuse handle if they have)
    if state.devices != devices()
        states[key] = state = new_state(cuda)
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
