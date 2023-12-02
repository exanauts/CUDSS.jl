# cuDSS functions for managing the library

function cudssCreate()
    handle = Ref{cudssHandle_t}()
    check(CUDSS_STATUS_NOT_INITIALIZED) do
        unsafe_cudssCreate(handle)
    end
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

# cache for created, but unused handles
const idle_handles = CUDA.HandleCache{CuContext,cudssHandle_t}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cudssHandle_t, stream::CuStream}
    states = get!(task_local_storage(), :CUDSS) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            cudssCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    cudssDestroy(new_handle)
                end
            end
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
