# Multi-GPU tests for CUDSS
# These tests require at least 2 CUDA devices

function has_multiple_gpus()
    return CUDA.ndevices() >= 2
end

function cudss_mg_device_management()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    @testset "Device management functions" begin
        # Test devices!
        original_devices = CUDSS.devices()

        CUDSS.devices!([Cint(0), Cint(1)])
        @test CUDSS.devices() == [Cint(0), Cint(1)]
        @test CUDSS.ndevices() == 2

        CUDSS.devices!([Cint(1), Cint(0)])
        @test CUDSS.devices() == [Cint(0), Cint(1)]  # should be sorted
        @test CUDSS.ndevices() == 2

        # Single device
        CUDSS.devices!([Cint(0)])
        @test CUDSS.devices() == [Cint(0)]
        @test CUDSS.ndevices() == 1

        # Restore original
        CUDSS.devices!(original_devices)
    end
end

function cudss_mg_data_creation()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    @testset "Multi-GPU CudssData creation" begin
        # Test with 2 devices
        device_indices = [Cint(0), Cint(1)]
        CUDSS.devices!(device_indices)

        data = CudssData(device_indices)
        @test data isa CudssData

        # Test with explicit mg_handle
        CUDSS.devices!(device_indices)
        mg_handle = CUDSS.mg_handle()
        data2 = CudssData(mg_handle)
        @test data2 isa CudssData
    end
end

function cudss_mg_solver()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    n = 100
    device_indices = [Cint(0), Cint(1)]

    @testset "Multi-GPU solver: precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)

        @testset "General matrix (G)" begin
            # Create test problem
            A_cpu = sprand(T, n, n, 0.01) + I
            b_cpu = rand(T, n)
            x_cpu = zeros(T, n)

            # Move to GPU (device 0)
            A_gpu = CuSparseMatrixCSR(A_cpu)
            b_gpu = CuVector(b_cpu)
            x_gpu = CuVector(x_cpu)

            # Configure for multi-GPU
            CUDSS.devices!(device_indices)
            config = CudssConfig(device_indices)
            data = CudssData(device_indices)

            # Create solver
            matrix = CudssMatrix(A_gpu, "G", 'F')
            solver = CudssSolver(matrix, config, data)

            # Solve
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu)
            cudss("solve", solver, x_gpu, b_gpu)

            # Verify solution
            r_gpu = b_gpu - A_gpu * x_gpu
            @test norm(r_gpu) ≤ sqrt(eps(R)) * 100
        end

        @testset "Symmetric/Hermitian (S/H)" begin
            # Create symmetric/Hermitian problem
            A_cpu = sprand(T, n, n, 0.01)
            A_cpu = A_cpu + A_cpu' + I
            b_cpu = rand(T, n)
            x_cpu = zeros(T, n)

            # Move to GPU
            A_gpu = CuSparseMatrixCSR(A_cpu)
            b_gpu = CuVector(b_cpu)
            x_gpu = CuVector(x_cpu)

            # Configure for multi-GPU
            CUDSS.devices!(device_indices)
            config = CudssConfig(device_indices)
            data = CudssData(device_indices)

            # Create solver
            structure = T <: Real ? "S" : "H"
            matrix = CudssMatrix(A_gpu, structure, 'L')
            solver = CudssSolver(matrix, config, data)

            # Solve
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu)
            cudss("solve", solver, x_gpu, b_gpu)

            # Verify solution
            r_gpu = b_gpu - A_gpu * x_gpu
            @test norm(r_gpu) ≤ sqrt(eps(R)) * 100
        end

        @testset "Symmetric/Hermitian Positive Definite (SPD/HPD)" begin
            # Create SPD/HPD problem
            A_cpu = sprand(T, n, n, 0.01)
            A_cpu = A_cpu * A_cpu' + I
            b_cpu = rand(T, n)
            x_cpu = zeros(T, n)

            # Move to GPU
            A_gpu = CuSparseMatrixCSR(A_cpu)
            b_gpu = CuVector(b_cpu)
            x_gpu = CuVector(x_cpu)

            # Configure for multi-GPU
            CUDSS.devices!(device_indices)
            config = CudssConfig(device_indices)
            data = CudssData(device_indices)

            # Create solver
            structure = T <: Real ? "SPD" : "HPD"
            matrix = CudssMatrix(A_gpu, structure, 'L')
            solver = CudssSolver(matrix, config, data)

            # Solve
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu)
            cudss("solve", solver, x_gpu, b_gpu)

            # Verify solution
            r_gpu = b_gpu - A_gpu * x_gpu
            @test norm(r_gpu) ≤ sqrt(eps(R)) * 100
        end
    end
end

function cudss_mg_multiple_rhs()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    n = 100
    p = 5
    device_indices = [Cint(0), Cint(1)]

    @testset "Multi-GPU with multiple RHS: precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)

        # Create test problem
        A_cpu = sprand(T, n, n, 0.01) + I
        B_cpu = rand(T, n, p)
        X_cpu = zeros(T, n, p)

        # Move to GPU
        A_gpu = CuSparseMatrixCSR(A_cpu)
        B_gpu = CuMatrix(B_cpu)
        X_gpu = CuMatrix(X_cpu)

        # Configure for multi-GPU
        CUDSS.devices!(device_indices)
        config = CudssConfig(device_indices)
        data = CudssData(device_indices)

        # Create solver
        matrix = CudssMatrix(A_gpu, "G", 'F')
        solver = CudssSolver(matrix, config, data)

        # Solve
        cudss("analysis", solver, X_gpu, B_gpu)
        cudss("factorization", solver, X_gpu, B_gpu)
        cudss("solve", solver, X_gpu, B_gpu)

        # Verify solution for each RHS
        R_gpu = B_gpu - A_gpu * X_gpu
        @test norm(R_gpu) ≤ sqrt(eps(R)) * 100
    end
end

function cudss_mg_refactorization()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    n = 100
    device_indices = [Cint(0), Cint(1)]

    @testset "Multi-GPU refactorization: precision = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)

        # Create initial problem
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        b_cpu = rand(T, n)
        x_cpu = zeros(T, n)

        # Move to GPU
        A_gpu = CuSparseMatrixCSR(A_cpu)
        b_gpu = CuVector(b_cpu)
        x_gpu = CuVector(x_cpu)

        # Configure for multi-GPU
        CUDSS.devices!(device_indices)
        config = CudssConfig(device_indices)
        data = CudssData(device_indices)

        # Create solver
        structure = T <: Real ? "SPD" : "HPD"
        matrix = CudssMatrix(A_gpu, structure, 'L')
        solver = CudssSolver(matrix, config, data)

        # Initial solve
        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)
        cudss("solve", solver, x_gpu, b_gpu)

        # Verify initial solution
        r_gpu = b_gpu - A_gpu * x_gpu
        @test norm(r_gpu) ≤ sqrt(eps(R)) * 100

        # Update matrix and refactorize
        d_gpu = rand(R, n) |> CuVector
        A_gpu_new = A_gpu + Diagonal(d_gpu)
        cudss_update(solver, A_gpu_new)

        b_cpu2 = rand(T, n)
        b_gpu2 = CuVector(b_cpu2)
        fill!(x_gpu, zero(T))

        cudss("refactorization", solver, x_gpu, b_gpu2)
        cudss("solve", solver, x_gpu, b_gpu2)

        # Verify updated solution
        r_gpu2 = b_gpu2 - A_gpu_new * x_gpu
        @test norm(r_gpu2) ≤ sqrt(eps(R)) * 100
    end
end

function cudss_mg_views()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    n = 80
    device_indices = [Cint(0), Cint(1)]

    @testset "Multi-GPU with different views: precision = $T" for T in (Float64, ComplexF64)
        R = real(T)

        # Create SPD/HPD matrix
        A_cpu = sprand(T, n, n, 0.01)
        A_cpu = A_cpu * A_cpu' + I
        b_cpu = rand(T, n)

        structure = T <: Real ? "SPD" : "HPD"

        @testset "view = $view" for view in ('L', 'U', 'F')
            x_cpu = zeros(T, n)

            # Prepare matrix based on view
            if view == 'L'
                A_view = tril(A_cpu)
            elseif view == 'U'
                A_view = triu(A_cpu)
            else
                A_view = A_cpu
            end

            A_gpu = CuSparseMatrixCSR(A_view)
            b_gpu = CuVector(b_cpu)
            x_gpu = CuVector(x_cpu)

            # Configure for multi-GPU
            CUDSS.devices!(device_indices)
            config = CudssConfig(device_indices)
            data = CudssData(device_indices)

            # Create solver
            matrix = CudssMatrix(A_gpu, structure, view)
            solver = CudssSolver(matrix, config, data)

            # Solve
            cudss("analysis", solver, x_gpu, b_gpu)
            cudss("factorization", solver, x_gpu, b_gpu)
            cudss("solve", solver, x_gpu, b_gpu)

            # Verify solution using full matrix
            r_gpu = b_gpu - CuSparseMatrixCSR(A_cpu) * x_gpu
            @test norm(r_gpu) ≤ sqrt(eps(R)) * 100
        end
    end
end

function cudss_mg_task_isolation()
    if !has_multiple_gpus()
        @test_skip "Multi-GPU tests require at least 2 CUDA devices"
        return
    end

    @testset "Task-local device configuration" begin
        # Test that device configuration is task-local
        CUDSS.devices!([Cint(0)])
        @test CUDSS.ndevices() == 1

        task = @async begin
            CUDSS.devices!([Cint(0), Cint(1)])
            CUDSS.ndevices()
        end

        result = fetch(task)
        @test result == 2

        # Original task should still have single device
        @test CUDSS.ndevices() == 1
    end
end
