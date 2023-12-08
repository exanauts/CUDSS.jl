function LinearAlgebra.lu(A::CuSparseMatrixCSR{T}) where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  solver = CudssSolver(A, "G", 'F')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

function LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T}) where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  structure = T <: Real ? "S" : "H"
  solver = CudssSolver(A, structure, 'F')
  (T <: Complex) && cudss_set(solver, "pivot_type", 'N')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

function LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T}) where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  structure = T <: Real ? "SPD" : "HPD"
  solver = CudssSolver(A, structure, 'F')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

for fun in (:lu!, :ldlt!, :cholesky!)
  @eval begin
    function LinearAlgebra.$fun(solver::CudssSolver{T}, A::CuSparseMatrixCSR{T}) where T <: BlasFloat
      n = LinearAlgebra.checksquare(A)
      cudss_set(solver.matrix, A)
      x = CudssMatrix(T, n)
      b = CudssMatrix(T, n)
      cudss("factorization", solver, x, b)
      return solver
    end
  end
end

for type in (:CuVector, :CuMatrix)
  @eval begin
    function LinearAlgebra.ldiv!(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, b, b)
    end

    function LinearAlgebra.ldiv!(x::$type{T}, solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, x, b)
    end
  end
end
