function LinearAlgebra.lu(A::CuSparseMatrixCSR{T}) where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  solver = CudssSolver(A, "G", 'F')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

function LinearAlgebra.ldlt(A::CuSparseMatrixCSR{T}; view::Char='F') where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  structure = T <: Real ? "S" : "H"
  solver = CudssSolver(A, structure, view)
  (T <: Complex) && cudss_set(solver, "pivot_type", 'N')
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

LinearAlgebra.ldlt(A::Symmetric{T,<:CuSparseMatrixCSR{T}}) where T <: BlasReal = LinearAlgebra.ldlt(A.data, view=A.uplo)
LinearAlgebra.ldlt(A::Hermitian{T,<:CuSparseMatrixCSR{T}}) where T <: BlasFloat = LinearAlgebra.ldlt(A.data, view=A.uplo)

function LinearAlgebra.cholesky(A::CuSparseMatrixCSR{T}; view::Char='F') where T <: BlasFloat
  n = LinearAlgebra.checksquare(A)
  structure = T <: Real ? "SPD" : "HPD"
  solver = CudssSolver(A, structure, view)
  x = CudssMatrix(T, n)
  b = CudssMatrix(T, n)
  cudss("analysis", solver, x, b)
  cudss("factorization", solver, x, b)
  return solver
end

LinearAlgebra.cholesky(A::Symmetric{T,<:CuSparseMatrixCSR{T}}) where T <: BlasReal = LinearAlgebra.cholesky(A.data, view=A.uplo)
LinearAlgebra.cholesky(A::Hermitian{T,<:CuSparseMatrixCSR{T}}) where T <: BlasFloat = LinearAlgebra.cholesky(A.data, view=A.uplo)

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
      return b
    end

    function LinearAlgebra.ldiv!(x::$type{T}, solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      cudss("solve", solver, x, b)
      return x
    end

    function Base.:\(solver::CudssSolver{T}, b::$type{T}) where T <: BlasFloat
      x = similar(b)
      ldiv!(x, solver, b)
      return x
    end
  end
end
