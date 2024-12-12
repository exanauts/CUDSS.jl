# CUDSS uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream
const cudaDataType_t = cudaDataType
const CUPTR_C_NULL = CuPtr{Ptr{Cvoid}}(0)