add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(rmsnorm STATIC rmsnorm_kernel.cu)
set_property(TARGET rmsnorm PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsnorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsnorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)