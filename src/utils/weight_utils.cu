#include "src/utils/weight_utils.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size)
{
    ONELLM_CHECK_WITH_INFO(size >= ((size_t)0), "Ask cudaMalloc size " + std::to_string(size) + "< 0 is invalid.");
    CHECK(cudaMalloc((void**)(ptr), sizeof(T) * size));
}
template void GPUMalloc(float** ptr, size_t size);
template void GPUMalloc(half** ptr, size_t size);

template<typename T>
void GPUFree(T* ptr)
{
    if (ptr != NULL) {
        CHECK(cudaFree(ptr));
        ptr = NULL;
    }
}
template void GPUFree(float* ptr);
template void GPUFree(half* ptr);