#include <iostream>
#include "src/kernels/act_kernel.h"

template<typename T>
__device__ __forceinline__ T silu(const T& in) {
  // x * sigmoid(x)
  return (T) (((float) in) / (1.0f + expf((float) -in)));
}

template<>
__device__ __forceinline__ half2 silu<half2>(const half2& in) {
  return make_half2(__float2half(silu<float>((float)(in.x))), __float2half(silu<float>((float)(in.y))));
}
//第一个intermediate size去做silu，结果与第二个intermediate mul
template<typename T>
__global__ void silu_and_mul_kernel(
  T* out,               // [bs, intermedia size]
  const T* input,       // [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
    const T x = input[batch_idx * 2 * intermedia_size + idx];
    const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
    out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
  }
}

template<>
__global__ void silu_and_mul_kernel<half>(
  half* out,               // [bs, intermedia size]
  const half* input,       // [2, bs, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  int vec_size = Vec<half>::size;
  using Vec_t = typename Vec<half>::Type;
  // Vec_t x_vec; 
  for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
    const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + idx]));
    const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]));
    *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu<Vec_t>(x), y);
  }
}

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    int batch_size = input->shape[1];
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);
    // std::cout << "calling silu_and_mul kernel" << "\n";
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
    // std::cout << "called silu_and_mul kernel" << "\n";
}
// We must instancite the template, if not, will report linking issue
template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);
