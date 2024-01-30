#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/kernels/cublas_utils.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"
//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
template<typename T>
void save_out_linear_i_w(TensorWrapper<T>* input, BaseWeight<T>& weight){
    int Bm = input->shape[0];
    int Bk = input->shape[1] * input->shape[2];
    T* icpu = (T*)malloc(sizeof(T) * Bm * Bk);
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);
    std::ofstream ouF1;
    ouF1.open("./in.bin", std::ofstream::binary);
    ouF1.write(reinterpret_cast<const char*>(icpu), sizeof(T)*Bn*Bk);
    ouF1.close();
    // std::cout << "called gemm" << "\n";
}

template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,
                      BaseWeight<T>& weight, 
                      TensorWrapper<T>* output,
                      cublasWrapper* cublas_wrapper,
                      bool trans_a = false,
                      bool trans_b = false);
template<typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T>* input1,
                                  TensorWrapper<T>* input2,
                                  TensorWrapper<T>* output,
                                  cublasWrapper* cublas_wrapper,
                                  bool trans_a = false,
                                  bool trans_b = false);
