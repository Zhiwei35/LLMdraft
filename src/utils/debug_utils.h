#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"
//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
template<typename T>
void save_tensor(TensorWrapper<T>* input, std::string filename){
    int Bm = 0;
    int Bk = 0;
    if (input->shape.size() == 4){
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3){
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 3){
        Bm = input->shape[0];
        Bk = input->shape[1];
    }
    T* icpu = (T*)malloc(sizeof(T) * Bm * Bk);
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);
    std::ofstream F;
    std::cout << "saving intermediate tensor in " << filename << "\n";
    F.open(filename, std::ofstream::binary);
    F.write(reinterpret_cast<const char*>(icpu), sizeof(T)*Bm*Bk);
    F.close();
}