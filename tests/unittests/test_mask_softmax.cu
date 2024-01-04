#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <math.h>
#include "src/kernels/attn_softmax_kernel.h"

#define TEST_MASKED_SOFTMAX(dtype)                                                                                                  \
    dtype *h_qk;                                                                                                                    \
    dtype *d_qk;                                                                                                                    \
    h_qk = (dtype *)malloc(sizeof(dtype) * qk_size);                                                                                \
    cudaMalloc((void **)&d_qk, sizeof(dtype) * qk_size);                                                                            \
    dtype *h_score;                                                                                                                 \
    dtype *d_score;                                                                                                                 \
    h_score = (dtype *)malloc(sizeof(dtype) * qk_size);                                                                             \
    cudaMalloc((void **)&d_score, sizeof(dtype) * qk_size);                                                                         \
    uint8_t *h_mask;                                                                                                                \
    uint8_t *d_mask;                                                                                                                \
    h_mask = (uint8_t *)malloc(sizeof(uint8_t) * batch_size * q_length * k_length);                                                 \
    cudaMalloc((void **)&d_mask, sizeof(uint8_t) * batch_size * q_length * k_length);                                               \
    for (int i = 0; i < qk_size; i++)                                                                                               \
    {                                                                                                                               \
        h_qk[i] = 4.0f;                                                                                                             \
    }                                                                                                                               \
    for (int i = 0; i < batch_size * q_length * k_length; i++)                                                                      \
    {                                                                                                                               \
        h_mask[i] = (uint8_t)(1);                                                                                                   \
    }                                                                                                                               \
    cudaMemcpy(d_qk, h_qk, sizeof(dtype) * qk_size, cudaMemcpyHostToDevice);                                                        \
    cudaMemcpy(d_mask, h_mask, sizeof(uint8_t) * batch_size * q_length * k_length, cudaMemcpyHostToDevice);                         \
    DataType type = getTensorType<dtype>();                                                                                         \
    TensorWrapper<dtype> *qk = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_qk);       \
    TensorWrapper<dtype> *mask = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, q_length, k_length}, d_mask);             \
    TensorWrapper<dtype> *score = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_score); \
    std::cout << "before launch softmax kernel" << std::endl;                                                                       \
    launchScaleMaskAndSoftmax(qk, mask, score, scale);                                                                              \
    std::cout << "after launch softmax kernel" << std::endl;                                                                        \
    std::cout << "cuda memcpy device to host" << std::endl;                                                                         \
    cudaMemcpy(h_score, score->data, sizeof(dtype) * qk_size, cudaMemcpyDeviceToHost);

int main(int argc, char *argv[])
{
    const int batch_size = 1;
    const int head_num = 2;
    const int q_length = 8;
    const int k_length = 8;
    const int head_size = 4;
    float scale = rsqrtf(float(head_size));
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int qk_size = batch_size * head_num * q_length * k_length;
    if (argv[1]) {
        TEST_MASKED_SOFTMAX(half);
    } else {
        TEST_MASKED_SOFTMAX(float);
    }
    for (int i = 0; i < qk_size; i++)
    {
        printf("attn score[%d] = %f\n", i, (float)h_score[i]);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_qk);
    free(h_score);
    free(h_mask);
    cudaFree(d_qk);
    cudaFree(d_score);
    cudaFree(d_mask);
}