#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include "src/utils/macro.h"
#include "src/kernels/linear.h"
#include "src/weights/base_weights.h"

std::vector<float> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    size_t dim0 = 1, dim1 = 1;
    if (shape.size() > 2) {
        dim0 = shape[0] * shape[1];
        dim1 = shape[2];
    }
    
    if (shape.size() == 2) {
        dim0 = shape[0];
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<float>();
    }

    std::vector<float> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "file" << filename << "cannot be opened, loading model fails!" << std::endl;
        return std::vector<float>();
    }

    size_t loaded_data_size = sizeof(float) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    std::cout << "Read " << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<float>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}
void internalFunc(float* ptr, std::vector<size_t> shape, std::string filename) {
    std::vector<float> host_array = loadWeightFromBinHelper(shape, filename);
    if (host_array.empty()) {
        std::cout << "[warning] data from file is empty!!" << "\n";
        return;
    }
    memcpy(ptr, host_array.data(), host_array.size());
    //CHECK(cudaMemcpy(ptr, host_array.data(), host_array.size(), cudaMemcpyHostToDevice));
    return;    
}
void loadWeights(float* ptr, float* ptr1, std::string weight_path) // weighttype参数比较多余
{
    // load out linear weight
    internalFunc(ptr, {8, 16}, weight_path + "weight.bin");
    // load attn output
    internalFunc(ptr1, {8, 16}, weight_path + "in.bin");

}
void loadWeights_trans(float* ptr, float* ptr1, std::string weight_path) // weighttype参数比较多余
{
    // load out linear weight
    internalFunc(ptr, {8, 16}, weight_path + "model.layers.0.self_attn.o_proj.weight.bin");
    // load attn output
    internalFunc(ptr1, {8, 16}, weight_path + "attn_output.bin");

}

void CPUlinear(float* input, float* weight, float* output,
                int m, int k, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                output[i * n + j] += input[i * k + l] * weight[l * n + j];
            }
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
  //  for(int i = 0; i < output_size; i++) {
	//if (i < 5) {
//	printf("0th res, CPUoutput = %f, GPUoutput = %f\n", CPUoutput[i], GPUoutput[i]);
	//}
  	//if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
        //    printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
        //    return false;
        //}

    //}
    return true;
}
//(right)2 fusedGateUpGemm / down =>{seqlen, hidden_units} * {2 * inter_size, hidden_units} = [16, 16] * [10*2, 16]
//(right)1 trans b => {seqlen, hidden_units} * {vocab_size, hidden_units} = [16, 16] * [32, 16]
//(right)0 most cases => {seqlen, hidden_units} * {hidden_units, hidden_units} = [16, 16] * [16, 16]
int main(int argc, char *argv[]) {
    const int seqlen = 13;
    const int hidden_units = 4096;
    const int vocab_size = 32;
    const int inter_size = 10;
    int hidden_units_2 = 0;
    int output_size = 0;
    if (atoi(argv[1]) == 2) { // fusedGateUpGemm and trans_b
        hidden_units_2 = 2 * inter_size * hidden_units;
        output_size = 2 * seqlen * inter_size;
    } else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear and down linear
        hidden_units_2 = vocab_size * hidden_units;
        output_size = seqlen * vocab_size;
    } else {
        hidden_units_2 = hidden_units * hidden_units;
        output_size = seqlen * hidden_units;
    
    }
    std::cout << "bbbbb" << "\n";
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w = (float*)malloc(sizeof(float) * 128);
    h_w = (float*)malloc(sizeof(float) * hidden_units_2);
    //cudaMalloc((void**)&d_w, sizeof(float) * hidden_units_2);
    for(int i = 0; i < hidden_units_2; i++) { 
       h_w[i] = (float)(i % 3); // 0 1 2 0 1 2
    }

    float* h_in = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    float* d_in = (float*)malloc(sizeof(float) * 128);
    //cudaMalloc((void**)&d_in, sizeof(float) * seqlen *  hidden_units);
    for(int i = 0; i < hidden_units * seqlen; i++) { 
       h_in[i] = (i / hidden_units) % 2 == 0 ? 0 : 1 ; // 
    }

    float* h_out = (float*) malloc(sizeof(float) * output_size);
    float* d_out;
    float* d_w_my;
    float* d_in_my;
    d_w_my = (float*)malloc(sizeof(float) * 128);
    d_in_my = (float*)malloc(sizeof(float) * 128);
    
    cudaMalloc((void**)&d_out, sizeof(float) * output_size);
    //std::cout << "aaaaaa" << "\n";
    //loadWeights_trans(d_w, d_in, "/home/");
    loadWeights(d_w_my, d_in_my, "/home/");
    for (int i = 0; i < 128; i++){
        printf("weight_trans = %f, my_weight = %f\n", d_w[i], d_w_my[i]);
	printf("in_trans = %f, my_in = %f\n", d_in[i], d_in_my[i]);
    }

    //disable due to attn out linear test
    //CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * hidden_units * seqlen, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_w, h_w, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));
    //DataType type = getTensorType<float>();
    //WeightType wtype = getWeightType<float>(); 
    //TensorWrapper<float>* in = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_in);
    //BaseWeight<float> weight;
    //if (atoi(argv[1]) == 2) {
    //    weight.shape = {2 * inter_size, hidden_units};
    //} else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear
    //    weight.shape = {vocab_size, hidden_units};
    //} else {
    //    weight.shape = {hidden_units, hidden_units};
   // }
    //loadWeights(d_w, d_in, );
    //weight.data = d_w;
    //weight.type = wtype;
    //TensorWrapper<float>* out;
    //if (atoi(argv[1]) == 2) {
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, 2, inter_size}, d_out);
    //} else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, vocab_size}, d_out);
    //} else {
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_out);
   // }
    //cublasHandle_t cublas_handle;
    //cublasLtHandle_t cublaslt_handle;
    //cublasCreate(&cublas_handle);
    //cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    //cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);  
    //cublas_wrapper->setFP32GemmConfig();  
    //printf("weight_trans = %f, my_weight = %f\n", d_w, d_w_my);# } else {
    //    weight.shape = {hidden_units, hidden_units};
    //}
    //loadWeights(d_w, d_in, );
    //weight.data = d_w;
    //weight.type = wtype;
    //TensorWrapper<float>* out;
    //if (atoi(argv[1]) == 2) {
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, 2, inter_size}, d_out);
    //} else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, vocab_size}, d_out);
    //} else {
    //    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_out);
    //}
    //std::cout << "before launch kernel" << std::endl;
    //if (atoi(argv[1]) == 2) {
    //    launchLinearGemm(in, weight, out, cublas_wrapper, false, true);
    //} else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear
    //    launchLinearGemm(in, weight, out, cublas_wrapper, false, true);
    //} else {
    //    launchLinearGemm(in, weight, out, cublas_wrapper, false, true);
   // }
    // debug info, better to retain:
    //std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain:
    //std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    //CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
    //float* CPUout = (float*) malloc(sizeof(float) * output_size);
    //DEbug info, better to retain: 
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    //CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
    //float* CPUout = (float*) malloc(sizeof(float) * output_size);
    //if (atoi(argv[1]) == 2) {
    //    CPUlinear(h_in, h_w, CPUout, seqlen, hidden_units, 2 * inter_size);
    //} else if (atoi(argv[1]) == 1) {// enable trans_b for test lmhead linear
    //    CPUlinear(h_in, h_w, CPUout, seqlen, hidden_units, vocab_size);
    //} else {
    //    CPUlinear(h_in, h_w, CPUout, seqlen, hidden_units, hidden_units+1);
    //} 
    
    //bool is_right = CheckResult(CPUout, h_out, output_size);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "linear passed" << std::endl;
    free(h_in);
    free(h_w);
    free(h_out);
    //free(CPUout);
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}
