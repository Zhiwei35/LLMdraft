#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include <fstream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"

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
    // CHECK(cudaMemcpy(ptr, host_array.data(), host_array.size(), cudaMemcpyHostToDevice));
    return;
}
void loadWeights(float* ptr1, std::string weight_path, int shape0, int shape1) // weighttype参数比较多余
{
    // load attn output
    internalFunc(ptr1, {(size_t)shape0, (size_t)shape1}, weight_path);

}
void loadWeights_trans(float* ptr1, std::string weight_path, int shape0, int shape1) // weighttype参数比较多余
{
    // load attn output
    internalFunc(ptr1, {(size_t)shape0, (size_t)shape1}, weight_path);

}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
        if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
            printf("the %dth res is wrong, onellm = %f, trans = %f\n", i, CPUoutput[i], GPUoutput[i]);
        //         //return false;
        // } else {
	    // printf("the %dth res is right, onellm = %f, trans = %f\n", i, CPUoutput[i], GPUoutput[i]);
   	    }
    }
    return true;
}
///home/data/trans/q_buf_after_rope_trans.bin
///home/data/onellm/q_buf_after_rope.bin
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
    //int in_size = 0; // TO MODIFY
    int shape0 = 13; // TO MODIFY
    int shape1 = 32*128; // TO MODIFY
    
    int in_size = shape0 * shape1;
    hidden_units_2 = hidden_units * hidden_units;
    output_size = seqlen * hidden_units;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w = (float*)malloc(sizeof(float) * hidden_units_2);
    float* d_w_trans= (float*)malloc(sizeof(float) * hidden_units_2);
    h_w = (float*)malloc(sizeof(float) * hidden_units_2);
    // cudaMalloc((void**)&d_w, sizeof(float) * hidden_units_2);
    for(int i = 0; i < hidden_units_2; i++) { 
       h_w[i] = (float)(i % 3); // 1 2 1 2
    }

    float* h_in = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    float* d_in = (float*) malloc(sizeof(float) * in_size);
    float* d_in_trans = (float*) malloc(sizeof(float) * in_size);
    // cudaMalloc((void**)&d_in, sizeof(float) * seqlen *  hidden_units);
    for(int i = 0; i < hidden_units * seqlen; i++) { 
       h_in[i] = (float)(i % 3);
    }

    float* h_out = (float*) malloc(sizeof(float) * output_size);
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * output_size);
    loadWeights(d_in, "/home/data/trans/qk_v_buf_after_rm_pad.bin", shape0, shape1); // TO MODIFY
    loadWeights_trans(d_in_trans, "/home/data/trans/qk_v_buf_after_rm_pad_trans.bin", shape0, shape1); // TO MODIFY
    std::cout << "====intermediate tensor====" << "\n";
    CheckResult(d_in, d_in_trans, output_size);

    free(h_in);
    free(h_w);
    free(h_out);
    free(d_in);
    free(d_w);
    free(d_in_trans);
    free(d_w_trans);
    cudaFree(d_out);
}
