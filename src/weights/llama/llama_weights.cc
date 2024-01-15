#include <iostream>
#include "src/weights/llama/llama_weights.h"
template<typename T>
LlamaWeight<T>::LlamaWeight(
    int     head_num,
    int     kv_head_num,
    int     head_size,
    int     inter_size,
    int     vocab_size,
    int     num_layer,
    bool    attn_bias,
    WeightType weight_type
):
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    num_layer(num_layer),
    weight_type(weight_type) 
{
    llama_layer_weight.reserve(num_layer);
    for (int l = 0; l < num_layer; ++l) {
        llama_layer_weight.push_back(new LlamaLayerWeight<T>(head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            inter_size,
                                                            weight_type,
                                                            //group_size,
                                                            attn_bias));
    }
    GPUMalloc(&out_rmsnorm_weight.gamma, hidden_units);
    GPUMalloc(&post_decoder_embedding_weight.data, vocab_size * hidden_units);
    GPUMalloc(&pre_decoder_embedding_weight.data, vocab_size * hidden_units);
    pre_decoder_embedding_weight.shape = {vocab_size, hidden_units};
    post_decoder_embedding_weight.shape = {hidden_units, vocab_size};
    pre_decoder_embedding_weight.type = weight_type;
    post_decoder_embedding_weight.type = weight_type;
}
template<typename T>
void LlamaWeight<T>::loadWeights(std::string weight_path) {
    //weight_path += '/';
    std::cout << "the weight path is " << weight_path <<"\n";
    // weight from HF is always half type
    loadWeightFromBin<T, half>::internalFunc(out_rmsnorm_weight.gamma, {hidden_units}, weight_path + "model.norm.weight.bin");
    std::cout << "loaded norm weight " << weight_path <<"\n";
    loadWeightFromBin<T, half>::internalFunc(post_decoder_embedding_weight.data, {hidden_units, vocab_size}, weight_path + "lm_head.weight.bin");
    std::cout << "loaded lmhead weight " << weight_path <<"\n";
    loadWeightFromBin<T, half>::internalFunc(pre_decoder_embedding_weight.data, {vocab_size, hidden_units}, weight_path + "model.embed_tokens.weight.bin");
    std::cout << "loaded embed tokens weight " << weight_path <<"\n";
    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeights(weight_path + "model.layers." + std::to_string(layer), weight_type);
    }
}

template<typename T>
void LlamaWeight<T>::loadWeightsFromDummy() {
    T* d_dummy_out_rmsnorm_weight_gamma; 
    T* d_dummy_post_decoder_embedding_weight; 
    T* d_dummy_pre_decoder_embedding_weight;
    GPUMalloc(&d_dummy_out_rmsnorm_weight_gamma, sizeof(T) * hidden_units);
    GPUMalloc(&d_dummy_post_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size);
    GPUMalloc(&d_dummy_pre_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size);
    T* h_dummy_out_rmsnorm_weight_gamma = (T*)malloc(sizeof(T) * hidden_units); 
    T* h_dummy_post_decoder_embedding_weight = (T*)malloc(sizeof(T) * hidden_units * vocab_size); 
    T* h_dummy_pre_decoder_embedding_weight = (T*)malloc(sizeof(T) * hidden_units * vocab_size);  
    for (int i = 0; i < hidden_units; i++){
        h_dummy_out_rmsnorm_weight_gamma[i] = (T)1.0f;
    }
    for (int i = 0; i < hidden_units * vocab_size; i++) {
        h_dummy_post_decoder_embedding_weight[i] = (T)1.0f; 
        h_dummy_pre_decoder_embedding_weight[i] = (T)1.0f; 
    } 
    cudaMemcpy(d_dummy_out_rmsnorm_weight_gamma, h_dummy_out_rmsnorm_weight_gamma, sizeof(T) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_post_decoder_embedding_weight, h_dummy_post_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_pre_decoder_embedding_weight, h_dummy_pre_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size, cudaMemcpyHostToDevice);

    out_rmsnorm_weight.gamma = d_dummy_out_rmsnorm_weight_gamma;
    post_decoder_embedding_weight.data = d_dummy_post_decoder_embedding_weight;
    pre_decoder_embedding_weight.data = d_dummy_pre_decoder_embedding_weight;
    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeights();
    }
}

template<typename T>
LlamaWeight<T>::~LlamaWeight()
{
    cudaFree(pre_decoder_embedding_weight.data);
    cudaFree(out_rmsnorm_weight.gamma);
    cudaFree(post_decoder_embedding_weight.data);

    for (auto& p : llama_layer_weight) {
        delete p;
    }
}

template struct LlamaWeight<float>;
template struct LlamaWeight<half>;
