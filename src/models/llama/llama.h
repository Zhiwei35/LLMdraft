#include "src/models/basemodel.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/llama/llama_weights.h"
#include "src/layers/decoder/context_decoder.h"
#include "src/layers/decoder/self_decoder.h"
#include "src/kernels/input_embedding.h"
#include "src/kernels/linear.h" //LM Head
#include "src/kernels/topK.h" //topK
#include "src/kernels/sampling.h" //sampling
#include "src/models/tokenizer.h"
template<typename T>
class Llama: public BaseModel{
private:
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int inter_size;
    const int num_layers;
    int vocab_size;
    int vocab_size_padded;
    float rmsnorm_eps = 1e-6f;   
    // const int start_id = 0; // from hf modeling_config
    // const int end_id = 2;// from hf modeling_config
    const int hidden_units; 
    const int max_seq_len;
    int output_token_limit = 256;
    int pad_token_id = 0;// from hf modeling_config 
    int bos_token_id = 1;
    int eos_token_id = 2;
    int layer_id = 0;
    int batch_size = 1; //tmp var, should included in dyn params
    int beamwidth = 1;
    int BlockPerBeam = 8;

    Tokenizer tokenizer;
    LlamaWeight<T>* llama_weights;
    LlamaSelfDecoder<T>* self_decoder;
    LlamaContextDecoder<T>* context_decoder;
    int max_context_token_num_ = 32;
    //int h_step;
    int K = 4;
    TensorWrapper<int>* step;
    TensorWrapper<T>* output_rmsnorm_weight;
    TensorWrapper<int>* layer;
    //T*   context_decoder_input_buf_{};   // CTXDEC
    TensorWrapper<T>* context_decoder_input;
    //T*   context_decoder_output_buf_{};  // CTXDEC
    TensorWrapper<T>* context_decoder_output;
    //int* context_decoder_ids_buf_{}; //这个倒没见过

    //T* decoder_input_buf_{};   // CTXDEC, GENERATE
    TensorWrapper<T>* decoder_input;
    //T* decoder_output_buf_{};  // CTXDEC, GENERATE
    TensorWrapper<T>* decoder_output;

    //int* input_ids_buf_{};       // input token ids, CTXDEC
    TensorWrapper<int>* input_ids;
    //int* input_length_buf_{};    // input length, CTXDEC, GENERATE
    TensorWrapper<int>* input_length;
    //int* history_length_buf_{};  // history length, CTXDEC
    TensorWrapper<int>* history_length;
    //int* context_length_buf_{};  // history length + input_length, CTXDEC, GENERATE
    TensorWrapper<int>* context_length;

    // float* logits_buf_{};        // combined logits
    // float* context_logits_buf_{};
    //int* total_padding_count_{};  // GENERATE
    TensorWrapper<T>* all_k_cache;
    TensorWrapper<T>* all_v_cache;

    // used by sampling
    IntDict int_params_of_sample;
    TensorWrapper<T>* probs;
    TensorWrapper<int>* token_ids;
    //int* token_ids_buf_{};   // all token IDs in [S, B], indexed using `step`
    //int* output_ids_buf_{};  // output ids in [B, S]
    TensorWrapper<int>* sequence_lengths; //record current sequence length in GENERATE
    //int* sequence_lengths_{};     // current sequence length，GENERATE
    //int*      end_ids_buf_{};
    TensorWrapper<bool>* is_finished;
    // TensorWrapper<T>* topk_workspace;
    TensorWrapper<int>* topk_id;
    TensorWrapper<T>* topk_val;
    TensorWrapper<int>* final_topk_id;
    TensorWrapper<T>* final_topk_val;

    // pinned or not pinned CPU buffers
    int* h_input_ids_buf_{};
    int* h_input_length_buf_{};
    int* h_history_length_buf_{};
    int* h_context_length_buf_{};
    int* h_sequence_lengths_{};
    bool* h_finished_buf_{};
    int* h_output_ids{};

public:
    Llama() = default;
    Llama(int head_num,
          int kv_head_num,
          int head_size,
          int inter_size,
          int num_layers,
          int vocab_size,
          const LLaMAAttentionStaticParams&  attn_static_params,
        // int                          max_batch_size,
        // int                          max_context_token_num,
          int max_seq_len,//session_len
          //int h_step,
        //for base model
          cudaStream_t stream,
          cublasWrapper* cublas_wrapper,
          BaseAllocator* allocator,
          cudaDeviceProp* cuda_device_prop):
    BaseModel(stream, cublas_wrapper, allocator, cuda_device_prop),
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    inter_size(inter_size),
    num_layers(num_layers),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    //h_step(h_step),
    hidden_units(head_num * head_size),
    max_seq_len(max_seq_len) {
        //int_params_of_sample.insert({"step", h_step});
        int_params_of_sample.insert({"vocab_size", vocab_size});
        int_params_of_sample.insert({"end_id", eos_token_id});
        layer = new TensorWrapper<int>(CPU, DataType::INT32, {1}, &layer_id);
        llama_weights = new LlamaWeight<T>(head_num,
                                          kv_head_num,
                                          head_size,
                                          inter_size,
                                          vocab_size,
                                          num_layers,
                                          /*attn_bias*/false,
                                          getWeightType<T>());

        self_decoder = new LlamaSelfDecoder<T>(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        num_layers,
                                        attn_static_params,
                                        rmsnorm_eps,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        is_free_buffer_after_forward);

        context_decoder = new LlamaContextDecoder<T>(head_num,
                                                    kv_head_num,
                                                    head_size,
                                                    inter_size,
                                                    num_layers,
                                                    attn_static_params,
                                                    rmsnorm_eps,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator,
                                                    is_free_buffer_after_forward);
        //only need to allocate buffer in initialize llama class
        //and the buffer value change can be finished by CUDA kernel
        //so we dont need to reallocate in multi epoch conversation
        //do the 3 "initializes" function can write in constructor?
        allocateCPUBuffer(1); // bs = 1
        allocateGPUBuffer(1);
    }

    ~Llama() {
        this->free();
    };
    void loadTokenizer(std::string file){
      tokenizer.Initialize(file);
    }
    void loadWeights(std::string file){
      llama_weights->loadWeights(file);
    }
    void loadWeightsFromDummy(){
      llama_weights->loadWeightsFromDummy();
    }
    void allocateCPUBuffer(int max_batch_size);
    void allocateGPUBuffer(int batch_size);
    void free();
    //weights在common_utils里面已经load好了
    //void loadWeights(std::string file);

    std::tuple<std::string, int, int> MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前轮次回复更新history
    // single request response
    std::string Response(const std::tuple<std::string, int, int>& input, CallBack PrintRes);

    //copy token ids to CPU(h_token_ids), 暂时不需要，因为反正也是bs=1
    int MakeOutput();

    void inputEmbedding(TensorWrapper<int>* input_ids, TensorWrapper<T>* decoder_input);

    void InitializeForContextDecoder(IntDict& int_params_first_token);
    int firstTokenGen(LLaMAAttentionDynParams& dparams, IntDict& int_params_first_token);
    void InitializeForSelfDecoder();
    int continueTokenGen(LLaMAAttentionDynParams& dparams);
    int LMHeadAndTopKSample(TensorMap& decoder_outputs);
};