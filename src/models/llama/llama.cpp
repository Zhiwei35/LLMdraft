// def __init__(
//     self,
//     vocab_size=32000,
//     hidden_size=4096,
//     intermediate_size=11008,
//     num_hidden_layers=32,
//     num_attention_heads=32,
//     hidden_act="silu",
//     max_position_embeddings=2048,
//     initializer_range=0.02,
//     rms_norm_eps=1e-6,
//     use_cache=True,
//     pad_token_id=0,
//     bos_token_id=1,
//     eos_token_id=2,
//     tie_word_embeddings=False,
//     **kwargs,
// ):
#include "src/models/llama/llama.h"

size_t RoundUpTo32x(size_t size)
{
    return ((size + 31) / 32) * 32;
}

// cpu pinned buffer
template <typename T>
void Llama<T>::allocateCPUBuffer(int max_batch_size)
{
    // 本来这里长度是session_len=4096, 定义在lmdeploy/model.py
    // session_len即max_output_len,max_seq_len,output_token_limit
    // tokenids->output ids, [s,b]=>[b,s]，此处因为没有batch，先省掉
    //  output_ids_buf_ = (int*)allocator->Malloc(output_ids_buf_, sizeof(int) * max_batch_size * output_token_limit, true);
    //  h_input_ids_buf_ =
    //      allocator->Malloc(h_input_ids_buf_, sizeof(int) * max_batch_size * max_seq_len, true);
    h_input_ids_buf_ =
        allocator->Malloc(h_input_ids_buf_, sizeof(int) * 16, true);
    h_input_length_buf_ =
        allocator->Malloc(h_input_length_buf_, sizeof(int) * max_batch_size, true);
    h_history_length_buf_ =
        allocator->Malloc(h_history_length_buf_, sizeof(int) * max_batch_size, true);
    h_context_length_buf_ =
        allocator->Malloc(h_context_length_buf_, sizeof(int) * max_batch_size, true);
    h_sequence_lengths_ =
        allocator->Malloc(h_sequence_lengths_, sizeof(int) * max_batch_size, true);
    // h_k_cache_ptr_buf_ =
    //     (uint64_t*)allocator->Malloc(h_k_cache_ptr_buf_, sizeof(T) * max_batch_size, true, true);
    // h_v_cache_ptr_buf_ =
    //     (uint64_t*)allocator->Malloc(h_v_cache_ptr_buf_, sizeof(T) * max_batch_size, true, true);
    h_finished_buf_ = allocator->Malloc(h_finished_buf_, sizeof(bool) * max_batch_size, true);
    for (int i = 0; i < max_batch_size; i++)
    {
        h_finished_buf_[i] = 0;
    }
    h_output_ids = allocator->Malloc(h_output_ids, sizeof(int) * max_batch_size, true);
    // h_seq_limit_len_ =
    //     (uint32_t*)allocator->Malloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, true);
}

// 后续剩余工作：
// 1. tokenizer目前有问题，试着加上fastllm.cpp/WeightMap::LoadFromFile(const std::string &fileName)这一段代码和torch2flm.py处理vocab的片段
// 2. loadweight按照tensorrt llm llama转换脚本去转换
// gpu buffer
template <typename T>
void Llama<T>::allocateGPUBuffer(int batch_size)
{
    step = new TensorWrapper<int>(CPU, getTensorType<int>(), {1});
    layer = new TensorWrapper<int>(CPU, getTensorType<int>(), {1}, &layer_id);
    context_decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 16, hidden_units});
    context_decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 16, hidden_units});
    // split from context_decoder_output
    context_decoder_lmhead_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 1, hidden_units});
    // TODO: self decoder tmply not consider batch size dim
    decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {1, hidden_units});
    decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {1, hidden_units});
    input_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {16}); //{batch_size, max_seq_len});//这里的seqlen应该是padding前的
    input_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    history_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    context_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    sequence_lengths = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    all_k_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    all_v_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    token_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    is_finished = new TensorWrapper<bool>(GPU, getTensorType<bool>(), {batch_size});
    output_rmsnorm_weight = new TensorWrapper<T>(GPU, getTensorType<T>(), {hidden_units}, llama_weights->out_rmsnorm_weight.gamma);
    probs = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, vocab_size});

    context_decoder_input->data =
        allocator->Malloc(context_decoder_input->data, sizeof(T) * 16 * hidden_units, false); // 512x4x32
    context_decoder_output->data =
        allocator->Malloc(context_decoder_output->data, sizeof(T) * 16 * hidden_units, false);
    // context_decoder_ids->data =
    //     (int*)allocator->Malloc(context_decoder_ids->data, sizeof(int) * 16, false);
    context_decoder_lmhead_input->data =
        allocator->Malloc(context_decoder_lmhead_input->data, sizeof(T) * 1 * hidden_units, false);
    decoder_input->data = allocator->Malloc(decoder_input->data, sizeof(T) * batch_size * hidden_units, false); // 4x32
    decoder_output->data = allocator->Malloc(decoder_output->data, sizeof(T) * batch_size * hidden_units, false);

    input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 16, false); // batch_size * max_seq_len, false);//4x100,进位到32x为416
    input_length->data = allocator->Malloc(input_length->data, sizeof(int) * batch_size, false);
    history_length->data = allocator->Malloc(history_length->data, sizeof(int) * batch_size, false);
    context_length->data = allocator->Malloc(context_length->data, sizeof(int) * batch_size, false);
    sequence_lengths->data = allocator->Malloc(sequence_lengths->data, sizeof(int) * batch_size, false);

    all_k_cache->data = allocator->Malloc(all_k_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size, false); // 4x2x100x2x8
    all_v_cache->data = allocator->Malloc(all_v_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size, false);

    // logits_buf_       = (float*)allocator->Malloc(logits_buf_, sizeof(float) * batch_size * vocab_size, false);
    // 输出id buffer
    token_ids->data = allocator->Malloc(token_ids->data, sizeof(int) * batch_size, false); // 4,,进位到32x为32

    // end_ids_buf_   = allocator->Malloc(end_ids_buf_, sizeof(int) * batch_size, false);
    is_finished->data = allocator->Malloc(is_finished->data, sizeof(bool) * batch_size, false); // 4,,进位到32x为32
    // seq_limit_len_ = (uint32_t*)allocator->Malloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);
    probs->data = allocator->Malloc(probs->data, sizeof(T) * batch_size * vocab_size, false);
    // 两个中间top ids和vals，和两个final topk ids和vals
    topk_id = new TensorWrapper<int>(GPU, getTensorType<int>(),
                                     {batch_size, beamwidth, BlockPerBeam, K});
    topk_val = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, beamwidth, BlockPerBeam, K});
    final_topk_id = new TensorWrapper<int>(GPU, getTensorType<int>(),
                                           {batch_size * beamwidth, K});
    final_topk_val = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size * beamwidth, K});
    topk_id->data = allocator->Malloc(topk_id->data, sizeof(int) * batch_size * beamwidth * BlockPerBeam * K, false);
    topk_val->data = allocator->Malloc(topk_val->data, sizeof(T) * batch_size * beamwidth * BlockPerBeam * K, false);
    final_topk_id->data = allocator->Malloc(final_topk_id->data, sizeof(int) * batch_size * beamwidth * K, false);
    final_topk_val->data = allocator->Malloc(final_topk_val->data, sizeof(T) * batch_size * beamwidth * K, false);
}
// seems we should self define max_context_len, since we only support bs=1 now
// 将CPU的各个length拷到GPU
template <typename T>
void Llama<T>::free()
{
    allocator->Free(h_input_ids_buf_, true);
    allocator->Free(h_input_length_buf_, true);
    allocator->Free(h_history_length_buf_, true);
    allocator->Free(h_context_length_buf_, true);
    allocator->Free(h_sequence_lengths_, true);
    DeviceSyncAndCheckCudaError();
    allocator->Free(context_decoder_input->data);
    allocator->Free(context_decoder_output->data);
    allocator->Free(decoder_input->data);
    allocator->Free(decoder_output->data);
    allocator->Free(input_ids->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(input_length->data);
    allocator->Free(history_length->data);
    allocator->Free(context_length->data);
    allocator->Free(sequence_lengths->data);
    allocator->Free(all_k_cache->data);
    allocator->Free(all_v_cache->data);
    allocator->Free(token_ids->data);
    allocator->Free(is_finished->data);
    allocator->Free(probs->data);
    DeviceSyncAndCheckCudaError();
}
template <typename T>
void Llama<T>::InitializeForContextDecoder(IntDict &int_params_first_token)
{
    // only support and assumed bs = 1
    h_input_length_buf_[0] = int_params_first_token["cur_input_length"];
    h_history_length_buf_[0] = int_params_first_token["history_length"];
    h_context_length_buf_[0] = int_params_first_token["context_length"];
    // printf("h_input_length_buf_[0] = %d\n", h_input_length_buf_[0]);
    // printf("h_input_ids_buf_[0] = %d\n", h_input_ids_buf_[0]);
    // printf("h_input_ids_buf_[12] = %d\n", h_input_ids_buf_[12]);
    // printf("h_input_ids_buf_[13] = %d\n", h_input_ids_buf_[13]);
    // printf("h_input_ids_buf_[20] = %d\n", h_input_ids_buf_[20]);
    CHECK(cudaMemcpy(input_ids->data,                                    //
                     h_input_ids_buf_,                                   // get from encode
                     sizeof(int) * h_input_length_buf_[0],//RoundUpTo32x(sizeof(int) * h_input_length_buf_[0]), // h_input_length_buf = 0B, cause allocation occurs before line137
                     cudaMemcpyHostToDevice));

    // 直接使用kv cache gpu buffer
    // h_k_cache_ptr_buf_[i] = ;
    // h_v_cache_ptr_buf_[i] = ;
    // step = h_context_length_buf_[0];
    // batch size = 1
    // printf("input ids h2d is done\n");
    CHECK(cudaMemcpy(input_length->data, h_input_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    // printf("input length h2d is done\n");
    CHECK(cudaMemcpy(history_length->data, h_history_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    // printf("history_length h2d is done\n");
    CHECK(cudaMemcpy(context_length->data, h_context_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    // printf("context_length is done\n");
    CHECK(cudaMemcpy(is_finished->data, h_finished_buf_, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));
    // printf("InitializeForContextDecoder is done\n");
}
//
template <typename T>
void Llama<T>::InitializeForSelfDecoder()
{
    // 搜一下d2d怎么copy,答：写个cudakernel吧
    //  CHECK(
    //      cudaMemcpy(sequence_lengths_, context_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    // step_ = max_context_len_;

    // 可自定义输出长度
    // CHECK(
    //     cudaMemcpy(is_finished->data, h_finished_buf_, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));
}

// 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
std::vector<std::string> Llama<T>::MakeInput(const std::string &history, int round, const std::string &input)
{
    std::vector<std::string> ret = {(round == 0 ? "" : history) + input, history, input};
    return ret;
}
template <typename T>
std::string Llama<T>::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output)
{
    // return (round == 0 ? prompt : history) + user_role + input + bot_role + output;// + history_sep;
    return (round == 0 ? prompt : history) + input + output; // + history_sep;
}
template <typename T>
void Llama<T>::inputEmbedding(TensorWrapper<int> *input_ids, TensorWrapper<T> *decoder_input)
{
    launchInputEmbedding<T>(input_ids, decoder_input, &(llama_weights->pre_decoder_embedding_weight));
    DeviceSyncAndCheckCudaError();
}
// 每轮对话的1st token
template <typename T>
int Llama<T>::firstTokenGen(LLaMAAttentionDynParams &dparams, IntDict &int_params_first_token)
{
    InitializeForContextDecoder(int_params_first_token);
    inputEmbedding(input_ids, context_decoder_input);
    ONELLM_CHECK_WITH_INFO(context_decoder_input->data != nullptr, "GPU context decoder input data is not initialized");
    ONELLM_CHECK_WITH_INFO(history_length->data != nullptr, "GPU history_length data is not initialized");
    ONELLM_CHECK_WITH_INFO(input_length->data != nullptr, "GPU input_length data is not initialized");
    ONELLM_CHECK_WITH_INFO(context_length->data != nullptr, "GPU context_length data is not initialized");
    ONELLM_CHECK_WITH_INFO(output_rmsnorm_weight->data != nullptr, "GPU output_rmsnorm_weight data is not initialized");
    TensorMap decoder_inputs{
        {"decoder_input", context_decoder_input},
        // {"padding_offset", padding_offset}, // get by cal_paddingoffset
        {"history_length", history_length},
        {"input_length", input_length},
        {"context_length", context_length},
        // {"attention_mask", attention_mask}, // get by build_casual_mask
        {"output_norm_weight", output_rmsnorm_weight}, // located at llamaweights class, rather not llamalayerweigths
        {"layer_id", layer}};
    // output buffer and input buffer are shared to reuse buffer between layers
    // I dont rewrite Tensor's copy constructor, default shallow copy, that can share buffer, which is I want
    TensorMap decoder_outputs{
        {"decoder_output", context_decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}};

    context_decoder->forward(decoder_inputs,
                             llama_weights->llama_layer_weight, // layerWeights,
                             decoder_outputs,
                             dparams);
    int res = LMHeadAndTopKSample(decoder_outputs);
    return res;
}

template <typename T>
int Llama<T>::continueTokenGen(LLaMAAttentionDynParams &dparams)
{
    InitializeForSelfDecoder();
    inputEmbedding(input_ids, decoder_input);
    TensorMap decoder_inputs{
        {"decoder_input", decoder_input},
        // {"sequence_lengths", Tensor(GPU, type, {hidden_units}, )},
        // {"total_padding_len", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, )},
        {"step", step}, // a batch shared same step, locate on CPU, no need GPU
        {"finished", is_finished},
        {"layer_id", layer},
        {"output_norm_weight", output_rmsnorm_weight} // located at llamaweights class, rather not llamalayerweigths
    };

    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache}, // 最开始是context decoder输出的kv cache，之后step是前一个step输出的kv cache，只要保证kv cache是llama的成员就可以保证同步更新
        {"all_v_cache", all_v_cache}};

    self_decoder->forward(decoder_inputs,
                          llama_weights->llama_layer_weight,
                          decoder_outputs,
                          dparams);
    std::cout << "sampling..." << std::endl;
    int res = LMHeadAndTopKSample(decoder_outputs);
    return res;
}

template <typename T>
int Llama<T>::LMHeadAndTopKSample(TensorMap &decoder_outputs)
{
    Tensor *decoder_output = decoder_outputs["decoder_output"];
    if (index == 0)
    {
        TensorWrapper<T> *decoder_output_tensorwrapper = decoder_output->as<T>();
        auto input_length = decoder_output_tensorwrapper->shape[0];
        auto hidden_units = decoder_output_tensorwrapper->shape[1];
        //ONELLM_CHECK(h_input_length_buf_[0] == input_length);
        // follow fastllm handle ctxdecoder sampling
        auto ptr = decoder_output_tensorwrapper->data + (input_length - 1) * hidden_units;
        context_decoder_lmhead_input->data = ptr;
        launchLinearGemm(/*Tensor**/ context_decoder_lmhead_input,                     //[1, hidden units] for ctx decoder
                         /*BaseWeight&*/ llama_weights->post_decoder_embedding_weight, // lm_head.weight.bin, [vocab_size, hidden_units]
                         /*Tensor**/ probs,                                            //[1, vocab size] for context decoder
                         cublas_wrapper,
                         false,
                         true);
        DeviceSyncAndCheckCudaError();
    } else {
        // for self decoder
        launchLinearGemm(/*Tensor**/ decoder_output->as<T>(),                          //[bs, hidden_units] for self decoder
                         /*BaseWeight&*/ llama_weights->post_decoder_embedding_weight, // lm_head.weight.bin, [vocab_size,hidden_units]
                         /*Tensor**/ probs,                                            //[bs, vocab size] for self decoder
                         cublas_wrapper,
                         false,
                         true);
        DeviceSyncAndCheckCudaError();
    }

    launchTopKforBeamSearch(probs, // [bs, vocab_size]
                            topk_id,
                            topk_val,
                            final_topk_id,
                            final_topk_val); // output，这个属于是中间buffer，定义在allocatebuffer就行
    DeviceSyncAndCheckCudaError();
    int_params_of_sample.insert({"step", *step->data});
    launchSampling(/*Tensor**/ final_topk_id,          // in
                   /*Tensor**/ final_topk_val,         // in
                   /*Tensor**/ sequence_lengths,       // out, +1
                   /*Tensor**/ is_finished,            // out, 判断一下是否结束
                   /*Tensor**/ token_ids,              // out, 新生成的token ids
                   /*IntDict&*/ int_params_of_sample); // in, including step vocabsize endid
    DeviceSyncAndCheckCudaError();

    CHECK(cudaMemcpy(h_output_ids, token_ids->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));
    std::cout << "sampling done" << std::endl;
    return h_output_ids[0]; // only for bs = 1
}

// 单轮对话, batch size = 1
// 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
std::string Llama<T>::Response(const std::vector<std::string> &input, CallBack PrintRes)
{
    // this input already include self-defined pre prompt
    // printf("input= %s", std::get<0>(input));
    std::cout << "input = " << input[0] << "\n";
    std::vector<int> res = tokenizer.Encode(input[2]);

    std::string history_str = input[1];
    std::vector<int> history_input_ids;
    if (!history_str.empty())
    {
        history_input_ids = tokenizer.Encode(history_str);
    }
    // h_input_ids_buf_ = res.data();// warning: dont use this method, should use for travese assign, or the former will generate trash val out of vector's scope
    std::string total_str = input[0];
    std::vector<int> context_ids;
    if (!total_str.empty())
    {
        context_ids = tokenizer.Encode(total_str);
    }
    for (int i = 0; i < res.size(); i++)
    {
        h_input_ids_buf_[i] = res[i]; // [max_context_token_nums_]
    }
    // printf("h_input_ids_vec_len = %d\n", res.size());//这个值有问题啊
    // printf("h_input_ids_buf_[1] = %d\n", h_input_ids_buf_[1]);
    // ensure prepared all needed input buffer
    //int index = 0;
    int ret;
    int context_length = context_ids.size();
    int history_length = history_input_ids.size();
    int cur_input_length = res.size(); // res.size() is the input ids len, which is the real input len, rather not len of input string
    std::cout << "context ids lenght = " << context_length
              << "history_length = " << history_length
              << "cur_input_length = " << cur_input_length
              << "\n";
    IntDict int_params_first_token;
    int_params_first_token["context_length"] = context_length;
    int_params_first_token["history_length"] = history_length;
    int_params_first_token["cur_input_length"] = cur_input_length;
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 1;
    attn_dyn_params.num_tokens = cur_input_length;          // 这个此时还是0
    attn_dyn_params.max_q_len = attn_dyn_params.num_tokens; // 这个指一个batch中的q的最大长度，因为此时不支持batch，所以就等于cur input len
    attn_dyn_params.max_k_len = context_length;             // max_seq_len; //这个指max context len，指当前batch的动态最大上下文长度
    step->data = &context_length;                           //
    // retString为当前轮次对话的所有token string
    std::string retString = "";
    while (index < output_token_limit)
    {
        // kv cache here input is empty, only buffer, output is not empty
        if (index == 0)
        {
            ret = firstTokenGen(attn_dyn_params, int_params_first_token);
        }
        else
        {
            // TODO move all needed data to GPU
            // no need input attnmask and positionid like fastllm, cause we build attnmask and dont support abili now
            ret = continueTokenGen(attn_dyn_params);
            if (ret == eos_token_id)
            {
                break;
            }
        }
        *step->data++;
        std::cout << "generated index: " << ret << "\n";

        // results.push_back(ret);
        std::string genString = tokenizer.Decode({ret}).c_str();
        retString += genString;
        PrintRes(index, genString.c_str());
        // deep copy
        // for ctx decoder, input_ids.shape = [max_context_token_nums]
        // for self decoder, input_ids.shape = [1]
        // but  input_ids->data.size = [max_context_token_nums]
        // input_ids->shape = {1};
        if (index == 0)
        {
            TensorWrapper<int> tmp = TensorWrapper<int>(CPU, getTensorType<int>(), {1}, &ret);
            ONELLM_CHECK(tmp.shape != input_ids->shape);
            ONELLM_CHECK(tmp.dtype == input_ids->dtype);
            ONELLM_CHECK(tmp.location != input_ids->location);
            allocator->Free(input_ids->data);
            input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 1, false);
            input_ids->shape = {1};
	    CHECK(cudaMemcpy(input_ids->data, tmp.data, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }
        else
        {
            CHECK(cudaMemcpy(input_ids->data, &ret, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }
        index++; // 生成的token数量
        // 但是这个我不希望对齐32b啊，看来allocator还是得改一下!!
        // lmdeploy对齐了32b，但是fastllm没有，我感觉也没有必要对齐32b
        // 但是至少目前对齐32b的情况下，所有kernel是正确的，后面考虑单单llama.cpp里面这些或者就input_ids不对齐32b

        // 把input_ids这块[max_context_token_nums]大小得buf输进self decoder是有问题得，应该输入一块decoder_input_buf.shape=[bs, hiddenunits]
        // 或者把input ids这块tensorwrapper的buffer重新allocate并重定义shape，参考fastllm.cpp#261-277
        // input_ids->shape = {1,1};[bs, max seq len]
        // CHECK(cudaMemcpy(input_ids->data, &ret, sizeof(int), cudaMemcpyHostToDevice));//note: dont use input_ids = new TensorWrapper<int>(...&ret), because this is cpu ret

        // input_ids = new TensorWrapper<int>(GPU, INT32, {1, 1}, &ret);
    }
    PrintRes(-1, retString.c_str());
    return retString;
}

template class Llama<float>;
template class Llama<half>;
