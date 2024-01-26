// This kernel only used in prompt phase
// 1.add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
// QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].

// 2.For q and k, apply RoPE, then send to attention.

// 3.rebuild padding to do mha

// input: qkv_buf : qkv continouns buf when no padding
// shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
// seqlen=max_q_len
#include <math.h>
#include <stdio.h>

#include "src/kernels/qkv_bias_and_RoPE.h"
// 这里算出来只有head size / 2个cos，同理sin个数也一样
// llama.py实现
//    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
//         """Compute the inverse frequency."""
//         inv_freq = 1.0 / (base**(torch.arange(
//             0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
//                                  self.rotary_dim))
//         return inv_freq

//     def _compute_cos_sin_cache(self) -> torch.Tensor:
//         """Compute the cos and sin cache."""
//         inv_freq = self._compute_inv_freq(self.base)
//         t = torch.arange(self.max_position_embeddings,
//                          dtype=torch.float,
//                          device="cuda")

//         freqs = torch.einsum("i,j -> ij", t, inv_freq)
//         cos = freqs.cos() // 2048，64
//         sin = freqs.sin()
//         cache = torch.cat((cos, sin), dim=-1)
//         return cache
//对比llama py实现，我们少了L30和34的这一步,即一个(2048,1)和(1,64)的外积
inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // 某个token的head size维度上连续俩元素的inv freq，t_Step表示tokenid，能对上transformers上的[0,2047]和freq的外积
    // 每个inv freq值对应于head size维度上0 2 4 6的值
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim); //rot_embed_dim = 128
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}
// inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
// {
//     float2 rot_v;
//     rot_v.x = coef.x * v.x - coef.y * v.y;
//     rot_v.y = coef.x * v.y + coef.y * v.x;
//     return rot_v;
// }

inline __device__ half2 GetRoPEres(const half2 v, const float2 coef)
{
    float2 fv = __half22float2(v);
    float2 rot_fv = GetRoPEres(fv, coef);
    return __float22half2_rn(rot_fv);
}

// inline __device__ void apply_RoPE(float q, float k, int tid, int rot_embed_dim, float base, int t_step)
// {
//     if (tid >= rot_embed_dim / 2)
//     {
//         return;
//     }

//     float2 coef0 = GetRoPEfreq(tid, rot_embed_dim, base, t_step);
//     q = GetRoPEres(q, coef0);
//     k = GetRoPEres(k, coef0);
// }

inline __device__ void apply_RoPE(half2 &q, half2 &k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (2 * tid >= rot_embed_dim)
    {
        return;
    }
    const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
    q = GetRoPEres(q, coef);
    k = GetRoPEres(k, coef);
}

inline __device__ void apply_RoPE(float4 &q, float4 &k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (4 * tid >= rot_embed_dim)
    {
        return;
    }

    TwoFloat2 &q_ = *reinterpret_cast<TwoFloat2 *>(&q); // q为float4 寄存器
    TwoFloat2 &k_ = *reinterpret_cast<TwoFloat2 *>(&k);

    float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
    // float freq0 = timestep / powf(rotary_embedding_base, 4 * tid / (float) rotary_embedding_dim); //分子zid = 0,2,4,, headsize/2-1,对应的theta下标为0,1,2.对应的headsize维度的索引为(0,1),(2,3)
    q_.x = GetRoPEres(q_.x, coef0);
    // rot0.x = coef0.x * q.x -  coef0.y * q.y; //q.x为x0,q.y为x1，head size维度上两个相邻
    // rot0.y = coef0.x * q.y +  coef0.y * q.x
    float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y = GetRoPEres(q_.y, coef1);
    // rot1.x = coef1.x * q.x -  coef1.y * q.y; //q.x为x2,q.y为x3，head size维度上两个相邻
    // rot1.y = coef1.x * q.y +  coef1.y * q.x;
    k_.x = GetRoPEres(k_.x, coef0);
    k_.y = GetRoPEres(k_.y, coef1);
}
template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                                                   T *k_buf,
                                                   T *v_buf,
                                                   T *QKV,
                                                   const T *qkv_bias,
                                                   const int *padding_offset, // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len, // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // default 10000 in llama
                                                   int max_position_embeddings, /*default 2048 in llama, placeholder for ntk RoPE*/
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{
    // int vec_size = Vec<T>::size;
    // using Vec_t = typename Vec<T>::Type;
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
    // bool is_data = tid * vec_size < head_size;
    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id
    // if (token_id == 0 && head_id == 0 && tid == 0)
    // {
    //     printf("QKV top2 res: \n");
    //     printf("%f\n", QKV[tid]);
    //     printf("%f\n", QKV[1]);
    // }
    // 2. bias add
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
    // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
    // TODO: reduce the pointer converter and fuse for loop
    // Vec_t q, k, v;
    // if (is_data)
    // {
    //     q = *reinterpret_cast<Vec_t *>(&QKV[q_id]);
	// if (qkv_bias != nullptr){
	//     Vec_t q_bias = *reinterpret_cast<Vec_t *>(const_cast<T *>(&qkv_bias[head_id * head_size + tid * vec_size]));
    //         for (int i = 0; i < vec_size; i++)
    //         {
    //         	reinterpret_cast<float *>(&q)[i] += reinterpret_cast<float *>(&q_bias)[i];
    //         }
	// }
    // }
    // // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
    // if (is_data && head_id < kv_head_num)
    // {
    //     k = *reinterpret_cast<Vec_t *>(&QKV[k_id]);
    //     // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
    //     if (qkv_bias != nullptr){
	//     Vec_t k_bias = *reinterpret_cast<Vec_t *>(const_cast<T *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
    //         for (int i = 0; i < vec_size; i++)
    //         {
    //             reinterpret_cast<float *>(&k)[i] += reinterpret_cast<float *>(&k_bias)[i];
    //         }
	// }

    //     v = *reinterpret_cast<Vec_t *>(&QKV[v_id]);
    //     if (qkv_bias != nullptr){
	//     Vec_t v_bias = *reinterpret_cast<Vec_t *>(const_cast<T *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
    //         for (int i = 0; i < vec_size; i++)
    //         {
    //             reinterpret_cast<float *>(&v)[i] += reinterpret_cast<float *>(&v_bias)[i];
    //         }
	// }
    // }
    float v = QKV[v_id];
    if (head_id < kv_head_num)
    { // for MQA and GQA
        v_buf[dst_kv_id] = v;
    }
    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id; //+ local_token_id得到m，即要结合history length做全局位置编码
    // timestep为cos(m*theta)中的m
    if (tid >= rot_embed_dim / 2)
    {
        return;
    }

    float2 cos_sin = GetRoPEfreq(tid, rot_embed_dim, base, timestep);
    float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin);
    float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin);

    // apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    // 4.write back to gmem and do transpose
    //  [bs, head num, seqlen, head size]
    //  pay attention to local token id and kv head num and max_seq_len(seq_len)
    // int dst_q_id = batch_id * seq_len * head_num * head_size +
    //                head_id * seq_len * head_size +
    //                local_token_id * head_size + tid * vec_size;

    // int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
    //                 head_id * seq_len * head_size +
    //                 local_token_id * head_size + tid * vec_size;
    int dst_q_id = batch_id * seq_len * head_num * head_size +
                   head_id * seq_len * head_size +
                   local_token_id * head_size + tid;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;
    // if (is_data)
    // {
        // *reinterpret_cast<Vec_t *>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
        // if (head_id < kv_head_num)
        // { // for MQA and GQA
        //     *reinterpret_cast<Vec_t *>(&k_buf[dst_kv_id]) = k;
        //     *reinterpret_cast<Vec_t *>(&v_buf[dst_kv_id]) = v;
        // }
        // if (token_id == 0 && head_id == 0 && tid == 0)
        // {
        //     printf("rope top2 res: \n");
        //     printf("%f\n", q_buf[tid]);
        //     printf("%f\n", q_buf[1]);
        // }
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if (head_id < kv_head_num)
    { // for MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
    // }
}

template <>
__global__ void add_fusedQKV_bias_transpose_kernel(half *q_buf,
                                                   half *k_buf,
                                                   half *v_buf,
                                                   half *QKV,
                                                   const half *qkv_bias,
                                                   const int *padding_offset, // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len, // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // default 10000 in llama
                                                   int max_position_embeddings, /*default 2048 in llama, placeholder for ntk RoPE*/
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
    bool is_data = tid * vec_size < head_size;
    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id

    // 2. bias add
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
    // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
    // TODO: reduce the pointer converter and fuse for loop
    Vec_t q, k, v;
    if (is_data)
    {
        q = *reinterpret_cast<Vec_t *>(&QKV[q_id]);
        Vec_t q_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size]));
        q = __hadd2(q, q_bias);
    }
    // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
    if (is_data && head_id < kv_head_num)
    {
        k = *reinterpret_cast<Vec_t *>(&QKV[k_id]);
        // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
        Vec_t k_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
        k = __hadd2(k, k_bias);
        v = *reinterpret_cast<Vec_t *>(&QKV[v_id]);
        Vec_t v_bias = *reinterpret_cast<Vec_t *>(const_cast<half *>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
        v = __hadd2(v, v_bias);
    }

    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id; //+ local_token_id得到m，即要结合history length做全局位置编码
    // timestep为cos(m*theta)中的m

    apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    // 4.write back to gmem and do transpose
    //  [bs, head num, seqlen, head size]
    //  pay attention to local token id and kv head num and max_seq_len(seq_len)
    int dst_q_id = batch_id * seq_len * head_num * head_size +
                   head_id * seq_len * head_size +
                   local_token_id * head_size + tid * vec_size;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid * vec_size;
    if (is_data)
    {
        *reinterpret_cast<Vec_t *>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
        if (head_id < kv_head_num)
        { // for MQA and GQA
            *reinterpret_cast<Vec_t *>(&k_buf[dst_kv_id]) = k;
            *reinterpret_cast<Vec_t *>(&v_buf[dst_kv_id]) = v;
        }
    }
}

// input: qkv_buf : qkv continouns buf when no padding
// shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size], 这里的seqlen应该是max_q_len
template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T> *q_buf,
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,
                                           BaseWeight<T> &qkv,
                                           // Tensor* qkv_bias,
                                           TensorWrapper<int> *padding_offset,
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LLaMAAttentionStaticParams &params)
{
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;

    dim3 grid(token_num, head_num);
    // dim3 block((head_size / Vec<float>::size + 32 - 1) / 32 * 32); // apply 2 eles vectorization to match RoPE
    dim3 block(head_size); // apply 2 eles vectorization to match RoPE
    // printf("calling qkvbias and rope\n");
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                                                           k_buf->data,
                                                           v_buf->data,
                                                           QKV->data,
                                                           qkv.bias,
                                                           padding_offset->data,
                                                           history_length->data,
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
    // printf("called qkv bias and rope\n");
}

template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float> *q_buf,
                                                    TensorWrapper<float> *k_buf,
                                                    TensorWrapper<float> *v_buf,
                                                    TensorWrapper<float> *QKV,
                                                    BaseWeight<float> &qkv,
                                                    // Tensor* qkv_bias,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LLaMAAttentionStaticParams &params);
template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<half> *q_buf,
                                                    TensorWrapper<half> *k_buf,
                                                    TensorWrapper<half> *v_buf,
                                                    TensorWrapper<half> *QKV,
                                                    BaseWeight<half> &qkv,
                                                    // Tensor* qkv_bias,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LLaMAAttentionStaticParams &params);
