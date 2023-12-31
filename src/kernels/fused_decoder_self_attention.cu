#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/kernels/fused_decoder_self_attention.h"
// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
// bug1: scale's dtype must be float ,not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which cause we dont run the cuda kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
// bug4: GQA, kv_head_num brd to head_num, we can automaticly do this by head id index like lmdeploy
// half or float version: the logits and mha output both are fp32 type, q k v are all accessed vectorizedly
template<typename T>
__device__ T warpReduceSum(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;

}
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];//why add static?or will report incomplete type
    // returned val is the sum computed by 0th thread.
    val = warpReduceSum<T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);

}
template<typename T>
__device__ T warpReduceMax(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    // returned val is the max computed by 0th thread.
    val = warpReduceMax(val); // remove <T> can ignore the multi-overloaded error?
    //note: here return val of warpreducemax should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}
// kv cache is the output of context attention(prompt phase), and the input of masked attention(token gen)
// struct masked_MHA_kernel_params
// {
//     float* q;       //[bs, q num heads, 1, head size]
//     float* k;       //[bs, kv num heads, step/seqlen, head size]
//     float* v;       //[bs, num heads, 1, head size]
//     float* k_cache; //output,[max_seq_len or step, bs, kv num heads, head size] from prompt phase
//     float* v_cache; //output,[max_seq_len or step, bs, num heads, head size] from prompt phase
//     int batch_size;
//     int num_heads;
//     int head_size;
//     float scale; // =rsqrt(head size);
//     //TODO: add qkv bias
//     int step;
//     float* mha_output;
//};


inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

// RoPE公式决定必须要做向量化
inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ half2 GetRoPEres(const half2 v, const float2 coef)
{
    float2 fv     = __half22float2(v);
    float2 rot_fv = GetRoPEres(fv, coef);
    return __float22half2_rn(rot_fv);
}

inline __device__ void apply_RoPE(half2& q, half2& k, int tid, int rot_embed_dim, float base, float t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
    q               = GetRoPEres(q, coef);
    k               = GetRoPEres(k, coef);
}

inline __device__ void apply_RoPE(float4& q, float4& k, int tid, int rot_embed_dim, float base, float t_step){
    if(4 * tid >= rot_embed_dim){
        return;
    }


    TwoFloat2& q_ = *reinterpret_cast<TwoFloat2*>(&q); // q为float4 寄存器
    TwoFloat2& k_ = *reinterpret_cast<TwoFloat2*>(&k);
    
    float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
    // float freq0 = timestep / powf(rotary_embedding_base, 4 * tid / (float) rotary_embedding_dim); //分子zid = 0,2,4,, headsize/2-1,对应的theta下标为0,1,2.对应的headsize维度的索引为(0,1),(2,3)
    q_.x = GetRoPEres(q_.x ,coef0);
    // rot0.x = coef0.x * q.x -  coef0.y * q.y; //q.x为x0,q.y为x1，head size维度上两个相邻
    // rot0.y = coef0.x * q.y +  coef0.y * q.x
    float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y = GetRoPEres(q_.y ,coef1);
    // rot1.x = coef1.x * q.x -  coef1.y * q.y; //q.x为x2,q.y为x3，head size维度上两个相邻
    // rot1.y = coef1.x * q.y +  coef1.y * q.x;
    k_.x = GetRoPEres(k_.x ,coef0);
    k_.y = GetRoPEres(k_.y ,coef1);
}

// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size
// 1 grid -> bs * num heads
// q;       [bs, q num heads, 1, head size]
// k;       [bs, kv num heads, step/seqlen, head size]
// v;       [bs, num heads, 1, head size]
// k_cache; output,[max_seq_len or step, bs, kv num heads, head size] from prompt phase
// v_cache; output,[max_seq_len or step, bs, num heads, head size] from prompt phase
template<typename T>
__global__ void masked_MHA_kernel(const T* q,
                    const T* k,
                    const T* v,
                    T* qkv_bias,
                    T* k_cache,
                    T* v_cache,
                    T* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    //const int num_heads,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){// rsqrt(dh)
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    int kv_head_id = q_head_id / head_num / kv_head_num;
    int kv_batch_id = q_batch_id;
    //int q_head_id = bid % head_num;
    //int q_batch_id = bid / head_num;
    //int kv_head_id = bid % kv_head_num;
    //int kv_batch_id = bid / kv_head_num;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = batch_size * kv_batch_stride;

    int vec_size = Vec<T>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    float scale = rsqrt(float(head_size));
    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    //Vec_t scale_vec = static_cast<Vec_t>(scale);
    //reuse q k v reg from rope
    const T* q_mem = q;
    const T* k_mem = k;
    const T* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec]));
        Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&qvec)[i] += reinterpret_cast<float*>(&q_bias)[i];
        }
        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec]));
        Vec_t k_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size]);
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&kvec)[i] += reinterpret_cast<float*>(&k_bias)[i];
        }
        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec]));
        Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&vvec)[i] += reinterpret_cast<float*>(&v_bias)[i];
        }
        // 测试的时候关掉rope
        // apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);
    }
    // q k smem for block reduce
    // define smem type is char type!! not T
    // 主要展示一些dynamic smem怎么memory plan
    extern __shared__ char sqk[];
    T* sq = reinterpret_cast<T*>(sqk); // 在step行把q存进smem，之前的step-1行可以直接从smem load to reg
    T* sk = sq + head_size; // 不是很有必要在reg上存k
    T* sv = sk + head_size;
    float* logits = reinterpret_cast<float*>(sv + step); // 所有线程reduce的结果存到logits，需要smem
    //T* sv = reinterpret_cast<T*>(logits + step);
    //sq[tid] = q_mem[qkv_offset];
    if (tid * vec_size < head_size) {
        *reinterpret_cast<Vec_t*>(&sq[tid * vec_size]) = qvec;
    }
    __syncthreads();
    // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
    // FT 5.3, 一个block处理k的多行，即多个head size
    for(int iter = 0; iter < step; iter++) {
        // every iter,  q and k's shape = [1, head size]
        // reuse k cache
        // float k = k_cache[iter * cache_offset + qkv_offset];
        //或许可以在每个step省略掉前step-1的qk dot
        sk[tid]= k_cache[iter * cache_offset + k_offset];
        __syncthreads();
        // when final step, update k cache
        if (iter == step - 1 && tid * vec_size < head_size) {
            // TODO: update k cache with k with bias add
            //k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
            //sk[tid] = k_mem[qkv_offset];
            *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) = kvec;
            *reinterpret_cast<Vec_t*>(&sk[tid * vec_size]) = kvec;         
        }
        // sq[tid] = q_mem[qkv_offset];
        __syncthreads();
        //在FT，k是从k cache加载到reg，q是从q smem加载到reg，q smem是每个新的step把新然后二者mul，这里直接用smem做mul也可以，反正compiler会帮我们load到reg
        T qk = (tid < head_size) ? (float)sq[tid] * (float)sk[tid] * (float)scale : (T)0.0f;
        //block reduce using multi warp reduce
        //TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
        T attn_score = blockReduceSum<T>(qk);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu;
    }
    __syncthreads();
    if(tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();

    // logits*V = [bs, num heads, 1, step] * [max_seq_len or step, bs, num heads, head size]
    if (tid < head_size) {
        // note: here is head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
        // so here we use acc O to acc the one ele logits * one ele v every step iter
        T O = 0.0f;
        for(int iter = 0; iter < step; iter++) {
            sv[tid]= v_cache[iter * cache_offset + k_offset];
            __syncthreads();
            // when final step, update k cache
            if (iter == step - 1) {
                // TODO: update k cache with k with bias add
                *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]) = vvec;
                // v_cache[iter * cache_offset + k_offset] = v_mem[k_offset];
                sv[tid] = v_mem[k_offset];
            }
	    __syncthreads();
            //if(bid==0 && tid == 0){
            //printf("when tid=0, v cache = %f\n", sv[tid]);
            //在FT，v是从v cache加载到reg，logits是从logits smem加载到reg，然后二者mul
            O += sv[tid] * logits[iter];
            __syncthreads();
        }
        mha_output[q_offset] = O;
    }
}

template<> //特化以下half类型的，不在fp32代码上改
__global__ void masked_MHA_kernel(const half* q,
                    const half* k,
                    const half* v,
                    half* qkv_bias,
                    half* k_cache,
                    half* v_cache,
                    half* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    //const int num_heads,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){// rsqrt(dh)
    int tid = threadIdx.x;
    //int bid = blockIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    int kv_head_id = q_head_id / head_num / kv_head_num;
    int kv_batch_id = q_batch_id;
    //int q_head_id = bid % head_num;
    //int q_batch_id = bid / head_num;
    //int kv_head_id = bid % kv_head_num;
    //int kv_batch_id = bid / kv_head_num;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = batch_size * kv_batch_stride;

    int vec_size = Vec<half>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    half scale = __float2half(rsqrt(float(head_size)));
    using Vec_t = typename Vec<half>::Type;
    Vec_t qvec, kvec, vvec;
    Vec_t scale_vec = scalar_cast_vec<Vec_t>(scale);
    //reuse q k v reg from rope
    const half* q_mem = q;
    const half* k_mem = k;
    const half* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&q_mem[q_offset_vec]));
        Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);
        qvec = __hadd2(qvec, q_bias);

        kvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&k_mem[k_offset_vec]));
        Vec_t k_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size]);
        kvec = __hadd2(kvec, k_bias);

        //apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);
        vvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&v_mem[k_offset_vec]));
        Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
        vvec = __hadd2(vvec, v_bias);
    }
    // q k smem for block reduce
    extern __shared__ char sqk[];
    half* sq = reinterpret_cast<half*>(sqk);
    half* sk = sq + head_size;
    //float* logits = reinterpret_cast<float*>(sk + head_size);
    half* sv = sk + head_size;
    float* logits = reinterpret_cast<float*>(sv + head_size);
    //sq[tid] = q_mem[qkv_offset];

    Vec_t* sq_vec = reinterpret_cast<Vec_t*>(sq);
    Vec_t* sk_vec = reinterpret_cast<Vec_t*>(sk);
    Vec_t* sv_vec = reinterpret_cast<Vec_t*>(sv);
    if (tid * vec_size < head_size) {
        // *reinterpret_cast<Vec_t*>(&sq[tid * vec_size]) = qvec;
        sq_vec[tid] = qvec;
    }
    __syncthreads();
    // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
    for(int iter = 0; iter < step; iter++) {
        // every iter,  q and k's shape = [1, head size]
        // reuse k cache
        // float k = k_cache[iter * cache_offset + qkv_offset];
        //或许可以在每个step省略掉前step-1的qk dot
        sk_vec[tid]= *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]);
        __syncthreads();
        // when final step, update k cache
        if (iter == step - 1 && tid * vec_size < head_size) {
            // TODO: update k cache with k with bias add
            //k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
            //sk[tid] = k_mem[qkv_offset];
            *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) = kvec;
            sk_vec[tid] = kvec;         
        }

        // sq[tid] = q_mem[qkv_offset];
        __syncthreads();
        Vec_t qk = (tid * vec_size < head_size) ? __hmul2(__hmul2(sq_vec[tid], sk_vec[tid]), scale_vec) : scalar_cast_vec<Vec_t>(0.0f);
        //block reduce using multi warp reduce
        float qk_fp32 = __half2float(qk.x) + __half2float(qk.y);
        float attn_score = blockReduceSum<float>(qk_fp32);
        if(tid == 0) {
            logits[iter] = attn_score;
	    //float q_tmp = (float)(sq_vec[0].x);
	    //float k_tmp = (float)(sk_vec[0].x);
	    //float scale_tmp = (float)(scale_vec.x);
            //printf("iter = %d, step=%d, blockIdx.x = %d, in cuda, logits[%d]=%f, qk_fp32 = %f, q_tmp=%f, k_tmp=%f, scale_tmp=%f\n",iter, step, blockIdx.x, iter, logits[iter], qk_fp32, q_tmp, k_tmp, scale_tmp);
	}
        __syncthreads();
    }
    //__syncthreads();
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    //if(tid < step){
    	//printf("logits[%d]=%f\n", tid, logits[tid]);
    //}
    float local_logits = tid < step ? logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    float block_max = blockReduceMax<float>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    float fenzi = tid < step ? expf(local_logits - row_max) : 0;
    //if(tid < step) {
    //	printf("after expf, row_max=%f, fenzi=%f, logits=%f\n", row_max, fenzi, local_logits);
    //}
    float block_fenmu = blockReduceSum<float>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu;
    }
    __syncthreads();
    if(tid < step) {
        
	logits[tid] = (float)(fenzi / fenmu);
//	printf("in cuda, row_max=%f, fenzi=%f, fenmu=%f, logits=%f\n", row_max, fenzi, fenmu, logits[tid]);
	
    }
    __syncthreads();

    // logits*V = [bs, num heads, 1, step] * [max_seq_len or step, bs, num heads, head size]
    if (tid * vec_size < head_size) {
        // note: here is head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
        // so here we use acc O to acc the one ele logits * one ele v every step iter
        float2 O = scalar_cast_vec<float2>(0.0f);
        //O.x = 0.0f;
        //O.y = 0.0f;
        for(int iter = 0; iter < step; iter++) {
            sv_vec[tid]= *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]);
            //sv[tid]= v_cache[iter * cache_offset + k_offset];
            __syncthreads();
            // when final step, update k cache
            if (iter == step - 1) {
                // TODO: update k cache with k with bias add
                // v_cache[iter * cache_offset + k_offset] = v_mem[k_offset];
                // sv[tid] = v_mem[k_offset];
                *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]) = vvec;
                sv_vec[tid] = vvec;  
            }
	    __syncthreads();
            //if(bid==0 && tid == 0){
            //printf("when tid=0, v cache = %f\n", sv[tid]);
            O.x += (logits[iter] * __half2float(sv_vec[tid].x));
            O.y += (logits[iter] * __half2float(sv_vec[tid].y));
            //O += sv[tid] * logits[iter];
            __syncthreads();
        }
        
        // float* mha_output_fp32 = reinterpret_cast<float*>(mha_output);
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = __float22half2_rn(O);
    }
}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3]; 
    int head_num = qkv_head_num - 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    T* qkv_data = qkv_buf->data;
    //[bs,1,qkv_head_num,head_size]
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;
    // bool is_half = sizeof(T) == 2;
    // T scale = is_half ? __float2half(rsqrt(float(head_size))) : rsqrt(float(head_size));

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    bool  use_dynamic_ntk = static_params.use_dynamic_ntk;
    dim3 grid(head_num, batch_size);//这里的block分配可以匹配得上lmdeploy
    dim3 block(head_size); //vec size = 4 for fp32
    // printf("calling fused masked self attn kernel\n");
    // printf("block nums = %d\n", grid.x);
    // printf("thread nums = %d\n", block.x);
    masked_MHA_kernel<T><<<grid, block, (3 * head_size * sizeof(T) + cur_step * sizeof(float))>>>(
                                                                                q,
                                                                                k,
                                                                                v,
                                                                                /*(T*)*/qkv.bias,
                                                                                k_cache->data + layer_offset,
                                                                                v_cache->data + layer_offset,
                                                                                mha_output->data,
                                                                                batch_size,
                                                                                head_num,
                                                                                kv_head_num,
                                                                                //num_heads,
                                                                                head_size,
                                                                                cur_step,
                                                                                rotary_embedding_base,
                                                                                rotary_embedding_dim);
    // printf("called fused masked self attn kernel\n");
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,
                            BaseWeight<float>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache,
                            TensorWrapper<float>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<float>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,
                            BaseWeight<half>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache,
                            TensorWrapper<half>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<half>* mha_output,
                            LLaMAAttentionStaticParams& static_params);
