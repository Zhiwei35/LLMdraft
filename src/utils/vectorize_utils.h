#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
template<typename T_OUT, typename T_IN>
inline __device__ T_OUT scalar_cast_vec(T_IN val)
{
    return val;
}

template<>
inline __device__ half2 scalar_cast_vec<half2, float>(float val)
{
    return __float2half2_rn(val);
}

template<>
inline __device__ float4 scalar_cast_vec<float4, float>(float val)
{
    return make_float4(val, val, val, val);
}

template<>
inline __device__ float2 scalar_cast_vec<float2, float>(float val)
{
    return make_float2(val, val);
}

template<>
inline __device__ half2 scalar_cast_vec<half2, half>(half val)
{
    return __half2half2(val);
}

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};
template<>
struct Vec<half> {
    using Type = half2; //half2 or uint32_t?
    static constexpr int size = 2;
};
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

struct TwoFloat2{
    float2 x;
    float2 y;
};