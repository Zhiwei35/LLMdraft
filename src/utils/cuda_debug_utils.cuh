#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
// usage: print_data<<<1, 1>>>()
template<typename T>
__global__ void print_data(T* src1) {
    int tid = threadIdx.x;
    if(tid == 0) {
    	// printf("qkv/outlinear data[%d] = %f\n", tid, src1[tid]);
    	// printf("qkv/outlinear data[%d] = %f\n", tid + 128, src1[tid + 128]);
    	// printf("qkv/outlinear data[%d] = %f\n", tid + 256, src1[tid + 256]);    	
    	printf("%dth = %f\n", tid, src1[tid]);
    	printf("%dth = %f\n", tid + 128, src1[tid + 128]);
    	printf("%dth = %f\n", tid + 256, src1[tid + 256]);    	

	    // printf("from/outlinearin data[%d] = %f\n", tid, src3[tid]);
    	// printf("from/outlinearin data[%d] = %f\n", tid + 1, src3[tid+1]);
   	    // printf("from/outlinearin data[%d] = %f\n", tid + 128, src3[tid+128]);
    	// printf("from/outlinearin data[%d] = %f\n", tid + 129, src3[tid+129]);
    	
	    // printf("qkvweight/outweight data[%d] = %f\n", tid, src2[tid]);
    	// printf("qkvweight/outweight data[%d] = %f\n", tid + 1, src2[tid+1]);    
    	// printf("qkvweight/outweight data[%d] = %f\n", tid + 128, src2[tid+128]);
    	// printf("qkvweight/outweight data[%d] = %f\n", tid + 129, src2[tid +129]);
    	// printf("linear done\n");

    }
}
