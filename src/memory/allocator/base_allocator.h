#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator 
{
public:
    //BaseAllocator() = default;//class with pure virtual func cant be initialized
    virtual ~BaseAllocator(){};
    // unified interface for alloc buffer
    template<typename T>
    T* Malloc(T* ptr, size_t size, bool is_host){
        return (T*)UnifyMalloc((void*)ptr, size, is_host);
    }
    virtual void* UnifyMalloc(void* ptr, size_t size, bool is_host = false) = 0;
    template<typename T>
    void Free(T* ptr, bool is_host = false){
        UnifyFree((void*)ptr, is_host);
    }
    virtual void UnifyFree(void* ptr, bool is_host = false) = 0;
};
