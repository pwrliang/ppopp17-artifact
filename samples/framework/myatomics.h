//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_MYATOMICS_H
#define GROUTE_MYATOMICS_H

template<typename T>
struct MyAtomicAdd {
    __forceinline__ __device__ T operator()(T *address, T val) {
        return atomicAdd(address, val);
    }
};

template<typename T>
struct MyAtomicMin {
    __forceinline__ __device__ T operator()(T *address, T val) {
        return atomicMin(address, val);
    }
};

template<typename T>
struct MyAtomicMax {
    __forceinline__ __device__ T operator()(T *address, T val) {
        return atomicMax(address, val);
    }
};

#endif //GROUTE_MYATOMICS_H
