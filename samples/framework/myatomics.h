//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_MYATOMICS_H
#define GROUTE_MYATOMICS_H

struct MyAtomicAdd {
    template<typename T>
    __forceinline__ __device__
    T operator()(T *address, T val) {
        return atomicAdd(address, val);
    }
};

struct MyAtomicMin {
    template<typename T>
    __forceinline__ __device__
    T operator()(T *address, T val) {
        return atomicMin(address, val);
    }
};

struct MyAtomicMax {
    template<typename T>
    __forceinline__ __device__
    T operator()(T *address, T val) {
        return atomicMax(address, val);
    }
};

#endif //GROUTE_MYATOMICS_H
