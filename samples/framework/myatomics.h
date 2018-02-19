//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_MYATOMICS_H
#define GROUTE_MYATOMICS_H

template<typename T>
__device__ void MyAtomicAdd(T *address, T val) {
    atomicAdd(address, val);
}

#endif //GROUTE_MYATOMICS_H
