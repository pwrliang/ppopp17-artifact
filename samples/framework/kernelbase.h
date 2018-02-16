//
// Created by liang on 2/5/18.
//

#ifndef GROUTE_KERNELBASE_H
#define GROUTE_KERNELBASE_H

#include <groute/graphs/common.h>

namespace maiter {
    template<typename V, typename D>
    struct IterateKernel {
//        virtual void read_data(string& line, K& k, D& data) = 0;
        __device__ virtual V InitValue(const index_t node, index_t out_degree) = 0;
        __device__ virtual D InitDelta(const index_t node, index_t out_degree) = 0;

        __device__ virtual void accumulate(V &a, const V &b) = 0;

        //__device__ virtual void priority(V &pri, const V &value, const V &delta) = 0;

      //  __device__ virtual D g_func(const index_t node,const V value ,const D delta, const index_t weight, const index_t out_degree) = 0;
    };

    struct Algo {
        static const char *Name() { return "Maiter Kernel"; }
    };


}

#endif //GROUTE_KERNELBASE_H
