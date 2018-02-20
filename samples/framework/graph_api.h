//
// Created by liang on 2/5/18.
//

#ifndef GROUTE_KERNELBASE_H
#define GROUTE_KERNELBASE_H

#include <groute/graphs/common.h>

namespace maiter {
    template<typename TValue, typename TDelta>
    struct IterateKernel {
//        virtual void read_data(string& line, K& k, D& data) = 0;
        __forceinline__ __device__ virtual TValue InitValue(const index_t node, index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TDelta InitDelta(const index_t node, index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TValue accumulate(const TValue value, const TDelta delta) const = 0;

        //__device__ virtual void priority(V &pri, const V &value, const V &delta) = 0;

        __forceinline__ __device__ virtual TDelta
        g_func(const TDelta delta, const index_t weight,
               const index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TDelta IdentityElement() const = 0;
    };

    struct Algo {
        static const char *Name() { return "Maiter Kernel"; }
    };


}

#endif //GROUTE_KERNELBASE_H
