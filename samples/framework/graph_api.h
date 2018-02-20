//
// Created by liang on 2/5/18.
//

#ifndef GROUTE_KERNELBASE_H
#define GROUTE_KERNELBASE_H

#include <groute/graphs/common.h>

namespace gframe {
    template<typename TValue, typename TDelta>
    struct GraphAPIBase {
        struct  {
            index_t nnodes;
            index_t nedges;

        } GraphInfo;

        __forceinline__ __device__ virtual TValue InitValue(const index_t node, index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TDelta InitDelta(const index_t node, index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TValue DeltaReducer(const TValue value, const TDelta delta) const = 0;

        __forceinline__ __device__ virtual bool Filter(const TDelta before_accumulate_delta, const TDelta new_delta) const = 0;
        //__device__ virtual void priority(V &pri, const V &value, const V &delta) = 0;

        __forceinline__ __device__ virtual TDelta
        DeltaMapper(const TDelta delta, const index_t weight,
               const index_t out_degree) const = 0;

        __forceinline__ __device__ virtual TDelta IdentityElement() const = 0;
    };

    struct Algo {
        static const char *Name() { return "gframe Kernel"; }
    };


}

#endif //GROUTE_KERNELBASE_H
