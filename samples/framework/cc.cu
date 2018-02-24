//
// Created by liang on 2/15/18.
//
#include <utils/cuda_utils.h>
#include "registry.h"
#include "gframe.h"
#include "myatomics.h"
#include "graph_api.h"
#include <stdint.h>

DECLARE_string(output);

template<typename TValue, typename TDelta>
struct CCImpl : gframe::api::GraphAPIBase {
    const TValue IdentityElementForValueReducer = 0;
    const TDelta IdentityElementForDeltaReducer = -UINT32_MAX;

    __forceinline__ __device__ TValue InitValue(const index_t node, index_t out_degree) const {
        return -UINT32_MAX;
    }

    __forceinline__ __device__ TDelta InitDelta(const index_t node, index_t out_degree) const {
        return node;
    }

    __forceinline__ __device__ TValue ValueReducer(const TValue a, const TValue b) const {
        if (a == IdentityElementForDeltaReducer && b != IdentityElementForDeltaReducer)
            return b;
        else if (a != IdentityElementForDeltaReducer && b == IdentityElementForDeltaReducer)
            return a;
        else if (a != IdentityElementForDeltaReducer && b != IdentityElementForDeltaReducer)
            return a + b;
        return 0;
    }

    __forceinline__ __device__ TDelta DeltaReducer(const TDelta a, const TDelta b) const {
        if (a == IdentityElementForDeltaReducer && b != IdentityElementForDeltaReducer)
            return b;
        else if (a != IdentityElementForDeltaReducer && b == IdentityElementForDeltaReducer)
            return a;
        else if (a != IdentityElementForDeltaReducer && b != IdentityElementForDeltaReducer)
            return a + b;
        return 0;
    }


    __forceinline__ __device__ TValue ValueDeltaCombiner(const TValue a, const TDelta b) const {
        return a < b ? b : a;
    }

    __forceinline__ __device__ TDelta DeltaMapper(const TDelta delta, const index_t weight, const index_t out_degree) const {
        return delta;
    }

    __forceinline__ __device__ bool Filter(const TDelta prev_delta, const TDelta new_delta) const {
        return new_delta <= prev_delta;
    }


    __forceinline__ __host__ __device__ bool IsTerminated(const TValue value, const TDelta delta) {
        return delta == 0;
    }
};

bool CC() {
    gframe::GFrameEngine<CCImpl<index_t, index_t>, MyAtomicMax, index_t, index_t> *kernel =
            new gframe::GFrameEngine<CCImpl<index_t, index_t>, MyAtomicMax, index_t, index_t>
                    (CCImpl<index_t, index_t>(),
                     MyAtomicMax(),
                     gframe::GFrameEngine<CCImpl<index_t, index_t>, MyAtomicMax, index_t, index_t>::Engine_TopologyDriven,
                     false,
                     false);
    kernel->InitValue();
    kernel->Run();
    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), false);
    delete kernel;
    return true;
}