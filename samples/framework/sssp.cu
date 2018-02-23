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
typedef uint32_t dist_t;
__device__ dist_t last_delta = UINT32_MAX;

template<typename TValue, typename TDelta>
struct SSSPImpl : gframe::api::GraphAPIBase {
    const TValue IdentityElementForValueReducer = UINT32_MAX;
    const TDelta IdentityElementForDeltaReducer = UINT32_MAX;
    const index_t SRC_NODE = 0;

    __forceinline__ __device__ TValue InitValue(const index_t node, index_t out_degree) const {
        return UINT32_MAX;
    }

    __forceinline__ __device__ TDelta InitDelta(const index_t node, index_t out_degree) const {
        if (node == SRC_NODE)
            return 0;
        return UINT_MAX;
    }

    __forceinline__ __device__ TValue ValueReducer(const TValue a, const TValue b) const {
//        return a < b ? a : b;
        if (a == IdentityElementForValueReducer && b != IdentityElementForValueReducer)
            return b;
        else if (a != IdentityElementForValueReducer && b == IdentityElementForValueReducer)
            return a;
        else if (a != IdentityElementForValueReducer && b != IdentityElementForValueReducer)
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
        return a < b ? a : b;
    }

    __forceinline__ __device__ TDelta DeltaMapper(const TDelta delta, const index_t weight, const index_t out_degree) const {
        //printf("%uld -> %uld\n", delta, delta + weight);
        return delta + weight;
    }

    __forceinline__ __device__ bool Filter(const TDelta prev_delta, const TDelta new_delta) const {
        return new_delta <= prev_delta;
    }


    __forceinline__ __host__ __device__ bool IsTerminated(const TValue value, const TDelta delta) {
//        return delta != UINT32_MAX;
        printf("value sum:%uld delta sum:%uld\n", value, delta);
        return delta == 0;
    }
};

bool SSSP() {
    gframe::GFrameEngine<SSSPImpl<dist_t, dist_t>, MyAtomicMin, dist_t, dist_t> *kernel =
            new gframe::GFrameEngine<SSSPImpl<dist_t, dist_t>, MyAtomicMin, dist_t, dist_t>
                    (SSSPImpl<dist_t, dist_t>(),
                     MyAtomicMin(),
                     gframe::GFrameEngine<SSSPImpl<dist_t, dist_t>, MyAtomicMin, dist_t, dist_t>::Engine_TopologyDriven,
                     true,
                     false);
    kernel->InitValue();
    kernel->Run();
    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), true);
    delete kernel;
    return true;
}