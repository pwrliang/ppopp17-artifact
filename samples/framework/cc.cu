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
    const TDelta IdentityElementForDeltaReducer = -999999999;

    __forceinline__ __device__ TValue InitValue(const int node, int out_degree) const {
        return IdentityElementForDeltaReducer;
    }

    __forceinline__ __device__ TDelta InitDelta(const int node, int out_degree) const {
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

    __forceinline__ __device__ TDelta DeltaMapper(const TDelta delta, const int weight, const int out_degree) const {
        return delta;
    }

    __forceinline__ __device__ bool Filter(const TDelta prev_delta, const TDelta new_delta) const {
        return new_delta <= prev_delta;
    }


    __forceinline__ __host__ __device__ bool IsTerminated(const TValue value, const TDelta delta) {
        printf("%d\n", value);
        return delta == 0;
    }
};

bool CC() {
    gframe::GFrameEngine<CCImpl<int, int>, MyAtomicMax, int, int> *kernel =
            new gframe::GFrameEngine<CCImpl<int, int>, MyAtomicMax, int, int>
                    (CCImpl<int, int>(),
                     MyAtomicMax(),
                     gframe::GFrameEngine<CCImpl<int, int>, MyAtomicMax, int, int>::Engine_TopologyDriven,
                     false,
                     false);
    kernel->InitValue();
    kernel->Run();
    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), false);
    delete kernel;
    return true;
}