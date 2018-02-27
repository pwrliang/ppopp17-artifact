//
// Created by liang on 2/15/18.
//
#include <utils/cuda_utils.h>
#include "registry.h"
#include "gframe.h"
#include "myatomics.h"
#include "graph_api.h"

DECLARE_string(output);
typedef float rank_t;

template<typename TValue, typename TDelta>
struct PageRankImpl : gframe::api::GraphAPIBase {
    const TValue IdentityElementForValueReducer = 0;
    const TDelta IdentityElementForDeltaReducer = 0;
    const TDelta IdentityElementForValueDeltaCombiner = 0;
    const rank_t ALPHA = 0.85f;

    __forceinline__ __device__ TValue InitValue(const index_t node, index_t out_degree) const {
        return 0;
    }

    __forceinline__ __device__ TDelta InitDelta(const index_t node, index_t out_degree) const {
        return 1 - ALPHA;
    }

    __forceinline__ __device__ TValue ValueReducer(const TValue a, const TValue b) const {
        return a + b;
    }

    __forceinline__ __device__ TDelta DeltaReducer(const TDelta a, const TDelta b) const {
        return a + b;
    }


    __forceinline__ __device__ TValue ValueDeltaCombiner(const TValue a, const TDelta b) const {
        return a + b;
    }

    __forceinline__ __device__ TDelta DeltaMapper(const TDelta delta, const index_t weight, const index_t out_degree) const {
        return ALPHA * delta / out_degree;
    }

    __forceinline__ __device__ bool Filter(const TDelta prev_delta, const TDelta new_delta) const {
        const rank_t EPSLION = 0.01f;
        return prev_delta < EPSLION && prev_delta + new_delta > EPSLION;
    }

    __forceinline__ __host__ __device__ bool IsTerminated(const TValue value, const TDelta delta) {
        return delta < 20000;
    }
};

bool PageRank() {
    gframe::GFrameEngine<PageRankImpl<rank_t, rank_t>, MyAtomicAdd, rank_t, rank_t> *kernel =
            new gframe::GFrameEngine<PageRankImpl<rank_t, rank_t>, MyAtomicAdd, rank_t, rank_t>
                    (PageRankImpl<rank_t, rank_t>(),
                     MyAtomicAdd(),
                     gframe::GFrameEngine<PageRankImpl<rank_t, rank_t>, MyAtomicAdd, rank_t, rank_t>::Engine_TopologyDriven,
                     false,
                     true);
    kernel->InitValue();
    kernel->Run();
    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), true);
    delete kernel;
    return true;
}