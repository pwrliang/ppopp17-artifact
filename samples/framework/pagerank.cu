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

struct MyIterateKernel : gframe::api::GraphAPIBase {
    const rank_t ALPHA = 0.85f;

    __forceinline__ __device__ rank_t InitValue(const index_t node, index_t out_degree) const {
//        printf("call %d\n", node);
        //printf("Identity Elem: %f\n", IdentityElement());
        return 0;
    }

    __forceinline__ __device__ rank_t InitDelta(const index_t node, index_t out_degree) const {
//        return (1 - ALPHA) / graphInfo.nnodes;
        return 1 - ALPHA;
    }

    __forceinline__ __device__ float DeltaReducer(const rank_t a, const rank_t b) const {
        return a + b;
    }

    __forceinline__ __device__ float
    DeltaMapper(const float delta, const index_t weight,
                const index_t out_degree) const {
        //assert(weight == 0);
        return ALPHA * delta / out_degree;
    }

    __forceinline__ __device__ float IdentityElement() const {
        return 0.0f;
    }

    __forceinline__ __device__ bool Filter(const rank_t prev_delta, const rank_t new_delta) const {
        const rank_t EPSLION = 0.01f;
        return prev_delta < EPSLION && prev_delta + new_delta > EPSLION;
    }

    __forceinline__ __host__ __device__ bool IsConverge(const rank_t value) {
        return value > 3.91682e+06;
    }
};

bool PageRank() {
    gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t> *kernel =
            new gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t>
                    (MyIterateKernel(),
                     MyAtomicAdd(),
                     gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t>::Engine_TopologyDriven,
                     false,
                     true);
    kernel->InitValue();
    kernel->Run();
    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), true);
    delete kernel;
    return true;
}