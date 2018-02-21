//
// Created by liang on 2/15/18.
//
#include <utils/cuda_utils.h>
#include "registry.h"
#include "kernel.h"
#include "myatomics.h"
#include "graph_api.h"

typedef float rank_t;

struct MyIterateKernel : gframe::api::GraphAPIBase {


    __forceinline__ __device__ rank_t InitValue(const index_t node, index_t out_degree) const {
//        printf("call %d\n", node);
        //printf("Identity Elem: %f\n", IdentityElement());
        return 0;
    }

    __forceinline__ __device__ rank_t InitDelta(const index_t node, index_t out_degree) const {
//        printf("%d %d\n", GraphInfo.nnodes, GraphInfo.nedges);
        return 0.2f / graphInfo.nnodes;
    }

    __forceinline__ __device__ float DeltaReducer(const rank_t a, const rank_t b) const {
        return a + b;
    }

    __forceinline__ __device__ float
    DeltaMapper(const float delta, const index_t weight,
                const index_t out_degree) const {
        return 0.8f * delta / out_degree;
    }

    __forceinline__ __device__ float IdentityElement() const {
        return 0.0f;
    }

    __forceinline__ __device__ bool Filter(const rank_t prev_delta, const rank_t new_delta) const {
        const rank_t EPSLION = 0.01f;
        return prev_delta < EPSLION && prev_delta + new_delta > EPSLION;
    }

    __forceinline__ __host__ bool IsConverge(const rank_t value) {
        return value > 0.841087f;
    }
};

bool PageRank() {
    gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t> *kernel =
            new gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t>(MyIterateKernel(), MyAtomicAdd(),
                                                                                             gframe::GFrameEngine<MyIterateKernel, MyAtomicAdd, rank_t, rank_t>::Engine_DataDriven, false, true);
    kernel->InitValue();
    kernel->Run();
//    kernel->DataDriven();
    delete kernel;
    return true;
}