//
// Created by liang on 2/15/18.
//
#include <utils/cuda_utils.h>
#include "registry.h"
#include "kernel.h"
#include "myatomics.h"

typedef float rank_t;

struct MyIterateKernel{
    __forceinline__ __device__ rank_t InitValue(const index_t node, index_t out_degree) const {
//        printf("call %d\n", node);
        //printf("Identity Elem: %f\n", IdentityElement());
        return 0;
    }

    __forceinline__ __device__ rank_t InitDelta(const index_t node, index_t out_degree) const {
//        printf("%d %d\n", GraphInfo.nnodes, GraphInfo.nedges);
        return 0.2;
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
};

bool PageRank() {

//    gframe::GFrameKernel<MyIterateKernel, MyAtomicAdd, rank_t, rank_t> kernel(MyAtomicAdd(),
//    false, true);
//
//    kernel.InitValue();
//    kernel.DataDriven();
//
//    if (FLAGS_output.length() > 0)
//        kernel.SaveResult(FLAGS_output.data(), true);

//    gframe::GFrameKernel<MyIterateKernel, MyAtomicAdd, rank_t, rank_t> kernel(MyIterateKernel(), MyAtomicAdd());
//    kernel.InitValue();
//    kernel.DataDriven();
    gframe::GFrameKernel<MyIterateKernel, MyAtomicAdd, rank_t, rank_t> *kernel =
            new gframe::GFrameKernel<MyIterateKernel, MyAtomicAdd, rank_t, rank_t>(MyIterateKernel(), MyAtomicAdd());
    kernel->InitValue();
    kernel->DataDriven();
    delete kernel;
    return true;
}