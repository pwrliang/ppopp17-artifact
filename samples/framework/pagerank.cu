//
// Created by liang on 2/15/18.
//
#include <utils/cuda_utils.h>
#include "registry.h"
#include "kernel.h"
#include "myatomics.h"

typedef float rank_t;

struct MyIterateKernel : public maiter::IterateKernel<rank_t, rank_t> {
    __forceinline__ __device__ rank_t InitValue(const index_t node, index_t out_degree) const {
//        printf("call %d\n", node);
        //printf("Identity Elem: %f\n", IdentityElement());
        return 0;
    }

    __forceinline__ __device__ rank_t InitDelta(const index_t node, index_t out_degree) const {
        return 0.2;
    }

    __forceinline__ __device__ float accumulate(const rank_t a, const rank_t b) const {
        return a + b;
    }

    __forceinline__ __device__ float
    g_func(const float delta, const index_t weight,
           const index_t out_degree) const {
        return 0.8f * delta / out_degree;
    }

    __forceinline__ __device__ virtual float IdentityElement() const {
        return 0.1f;
    }
};

bool PageRank() {
    maiter::MaiterKernel<MyIterateKernel, rank_t, rank_t> *kernel = new maiter::MaiterKernel<MyIterateKernel, rank_t, rank_t>(false);

//    createFunc<MyIterateKernel> << < 1, 1, 0, kernel->getStream().cuda_stream >> > (kernel->DeviceKernelObject());

    kernel->InitValue();

    kernel->DataDriven(MyAtomicAdd<rank_t>());

    if (FLAGS_output.length() > 0)
        kernel->SaveResult(FLAGS_output.data(), true);

    delete kernel;
    return true;
}