//
// Created by liang on 2/15/18.
//

#include "registry.h"
#include "kernel.h"

typedef float rank_t;

struct MyIterateKernel : public maiter::IterateKernel<rank_t, rank_t> {
    __forceinline__ __device__ rank_t InitValue(const index_t node, index_t out_degree) const {
//        printf("call %d\n", node);
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
        return 0.8 * delta / out_degree;
    }

    __forceinline__ __device__ virtual float IdentityElement() const {
        return 0;
    }
};

__global__ void createFunc(maiter::IterateKernel<rank_t, rank_t> **baseFunc) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *baseFunc = new MyIterateKernel();
    }
}

bool PageRank() {
    maiter::MaiterKernel<rank_t, rank_t> *kernel = new maiter::MaiterKernel<rank_t, rank_t>();

    createFunc << < 1, 1 >> > (kernel->DeviceKernelObject());
    GROUTE_CUDA_CHECK(cudaDeviceSynchronize());

    kernel->InitValue();

    kernel->DataDriven();

    delete kernel;
//    kernel->
    return true;
}