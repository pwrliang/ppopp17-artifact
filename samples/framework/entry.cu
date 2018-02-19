//
// Created by liang on 2/15/18.
//

#include <cstdio>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <iostream>
#include <utils/utils.h>
#include <utils/interactor.h>
#include <utils/app_skeleton.h>
#include "kernel.h"
#include "registry.h"
DEFINE_double(wl_alloc_factor, 0.2, "Local worklists will allocate '(nedges / ngpus)' times this factor");
DEFINE_uint64(wl_alloc_abs, 0, "Absolute size for local worklists (if not zero, overrides --wl_alloc_factor");

bool PageRank();
void CleanupGraphs();

namespace maiter {
    struct App {
        static const char *Name() { return "page rank"; }

        static const char *NameUpper() { return "Page Rank"; }

        static bool Single() {
            LOG(INFO) << "Run Single" << std::endl;
            return PageRank();
        }

        static bool AsyncMulti(int G) {
//            return FLAGS_opt ? TestPageRankAsyncMultiOptimized(G) : TestPageRankAsyncMulti(G);
            return false;
        }

        static void Cleanup() { CleanupGraphs(); }
    };
}

int main(int argc, char **argv) {
    Skeleton<maiter::App> app;
    int exit = app(argc, argv);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return exit;
}
