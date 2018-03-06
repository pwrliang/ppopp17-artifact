// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <cstdio>
#include <limits>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <utils/utils.h>
#include <utils/app_skeleton.h>

DEFINE_bool(data_driven, true, "Data-Driven mode (default)");
DEFINE_double(wl_alloc_factor, 0.2, "Local worklists will allocate '(nedges / ngpus)' times this factor");
DEFINE_uint64(wl_alloc_abs, 0, "Absolute size for local worklists (if not zero, overrides --wl_alloc_factor");
DEFINE_int32(max_pr_iterations, 200,
             "The maximum number of PR iterations"); // used just for host and some single versions
DEFINE_double(epsilon, 0.01, "EPSILON (default 0.01)");
DEFINE_bool(outlining, false, "Enable outlining");
DEFINE_double(threshold, std::numeric_limits<double>::max(), "PR sum as threshold");
DEFINE_bool(append_warp, true, "Parallel append warp");
DEFINE_int32(mode, 0, "0 sync 1 async 2 hybrid");
DEFINE_int32(switch_threshold, -1, "using sync mode when iteration times below threshold otherwise async mode");
DEFINE_bool(topology, true, "default topology");
DEFINE_bool(force_sync, false, "");
DEFINE_bool(force_async, false, "");

bool HybridDataDriven();

bool HybridTopologyDriven1();

bool TestPageRankAsyncMulti();

//void CleanupGraphs();

namespace pr {
    struct App {
        static const char *Name() { return "page rank"; }

        static const char *NameUpper() { return "Page Rank"; }

        static bool Single() {
//            return HybridDataDriven();
            return HybridTopologyDriven1();
//            return TestPageRankAsyncMulti();
//            return DualGPU();
//            HybridDataDriven();
//            if (FLAGS_topology)
//                HybridTopologyDriven();
//            else
//                HybridDataDriven();
        }

        static bool AsyncMulti(int G) {
            return false;
        }

        static void Cleanup() { CleanupGraphs(); }
    };
}


int main(int argc, char **argv) {
    Skeleton<pr::App> app;
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
