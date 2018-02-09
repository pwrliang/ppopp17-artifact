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
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <gflags/gflags.h>
#include <cub/cub.cuh>
#include <groute/event_pool.h>
#include <groute/device/cta_scheduler.cuh>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/work_source.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/cuda_utils.h>
#include <utils/balancer.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include "pr_common.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);
DECLARE_int32(grid_size);


#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define CHECK_INTERVAL 10000000
#define FILTER_THRESHOLD 0.0000000001

typedef float rank_t;

namespace persistpr {
    struct Algo {

        static const char *Name() { return "PR"; }
    };

    __inline__ __device__ float warpReduce(float localSum) {
        localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

        return localSum;
    }

    template<typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        index_t start_node = graph.owned_start_node();
        index_t end_node = start_node + graph.owned_nnodes();
        for (index_t node = start_node + tid; node < end_node; node += nthreads) {
            current_ranks[node] = 0.0;
            residual[node] = (1.0 - ALPHA) / graph.owned_nnodes();
//            residual[node] = 1.0 - ALPHA;
        }
    }

    template<typename TMetaData>
    struct work_chunk {
        index_t start;
        index_t size;
        TMetaData meta_data;
    };

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankPersistKernelCTA__Single__(
            TGraph graph,
            index_t *lbounds, index_t *ubounds,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            rank_t *block_sum,
            int *running) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        index_t node_start = lbounds[blockIdx.x];
        index_t node_end = ubounds[blockIdx.x];
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        extern __shared__ rank_t smem[];
        const int SMEMDIM = blockDim.x / warpSize;

        int current_round = 0;

        uint32_t work_size = node_end - node_start;
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        while (*running) {
            rank_t local_sum = 0;

            for (uint32_t i = 0 + threadIdx.x; i < work_size_rup; i += blockDim.x) {

                groute::dev::np_local<rank_t> local_work = {0, 0, 0.0};

                if (i < work_size) {
                    index_t node = i + node_start;
                    rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                    current_ranks[node] += res;
                    local_sum += current_ranks[node];

                    if (res > 0) {
                        local_work.start = graph.begin_edge(node);
                        local_work.size = graph.end_edge(node) - local_work.start;

                        if (local_work.size == 0) {
//                            rank_t update = ALPHA * res;
//
//                            atomicAdd(residual.get_item_ptr(node), update);
                        } else {
                            rank_t update = ALPHA * res / local_work.size;

                            local_work.meta_data = update;
                        }
                    }
                }

                groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                        local_work,
                        [&graph, &residual](index_t edge, rank_t update) {
                            index_t dest = graph.edge_dest(edge);

                            atomicAdd(residual.get_item_ptr(dest), update);
                        }
                );
            }

            if (current_round++ % CHECK_INTERVAL == 0) {
                //naive check implementation
//                if (tid == 0) {
//                    double sum = 0;
//                    for (int node = 0; node < graph.nnodes; node++) {
//                        sum += current_ranks[node];
//                    }
//                    printf("sum:%f\n", sum);
//                    *running = sum < THRESHOLD;
//                }

                local_sum = warpReduce(local_sum);

                if (laneIdx == 0)
                    smem[warpIdx] = local_sum;
                __syncthreads();

                if (threadIdx.x < warpSize)
                    local_sum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

                if (warpIdx == 0)
                    local_sum = warpReduce(local_sum);

                if (threadIdx.x == 0)
                    block_sum[blockIdx.x] = local_sum;

                if (tid == 0) {
                    double sum = 0;
                    for (int bid = 0; bid < gridDim.x; bid++) {
                        sum += block_sum[bid];
                    }
                    printf("%f\n", sum);
                    *running = sum < FLAGS_THRESHOLD;
                }
                current_round = 0;
            }
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankPersistKernel__Single__(
            TGraph graph,
            index_t *lbounds, index_t *ubounds,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            rank_t *block_sum,
            int *running) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        index_t node_start = lbounds[blockIdx.x];
        index_t node_end = ubounds[blockIdx.x];
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        extern __shared__ rank_t smem[];
        const int SMEMDIM = blockDim.x / warpSize;

        uint32_t work_size = node_end - node_start;
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
        int current_round = 0;

        while (*running) {
            rank_t local_sum = 0;

            for (uint32_t i = 0 + threadIdx.x; i < work_size_rup; i += blockDim.x) {

                work_chunk<rank_t> local_work = {0, 0, 0.0};

                if (i < work_size) {
                    index_t node = i + node_start;
                    rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                    current_ranks[node] += res;
                    local_sum += current_ranks[node];

                    if (res > 0) {

                        local_work.start = graph.begin_edge(node);
                        local_work.size = graph.end_edge(node) - local_work.start;

                        if (local_work.size == 0) {
                            //rank_t update = ALPHA * res;

                           // atomicAdd(residual.get_item_ptr(node), update);
                        } else {
                            rank_t update = ALPHA * res / local_work.size;

                            local_work.meta_data = update;
                        }
                    }
                }

                const int lane_id = cub::LaneId();

                while (__any(local_work.size >= warpSize)) {
                    int mask = __ballot(local_work.size >= warpSize ? 1 : 0);
                    int leader = __ffs(mask) - 1;

                    index_t start = cub::ShuffleIndex(local_work.start, leader);
                    index_t size = cub::ShuffleIndex(local_work.size, leader);
                    rank_t meta_data = cub::ShuffleIndex(local_work.meta_data, leader);

                    if (leader == lane_id) {
                        local_work.start = 0;
                        local_work.size = 0;
                    }

                    // use all thread in warp to do the task
                    for (int edge = lane_id; edge < size; edge += warpSize) {
                        index_t dest = graph.edge_dest(start + edge);

                        atomicAdd(residual.get_item_ptr(dest), meta_data);
                    }
                }

                __syncthreads();

                for (int edge = 0; edge < local_work.size; edge++) {
                    index_t dest = graph.edge_dest(local_work.start + edge);

                    atomicAdd(residual.get_item_ptr(dest), local_work.meta_data);
                }
            }

            if (current_round++ % CHECK_INTERVAL == 0) {
                //naive check implementation
//                if (tid == 0) {
//                    double sum = 0;
//                    for (int node = 0; node < graph.nnodes; node++) {
//                        sum += current_ranks[node];
//                    }
//                    printf("sum:%f\n", sum);
//                    *running = sum < THRESHOLD;
//                }

                local_sum = warpReduce(local_sum);

                if (laneIdx == 0)
                    smem[warpIdx] = local_sum;
                __syncthreads();

                if (threadIdx.x < warpSize)
                    local_sum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

                if (warpIdx == 0)
                    local_sum = warpReduce(local_sum);

                if (threadIdx.x == 0) {
                    block_sum[blockIdx.x] = local_sum;
                }

                if (tid == 0) {
                    double sum = 0;
                    for (int bid = 0; bid < gridDim.x; bid++) {
                        sum += block_sum[bid];
                    }
                    printf("%f\n", sum);
                    *running = sum < FLAGS_THRESHOLD;
                }
                current_round = 0;
            }
        }
    }

/*
* The per-device Page Rank problem
*/
    template<typename TGraph,
            template<typename> class ResidualDatum,
            template<typename> class RankDatum>
    struct Problem {
        TGraph m_graph;
        ResidualDatum<rank_t> m_residual;
        RankDatum<rank_t> m_current_ranks;
        index_t *m_lbounds;
        index_t *m_ubounds;

        Problem(const TGraph &graph,
                index_t *lbounds,
                index_t *ubounds,
                RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) :
                m_graph(graph), m_lbounds(lbounds), m_ubounds(ubounds),
                m_residual(residual), m_current_ranks(current_ranks) {
        }

        void Init__Single__(groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");
            PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (m_graph, m_current_ranks, m_residual);
        }

        template<typename WorkSource>
        bool RankCheck__Single__(WorkSource work_source, mgpu::context_t &context) {
            rank_t *tmp = m_current_ranks.data_ptr;

            printf("work size:%d\n", work_source.get_size());

            auto check_segment_sizes = [=]__device__(int idx) {
                return tmp[idx];
            };

            mgpu::mem_t<rank_t> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);
            int *scanned_offsets = deviceOffsets.data();
            mgpu::transform_scan<rank_t>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<rank_t>(), checkSum.data(), context);
            rank_t pr_sum = mgpu::from_mem(checkSum)[0];
            std::cout << "checking...sum " << pr_sum << std::endl;
            return pr_sum < FLAGS_THRESHOLD;
        }

        void DoPageRank(index_t blocksPerGrid, groute::Stream &stream) {
            utils::SharedValue<int> running;
            utils::SharedArray<rank_t> sum_buffer(blocksPerGrid);

            running.set_val_H2D(1);
            GROUTE_CUDA_CHECK(cudaMemset(sum_buffer.dev_ptr, 0, sum_buffer.buffer_size * sizeof(rank_t)));

            const int SMEMDIM = FLAGS_block_size / 32;

            if (FLAGS_cta_np) {
                PageRankPersistKernelCTA__Single__ << < blocksPerGrid, FLAGS_block_size, sizeof(rank_t) * SMEMDIM,
                        stream.cuda_stream >> >
                        (m_graph, m_lbounds, m_ubounds, m_current_ranks, m_residual
                                , sum_buffer.dev_ptr, running.dev_ptr);
            } else {
                PageRankPersistKernel__Single__ << < blocksPerGrid, FLAGS_block_size, sizeof(rank_t) * SMEMDIM,
                        stream.cuda_stream >> >
                        (m_graph, m_lbounds, m_ubounds, m_current_ranks, m_residual
                                , sum_buffer.dev_ptr, running.dev_ptr);
            }
        }
    };

}


bool MyTestPageRankSinglePersist() {
    printf("running MyTestPageRankSinglePersist\n");
    utils::traversal::Context<persistpr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    utils::SharedArray<index_t> node_lbounds(FLAGS_grid_size);
    utils::SharedArray<index_t> node_ubounds(FLAGS_grid_size);
    utils::SharedArray<index_t> node_outdegrees(context.host_graph.nnodes);

    for (index_t node = 0; node < context.host_graph.nnodes; node++) {
        index_t out_degree = context.host_graph.end_edge(node) - context.host_graph.begin_edge(node);
        node_outdegrees.host_ptr()[node] = out_degree;
    }

    index_t blocksPerGrid = FLAGS_grid_size;

    groute::balanced_alloctor(context.host_graph.nnodes, node_outdegrees.host_ptr(), blocksPerGrid,
                                 node_lbounds.host_ptr(), node_ubounds.host_ptr());

    node_lbounds.H2D();
    node_ubounds.H2D();

    persistpr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(), node_lbounds.dev_ptr, node_ubounds.dev_ptr,
                   current_ranks.DeviceObject(), residual.DeviceObject());

    Stopwatch stopwatch;

    if (FLAGS_cta_np) {
        printf("CTA enabled\n");
    }

    stopwatch.start();

    solver.Init__Single__(stream);
    stream.Sync();
    solver.DoPageRank(blocksPerGrid, stream);

    stopwatch.stop();

    printf("persist kernel PR:%f\n", stopwatch.ms());

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}