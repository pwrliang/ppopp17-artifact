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
#include <groute/graphs/csr_graph_align.h>
#include <groute/dwl/work_source.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/cuda_utils.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <cub/grid/grid_barrier.cuh>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include "pr_common.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);
DECLARE_int32(grid_size);
DECLARE_double(threshold);

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define CHECK_INTERVAL 10000000

typedef float rank_t;

namespace aligned_outlinedpr {
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
//            residual[node] = (1.0 - ALPHA) / graph.owned_nnodes();
            residual[node] = 1.0 - ALPHA;
        }
    }

    template<
            typename TGraph,
            typename WorkSource,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __device__ void PageRankKernelCTA__Single__(
            TGraph graph, WorkSource work_source,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {

            groute::dev::np_local<rank_t> local_work = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);
                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                if (res > 0) {
                    current_ranks[node] += res;

                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;

                    if (local_work.size == 0) {
//                        rank_t update = ALPHA * res;
//
//                        atomicAdd(residual.get_item_ptr(node), update);
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
    }

    template<
            typename TGraph,
            typename WorkSource,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __device__ void PageRankKernel8__Single__(
            TGraph graph, WorkSource work_source,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            rank_t res = atomicExch(residual.get_item_ptr(node), 0);

            if (res == 0)continue;

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge8(node),
                    out_degree = end_edge - begin_edge,
                    aligned_begin_edge = graph.aligned_begin_edge(node);

            const index_t VEC_SIZE = groute::graphs::VEC_SIZE;

            if (out_degree == 0) continue;

            rank_t update = ALPHA * res / out_degree;

            index_t offset = begin_edge;
            index_t aligned_offset = aligned_begin_edge;

            while (end_edge - offset >= VEC_SIZE) {
                uint8 dest8 = graph.edge_dest8(aligned_offset);

                atomicAdd(residual.get_item_ptr(dest8.a0), update);
                atomicAdd(residual.get_item_ptr(dest8.a1), update);
                atomicAdd(residual.get_item_ptr(dest8.a2), update);
                atomicAdd(residual.get_item_ptr(dest8.a3), update);
                atomicAdd(residual.get_item_ptr(dest8.a4), update);
                atomicAdd(residual.get_item_ptr(dest8.a5), update);
                atomicAdd(residual.get_item_ptr(dest8.a6), update);
                atomicAdd(residual.get_item_ptr(dest8.a7), update);

                aligned_offset++;
                offset += VEC_SIZE;
            }

            if (offset < end_edge) {
                uint8 last_trunk = graph.edge_dest8(aligned_offset);
                index_t rest_len = end_edge - offset;

                switch (rest_len){
                    case 7:
                        atomicAdd(residual.get_item_ptr(last_trunk.a6), update);
                    case 6:
                        atomicAdd(residual.get_item_ptr(last_trunk.a5), update);
                    case 5:
                        atomicAdd(residual.get_item_ptr(last_trunk.a4), update);
                    case 4:
                        atomicAdd(residual.get_item_ptr(last_trunk.a3), update);
                    case 3:
                        atomicAdd(residual.get_item_ptr(last_trunk.a2), update);
                    case 2:
                        atomicAdd(residual.get_item_ptr(last_trunk.a1), update);
                    case 1:
                        atomicAdd(residual.get_item_ptr(last_trunk.a0), update);
                }
            }
        }
    }
    template<
            typename TGraph,
            typename WorkSource,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __device__ void PageRankKernel4__Single__(
            TGraph graph, WorkSource work_source,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            rank_t res = atomicExch(residual.get_item_ptr(node), 0);

            if (res == 0)continue;

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge4(node),
                    out_degree = end_edge - begin_edge,
                    aligned_begin_edge = graph.aligned_begin_edge(node);

            const index_t VEC_SIZE = groute::graphs::VEC_SIZE;

            if (out_degree == 0) continue;

            rank_t update = ALPHA * res / out_degree;

            index_t offset = begin_edge;
            index_t aligned_offset = aligned_begin_edge;

            while (end_edge - offset >= VEC_SIZE) {
                uint4 dest4 = graph.edge_dest4(aligned_offset);

                atomicAdd(residual.get_item_ptr(dest4.x), update);
                atomicAdd(residual.get_item_ptr(dest4.y), update);
                atomicAdd(residual.get_item_ptr(dest4.z), update);
                atomicAdd(residual.get_item_ptr(dest4.w), update);

                aligned_offset++;
                offset += VEC_SIZE;
            }

            if (offset < end_edge) {
                uint4 last_trunk = graph.edge_dest4(aligned_offset);
                index_t rest_len = end_edge - offset;

                switch (rest_len){
                    case 3:
                        atomicAdd(residual.get_item_ptr(last_trunk.z), update);
                    case 2:
                        atomicAdd(residual.get_item_ptr(last_trunk.y), update);
                    case 1:
                        atomicAdd(residual.get_item_ptr(last_trunk.x), update);
                }
            }
        }
    }

    template<typename WorkSource, template<typename> class RankDatum>
    __device__ void PageRankCheck__Single__(WorkSource work_source, RankDatum<rank_t> current_ranks,
                                            rank_t *block_sum_buffer, rank_t *rtn_sum) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        const int SMEMDIM = blockDim.x / warpSize;
        extern __shared__ rank_t smem[];

        uint32_t work_size = work_source.get_size();
        rank_t local_sum = 0;

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            local_sum += current_ranks[node];
        }

        local_sum = warpReduce(local_sum);

        if (laneIdx == 0)
            smem[warpIdx] = local_sum;
        __syncthreads();

        local_sum = (threadIdx.x < SMEMDIM) ? smem[threadIdx.x] : 0;

        if (warpIdx == 0)
            local_sum = warpReduce(local_sum);

        if (threadIdx.x == 0) {
            block_sum_buffer[blockIdx.x] = local_sum;
        }

        if (tid == 0) {
            double sum = 0;
            for (int bid = 0; bid < gridDim.x; bid++) {
                sum += block_sum_buffer[bid];
            }
            *rtn_sum = sum;
        }
    }

    template<
            typename TGraph,
            typename WorkSource,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankControlCTA__Single__(
            double threshold,
            TGraph graph,
            WorkSource work_source,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            rank_t *block_sum_buffer,
            int *running,
            cub::GridBarrier gridBarrier) {
        unsigned tid = TID_1D;

        rank_t pr_sum;
        uint32_t counter = 0;
        while (*running) {
            PageRankKernelCTA__Single__(graph, work_source, current_ranks, residual);
//            gridBarrier.Sync();
            PageRankCheck__Single__(work_source, current_ranks, block_sum_buffer, &pr_sum);
//            gridBarrier.Sync();
            if (counter++ % 500000 == 0) {
                if (tid == 0) {
//                    pr_sum = 0;
//                    for (int i = 0; i < graph.nnodes; i++) {
//                        pr_sum += current_ranks[i];
//                    }

                    printf("pr_sum:%f\n", pr_sum);
                    *running = pr_sum < threshold ? 1 : 0;
                }
                counter = 0;
            }
            gridBarrier.Sync();
        }
    };

    template<
            typename TGraph,
            typename WorkSource,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankControl__Single__(
            double threshold,
            TGraph graph,
            WorkSource work_source,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            rank_t *block_sum_buffer,
            int *running,
            cub::GridBarrier gridBarrier) {
        unsigned tid = TID_1D;

        rank_t pr_sum;
        uint32_t counter = 0;
        while (*running) {
            PageRankKernel8__Single__(graph, work_source, current_ranks, residual);
//            gridBarrier.Sync();
            PageRankCheck__Single__(work_source, current_ranks, block_sum_buffer, &pr_sum);
//            gridBarrier.Sync();
            if (counter++ % 500000 == 0) {
                if (tid == 0) {
//                    pr_sum = 0;
//                    for (int i = 0; i < graph.nnodes; i++) {
//                        pr_sum += current_ranks[i];
//                    }

                    printf("pr_sum:%f\n", pr_sum);
                    *running = pr_sum < threshold ? 1 : 0;
                }
                counter = 0;
            }
            gridBarrier.Sync();
        }
    };

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

        Problem(const TGraph &graph,
                RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) :
                m_graph(graph), m_residual(residual), m_current_ranks(current_ranks) {
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
            return pr_sum < FLAGS_threshold;
        }

        template<typename WorkSource>
        void DoPageRank(WorkSource work_source, index_t blocksPerGrid, groute::Stream &stream) {
            const int SMEMDIM = FLAGS_block_size / 32;
            cub::GridBarrierLifetime gridBarrier;
            utils::SharedArray<rank_t> sum_buffer(blocksPerGrid);
            utils::SharedValue<int> running_flag;

            GROUTE_CUDA_CHECK(cudaMemset(sum_buffer.dev_ptr, 0, sum_buffer.buffer_size * sizeof(rank_t)));
            running_flag.set_val_H2D(1);

            gridBarrier.Setup(blocksPerGrid);
            if (FLAGS_cta_np) {
                printf("run with cta\n");
                PageRankControlCTA__Single__ << < blocksPerGrid, FLAGS_block_size, sizeof(rank_t) * SMEMDIM,
                        stream.cuda_stream >> >
                        (FLAGS_threshold, m_graph, work_source, m_current_ranks, m_residual
                                , sum_buffer.dev_ptr, running_flag.dev_ptr, gridBarrier);
            } else {
                printf("run with vectorize\n");
                PageRankControl__Single__ << < blocksPerGrid, FLAGS_block_size, sizeof(rank_t) * SMEMDIM,
                        stream.cuda_stream >> >
                        (FLAGS_threshold, m_graph, work_source, m_current_ranks, m_residual
                                , sum_buffer.dev_ptr, running_flag.dev_ptr, gridBarrier);
            }
        }
    };

}


bool AlignedMyTestPageRankSingleOutlined() {
    utils::traversal::Context<aligned_outlinedpr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocatorAlign dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    index_t blocksPerGrid = FLAGS_grid_size;


    aligned_outlinedpr::Problem<groute::graphs::dev::CSRGraphAlign, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(), current_ranks.DeviceObject(), residual.DeviceObject());

    Stopwatch stopwatch;

    if (FLAGS_cta_np) {
        printf("CTA enabled\n");
    }

    stopwatch.start();

    solver.Init__Single__(stream);
    stream.Sync();
    solver.DoPageRank(groute::dev::WorkSourceRange<index_t>(dev_graph_allocator.DeviceObject().owned_start_node(),
                                                            dev_graph_allocator.DeviceObject().owned_nnodes()),
                      blocksPerGrid,
                      stream);
    stream.Sync();
    stopwatch.stop();

    printf("persist kernel PR:%f\n", stopwatch.ms());

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}