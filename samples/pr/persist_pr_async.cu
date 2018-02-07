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

#include <groute/event_pool.h>

#include <groute/graphs/csr_graph.h>
#include <groute/dwl/work_source.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/cuda_utils.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include "pr_common.h"
#include "my_pr_balanced.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define THRESHOLD 0.9999
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
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankPersistKernel__Single__(
            TGraph graph,
            index_t *lbounds, index_t *ubounds,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            int *running) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        index_t node_start = lbounds[blockIdx.x];
        index_t node_end = ubounds[blockIdx.x];
        extern __shared__ rank_t shared_mem[];
//        int cache_size = cub::
        int current_round;

        while (*running) {
            rank_t local_sum = 0;

            for (index_t node = node_start + threadIdx.x; node < node_end; node += blockDim.x) {

                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                if (res == 0)continue;

                current_ranks[node] += res;

                local_sum+=current_ranks[node];

                index_t begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                if (out_degree == 0) continue;

                rank_t update = ALPHA * res / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                }
            }

            if (current_round++ % 10000 == 0) {
                local_sum = warpReduce(local_sum);

                if(cub::LaneId()==0){
                    shared_mem[cub::WarpId()] = local_sum
                }
                __syncthreads();

                if(threadIdx.x<warpSize)
                    local_sum = (threadIdx.x<)
            }
        }
    }

    void balanced_alloctor(index_t elems_num, index_t *p_degree, index_t blocksPerGrid, index_t *p_o_lbounds,
                           index_t *p_o_ubounds) {
        int total_degree = 0;

        for (int v_idx = 0; v_idx < elems_num; v_idx++) {
            total_degree += p_degree[v_idx];
        }

        int avg_degree = total_degree / blocksPerGrid;
        assert(avg_degree > 0);

        int start_idx = 0;
        int end_idx = 0;
        int degree_in_block = 0;
        int bid = 0;
        for (int v_idx = 0; v_idx < elems_num; v_idx++) {
            degree_in_block += p_degree[v_idx];

            if (degree_in_block >= avg_degree) {
                end_idx = v_idx + 1;    // include this vertex

                end_idx = std::min<index_t>(end_idx, elems_num);
                p_o_lbounds[bid] = start_idx;
                p_o_ubounds[bid] = end_idx;
                bid++;
                start_idx = end_idx;
                degree_in_block = 0;
            }
        }
        p_o_lbounds[bid] = start_idx;
        p_o_ubounds[bid] = elems_num;
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
            return pr_sum < THRESHOLD;
        }

        void DoPageRank(index_t blocksPerGrid, groute::Stream &stream) {
            utils::SharedValue<int> running;
            running.set_val_H2D(1);

            PageRankPersistKernel__Single__ << < blocksPerGrid, FLAGS_block_size >> > (m_graph, m_lbounds, m_ubounds,
                    m_current_ranks, m_residual, running);
        }
    };
}


bool MyTestPageRankSinglePersist() {
    utils::traversal::Context<persistpr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context;

    utils::SharedArray<index_t> node_lbounds(context.host_graph.nnodes);
    utils::SharedArray<index_t> node_ubounds(context.host_graph.nnodes);
    utils::SharedArray<index_t> node_outdegrees(context.host_graph.nnodes);

    for (index_t node = 0; node < context.host_graph.nnodes; node++) {
        index_t out_degree = context.host_graph.end_edge(node) - context.host_graph.begin_edge(node);
        node_outdegrees.host_ptr()[node] = out_degree;
    }

    index_t blocksPerGrid = 20;

    persistpr::balanced_alloctor(context.host_graph.nnodes, node_outdegrees.host_ptr(), blocksPerGrid,
                                 node_lbounds.host_ptr(), node_ubounds.host_ptr());

    persistpr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(), node_lbounds.dev_ptr, node_ubounds.dev_ptr,
                   current_ranks.DeviceObject(), residual.DeviceObject());

    Stopwatch stopwatch;

    stopwatch.start();

    solver.Init__Single__(stream);

    solver.DoPageRank(blocksPerGrid, stream);

    stopwatch.stop();

    printf("non-balancing PR:%f\n", stopwatch.ms());

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}