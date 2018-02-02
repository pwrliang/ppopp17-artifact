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
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include "pr_common.h"
//#include "pr_common.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define THRESHOLD 0.9999
#define FILTER_THRESHOLD 0.0000000001

typedef float rank_t;

namespace mypr {
    struct Algo {

        static const char *Name() { return "PR"; }
    };

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
//            current_ranks[node] = 0.0;
//            residual[node] = (1.0 - ALPHA) / graph.owned_nnodes();
            current_ranks[node] = 1.0 - ALPHA;

            index_t
                    begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;

            rank_t update = ((1.0 - ALPHA) * ALPHA) / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t dest = graph.edge_dest(edge);
                atomicAdd(residual.get_item_ptr(dest), update);
            }
        }
    }


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource,
            template<typename> class TWorkList>
    __global__ void PageRankKernel__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            WorkSource work_source, TWorkList<index_t> output_worklist) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();


        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);

            rank_t res = atomicExch(residual.get_item_ptr(node), 0);

            if (res == 0)continue;
//            if (res > 0.0000000001) {
//                output_worklist.append_warp(node);
//            }

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) {
//                rank_t update = ALPHA * res;
//                atomicAdd(residual.get_item_ptr(node), update);
            } else {
                rank_t update = ALPHA * res / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                    if (prev + update > EPSILON && prev < EPSILON) {
                        output_worklist.append_warp(dest);
                    }
//                rank_t prev = atomicAdd(&residual[dest], update);
//                printf("%d %d\n", node, dest);
                }
            }
        }
    }

    template<template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankPrint__Single__(RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
                                            uint32_t work_size) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;


        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = i;
            printf("%d %f %f\n", node, current_ranks[node], residual[node]);
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

        Problem(const TGraph &graph, RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) :
                m_graph(graph), m_residual(residual), m_current_ranks(current_ranks) {
        }

        void Init__Single__(groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");
            PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (m_graph, m_current_ranks, m_residual);
//            stream.Sync();
//            PageRankPrint__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
//                                                                   (m_current_ranks, m_residual, m_graph.owned_nnodes());
        }

        template<typename WorkSource,
                template<typename> class TWorkList>
        void
        Relax__Single__(const WorkSource &work_source, TWorkList<index_t> &output_worklist, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

//            printf("size %d\n", work_source.get_size());

            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");
            PageRankKernel__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                    (m_graph, m_current_ranks, m_residual, work_source, output_worklist.DeviceObject());
            // stream.Sync();
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
//            printf("pr_sum:%f\n", pr_sum / work_source.get_size());
//            return pr_sum / work_source.get_size() < 0.982861;
            return pr_sum < THRESHOLD;
        }
    };
}

bool MyTestPageRankSingle() {
    utils::traversal::Context<mypr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context;

    mypr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(),
                   current_ranks.DeviceObject(), residual.DeviceObject());

    solver.Init__Single__(stream);


    int iteration = 0;

    index_t *queue_space = static_cast<index_t *>(context.Alloc(0, sizeof(index_t) * context.host_graph.nnodes,
                                                                sizeof(index_t)));
//    GROUTE_CUDA_CHECK(cudaMalloc(&queue_space, sizeof(index_t) * context.host_graph.nnodes));



//    groute::dev::Queue<index_t> worklist(queue_space, 0, context.host_graph.nnodes);
    groute::Queue<index_t> wl1(context.host_graph.nnodes, 0, "input queue");
    groute::Queue<index_t> wl2(context.host_graph.nnodes, 0, "output queue");

    wl1.ResetAsync(stream);
    wl2.ResetAsync(stream);
    stream.Sync();

    groute::Queue<index_t> *in_wl = &wl1, *out_wl = &wl2;

    solver.Relax__Single__(groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(),
            dev_graph_allocator.DeviceObject().owned_nnodes()), *in_wl, stream);

    groute::Segment<index_t> work_seg = in_wl->GetSeg(stream);

    while (work_seg.GetSegmentSize() > 0) {
        solver.Relax__Single__(
                groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(), work_seg.GetSegmentSize()), *out_wl,
                stream);

        printf("iteration:%d active nodes:%d\n", iteration++, out_wl->GetCount(stream));

        in_wl->ResetAsync(stream);
        stream.Sync();
        std::swap(in_wl, out_wl);
        work_seg = in_wl->GetSeg(stream);
//        running = solver.RankCheck__Single__(groute::dev::WorkSourceRange<index_t>(
//                dev_graph_allocator.DeviceObject().owned_start_node(),
//                dev_graph_allocator.DeviceObject().owned_nnodes()),
//                                             mgpu_context);
    }

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
}