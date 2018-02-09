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
#include "my_pr_balanced.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define FILTER_THRESHOLD 0.0000000001

typedef float rank_t;

namespace persistpr {
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

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;

            rank_t update = ALPHA * res / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t dest = graph.edge_dest(edge);

                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                if (prev + update > EPSILON && prev < EPSILON) {
                    output_worklist.append_warp(dest);
                }
            }
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            template<typename> class TWorkList>
    __global__ void PageRankDevice__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            TWorkList<index_t> input_worklist, TWorkList<index_t> output_worklist) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = input_worklist.count();

        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = input_worklist.read(i);

            rank_t res = atomicExch(residual.get_item_ptr(node), 0);

            if (res == 0)continue;

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;
            rank_t update = ALPHA * res / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t dest = graph.edge_dest(edge);

                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                if (prev + update > EPSILON && prev < EPSILON) {
                    output_worklist.append_warp(dest);
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

    template<typename TGraph,
            template<typename> class DQueue>
    __global__ void PageRankWorkList__Init__(TGraph graph,
                                             DQueue<index_t> input_worklist) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (int node = 0 + tid; node < graph.nnodes; node += nthreads) {
            input_worklist.append_warp(node);
        }
    }

    template<const int BLOCK_SIZE,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            template<typename> class DQueue>
    __global__ void PageRank__Single__(TGraph graph,
                                       RankDatum<rank_t> current_ranks,
                                       ResidualDatum<rank_t> residual,
                                       DQueue<index_t> input_worklist, DQueue<index_t> output_worklist) {
        unsigned tid = TID_1D;

        dim3 block_dims(BLOCK_SIZE, 1, 1);
        dim3 grid_dims(round_up(input_worklist.count(), block_dims.x), 1, 1);

        DQueue<index_t> *wl1 = &input_worklist, *wl2 = &output_worklist, *wl_tmp;

        if (tid == 0) {
            int iteration = 0;

            while (wl1->count() > 0) {
//                PageRankDeviceBalanced1__Single__ << < grid_dims, block_dims >> > (
//                        graph, current_ranks, residual, *wl1, *wl2);
                cudaDeviceSynchronize();

                printf("iteration %d active nodes %d\n", iteration++, wl2->count());

                grid_dims = dim3(round_up(wl2->count(), block_dims.x), 1, 1);
                wl1->reset();
                wl_tmp = wl1;
                wl1 = wl2;
                wl2 = wl_tmp;
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
            return pr_sum < THRESHOLD;
        }

        void DoPageRank(groute::Stream &stream) {
            groute::Queue<index_t> wl1(m_graph.nnodes, 0, "input queue");
            groute::Queue<index_t> wl2(m_graph.nnodes, 0, "output queue");

            wl1.ResetAsync(stream);
            wl2.ResetAsync(stream);
            stream.Sync();

            Stopwatch stopwatch;


            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.nnodes);

            PageRankWorkList__Init__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                    (m_graph, wl1.DeviceObject());
            stream.Sync();

            stopwatch.start();

            PageRank__Single__<256> << < 1, 1, 0, stream.cuda_stream >> >
                                                  (m_graph, m_current_ranks, m_residual, wl1.DeviceObject(),
                                                          wl2.DeviceObject());
            stream.Sync();

            stopwatch.stop();

            printf("PR:%fms\n", stopwatch.ms());
        }
    };
}

//-num_gpus=1 -graphfile /home/xiayang/diskb/liang/ppopp17-artifact/dataset/soc-LiveJournal1/soc-LiveJournal1-weighted-1.gr -single
bool MyTestPageRankSingleOutlining() {
    utils::traversal::Context<persistpr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context;

    persistpr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(),
                   current_ranks.DeviceObject(), residual.DeviceObject());
    Stopwatch sw(true);

    solver.Init__Single__(stream);
    solver.DoPageRank(stream);

    sw.stop();

    printf("cta-balancing PR:%fms\n", sw.ms());

    dev_graph_allocator.GatherDatum(current_ranks);
    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();
    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}

bool MyTestPageRankSingle() {
    utils::traversal::Context<persistpr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);
    context.SetDevice(0);
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);
    context.SyncDevice(0);
    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context;

    persistpr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(),
                   current_ranks.DeviceObject(), residual.DeviceObject());

    Stopwatch stopwatch;
    stopwatch.start();
    solver.Init__Single__(stream);


    int iteration = 0;

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
    }

    stopwatch.stop();

    printf("non-balancing PR:%f\n", stopwatch.ms());

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}