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
#include "readranks.h"
#include "pr_common.h"
#include "my_pr_balanced.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);
DEFINE_bool(sync, false, "using sync mode to run pagerank algorithm");
DECLARE_bool(cta_np);
DECLARE_double(threshold);
#define GTID (blockIdx.x * blockDim.x + threadIdx.x)

namespace deltapr {
    struct Algo {

        static const char *Name() { return "PR"; }
    };

    template<typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual, ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        index_t start_node = graph.owned_start_node();
        index_t end_node = start_node + graph.owned_nnodes();
        for (index_t node = start_node + tid; node < end_node; node += nthreads) {
            current_ranks[node] = 0.0f;
            residual[node] = (1 - ALPHA);// / graph.owned_nnodes();
            last_residual[node] = 0.0f;
        }
    }


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankKernel__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            WorkSource work_source) {
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

            if (out_degree == 0) {
//                rank_t update = ALPHA * res;
//                atomicAdd(residual.get_item_ptr(node), update);
            } else {

                rank_t update = ALPHA * res / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                }
            }
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankKernelCTA__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            WorkSource work_source) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup =
                round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop


        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> np_local = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);
                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                if (res > 0) {

                    current_ranks[node] += res;

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;

                    if (np_local.size == 0) {
//                        rank_t update = ALPHA * res;
//
//                        atomicAdd(residual.get_item_ptr(node), update);
                    } else {
                        rank_t update = ALPHA * res / np_local.size;

                        np_local.meta_data = update;
                    }
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    np_local,
                    [&graph, &residual](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);

                        atomicAdd(residual.get_item_ptr(dest), update);
                    }
            );
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankSyncKernel__Single__(
            TGraph graph,
            index_t iteration,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual,
            WorkSource work_source) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();


        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);

            rank_t res;

            if (iteration % 2 == 0) {
                res = atomicExch(residual.get_item_ptr(node), 0);
            } else {
                res = atomicExch(last_residual.get_item_ptr(node), 0);
            }

            if (res == 0)continue;

            current_ranks[node] += res;

            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) {
//                rank_t update = ALPHA * res;
//
//                if (iteration % 2 == 0) {
//                    atomicAdd(last_residual.get_item_ptr(node), update);
//                } else {
//                    atomicAdd(residual.get_item_ptr(node), update);
//                }

            } else {

                rank_t update = ALPHA * res / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);

                    if (iteration % 2 == 0) {
                        atomicAdd(last_residual.get_item_ptr(dest), update);
                    } else {
                        atomicAdd(residual.get_item_ptr(dest), update);
                    }
                }
            }
        }
    }


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankSyncKernelCTA__Single__(
            TGraph graph,
            index_t iteration,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual,
            WorkSource work_source) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup =
                round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop


        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> np_local = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);

                rank_t res;

                if (iteration % 2 == 0) {
                    res = atomicExch(residual.get_item_ptr(node), 0);
                } else {
                    res = atomicExch(last_residual.get_item_ptr(node), 0);
                }

                if (res > 0) {
                    current_ranks[node] += res;

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;

                    if (np_local.size == 0) {
//                        rank_t update = ALPHA * res;
//
//                        if (iteration % 2 == 0) {
//                            atomicAdd(last_residual.get_item_ptr(node), update);
//                        } else {
//                            atomicAdd(residual.get_item_ptr(node), update);
//                        }

                    } else {
                        rank_t update = ALPHA * res / np_local.size;

                        np_local.meta_data = update;
                    }
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    np_local,
                    [&iteration, &graph, &residual, &last_residual](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);

                        if (iteration % 2 == 0) {
                            atomicAdd(last_residual.get_item_ptr(dest), update);
                        } else {
                            atomicAdd(residual.get_item_ptr(dest), update);
                        }
                    }
            );
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

    /*
    * The per-device Page Rank problem
    */
    template<typename TGraph,
            template<typename> class ResidualDatum,
            template<typename> class RankDatum>
    struct Problem {
        TGraph m_graph;
        RankDatum<rank_t> m_current_ranks;
        ResidualDatum<rank_t> m_residual;
        ResidualDatum<rank_t> m_last_residual;

        Problem(const TGraph &graph, RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
                ResidualDatum<rank_t> last_residual) :
                m_graph(graph), m_residual(residual), m_current_ranks(current_ranks), m_last_residual(last_residual) {
        }

        void Init__Single__(groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");
            PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (m_graph, m_current_ranks, m_residual, m_last_residual);
//            stream.Sync();
//            PageRankPrint__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
//                                                                   (m_current_ranks, m_residual, m_graph.owned_nnodes());
        }


        template<typename WorkSource>
        void
        RelaxSync__Single__(index_t iteration, const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");
            if (FLAGS_cta_np) {
                PageRankSyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                               (m_graph, iteration, m_current_ranks, m_residual, m_last_residual, work_source);
            } else {
                PageRankSyncKernel__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                            (m_graph, iteration, m_current_ranks, m_residual, m_last_residual, work_source);
            }
        }

        template<typename WorkSource,
                template<typename> class TWorkList>
        void
        Relax__Single__(const WorkSource &work_source, TWorkList<index_t> &output_worklist, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");

            if (FLAGS_cta_np) {
                PageRankKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                           (m_graph, m_current_ranks, m_residual, work_source);
            } else {
                PageRankKernel__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                        (m_graph, m_current_ranks, m_residual, work_source);
            }
        }

        template<typename WorkSource>
        double RankCheck__Single__(WorkSource work_source, mgpu::context_t &context) {
            rank_t *tmp = m_current_ranks.data_ptr;

            printf("work size:%d\n", work_source.get_size());

            auto check_segment_sizes = [=]__device__(int idx) {
                return tmp[idx];
            };

            mgpu::mem_t<double> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);
            int *scanned_offsets = deviceOffsets.data();
            mgpu::transform_scan<double>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), context);
            return mgpu::from_mem(checkSum)[0];
        }

        template<typename WorkSource>
        rank_t RankCmp__Single__(WorkSource work_source, RankDatum<rank_t> accu_ranks, mgpu::context_t &context) {
            rank_t *tmp = m_current_ranks.data_ptr;
            rank_t *local_accu_ranks = accu_ranks.data_ptr;

            printf("work size:%d\n", work_source.get_size());

            auto check_segment_sizes = [=]__device__(int idx) {
                return fabs(tmp[idx] - local_accu_ranks[idx]);
            };

            mgpu::mem_t<rank_t> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);
            int *scanned_offsets = deviceOffsets.data();
            mgpu::transform_scan<rank_t>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<rank_t>(), checkSum.data(), context);
            return mgpu::from_mem(checkSum)[0];
        }

    };
}

bool PageRankDeltaBased() {
    utils::traversal::Context<deltapr::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> last_residual;
    groute::graphs::single::NodeOutputDatum<rank_t> accu_ranks;

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_residual, accu_ranks);
    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context;


    groute::Queue<index_t> wl1(context.host_graph.nnodes, 0, "input queue");
    groute::Queue<index_t> wl2(context.host_graph.nnodes, 0, "output queue");

    wl1.ResetAsync(stream);
    wl2.ResetAsync(stream);
    stream.Sync();

    groute::Queue<index_t> *in_wl = &wl1, *out_wl = &wl2;
    deltapr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(dev_graph_allocator.DeviceObject(),
                   current_ranks.DeviceObject(), residual.DeviceObject(), last_residual.DeviceObject());

    solver.Init__Single__(stream);
    stream.Sync();

    Stopwatch stopwatch(true);


//    std::vector<float> host_accu_ranks(context.host_graph.nnodes);
//    ReadRanks("/home/liang/kron_g500-simple-logn21-weighted-random_0.85_accurate", &host_accu_ranks);
//
//    GROUTE_CUDA_CHECK(cudaMemcpy(accu_ranks.DeviceObject().data_ptr, host_accu_ranks.data(),
//                                 host_accu_ranks.size() * sizeof(rank_t), cudaMemcpyHostToDevice));

    if (FLAGS_cta_np)
        printf("CTA enabled\n");

    double last_sum = 0;

    if (FLAGS_sync) {
        printf("Running in Sync mode\n");

//        for (index_t iteration = 0; iteration < 1000; iteration++)
        bool running = true;
        int iteration = 0;

        while (running) {
            solver.RelaxSync__Single__(iteration, groute::dev::WorkSourceRange<index_t>(
                    dev_graph_allocator.DeviceObject().owned_start_node(),
                    dev_graph_allocator.DeviceObject().owned_nnodes()), stream);
            stream.Sync();
            double pr_sum = solver.RankCheck__Single__(groute::dev::WorkSourceRange<index_t>(
                    dev_graph_allocator.DeviceObject().owned_start_node(),
                    dev_graph_allocator.DeviceObject().owned_nnodes()), mgpu_context);
            printf("iteration:%d pr sum:%f\n", iteration++, pr_sum);
            running = (last_sum != pr_sum);
            if (FLAGS_threshold != 999999999) {
                running = pr_sum < FLAGS_threshold;
            }
            last_sum = pr_sum;
        }
    } else {
        printf("Running in Async mode\n");
        int iteration = 0;
        bool running = true;

        while (running) {
            solver.Relax__Single__(groute::dev::WorkSourceRange<index_t>(
                    dev_graph_allocator.DeviceObject().owned_start_node(),
                    dev_graph_allocator.DeviceObject().owned_nnodes()), *in_wl, stream);
            double pr_sum = solver.RankCheck__Single__(groute::dev::WorkSourceRange<index_t>(
                    dev_graph_allocator.DeviceObject().owned_start_node(),
                    dev_graph_allocator.DeviceObject().owned_nnodes()), mgpu_context);


//            rank_t diff_sum = solver.RankCmp__Single__(groute::dev::WorkSourceRange<index_t>(
//                    dev_graph_allocator.DeviceObject().owned_start_node(),
//                    dev_graph_allocator.DeviceObject().owned_nnodes()), accu_ranks.DeviceObject(), mgpu_context);
            running = pr_sum < FLAGS_threshold;
//            running = diff_sum > 0.004296;
            printf("iteration:%d pr sum:%f\n", iteration++, pr_sum);
//            printf("iteration:%d pr sum:%f diff:%f\n", iteration++, pr_sum, diff_sum);
        }
    }

    stopwatch.stop();
    printf("delta-based PR:%f\n", stopwatch.ms());

    dev_graph_allocator.GatherDatum(current_ranks);

    std::vector<rank_t> host_current_ranks = current_ranks.GetHostData();

    if (FLAGS_output.length() != 0)
        PageRankOutput(FLAGS_output.c_str(), host_current_ranks);
    return true;
}