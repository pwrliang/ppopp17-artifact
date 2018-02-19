//
// Created by liang on 2/16/18.
//
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/work_source.cuh>
#include <groute/device/cta_scheduler.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/graphs/traversal.h>
#include <utils/stopwatch.h>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/format.hpp>
#include <utils/cuda_utils.h>
#include "pr_common.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);
DECLARE_int32(max_pr_iterations);
DECLARE_double(threshold);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);
DECLARE_double(epsilon);
DECLARE_bool(cta_np);
DEFINE_int32(first_iteration, 20, "Iteration times for Topology-Driven");

namespace hybrid_unopt_pr {
    template<typename WorkSource,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
            index_t node = work_source.get_work(ii);

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
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankKernelTopologyDriven__Single__(
            WorkSource work_source, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);

            rank_t res = atomicExch(residual.get_item_ptr(node), 0);
            if (res == 0) continue; // might happen if work_source has duplicates

            current_ranks[node] += res;

            index_t
                    begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;

            rank_t update = res * ALPHA / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t dest = graph.edge_dest(edge);
                atomicAdd(residual.get_item_ptr(dest), update);
            }
        }
    }

    template<
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankKernelTopologyDrivenCTA__Single__(
            WorkSource work_source, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
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
                    if (local_work.size > 0) {
                        rank_t update = res * ALPHA / local_work.size;

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
            typename WorkSource, typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankKernelDataDriven__Single__(
            WorkSource work_source, WorkTarget work_target,
            float EPSILON, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);

            rank_t res = atomicExch(residual.get_item_ptr(node), 0);
            if (res == 0) continue; // might happen if work_source has duplicates

            current_ranks[node] += res;

            index_t
                    begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;

            rank_t update = res * ALPHA / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                index_t dest = graph.edge_dest(edge);
                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                if (prev <= EPSILON && prev + update > EPSILON) {
                    work_target.append(dest);
                }
            }
        }
    }

    template<
            typename WorkSource, typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankKernelDataDrivenCTA__Single__(
            WorkSource work_source, WorkTarget work_target,
            float EPSILON, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

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

                    index_t
                            begin_edge = graph.begin_edge(node),
                            end_edge = graph.end_edge(node),
                            out_degree = end_edge - begin_edge;
                    if (local_work.size > 0) {
                        rank_t update = res * ALPHA / out_degree;

                        local_work.meta_data = update;
                    }
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&work_target, &graph, &residual, &EPSILON](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        if (prev <= EPSILON && prev + update > EPSILON) {
                            work_target.append(dest);
                        }
                    }
            );
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

        template<typename WorkSource>
        void Init__Single__(const WorkSource &workSource, groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");

            PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (workSource, m_graph, m_current_ranks, m_residual);
        }

        template<typename WorkSource>
        void
        RelaxTopologyDriven__Single__(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernelTopologyDriven__Single__");

            if (FLAGS_cta_np)
                PageRankKernelTopologyDrivenCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                         (work_source, m_graph, m_current_ranks, m_residual);
            else
                PageRankKernelTopologyDriven__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                      (work_source, m_graph, m_current_ranks, m_residual);
        }

        template<typename WorkSource,
                typename WorkTarget>
        void
        RelaxDataDriven__Single__(const WorkSource &work_source, WorkTarget &output_worklist, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            float EPSILON = FLAGS_epsilon;
            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");

            if (FLAGS_cta_np)
                PageRankKernelDataDrivenCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                     (work_source, output_worklist.DeviceObject(), EPSILON, m_graph, m_current_ranks, m_residual);
            else
                PageRankKernelDataDriven__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                                  (work_source, output_worklist.DeviceObject(), EPSILON, m_graph, m_current_ranks, m_residual);
        }
    };

    struct Algo {
        static const char *NameLower() { return "pr"; }

        static const char *Name() { return "PR"; }


        template<
                typename TGraphAllocator, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static const std::vector<rank_t> &Gather(
                TGraphAllocator &graph_allocator, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            graph_allocator.GatherDatum(current_ranks);
            return current_ranks.GetHostData();
        }

        template<
                typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static std::vector<rank_t> Host(
                groute::graphs::host::CSRGraph &graph, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            return PageRankHost(graph);
        }

        static int Output(const char *file, const std::vector<rank_t> &ranks) {
            return PageRankOutput(file, ranks);
        }

        static int CheckErrors(std::vector<rank_t> &ranks, std::vector<rank_t> &regression) {
            return PageRankCheckErrors(ranks, regression);
        }
    };
}

bool HybridDrivenPR() {
    VLOG(0) << "HybridDrivenPR";
    if (FLAGS_cta_np)
        VLOG(0) << "CTA_NP Enabled";
    typedef groute::Queue<index_t> Worklist;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    utils::traversal::Context<hybrid_unopt_pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
            dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);

    hybrid_unopt_pr::Problem<
            groute::graphs::dev::CSRGraph,
            groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(
            dev_graph_allocator.DeviceObject(),
            current_ranks.DeviceObject(),
            residual.DeviceObject());

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;

    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;

    Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

    Worklist *in_wl = &wl1, *out_wl = &wl2;

    Stopwatch sw(true);

    solver.Init__Single__(groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(),
            dev_graph_allocator.DeviceObject().owned_nnodes()), stream);

    int iteration = 0;

    for (iteration = 0; iteration < FLAGS_first_iteration; iteration++) {
        solver.RelaxTopologyDriven__Single__(
                groute::dev::WorkSourceRange<index_t>(dev_graph_allocator.DeviceObject().owned_start_node(),
                                                      dev_graph_allocator.DeviceObject().owned_nnodes()), stream);
        stream.Sync();
        VLOG(1) << "Topology-Driven Iteration: " << iteration;
    }

    solver.RelaxDataDriven__Single__(
            groute::dev::WorkSourceRange<index_t>(dev_graph_allocator.DeviceObject().owned_start_node(),
                                                  dev_graph_allocator.DeviceObject().owned_nnodes()),
            *in_wl, stream);

    groute::Segment<index_t> work_seg;

    work_seg = in_wl->GetSeg(stream);

    while (work_seg.GetSegmentSize() > 0) {
        solver.RelaxDataDriven__Single__(groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(),
                                                                               work_seg.GetSegmentSize()),
                                         *out_wl, stream);
        VLOG(1) << "Data-Driven Iteration: " << iteration << " INPUT " << work_seg.GetSegmentSize() << " OUTPUT "
                << out_wl->GetCount(stream);

        if (++iteration > FLAGS_max_pr_iterations) {
            LOG(WARNING) << "maximum iterations reached";
            break;
        }

        in_wl->ResetAsync(stream.cuda_stream);
        std::swap(in_wl, out_wl);
        work_seg = in_wl->GetSeg(stream);
    }

    sw.stop();

    VLOG(1)
    << boost::format("%s terminated after %d iterations (max: %d)") % hybrid_unopt_pr::Algo::Name() % iteration %
       FLAGS_max_pr_iterations;
    VLOG(0) << hybrid_unopt_pr::Algo::Name() << ": " << sw.ms() << " ms. <filter>";
    // Gather
    auto gathered_output = hybrid_unopt_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        hybrid_unopt_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = hybrid_unopt_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return hybrid_unopt_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    } else {
        LOG(WARNING) << "Result not checked";
        return true;
    }
}