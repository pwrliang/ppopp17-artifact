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

namespace data_driven_unopt_pr {
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
    __global__ void PageRankKernel__Single__(
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
        bool RankCheck__Single__(WorkSource work_source, mgpu::context_t &context) {
            rank_t *tmp = m_current_ranks.data_ptr;

            auto check_segment_sizes = [=]__device__(int idx) {
                return tmp[idx];
            };

            mgpu::mem_t<double> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);

            int *scanned_offsets = deviceOffsets.data();

            mgpu::transform_scan<double>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), context);

            double pr_sum = mgpu::from_mem(checkSum)[0];

            VLOG(1) << "Checking... SUM: " << pr_sum << " Relative SUM: " << pr_sum / work_source.get_size();

            return pr_sum / work_source.get_size() < FLAGS_threshold;
        }

        template<typename WorkSource>
        void
        Relax__Single__(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");
            PageRankKernel__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                    (work_source, m_graph, m_current_ranks, m_residual);
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

bool TopologyDrivenUnoptPR() {
    VLOG(0) << "TopologyDrivenUnoptPR";

    typedef groute::Queue<index_t> Worklist;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    utils::traversal::Context<data_driven_unopt_pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
            dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);

    data_driven_unopt_pr::Problem<
            groute::graphs::dev::CSRGraph,
            groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(
            dev_graph_allocator.DeviceObject(),
            current_ranks.DeviceObject(),
            residual.DeviceObject());

    Stopwatch sw(true);

    solver.Init__Single__(groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(),
            dev_graph_allocator.DeviceObject().owned_nnodes()),
                          stream);

    int iteration = 0;
    bool running = true;

    while (running) {
        solver.Relax__Single__(groute::dev::WorkSourceRange<index_t>(
                dev_graph_allocator.DeviceObject().owned_start_node(),
                dev_graph_allocator.DeviceObject().owned_nnodes()), stream);

        stream.Sync();

        running = solver.RankCheck__Single__(groute::dev::WorkSourceRange<index_t>(
                dev_graph_allocator.DeviceObject().owned_start_node(),
                dev_graph_allocator.DeviceObject().owned_nnodes()), mgpu_context);

        if (++iteration > FLAGS_max_pr_iterations) {
            LOG(WARNING) << "maximum iterations reached";
            break;
        }

        VLOG(1) << "Iteration: " << iteration;
    }

    sw.stop();

    VLOG(1)
    << boost::format("%s terminated after %d iterations (max: %d)") % data_driven_unopt_pr::Algo::Name() % iteration %
       FLAGS_max_pr_iterations;
    VLOG(0) << data_driven_unopt_pr::Algo::Name() << ": " << sw.ms() << " ms. <filter>";
    // Gather
    auto gathered_output = data_driven_unopt_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        data_driven_unopt_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = data_driven_unopt_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return data_driven_unopt_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    } else {
        LOG(WARNING) << "Result not checked";
        return true;
    }
}