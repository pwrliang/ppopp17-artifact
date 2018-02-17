//
// Created by liang on 2/16/18.
//
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <cuda.h>
#include <gflags/gflags.h>
#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/work_source.cuh>
#include <groute/device/cta_scheduler.cuh>
#include <cub/grid/grid_barrier.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/cuda_utils.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <glog/logging.h>
#include "pr_common.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);
DECLARE_int32(max_pr_iterations);
DECLARE_double(epsilon);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);

namespace data_driven_outlining_pr {
    template<typename WorkSource,
            typename WorkTarget,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source, WorkTarget work_target,
            float EPSILON, TGraph graph,
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
                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                if (prev <= EPSILON && prev + update > EPSILON)
                    work_target.append_warp(dest);
            }
        }
    }

    template<
            typename WorkSource, typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __device__ void PageRankKernel__Single__(
            WorkSource work_source, WorkTarget work_target,
            float EPSILON, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.count();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.read(i);

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
                    work_target.append_warp(dest);
                }
            }
        }
    }


//    template<typename WorkSource, typename WorkTarget>
    template<typename WorkSource, typename WorkTarget,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankControl__Single__(
            WorkSource work_source, WorkTarget work_target, float EPSILON, cub::GridBarrier grid_barrier, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        WorkSource *in_wl = &work_source;
        WorkTarget *out_wl = &work_target;


        while (in_wl->count() > 0) {
            PageRankKernel__Single__(*in_wl, *out_wl, EPSILON, graph, current_ranks, residual);

            grid_barrier.Sync();

            if (tid == 0) {
                printf("INPUT %d OUTPUT %d\n",in_wl->count(), out_wl->count());
                in_wl->reset();
            }

            WorkSource *tmp_wl = in_wl;
            in_wl = out_wl;
            out_wl = tmp_wl;
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

        Problem(const TGraph &graph, RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) :
                m_graph(graph), m_residual(residual), m_current_ranks(current_ranks) {
        }

        template<typename WorkSource, typename WorkTarget>
        void Init__Single__(const WorkSource &workSource, WorkTarget workTarget, groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");

            PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (workSource, workTarget, FLAGS_epsilon, m_graph, m_current_ranks, m_residual);
        }

        template<typename WorkSource, typename WorkTarget>
        void DoPageRank(WorkSource work_source, WorkTarget work_target,
                        index_t blocksPerGrid, index_t threadsPerBlock, groute::Stream &stream) {
            cub::GridBarrierLifetime grid_barrier;

            grid_barrier.Setup(blocksPerGrid);

            float EPSILON = FLAGS_epsilon;

            PageRankControl__Single__ << < blocksPerGrid, threadsPerBlock, 0,
                    stream.cuda_stream >> >
                    (work_source, work_target, EPSILON, grid_barrier,
                            m_graph, m_current_ranks, m_residual);
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

bool DataDrivenOutliningPR() {
    VLOG(0) << "DataDrivenOutliningPR";

    typedef groute::Queue<index_t> Worklist;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    utils::traversal::Context<data_driven_outlining_pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
            dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    data_driven_outlining_pr::Problem<
            groute::graphs::dev::CSRGraph,
            groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
            solver(
            dev_graph_allocator.DeviceObject(),
            current_ranks.DeviceObject(),
            residual.DeviceObject());

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;

    groute::Stream stream = context.CreateStream(0);

    Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

    Stopwatch sw(true);

    Worklist *in_wl = &wl1, *out_wl = &wl2;

    solver.Init__Single__(groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(),
            dev_graph_allocator.DeviceObject().owned_nnodes()),
                          out_wl->DeviceObject(), stream);
    solver.DoPageRank(in_wl->DeviceObject(), out_wl->DeviceObject(),
                      FLAGS_grid_size, FLAGS_block_size, stream);
    stream.Sync();
    sw.stop();

    VLOG(0) << "Blocks per grid: " << FLAGS_grid_size << " Threads per grid: " << FLAGS_block_size;
    VLOG(0) << "EPSILON: " << FLAGS_epsilon;
    VLOG(0) << data_driven_outlining_pr::Algo::Name() << ": " << sw.ms() << " ms. <filter>";
    // Gather
    auto gathered_output = data_driven_outlining_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        data_driven_outlining_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = data_driven_outlining_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return data_driven_outlining_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    } else {
        LOG(WARNING) << "Result not checked";
        return true;
    }
}