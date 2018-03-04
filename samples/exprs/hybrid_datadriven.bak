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
#include <groute/device/queue.cuh>
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
DECLARE_int32(mode);
DECLARE_int32(switch_threshold);
#define EPSILON 0.01

namespace hybrid_datadriven {
    template<typename WorkSource,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual, ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
            index_t node = work_source.get_work(ii);

            current_ranks[node] = 1.0 - ALPHA;
            last_residual[node] = 0.0;

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

    template<typename WorkSource, typename WorkTarget>
    __global__ void InitWorklist(WorkSource work_source, WorkTarget work_target) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
            index_t node = work_source.get_work(ii);
            work_target.append_warp(node);
        }
    };

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource,
            typename WorkTarget>
    __global__ void PageRankSyncKernel__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            index_t iteration,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual) {
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

            if (out_degree > 0) {
                rank_t update = ALPHA * res / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);
                    rank_t prev;
                    if (iteration % 2 == 0) {
                        prev = atomicAdd(last_residual.get_item_ptr(dest), update);
                    } else {
                        prev = atomicAdd(residual.get_item_ptr(dest), update);
                    }

                    if (prev <= EPSILON && update + prev > EPSILON) {
                        work_target.append_warp(dest);
                    }
                }
            }
        }
    }


    template<
            typename TGraph,
            typename WorkSource,
            typename WorkTarget,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankSyncKernelCTA__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            index_t iteration,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, 0.0};

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


                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&graph, &work_target, &residual, &last_residual, &iteration](index_t edge, index_t size, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev;
                        if (iteration % 2 == 0) {
                            prev = atomicAdd(last_residual.get_item_ptr(dest), update);
                        } else {
                            prev = atomicAdd(residual.get_item_ptr(dest), update);
                        }

                        if (prev <= EPSILON && prev + update > EPSILON) {
                            work_target.append(dest);
                        }
                    }
            );
        }
    }

    template<
            typename TGraph,
            typename WorkSource,
            typename WorkTarget,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankAsyncKernel__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
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

    template<typename TGraph,
            typename WorkSource,
            typename WorkTarget,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankAsyncKernelCTA__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
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
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&graph, work_target, &residual](index_t edge, index_t size, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        if (prev <= EPSILON && prev + update > EPSILON) {
                            work_target.append(dest);
                        }
                    }
            );
        }
    }

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

bool HybridDataDriven() {
    VLOG(0) << "HybridDataDriven";

    typedef groute::Queue<index_t> Worklist;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> last_residual;


    utils::traversal::Context<hybrid_datadriven::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
            dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_residual);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);


    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    hybrid_datadriven::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(),
            dev_graph_allocator.DeviceObject().owned_nnodes()),
            dev_graph_allocator.DeviceObject(),
            current_ranks.DeviceObject(),
            residual.DeviceObject(),
            last_residual.DeviceObject());

    int syncIteration = 0;

    //mode = 0 means sync, mode = 1 means async
    int mode = FLAGS_mode;
    if (mode == 2 && FLAGS_switch_threshold > 0)mode = 0;
    int last_mode;
    srand(time(NULL));
    groute::graphs::single::NodeOutputDatum<rank_t> *available_residual = &residual;

    int totalIteration = 0;

    double totalSync = 0;
    double totalAsync = 0;
    mgpu::mem_t<double> checkSum(1, mgpu_context);
    mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(context.host_graph.nnodes, mgpu_context);

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;
    Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

    Worklist *in_wl = &wl1, *out_wl = &wl2;


    hybrid_datadriven::InitWorklist << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                   (groute::dev::WorkSourceRange<index_t>(dev_graph_allocator.DeviceObject().owned_start_node(),
                                                                                                          dev_graph_allocator.DeviceObject().owned_nnodes()), in_wl->DeviceObject());
    stream.Sync();

    groute::Segment<index_t> work_seg = in_wl->GetSeg(stream);

    Stopwatch sw(true);

    int *scanned_offsets = deviceOffsets.data();

    double last_variance = 9999999;

    while (work_seg.GetSegmentSize() > 0) {
//        if (FLAGS_mode == 2)
//            mode = rand() % 2;
//        else
//            mode = FLAGS_mode;
        ++totalIteration;
//        if (totalIteration > FLAGS_switch_threshold && mode == 1) {
//            mode = 0;
//        }

        if (mode == 0) {
            last_mode = 0;
            Stopwatch sw1(true);
            hybrid_datadriven::PageRankSyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                    dev_graph_allocator.DeviceObject(),
                            groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(), work_seg.GetSegmentSize()),
                            out_wl->DeviceObject(),
                            syncIteration++,
                            current_ranks.DeviceObject(),
                            residual.DeviceObject(),
                            last_residual.DeviceObject());
            if(syncIteration%2)
                available_residual = &last_residual;
            else
                available_residual = &residual;
            stream.Sync();
            sw1.stop();
            VLOG(1) << "THROUGHPUT: " << (work_seg.GetSegmentSize() + out_wl->GetCount(stream)) / sw1.ms() << " nodes/ms";

            totalSync += sw1.ms();
        } else {
            last_mode = 1;
            Stopwatch sw2(true);
            hybrid_datadriven::PageRankAsyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                    dev_graph_allocator.DeviceObject(),
                            groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(), work_seg.GetSegmentSize()),
                            out_wl->DeviceObject(),
                            current_ranks.DeviceObject(),
                            available_residual->DeviceObject());
            stream.Sync();
            sw2.stop();
            totalAsync += sw2.ms();
            VLOG(1) << "THROUGHPUT: " << (work_seg.GetSegmentSize() + out_wl->GetCount(stream)) / sw2.ms() << " nodes/ms";
        }


        rank_t *p_residual = current_ranks.DeviceObject().data_ptr;
//        if (mode == 0) {
//            if (syncIteration % 2)
//                p_residual = last_residual.DeviceObject().data_ptr;
//            else
//                p_residual = residual.DeviceObject().data_ptr;
//        } else {
//            p_residual = available_residual->DeviceObject().data_ptr;
//        }


        auto func_residual_sum = [=]__device__(index_t idx) {
            return p_residual[idx];
        };

        mgpu::transform_scan<double>(func_residual_sum, context.host_graph.nnodes, scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double residual_sum = mgpu::from_mem(checkSum)[0];

        double avg_residual_sum = residual_sum / context.host_graph.nnodes;

        auto func_variance_compute = [=]__device__(index_t idx) {
            return (p_residual[idx] - avg_residual_sum) * (p_residual[idx] - avg_residual_sum);
        };

        mgpu::transform_scan<double>(func_variance_compute, context.host_graph.nnodes, scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double residual_variance = mgpu::from_mem(checkSum)[0] / context.host_graph.nnodes;

//        if (last_variance - residual_variance < 0) {
//            mode = 1;
//            VLOG(0) << "SWITCH!!!";
//        }

        last_variance = residual_variance;


        VLOG(0) << "Residual Variance: " << residual_variance;


        VLOG(0) << "Iteration: " << totalIteration << " " << (mode == 0 ? "SYNC" : "ASYNC") << "  INPUT: " << in_wl->GetCount(stream) << " OUTPUT: " << out_wl->GetCount(stream);


        in_wl->ResetAsync(stream);
        std::swap(in_wl, out_wl);
        work_seg = in_wl->GetSeg(stream);

//        if (last_mode == 0) {
//            if (syncIteration % 2 == 0)//last round is last_residual-->residual
//                available_residual = &residual;
//            else
//                available_residual = &last_residual;
//        }


    }

    sw.stop();

    VLOG(1)
    << boost::format("%s terminated after %d iterations (max: %d, sync: %d, async: %d)") % hybrid_datadriven::Algo::Name() % totalIteration %
       FLAGS_max_pr_iterations % syncIteration % (totalIteration - syncIteration);
    VLOG(0) << hybrid_datadriven::Algo::Name() << ": " << sw.ms() << " ms. <filter>";
    VLOG(0) << "AVG SYNC: " << totalSync / syncIteration << "ms TOTAL ASYNC: " << totalAsync / (totalIteration - syncIteration) << " ms.";

    // Gather
    auto gathered_output = hybrid_datadriven::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        hybrid_datadriven::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = hybrid_datadriven::Algo::Host(context.host_graph, residual, current_ranks);
        return hybrid_datadriven::Algo::CheckErrors(gathered_output, regression) == 0;
    } else {
        LOG(WARNING) << "Result not checked";
        return true;
    }
}