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
#include <cub/grid/grid_barrier.cuh>
#include "pr_common.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);
DECLARE_int32(max_pr_iterations);
DECLARE_double(threshold);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);
DECLARE_int32(async_to_sync);
DECLARE_int32(sync_to_async);
DECLARE_bool(force_sync);
DECLARE_bool(force_async);
#define EPSILON 0.01

namespace hybrid_datadriven {

    template<typename T>
    __device__ void swap(T &a, T &b) {
        T tmp = a;
        a = b;
        b = tmp;
    }

    template<typename WorkSource,
            typename WorkTarget,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            WorkTarget work_target,
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual, ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (index_t ii = 0 + tid; ii < work_size; ii += nthreads) {
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
                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                if (prev <= EPSILON && prev + update > EPSILON) {
                    work_target.append(dest);
                }
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
            typename WorkSource,
            typename WorkTarget,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    //__device__
    __global__
    void PageRankSyncKernelCTA__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            index_t iteration,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.count();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.read(i);

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
                    [&graph, &work_target, &residual, &last_residual, &iteration](index_t edge, index_t size,
                                                                                  rank_t update) {
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

    template<typename TGraph,
            typename WorkSource,
            typename WorkTarget,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    //__device__
    __global__
    void PageRankAsyncKernelCTA__Single__(
            TGraph graph,
            WorkSource work_source,
            WorkTarget work_target,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.count();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, 0.0};

            if (i < work_size) {
                index_t node = work_source.read(i);

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

    template<
            template<typename> class WorkList,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankControl__Single__(
            bool fource_sync,
            uint32_t async_to_sync,
            uint32_t sync_to_async,
            cub::GridBarrier grid_barrier,
            WorkList<index_t> work_source,
            WorkList<index_t> work_target,
            TGraph graph,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_residual) {
        uint32_t tid = TID_1D;
        WorkList<index_t> *in_wl = &work_source;
        WorkList<index_t> *out_wl = &work_target;
        uint32_t iteration = 0;
        ResidualDatum<rank_t> *available_delta = &residual;

        if (tid == 0) {
            printf("CALL PageRankControl%s__Single__ \n", "Hybrid");
        }

        int mode;

        while (in_wl->count() > 0) {
            if (!fource_sync && (iteration < async_to_sync || iteration >= sync_to_async)) {
                PageRankAsyncKernelCTA__Single__(graph,
                                                 *in_wl,
                                                 *out_wl,
                                                 current_ranks,
                                                 *available_delta);
                mode = 1;
            } else {
                PageRankSyncKernelCTA__Single__(graph,
                                                *in_wl,
                                                *out_wl,
                                                iteration,
                                                current_ranks,
                                                residual,
                                                last_residual);
                if (iteration % 2 == 0) {
                    available_delta = &last_residual;
                }
                mode = 0;
            }
            grid_barrier.Sync();
            if (tid == 0) {
                printf("%s Iter:%d Input:%d Output:%d\n", mode == 0 ? "Sync" : "Async", iteration, in_wl->count(),
                       out_wl->count());
                in_wl->reset();
            }
            swap(in_wl, out_wl);
            iteration++;
        }

        if (tid == 0) {
            printf("Total iterations: %d\n", iteration);
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

    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;
    Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

//    hybrid_datadriven::InitWorklist << < grid_dims, block_dims, 0, stream.cuda_stream >> >
//                                                                   (groute::dev::WorkSourceRange<index_t>(
//                                                                           dev_graph_allocator.DeviceObject().owned_start_node(),
//                                                                           dev_graph_allocator.DeviceObject().owned_nnodes()), wl1.DeviceObject());

    hybrid_datadriven::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                             (groute::dev::WorkSourceRange<index_t>(
                                                                                     dev_graph_allocator.DeviceObject().owned_start_node(),
                                                                                     dev_graph_allocator.DeviceObject().owned_nnodes()),
                                                                                     wl1.DeviceObject(),
                                                                                     dev_graph_allocator.DeviceObject(),
                                                                                     current_ranks.DeviceObject(),
                                                                                     residual.DeviceObject(),
                                                                                     last_residual.DeviceObject());

    stream.Sync();


    auto *in_wl = &wl1;
    auto *out_wl = &wl2;
    uint32_t iteration = 0;
    auto *available_delta = &residual;
    auto *last_available_delta = &last_residual;

    int mode;


    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);

    mgpu::mem_t<double> checkSum(1, mgpu_context);
    mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(context.host_graph.nnodes, mgpu_context);

    int *scanned_offsets = deviceOffsets.data();

    Stopwatch sw(true);

    while (in_wl->GetCount(stream) > 0) {
        KernelSizing(grid_dims, block_dims, in_wl->GetCount(stream));
        if (FLAGS_force_async)
            goto async;
        else if (FLAGS_force_sync)
            goto sync;
        if (iteration < FLAGS_async_to_sync || iteration >= FLAGS_sync_to_async) {
            async:
            hybrid_datadriven::PageRankAsyncKernelCTA__Single__
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                   (dev_graph_allocator.DeviceObject(),
                                                           in_wl->DeviceObject(),
                                                           out_wl->DeviceObject(),
                                                           current_ranks.DeviceObject(),
                                                           available_delta->DeviceObject());
            mode = 1;
        } else {
            sync:
            hybrid_datadriven::PageRankSyncKernelCTA__Single__
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                   (dev_graph_allocator.DeviceObject(),
                                                           in_wl->DeviceObject(),
                                                           out_wl->DeviceObject(),
                                                           iteration,
                                                           current_ranks.DeviceObject(),
                                                           available_delta->DeviceObject(),
                                                           last_available_delta->DeviceObject());
            if (iteration % 2 == 0) {
                available_delta = &last_residual;
            }
            mode = 0;
        }
        stream.Sync();


        rank_t *tmp = current_ranks.DeviceObject().data_ptr;

        auto func_data_sum_func = [=]__device__(int idx) {
            return tmp[idx];
        };


        mgpu::transform_scan<double>(func_data_sum_func, context.host_graph.nnodes,
                                     scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double pr_sum = mgpu::from_mem(checkSum)[0];

        if (pr_sum / context.host_graph.nnodes >= FLAGS_threshold)
            break;

        VLOG(1) << "Checking... SUM: " << pr_sum << " Relative SUM: " << pr_sum / context.host_graph.nnodes;

        printf("%s Iter:%d Input:%d Output:%d\n", mode == 0 ? "Sync" : "Async", iteration, in_wl->GetCount(stream),
               out_wl->GetCount(stream));
        in_wl->ResetAsync(stream);
        std::swap(in_wl, out_wl);
        iteration++;
    }

    printf("Total iterations: %d\n", iteration);


//    int occupancy_per_MP = FLAGS_grid_size;
//
//    cub::GridBarrierLifetime grid_barrier;
//
//    grid_barrier.Setup(occupancy_per_MP);

//    printf("grid size %d block size %d\n", occupancy_per_MP, FLAGS_block_size);


//    hybrid_datadriven::PageRankControl__Single__
//            << < occupancy_per_MP, FLAGS_block_size, 0, stream.cuda_stream >> >
//                                                        (FLAGS_sync,
//                                                                FLAGS_async_to_sync,
//                                                                FLAGS_sync_to_async,
//                                                                grid_barrier,
//                                                                wl1.DeviceObject(),
//                                                                wl2.DeviceObject(),
//                                                                dev_graph_allocator.DeviceObject(),
//                                                                current_ranks.DeviceObject(),
//                                                                residual.DeviceObject(),
//                                                                last_residual.DeviceObject());
//    stream.Sync();

    sw.stop();

    VLOG(0) << "Hybrid: " << sw.ms() << " ms.";
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