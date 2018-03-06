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
DECLARE_int32(mode);
DECLARE_int32(switch_threshold);
DECLARE_int32(async_to_sync);
DECLARE_int32(sync_to_async);
DECLARE_bool(force_sync);
DECLARE_bool(force_async);
namespace hybrid {
    rank_t IDENTITY_ELEMENT = 0;
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

    template<typename TValue>
    __inline__ __device__ TValue warpReduce(TValue localSum) {
        localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

        return localSum;
    }

    template<template<typename> class TRankDatum>
    __device__ void PageRankCheck__Single__(TRankDatum<rank_t > current_ranks,
                                            rank_t *block_sum_buffer, rank_t *rtn_sum) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        const int SMEMDIM = blockDim.x / warpSize;
        __shared__ rank_t smem[32];

        uint32_t work_size = current_ranks.size;
        rank_t local_sum = 0;

        for (uint32_t node = 0 + tid; node < work_size; node += nthreads) {
            rank_t dist = current_ranks[node];
            if (dist != IDENTITY_ELEMENT)
                local_sum += dist;
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
            uint32_t sum = 0;
            for (int bid = 0; bid < gridDim.x; bid++) {
                sum += block_sum_buffer[bid];
            }
            *rtn_sum = sum;
        }
    }


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankSyncKernelCTA__Single__(
            WorkSource work_source,
            TGraph graph,
            index_t iteration,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
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
                    res = residual[node];//atomicExch(residual.get_item_ptr(node), 0);
                    residual[node] = 0;
                } else {
                    res = last_residual[node];
                    last_residual[node] = 0;
                    //res = atomicExch(last_residual.get_item_ptr(node), 0);
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
                    [&graph, &residual, &last_residual, &iteration](index_t edge, index_t size, rank_t update) {
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

    template<
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankAsyncKernelCTA__Single__(
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
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&graph, &residual](index_t edge, index_t size, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        atomicAdd(residual.get_item_ptr(dest), update);
                    }
            );
        }
    }


    template<
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankAsyncKernelCTAPersist__Single__(
            WorkSource work_source, TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        while(true) {
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
                        [&graph, &residual](index_t edge, index_t size, rank_t update) {
                            index_t dest = graph.edge_dest(edge);
                            atomicAdd(residual.get_item_ptr(dest), update);
                        }
                );
//                PageRankCheck__Single__
                //check here
            }
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

bool HybridTopologyDriven1() {
    VLOG(0) << "HybridTopologyDriven";

    typedef groute::Queue<index_t> Worklist;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> last_residual;

    utils::traversal::Context<hybrid::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
            dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_residual);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(0);

    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);


    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    hybrid::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                  (groute::dev::WorkSourceRange<index_t>(
                                                                          dev_graph_allocator.DeviceObject().owned_start_node(),
                                                                          dev_graph_allocator.DeviceObject().owned_nnodes()),
                                                                          dev_graph_allocator.DeviceObject(),
                                                                          current_ranks.DeviceObject(),
                                                                          residual.DeviceObject(),
                                                                          last_residual.DeviceObject());

    int iteration = 0;
    bool running = true;


    mgpu::mem_t<double> checkSum(1, mgpu_context);
    mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(context.host_graph.nnodes, mgpu_context);

    int *scanned_offsets = deviceOffsets.data();
    rank_t last_sum = 0;
    int mode;

    double compute_consume = 0;

    while (running) {
        Stopwatch sw_update(true);
        groute::graphs::single::NodeOutputDatum<rank_t> *available_residual = &residual;
        if (FLAGS_force_async)
            goto async;
        else if (FLAGS_force_sync)
            goto sync;
        if (iteration < FLAGS_sync_to_async || iteration >= FLAGS_async_to_sync) {
            sync:
            hybrid::PageRankSyncKernelCTA__Single__
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                   (groute::dev::WorkSourceRange<index_t>(
                                                           dev_graph_allocator.DeviceObject().owned_start_node(),
                                                           dev_graph_allocator.DeviceObject().owned_nnodes()),
                                                           dev_graph_allocator.DeviceObject(),
                                                           iteration,
                                                           current_ranks.DeviceObject(),
                                                           residual.DeviceObject(),
                                                           last_residual.DeviceObject());
            if (iteration % 2 == 0) {
                available_residual = &last_residual;
            }
            mode = 0;
        } else {
            async:
            hybrid::PageRankAsyncKernelCTA__Single__
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                   (groute::dev::WorkSourceRange<index_t>(
                                                           dev_graph_allocator.DeviceObject().owned_start_node(),
                                                           dev_graph_allocator.DeviceObject().owned_nnodes()),
                                                           dev_graph_allocator.DeviceObject(),
                                                           current_ranks.DeviceObject(),
                                                           available_residual->DeviceObject());
            mode = 1;
        }

        stream.Sync();

        sw_update.stop();

        compute_consume += sw_update.ms();


        VLOG(0) << boost::format("->%s Iter:%d THROUGHPUT:%f nodes/ms\n")
                   % (mode == 0 ? "Sync" : "Async") % iteration %
                   (dev_graph_allocator.DeviceObject().owned_nnodes() / sw_update.ms());

        rank_t *tmp = current_ranks.DeviceObject().data_ptr;

        auto func_data_sum_func = [=]__device__(int idx) {
            return tmp[idx];
        };


        mgpu::transform_scan<double>(func_data_sum_func, context.host_graph.nnodes,
                                     scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double pr_sum = mgpu::from_mem(checkSum)[0];

        VLOG(1) << "Checking... SUM: " << pr_sum << " Relative SUM: " << pr_sum / context.host_graph.nnodes;

        rank_t *p_residual = available_residual->DeviceObject().data_ptr;

        auto func_residual_sum = [=]__device__(index_t idx) {
            return p_residual[idx];
        };

        mgpu::transform_scan<double>(func_residual_sum, context.host_graph.nnodes, scanned_offsets,
                                     mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double residual_sum = mgpu::from_mem(checkSum)[0];

        double avg_residual_sum = residual_sum / context.host_graph.nnodes;

        auto func_variance_compute = [=]__device__(index_t idx) {
            return (p_residual[idx] - avg_residual_sum) * (p_residual[idx] - avg_residual_sum);
        };

        mgpu::transform_scan<double>(func_variance_compute, context.host_graph.nnodes, scanned_offsets,
                                     mgpu::plus_t<double>(), checkSum.data(), mgpu_context);

        double residual_variance = mgpu::from_mem(checkSum)[0] / context.host_graph.nnodes;

        VLOG(0) << "Residual Variance: " << residual_variance;

        if (last_sum > 0) {
            rank_t sum_delta = pr_sum - last_sum;

            VLOG(1) << "X FACTOR: " << sum_delta / sw_update.ms();
        }

        last_sum = pr_sum;

        if (pr_sum / context.host_graph.nnodes > FLAGS_threshold) {
            VLOG(0) << "Threshold reached";
            break;
        }

        iteration++;
        if (iteration >= FLAGS_max_pr_iterations) {
            LOG(WARNING) << "maximum iterations reached";
            break;
        }
    }


    VLOG(1) << boost::format("%s terminated after %d iterations ") % hybrid::Algo::Name() % iteration;
    VLOG(0) << hybrid::Algo::Name() << ": " << compute_consume << " ms. <filter>";

    // Gather
    auto gathered_output = hybrid::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        hybrid::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = hybrid::Algo::Host(context.host_graph, residual, current_ranks);
        return hybrid::Algo::CheckErrors(gathered_output, regression) == 0;
    } else {
        LOG(WARNING) << "Result not checked";
        return true;
    }
}

//bool HybridTopologyDriven() {
//    VLOG(0) << "HybridTopologyDriven";
//
//    typedef groute::Queue<index_t> Worklist;
//    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
//    groute::graphs::single::NodeOutputDatum<rank_t> residual;
//    groute::graphs::single::NodeOutputDatum<rank_t> last_residual;
//
//    utils::traversal::Context<hybrid::Algo> context(1);
//
//    groute::graphs::single::CSRGraphAllocator
//            dev_graph_allocator(context.host_graph);
//
//    context.SetDevice(0);
//
//    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_residual);
//
//    context.SyncDevice(0); // graph allocations are on default streams, must sync device
//
//    groute::Stream stream = context.CreateStream(0);
//
//    mgpu::standard_context_t mgpu_context(true, stream.cuda_stream);
//
//
//    dim3 grid_dims, block_dims;
//    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);
//
//    Stopwatch sw(true);
//
//    hybrid::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (groute::dev::WorkSourceRange<index_t>(
//            dev_graph_allocator.DeviceObject().owned_start_node(),
//            dev_graph_allocator.DeviceObject().owned_nnodes()),
//            dev_graph_allocator.DeviceObject(),
//            current_ranks.DeviceObject(),
//            residual.DeviceObject(),
//            last_residual.DeviceObject());
//
//    int iteration = 0;
//    bool running = true;
//
//    //mode = 0 means sync, mode = 1 means async
//    int mode = FLAGS_mode;
//    int last_mode;
//    srand(time(NULL));
//    groute::graphs::single::NodeOutputDatum<rank_t> *available_residual = &residual;
//
//    int totalIteration = 0;
//
//    double totalSync = 0;
//    double totalAsync = 0;
//    mgpu::mem_t<double> checkSum(1, mgpu_context);
//    mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(context.host_graph.nnodes, mgpu_context);
//
//    int *scanned_offsets = deviceOffsets.data();
//    rank_t last_sum = 0;
//    if (mode == 2 && FLAGS_switch_threshold > 0)mode = 1;
//    while (running) {
////        if (FLAGS_mode == 2)
////            mode = rand() % 2;
////        else
//
//        if (totalIteration > FLAGS_switch_threshold && mode == 1) {
//            last_sum = 0;
//            mode = 0;
//        }
//        Stopwatch sw_update(false);
//        VLOG(1) << "Iteration: " << ++totalIteration;
//
//
//        if (mode == 0) {
//            last_mode = 0;
//
//            sw_update.start();
//            hybrid::PageRankSyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (groute::dev::WorkSourceRange<index_t>(
//                    dev_graph_allocator.DeviceObject().owned_start_node(),
//                    dev_graph_allocator.DeviceObject().owned_nnodes()),
//                    dev_graph_allocator.DeviceObject(),
//                    iteration++,
//                    current_ranks.DeviceObject(),
//                    residual.DeviceObject(),
//                    last_residual.DeviceObject());
//            stream.Sync();
//            sw_update.stop();
//            totalSync += sw_update.ms();
//            VLOG(1) << "SYNC THROUGHPUT: " << dev_graph_allocator.DeviceObject().owned_nnodes() / sw_update.ms() << " nodes/ms";
//        } else {
//            last_mode = 1;
//            sw_update.start();
//            hybrid::PageRankAsyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (groute::dev::WorkSourceRange<index_t>(
//                    dev_graph_allocator.DeviceObject().owned_start_node(),
//                    dev_graph_allocator.DeviceObject().owned_nnodes()),
//                    dev_graph_allocator.DeviceObject(),
//                    current_ranks.DeviceObject(),
//                    available_residual->DeviceObject());
//            stream.Sync();
//            sw_update.stop();
//            totalAsync += sw_update.ms();
//            VLOG(1) << "ASYNC THROUGHPUT: " << dev_graph_allocator.DeviceObject().owned_nnodes() / sw_update.ms() << " nodes/ms";
//        }
//
//        if (last_mode == 0) {
//            if (iteration % 2 == 0)//last round is last_residual-->residual
//                available_residual = &residual;
//            else
//                available_residual = &last_residual;
//        }
//
//
//        rank_t *tmp = current_ranks.DeviceObject().data_ptr;
//
//        auto func_data_sum_func = [=]__device__(int idx) {
//            return tmp[idx];
//        };
//
//
//        mgpu::transform_scan<double>(func_data_sum_func, context.host_graph.nnodes,
//                                     scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);
//
//        double pr_sum = mgpu::from_mem(checkSum)[0];
//
//        VLOG(1) << "Checking... SUM: " << pr_sum << " Relative SUM: " << pr_sum / context.host_graph.nnodes;
//
//        rank_t *p_residual;
//        if (mode == 0) {
//            if (iteration % 2)
//                p_residual = last_residual.DeviceObject().data_ptr;
//            else
//                p_residual = residual.DeviceObject().data_ptr;
//        } else {
//            p_residual = residual.DeviceObject().data_ptr;
//        }
//
//
//        auto func_residual_sum = [=]__device__(index_t idx) {
//            return p_residual[idx];
//        };
//
//        mgpu::transform_scan<double>(func_residual_sum, context.host_graph.nnodes, scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);
//
//        double residual_sum = mgpu::from_mem(checkSum)[0];
//
//        double avg_residual_sum = residual_sum / context.host_graph.nnodes;
//
//        auto func_variance_compute = [=]__device__(index_t idx) {
//            return (p_residual[idx] - avg_residual_sum) * (p_residual[idx] - avg_residual_sum);
//        };
//
//        mgpu::transform_scan<double>(func_variance_compute, context.host_graph.nnodes, scanned_offsets, mgpu::plus_t<double>(), checkSum.data(), mgpu_context);
//
//        double residual_variance = mgpu::from_mem(checkSum)[0] / context.host_graph.nnodes;
//
//        VLOG(0) << "Residual Variance: " << residual_variance;
//
//        if (last_sum > 0) {
//            rank_t sum_delta = pr_sum - last_sum;
//
//            VLOG(1) << "X FACTOR: " << sum_delta / sw_update.ms();
//        }
//        last_sum = pr_sum;
//
//
//        if (pr_sum / context.host_graph.nnodes > FLAGS_threshold) {
//            VLOG(0) << "Threshold reached";
//            break;
//        }
//
//
//        if (totalIteration > FLAGS_max_pr_iterations) {
//            LOG(WARNING) << "maximum iterations reached";
//            break;
//        }
//
//    }
//
//    sw.stop();
//
//    VLOG(1)
//    << boost::format("%s terminated after %d iterations (max: %d, sync: %d, async: %d)") % hybrid::Algo::Name() % totalIteration %
//       FLAGS_max_pr_iterations % iteration % (totalIteration - iteration);
//    VLOG(0) << hybrid::Algo::Name() << ": " << sw.ms() << " ms. <filter>";
//    VLOG(0) << "AVG SYNC: " << totalSync / iteration << "ms / ASYNC: " << totalAsync / (totalIteration - iteration) << " ms.";
//
//    // Gather
//    auto gathered_output = hybrid::Algo::Gather(dev_graph_allocator, residual, current_ranks);
//
//    if (FLAGS_output.length() != 0)
//        hybrid::Algo::Output(FLAGS_output.c_str(), gathered_output);
//
//    if (FLAGS_check) {
//        auto regression = hybrid::Algo::Host(context.host_graph, residual, current_ranks);
//        return hybrid::Algo::CheckErrors(gathered_output, regression) == 0;
//    } else {
//        LOG(WARNING) << "Result not checked";
//        return true;
//    }
//}