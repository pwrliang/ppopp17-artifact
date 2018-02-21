//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_WORK_KERNEL_H
#define GROUTE_WORK_KERNEL_H

#include <cub/grid/grid_barrier.cuh>
#include <groute/common.h>
#include <groute/device/cta_scheduler.cuh>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>

namespace gframe {
    namespace kernel {
        template<typename TGraphAPI,
                typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__ void GraphInitDataDriven(TGraphAPI graph_api,
                                            TAtomicFunc atomicFunc,
                                            WorkSource work_source,
                                            WorkTarget work_target,
                                            groute::graphs::dev::CSRGraph graph,
                                            groute::graphs::dev::GraphDatum<TValue> value_datum,
                                            groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                            groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                            bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
                index_t node = work_source.get_work(ii);
                index_t
                        begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                value_datum[node] = graph_api.InitValue(node, out_degree);
                TDelta init_delta = graph_api.InitDelta(node, out_degree);
                delta_datum[node] = init_delta;

                if (out_degree > 0) {
                    TDelta new_delta;

                    if (!IsWeighted)
                        new_delta = graph_api.DeltaMapper(init_delta, 0, out_degree);

                    for (index_t edge = begin_edge; edge < end_edge; edge++) {
                        index_t dest = graph.edge_dest(edge);
                        if (IsWeighted)new_delta = graph_api.DeltaMapper(init_delta, weight_datum.get_item(edge), out_degree);
                        TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                        if (graph_api.Filter(prev_delta, new_delta))
                            work_target.append_warp(dest);
                    }
                }
            }
        }

        template<typename TGraphAPI, typename TAtomicFunc, typename TValue, typename TDelta, typename TWeight>
        __global__
        void GraphInitTopologyDriven(TGraphAPI graph_api,
                                     TAtomicFunc atomicFunc,
                                     groute::graphs::dev::CSRGraph graph,
                                     groute::graphs::dev::GraphDatum<TValue> value_datum,
                                     groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                     groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                     bool IsWeighted) {
            const unsigned TID = TID_1D;
            const unsigned NTHREADS = TOTAL_THREADS_1D;

            index_t start_node = graph.owned_start_node();
            index_t end_node = start_node + graph.owned_nnodes();
            for (index_t node = start_node + TID;
                 node < end_node; node += NTHREADS) {

                index_t
                        begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                value_datum[node] = graph_api.InitValue(node, out_degree);
                TDelta init_delta = graph_api.InitDelta(node, out_degree);
                delta_datum[node] = init_delta;

                if (out_degree > 0) {
                    TDelta new_delta;

                    if (!IsWeighted)
                        new_delta = graph_api.DeltaMapper(init_delta, 0, out_degree);

                    for (index_t edge = begin_edge; edge < end_edge; edge++) {
                        index_t dest = graph.edge_dest(edge);
                        if (IsWeighted)new_delta = graph_api.DeltaMapper(init_delta, weight_datum.get_item(edge), out_degree);
                        TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                    }
                }
            }
        }

        template<typename TGraphAPI,
                typename TAtomicFunc,
                typename WorkSource,
                typename TValue,
                typename TDelta,
                typename TWeight>
#ifdef __OUTLINING__
        __device__
#else
        __global__
#endif
        void GraphKernelTopology(TGraphAPI graph_api,
                                 TAtomicFunc atomicFunc,
                                 WorkSource work_source,
                                 groute::graphs::dev::CSRGraph graph,
                                 groute::graphs::dev::GraphDatum<TValue> value_datum,
                                 groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                 groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                 bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                if (old_delta != graph_api.IdentityElement()) {
                    value_datum[node] = graph_api.DeltaReducer(value_datum[node], old_delta);


                    index_t begin_edge = graph.begin_edge(node),
                            end_edge = graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta;

                        if (!IsWeighted)
                            new_delta = graph_api.DeltaMapper(old_delta, 0, out_degree);

                        for (index_t edge = begin_edge; edge < end_edge; edge++) {
                            index_t dest = graph.edge_dest(edge);
                            if (IsWeighted)new_delta = graph_api.DeltaMapper(old_delta, weight_datum.get_item(edge), out_degree);

                            atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                        }
                    }
                }
            }
        }

        template<typename TGraphAPI,
                typename WorkSource,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
#ifdef __OUTLINING__
        __device__
#else
        __global__
#endif
        void GraphKernelTopologyCTA(TGraphAPI graph_api,
                                    TAtomicFunc atomicFunc,
                                    WorkSource work_source,
                                    groute::graphs::dev::CSRGraph graph,
                                    groute::graphs::dev::GraphDatum<TValue> value_datum,
                                    groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                    groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                    bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<TDelta> local_work = {0, 0};

                if (i < work_size) {
                    index_t node = work_source.get_work(i);
                    TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                    if (old_delta != graph_api.IdentityElement()) {
                        value_datum[node] = graph_api.DeltaReducer(value_datum[node], old_delta);

                        local_work.start = graph.begin_edge(node);
                        local_work.size = graph.end_edge(node) - local_work.start;
                        if (IsWeighted)
                            local_work.meta_data = old_delta;
                        else
                            local_work.meta_data = graph_api.DeltaMapper(old_delta, 0, local_work.size);
                    }
                }

                if (IsWeighted) {
                    groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                            local_work,
                            [&graph, &weight_datum, &delta_datum, &atomicFunc, &graph_api](
                                    index_t edge, index_t out_degree, TDelta old_delta) {
                                index_t dest = graph.edge_dest(edge);

                                TDelta new_delta = graph_api.DeltaMapper(old_delta, weight_datum.get_item(edge), out_degree);
                                TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                            }
                    );
                } else {
                    groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                            local_work,
                            [&graph, &delta_datum, &atomicFunc, &graph_api](
                                    index_t edge, index_t out_degree, TDelta new_delta) {
                                index_t dest = graph.edge_dest(edge);
                                TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                            }
                    );
                }
            }
        }

        template<typename TGraphAPI,
                typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
#ifdef __OUTLINING__
        __device__
#else
        __global__
#endif
        void GraphKernelDataDriven(TGraphAPI graph_api,
                                   TAtomicFunc atomicFunc,
                                   WorkSource work_source,
                                   WorkTarget work_target,
                                   groute::graphs::dev::CSRGraph graph,
                                   groute::graphs::dev::GraphDatum<TValue> value_datum,
                                   groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                   groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                   bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                if (old_delta != graph_api.IdentityElement()) {
                    TValue value = graph_api.DeltaReducer(value_datum[node], old_delta);
                    value_datum[node] = value;

                    index_t begin_edge = graph.begin_edge(node),
                            end_edge = graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta;
                        if (!IsWeighted) new_delta = graph_api.DeltaMapper(old_delta, 0, out_degree);

                        for (index_t edge = begin_edge; edge < end_edge; edge++) {
                            index_t dest = graph.edge_dest(edge);
                            if (IsWeighted)new_delta = graph_api.DeltaMapper(old_delta, weight_datum.get_item(edge), out_degree);
                            TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                            if (graph_api.Filter(prev_delta, new_delta)) work_target.append_warp(dest);
                        }
                    }
                }
            }
        }


        template<typename TGraphAPI,
                typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
#ifdef __OUTLINING__
        __device__
#else
        __global__
#endif
        void GraphKernelDataDrivenCTA(TGraphAPI graph_api,
                                      TAtomicFunc atomicFunc,
                                      WorkSource work_source,
                                      WorkTarget work_target,
                                      groute::graphs::dev::CSRGraph graph,
                                      groute::graphs::dev::GraphDatum<TValue> value_datum,
                                      groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                      groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                      bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<TDelta> local_work = {0, 0};
                if (i < work_size) {
                    index_t node = work_source.get_work(i);
                    TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                    if (old_delta != graph_api.IdentityElement()) {
                        TValue value = graph_api.DeltaReducer(value_datum[node], old_delta);
                        value_datum[node] = value;

                        local_work.start = graph.begin_edge(node);
                        local_work.size = graph.end_edge(node) - local_work.start;
                        if (IsWeighted)
                            local_work.meta_data = old_delta;
                        else
                            local_work.meta_data = graph_api.DeltaMapper(old_delta, 0, local_work.size);
                    }
                }

                if (IsWeighted) {
                    groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                            local_work,
                            [&work_target, &graph, &weight_datum, &delta_datum, &atomicFunc, &graph_api](
                                    index_t edge, index_t out_degree, TDelta old_delta) {
                                index_t dest = graph.edge_dest(edge);

                                TDelta new_delta = graph_api.DeltaMapper(old_delta, weight_datum.get_item(edge), out_degree);
                                TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                                if (graph_api.Filter(prev_delta, new_delta)) {
                                    work_target.append_warp(dest);
                                }
                            }
                    );
                } else {
                    groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                            local_work,
                            [&work_target, &graph, &delta_datum, &atomicFunc, &graph_api](
                                    index_t edge, index_t out_degree, TDelta new_delta) {
                                index_t dest = graph.edge_dest(edge);
                                TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                                if (graph_api.Filter(prev_delta, new_delta)) work_target.append_warp(dest);
                            }
                    );
                }
            }
        }

//#ifdef __OUTLINING__
        template<typename TGraphAPI,
                typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__
        void KernelControllerDataDriven(TGraphAPI graph_api,
                                        TAtomicFunc atomicFunc,
                                        WorkSource work_source,
                                        WorkTarget work_target,
                                        groute::graphs::dev::CSRGraph graph,
                                        groute::graphs::dev::GraphDatum<TValue> value_datum,
                                        groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                        groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                        bool IsWeighted,
                                        bool cta_np,
                                        uint32_t max_iteration,
                                        cub::GridBarrier grid_barrier) {
            uint32_t tid = TID_1D;
            WorkSource *in_wl = &work_source;
            WorkTarget *out_wl = &work_target;
            int iteration;

            for (iteration = 0; iteration < max_iteration; iteration++) {
                if (cta_np) {
                    GraphKernelDataDrivenCTA(graph_api,
                                             *in_wl,
                                             *out_wl,
                                             atomicFunc,
                                             graph,
                                             value_datum,
                                             delta_datum,
                                             weight_datum);
                } else {
                    GraphKernelDataDriven(graph_api,
                                          *in_wl,
                                          *out_wl,
                                          atomicFunc,
                                          graph,
                                          value_datum,
                                          delta_datum,
                                          weight_datum);
                }
                grid_barrier.Sync();

                if (tid == 0) {
                    printf("Iteration: %d INPUT %d OUTPUT %d\n", iteration, in_wl->count(), out_wl->count());
                    in_wl->reset();
                }

                if (out_wl->get_size() == 0)
                    break;

                WorkSource *tmp_wl = in_wl;
                in_wl = out_wl;
                out_wl = tmp_wl;
            }

            if (tid == 0) {
                printf("Total iterations: %d\n", iteration);
            }
        }

//#endif

        template<typename TGraphAPI,
                typename WorkSource,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__
        void KernelControllerTopologyDriven(TGraphAPI graph_api,
                                            TAtomicFunc atomicFunc,
                                            WorkSource work_source,
                                            groute::graphs::dev::CSRGraph graph,
                                            groute::graphs::dev::GraphDatum<TValue> value_datum,
                                            groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                            groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                            bool IsWeighted,
                                            bool cta_np,
                                            uint32_t max_iteration,
                                            TValue *grid_buffer,
                                            int *running,
                                            cub::GridBarrier grid_barrier) {
            uint32_t tid = TID_1D;
            WorkSource *in_wl = &work_source;
            TValue rtn_res;

            int iteration = 0;
            int counter = 0;

            while (*running) {
                if (cta_np) {
                    GraphKernelTopologyCTA(graph_api,
                                           atomicFunc,
                                           *in_wl,
                                           graph,
                                           value_datum,
                                           delta_datum,
                                           weight_datum);
                } else {
                    GraphKernelTopology(graph_api,
                                        atomicFunc,
                                        *in_wl,
                                        graph,
                                        value_datum,
                                        delta_datum,
                                        weight_datum);
                }
                grid_barrier.Sync();
                iteration++;
                if (iteration >= max_iteration) {
                    break;
                }

                ConvergeCheckDevice(graph_api,
                                    work_source,
                                    grid_barrier,
                                    &rtn_res,
                                    value_datum);
                if (counter++ % 10000 == 0) {
                    if (tid == 0) {
                        if (graph_api.IsConverge(rtn_res))*running = 0;
                    }
                    counter = 0;
                }
                grid_barrier.Sync();
            }

            if (tid == 0) {
                printf("Total iterations: %d\n", iteration);
            }
        }

        template<typename TGraphAPI, typename WorkSource, typename TValue>
        TValue ConvergeCheck(TGraphAPI graph_api,
                             mgpu::context_t &context,
                             WorkSource work_source,
                             groute::graphs::dev::GraphDatum<TValue> value_datum) {
            TValue *tmp = value_datum.data_ptr;

            auto check_segment_sizes = [=]__device__(int idx) {
                TValue value = tmp[idx];

                if (value == graph_api.IdentityElement())
                    return (TValue) 0;
                return tmp[idx];
            };

            mgpu::mem_t<TValue> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);

            int *scanned_offsets = deviceOffsets.data();

            mgpu::transform_scan<TValue>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<TValue>(), checkSum.data(), context);

            return mgpu::from_mem(checkSum)[0];
        }

        template<typename TValue>
        __forceinline__ __device__ TValue warpReduce(TValue localSum) {
            localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
            localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
            localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
            localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
            localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

            return localSum;
        }

        template<typename TGraphAPI, typename WorkSource, typename TValue>
        __device__
        void ConvergeCheckDevice(TGraphAPI graph_api,
                                 WorkSource work_source,
                                 TValue *grid_buffer,
                                 TValue *rtn_res,
                                 groute::graphs::dev::GraphDatum<TValue> value_datum) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;
            int laneIdx = threadIdx.x % warpSize;
            int warpIdx = threadIdx.x / warpSize;
            const int SMEMDIM = blockDim.x / warpSize;
            extern __shared__ TValue smem[];

            uint32_t work_size = work_source.get_size();
            TValue local_sum = 0;

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                local_sum += value_datum[node];
            }

            local_sum = warpReduce(local_sum);

            if (laneIdx == 0)
                smem[warpIdx] = local_sum;
            __syncthreads();

            local_sum = (threadIdx.x < SMEMDIM) ? smem[threadIdx.x] : 0;

            if (warpIdx == 0)
                local_sum = warpReduce(local_sum);

            if (threadIdx.x == 0) {
                grid_buffer[blockIdx.x] = local_sum;
            }

            if (tid == 0) {
                TValue sum = 0;

                for (int bid = 0; bid < gridDim.x; bid++) {
                    sum += grid_buffer[bid];
                }
                *rtn_res = sum;
            }
        };
    }
}

#endif //GROUTE_WORK_KERNEL_H
