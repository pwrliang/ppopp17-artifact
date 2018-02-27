//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_WORK_KERNEL_H
#define GROUTE_WORK_KERNEL_H

#include <cub/grid/grid_barrier.cuh>
#include <groute/common.h>
#include <groute/dwl/work_source.cuh>
#include <groute/device/queue.cuh>
#include <groute/device/cta_scheduler.cuh>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include "graph_common.h"

namespace gframe {
    namespace kernel {
        template<typename WorkSource>
        __global__
        void InitWorklist(groute::dev::Queue<index_t> worklist, WorkSource work_source) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
                worklist.append_warp(work_source.get_work(ii));
            }
        }

        template<typename TGraphAPI, typename TAtomicFunc, typename TValue, typename TDelta, typename TWeight>
        __global__
        void GraphInit(TGraphAPI graph_api,
                       TAtomicFunc atomicFunc,
                       groute::dev::WorkSourceRange<index_t> work_source,
                       groute::graphs::dev::CSRGraph graph,
                       groute::graphs::dev::GraphDatum<TValue> value_datum,
                       groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                       groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                       bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
                index_t node = work_source.get_work(ii);
                index_t begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                value_datum[node] = graph_api.InitValue(node, out_degree);
                delta_datum[node] = graph_api.InitDelta(node, out_degree);
            }
        }
        template<typename TGraphAPI,
                typename TAtomicFunc,
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
                                 groute::dev::WorkSourceRange<index_t> work_source,
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
                TValue old_value = value_datum[node];
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), graph_api.IdentityElementForDeltaReducer);
                TValue new_value = graph_api.ValueDeltaCombiner(old_value, old_delta);

                if (new_value != old_value) {
                    value_datum[node] = new_value;
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

                            TDelta delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                        }
                    }
                }
            }
        }

        template<typename TGraphAPI,
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
                                    groute::dev::WorkSourceRange<index_t> work_source,
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
                    TValue old_value = value_datum[node];
                    TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), graph_api.IdentityElementForDeltaReducer);
                    TValue new_value = graph_api.ValueDeltaCombiner(old_value, old_delta);

                    if (new_value != old_value) {
                        value_datum[node] = new_value;
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
                                atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                            }
                    );
                } else {
                    groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                            local_work,
                            [&graph, &delta_datum, &atomicFunc, &graph_api](
                                    index_t edge, index_t out_degree, TDelta new_delta) {
                                index_t dest = graph.edge_dest(edge);
                                atomicFunc(delta_datum.get_item_ptr(dest), new_delta);
                            }
                    );
                }
            }
        }

        template<typename TGraphAPI,
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
                                   groute::dev::Queue<index_t> work_source,
                                   groute::dev::Queue<index_t> work_target,
                                   groute::graphs::dev::CSRGraph graph,
                                   groute::graphs::dev::GraphDatum<TValue> value_datum,
                                   groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                   groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                   bool IsWeighted) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.count();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.read(i);
                TValue old_value = value_datum[node];
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), graph_api.IdentityElementForDeltaReducer);
                TValue new_value = graph_api.ValueDeltaCombiner(old_value, old_delta);

                if (new_value != old_value) {
                    value_datum[node] = new_value;
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
                                      groute::dev::Queue<index_t> work_source,
                                      groute::dev::Queue<index_t> work_target,
                                      groute::graphs::dev::CSRGraph graph,
                                      groute::graphs::dev::GraphDatum<TValue> value_datum,
                                      groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                      groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                      bool IsWeighted) {
            const unsigned tid = TID_1D;
            const unsigned nthreads = TOTAL_THREADS_1D;

            const uint32_t work_size = work_source.count();
            const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<TDelta> local_work = {0, 0};
                if (i < work_size) {
                    index_t node = work_source.read(i);
                    TValue old_value = value_datum[node];
                    TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), graph_api.IdentityElementForDeltaReducer);
                    TValue new_value = graph_api.ValueDeltaCombiner(old_value, old_delta);

                    if (new_value != old_value) {
                        value_datum[node] = new_value;
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

                                if (graph_api.Filter(prev_delta, new_delta)) work_target.append_warp(dest);
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

#ifdef __OUTLINING__

        template<typename TGraphAPI,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__
        void KernelControllerDataDriven(TGraphAPI graph_api,
                                        TAtomicFunc atomicFunc,
                                        groute::dev::Queue<index_t> work_source,
                                        groute::dev::Queue<index_t> work_target,
                                        groute::graphs::dev::CSRGraph graph,
                                        groute::graphs::dev::GraphDatum<TValue> value_datum,
                                        groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                        groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                        bool IsWeighted,
                                        bool cta_np,
                                        uint32_t max_iteration,
                                        cub::GridBarrier grid_barrier) {
            uint32_t tid = TID_1D;
            groute::dev::Queue<index_t> *in_wl = &work_source;
            groute::dev::Queue<index_t> *out_wl = &work_target;
            int iteration;

            for (iteration = 1; iteration <= max_iteration; iteration++) {
                if (cta_np) {
                    GraphKernelDataDrivenCTA(graph_api,
                                             atomicFunc,
                                             *in_wl,
                                             *out_wl,
                                             graph,
                                             value_datum,
                                             delta_datum,
                                             weight_datum,
                                             IsWeighted);
                } else {
                    GraphKernelDataDriven(graph_api,
                                          atomicFunc,
                                          *in_wl,
                                          *out_wl,
                                          graph,
                                          value_datum,
                                          delta_datum,
                                          weight_datum,
                                          IsWeighted);
                }
                grid_barrier.Sync();

                if (tid == 0) {
                    printf("Iteration: %d INPUT %d OUTPUT %d\n", iteration, in_wl->count(), out_wl->count());
                    in_wl->reset();
                }

                if (out_wl->count() == 0)
                    break;

                groute::dev::Queue<index_t> *tmp_wl = in_wl;
                in_wl = out_wl;
                out_wl = tmp_wl;
                grid_barrier.Sync();//???????????/?
            }

            if (tid == 0) {
                printf("Total iterations: %d\n", iteration);
            }
        }

        template<typename TGraphAPI,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__
        void KernelControllerTopologyDriven(TGraphAPI graph_api,
                                            TAtomicFunc atomicFunc,
                                            groute::dev::WorkSourceRange<index_t> work_source,
                                            groute::graphs::dev::CSRGraph graph,
                                            groute::graphs::dev::GraphDatum<TValue> value_datum,
                                            groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                            groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                            bool IsWeighted,
                                            bool cta_np,
                                            uint32_t max_iteration,
                                            TValue *grid_value_buffer,
                                            TDelta *grid_delta_buffer,
                                            int *running,
                                            cub::GridBarrier grid_barrier) {
            //printf("start KernelControllerTopologyDriven\n");
            uint32_t tid = TID_1D;
            TValue rtn_value;
            TDelta rtn_delta;

            int iteration = 1;
            int counter = 0;

            while (*running) {
                if (cta_np) {
                    GraphKernelTopologyCTA(graph_api,
                                           atomicFunc,
                                           work_source,
                                           graph,
                                           value_datum,
                                           delta_datum,
                                           weight_datum,
                                           IsWeighted);
                } else {
                    GraphKernelTopology(graph_api,
                                        atomicFunc,
                                        work_source,
                                        graph,
                                        value_datum,
                                        delta_datum,
                                        weight_datum,
                                        IsWeighted);
                }
                grid_barrier.Sync();
                ConvergeCheckDevice(graph_api,
                                    work_source,
                                    grid_value_buffer,
                                    grid_delta_buffer,
                                    &rtn_value,
                                    &rtn_delta,
                                    value_datum,
                                    delta_datum);
                grid_barrier.Sync();
                if (counter++ % CHECK_INTERVAL == 0) {
                    if (tid == 0) {
                        printf("Iteration: %d Current accumulated value: %lf Current accumulated delta: %lf\n", iteration, (double) rtn_value, (double) rtn_delta);
                        if (graph_api.IsTerminated(rtn_value, rtn_delta))*running = 0;
                    }
                    counter = 0;
                }

                iteration++;
                if (iteration >= max_iteration) {
                    break;
                }
                grid_barrier.Sync();
            }

            if (tid == 0) {
                printf("Total iterations: %d\n", iteration);
            }
        }

#endif

        template<typename TGraphAPI, typename TValue>
        __forceinline__ __device__ TValue WarpValueReducer(TGraphAPI graph_api, TValue partial_value) {
            partial_value = graph_api.ValueReducer(partial_value, __shfl_xor_sync(0xfffffff, partial_value, 16));
            partial_value = graph_api.ValueReducer(partial_value, __shfl_xor_sync(0xfffffff, partial_value, 8));
            partial_value = graph_api.ValueReducer(partial_value, __shfl_xor_sync(0xfffffff, partial_value, 4));
            partial_value = graph_api.ValueReducer(partial_value, __shfl_xor_sync(0xfffffff, partial_value, 2));
            partial_value = graph_api.ValueReducer(partial_value, __shfl_xor_sync(0xfffffff, partial_value, 1));
            return partial_value;
        }

        template<typename TGraphAPI, typename TDelta>
        __forceinline__ __device__ TDelta WarpDeltaReducer(TGraphAPI graph_api, TDelta partial_delta) {
            partial_delta = graph_api.DeltaReducer(partial_delta, __shfl_xor_sync(0xfffffff, partial_delta, 16));
            partial_delta = graph_api.DeltaReducer(partial_delta, __shfl_xor_sync(0xfffffff, partial_delta, 8));
            partial_delta = graph_api.DeltaReducer(partial_delta, __shfl_xor_sync(0xfffffff, partial_delta, 4));
            partial_delta = graph_api.DeltaReducer(partial_delta, __shfl_xor_sync(0xfffffff, partial_delta, 2));
            partial_delta = graph_api.DeltaReducer(partial_delta, __shfl_xor_sync(0xfffffff, partial_delta, 1));
            return partial_delta;
        }

        template<typename TGraphAPI, typename TValue, typename TDelta>
        __device__
        void ConvergeCheckDevice(TGraphAPI graph_api,
                                 groute::dev::WorkSourceRange<index_t> work_source,
                                 TValue *grid_value_buffer,
                                 TDelta *grid_delta_buffer,
                                 TValue *rtn_value_res,
                                 TDelta *rtn_delta_res,
                                 groute::graphs::dev::GraphDatum<TValue> value_datum,
                                 groute::graphs::dev::GraphDatum<TDelta> delta_datum) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;
            int laneIdx = threadIdx.x % warpSize;
            int warpIdx = threadIdx.x / warpSize;
            const int SMEMDIM = blockDim.x / warpSize;
//            extern __shared__ TValue smem[];
            __shared__ TValue smem_value[32];
            __shared__ TDelta smem_delta[32];
//            TValue *smem_value = (TValue *) smem[0];
//            TDelta *smem_delta = (TDelta *) smem[SMEMDIM];

            uint32_t work_size = work_source.get_size();
            TValue local_sum_value = graph_api.IdentityElementForValueReducer;
            TDelta local_sum_delta = graph_api.IdentityElementForDeltaReducer;

            for (uint32_t work_id = tid; work_id < work_size; work_id += nthreads) {
                index_t node = work_source.get_work(work_id);

                local_sum_value = graph_api.ValueReducer(local_sum_value, value_datum[node]);
                local_sum_delta = graph_api.DeltaReducer(local_sum_delta, delta_datum[node]);
            }

            local_sum_value = WarpValueReducer(graph_api, local_sum_value);
            local_sum_delta = WarpDeltaReducer(graph_api, local_sum_delta);

            if (laneIdx == 0) {
                smem_value[warpIdx] = local_sum_value;
                smem_delta[warpIdx] = local_sum_delta;
            }

            __syncthreads();

            local_sum_value = (threadIdx.x < SMEMDIM) ? smem_value[threadIdx.x] : graph_api.IdentityElementForValueReducer;
            local_sum_delta = (threadIdx.x < SMEMDIM) ? smem_delta[threadIdx.x] : graph_api.IdentityElementForDeltaReducer;

            // let first warp to reduce the whole block, this is ok because warpSize = 32, and max threads per block no more than 1024
            // so SMEMDIM <= 32, a warp can hold this job.
            if (warpIdx == 0) {
                local_sum_value = WarpValueReducer(graph_api, local_sum_value);
                local_sum_delta = WarpDeltaReducer(graph_api, local_sum_delta);
            }

            if (threadIdx.x == 0) {
                grid_value_buffer[blockIdx.x] = local_sum_value;
                grid_delta_buffer[blockIdx.x] = local_sum_delta;
            }

            if (tid == 0) {
                TValue accumulated_value = graph_api.IdentityElementForValueReducer;
                TDelta accumulated_delta = graph_api.IdentityElementForDeltaReducer;

                for (int bid = 0; bid < gridDim.x; bid++) {
                    accumulated_value = graph_api.ValueReducer(accumulated_value, grid_value_buffer[bid]);
                    accumulated_delta = graph_api.DeltaReducer(accumulated_delta, grid_delta_buffer[bid]);
                }
                *rtn_value_res = accumulated_value;
                *rtn_delta_res = accumulated_delta;
            }
        }

        template<typename TGraphAPI, typename TValue, typename TDelta>
        __global__
        void ConvergeCheck(TGraphAPI graph_api,
                           groute::dev::WorkSourceRange<index_t> work_source,
                           TValue *grid_value_buffer,
                           TDelta *grid_delta_buffer,
                           TValue *rtn_value_res,
                           TDelta *rtn_delta_res,
                           groute::graphs::dev::GraphDatum<TValue> value_datum,
                           groute::graphs::dev::GraphDatum<TDelta> delta_datum) {

            ConvergeCheckDevice(graph_api,
                                work_source,
                                grid_value_buffer,
                                grid_delta_buffer,
                                rtn_value_res,
                                rtn_delta_res,
                                value_datum,
                                delta_datum);
        }
    }
}

#endif //GROUTE_WORK_KERNEL_H
