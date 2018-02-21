//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_WORK_KERNEL_H
#define GROUTE_WORK_KERNEL_H

#include <groute/common.h>
#include <groute/device/cta_scheduler.cuh>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>

namespace gframe {
    namespace kernel {
        template<typename TGraphAPI, typename TValue, typename TDelta>
        __global__ void GraphInit(TGraphAPI graph_api,
                                  groute::graphs::dev::CSRGraph graph,
                                  groute::graphs::dev::GraphDatum<TValue> value_datum,
                                  groute::graphs::dev::GraphDatum<TDelta> delta_datum) {
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
                delta_datum[node] = graph_api.InitDelta(node, out_degree);
            }
        };

        template<typename TGraphAPI,
                typename WorkSource,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__ void GraphKernelTopology(TGraphAPI graph_api,
                                              WorkSource work_source,
                                              TAtomicFunc atomicFunc,
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
                            if (IsWeighted)new_delta = graph_api.DeltaMapper(old_delta, weight_datum.get_item(dest), out_degree);

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
        __global__ void GraphKernelTopologyCTA(TGraphAPI graph_api,
                                               groute::graphs::dev::CSRGraph graph,
                                               WorkSource work_source,
                                               TAtomicFunc atomicFunc,
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
        __global__ void GraphKernelDataDriven(TGraphAPI graph_api,
                                              WorkSource work_source,
                                              WorkTarget work_target,
                                              TAtomicFunc atomicFunc,
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
        __global__ void GraphKernelDataDrivenCTA(TGraphAPI graph_api,
                                                 WorkSource work_source,
                                                 WorkTarget work_target,
                                                 TAtomicFunc atomicFunc,
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

        template<typename TGraphAPI, typename WorkSource, typename TValue>
        TValue ConvergeCheck(TGraphAPI graph_api,
                             mgpu::context_t &context,
                             WorkSource work_source,
                             groute::graphs::dev::GraphDatum<TValue> value_datum) {
            TValue *tmp = value_datum.data_ptr;

            auto check_segment_sizes = [=]__device__(int idx) {
                TValue value = tmp[idx];

                if (value == graph_api.IdentityElement())
                    return (TValue)0;
                return tmp[idx];
            };

            mgpu::mem_t<TValue> checkSum(1, context);
            mgpu::mem_t<int> deviceOffsets = mgpu::mem_t<int>(work_source.get_size(), context);

            int *scanned_offsets = deviceOffsets.data();

            mgpu::transform_scan<TValue>(check_segment_sizes, work_source.get_size(),
                                         scanned_offsets, mgpu::plus_t<TValue>(), checkSum.data(), context);

            return mgpu::from_mem(checkSum)[0];
        }

    }
}

#endif //GROUTE_WORK_KERNEL_H
