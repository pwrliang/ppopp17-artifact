//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_WORK_KERNEL_H
#define GROUTE_WORK_KERNEL_H

#include <groute/common.h>
#include <groute/device/cta_scheduler.cuh>

namespace gframe {
    namespace kernel {
        template<typename TValue, typename TDelta>
        __global__ void GraphInit(groute::graphs::dev::CSRGraph dev_graph,
                                  gframe::GraphAPIBase<TValue, TDelta> **graph_api_base,
                                  groute::graphs::dev::GraphDatum<TValue> arr_value,
                                  groute::graphs::dev::GraphDatum<TDelta> arr_delta) {
            const unsigned TID = TID_1D;
            const unsigned NTHREADS = TOTAL_THREADS_1D;

            index_t start_node = dev_graph.owned_start_node();
            index_t end_node = start_node + dev_graph.owned_nnodes();
            for (index_t node = start_node + TID;
                 node < end_node; node += NTHREADS) {

                index_t
                        begin_edge = dev_graph.begin_edge(node),
                        end_edge = dev_graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                arr_value[node] = (*graph_api_base)->InitValue(node, out_degree);
                arr_delta[node] = (*graph_api_base)->InitDelta(node, out_degree);
            }
        };

        template<typename WorkSource,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta>
        __global__ void GraphKernelTopology(groute::graphs::dev::CSRGraph dev_graph,
                                            WorkSource work_source,
                                            TAtomicFunc atomicFunc,
                                            gframe::GraphAPIBase<TValue, TDelta> **graph_api_base,
                                            groute::graphs::dev::GraphDatum<TValue> arr_value,
                                            groute::graphs::dev::GraphDatum<TDelta> arr_delta) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(arr_delta.get_item_ptr(node), 0);

                if (old_delta != (*graph_api_base)->IdentityElement()) {
                    arr_value[node] = (*graph_api_base)->DeltaReducer(arr_value[node], old_delta);


                    index_t begin_edge = dev_graph.begin_edge(node),
                            end_edge = dev_graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta = (*graph_api_base)->DeltaMapper(old_delta, -1, out_degree);

                        for (index_t edge = begin_edge; edge < end_edge; edge++) {
                            index_t dest = dev_graph.edge_dest(edge);

                            atomicFunc(arr_delta.get_item_ptr(dest), new_delta);
                        }
                    }
                }
            }
        }

        template<typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__ void GraphKernelDataDriven(WorkSource work_source,
                                              WorkTarget work_target,
                                              TAtomicFunc atomicFunc,
                                              gframe::GraphAPIBase<TValue, TDelta> **graph_api_base,
                                              groute::graphs::dev::CSRGraph graph,
                                              groute::graphs::dev::GraphDatum<TValue> value_datum,
                                              groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                              groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                              bool IsWeighted = false) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                if (old_delta != (*graph_api_base)->IdentityElement()) {
                    TValue value = (*graph_api_base)->DeltaReducer(value_datum[node], old_delta);
                    value_datum[node] = value;

                    index_t begin_edge = graph.begin_edge(node),
                            end_edge = graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta;

                        if (!IsWeighted)
                            new_delta = (*graph_api_base)->DeltaMapper(old_delta, 0, out_degree);

                        for (index_t edge = begin_edge; edge < end_edge; edge++) {
                            index_t dest = graph.edge_dest(edge);

                            if (IsWeighted) {
                                new_delta = (*graph_api_base)->DeltaMapper(old_delta, weight_datum.get_item(edge),
                                                                           out_degree);
                            }

                            TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                            if ((*graph_api_base)->Filter( prev_delta, new_delta)) {
                                work_target.append_warp(dest);
                            }
                        }
                    }
                }
            }
        }

        template<typename WorkSource,
                typename WorkTarget,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta,
                typename TWeight>
        __global__ void GraphKernelDataDrivenCTA(WorkSource work_source,
                                                 WorkTarget work_target,
                                                 TAtomicFunc atomicFunc,
                                                 gframe::GraphAPIBase<TValue, TDelta> **graph_api_base,
                                                 groute::graphs::dev::CSRGraph graph,
                                                 groute::graphs::dev::GraphDatum<TValue> value_datum,
                                                 groute::graphs::dev::GraphDatum<TDelta> delta_datum,
                                                 groute::graphs::dev::GraphDatum<TWeight> weight_datum,
                                                 bool IsWeighted = false) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<TDelta> local_work = {0, 0};
                if (i < work_size) {
                    index_t node = work_source.get_work(i);
                    TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                    if (old_delta != (*graph_api_base)->IdentityElement()) {
                        TValue value = (*graph_api_base)->DeltaReducer(value_datum[node], old_delta);
                        value_datum[node] = value;

                        local_work.start = graph.begin_edge(node);
                        local_work.size = graph.end_edge(node) - local_work.start;
                        local_work.meta_data = old_delta;
                    }
                }

                groute::dev::CTAWorkScheduler<TDelta>::template schedule(
                        local_work,
                        [&work_target, &graph, &weight_datum, &delta_datum, &atomicFunc, &graph_api_base, &IsWeighted](index_t edge, index_t out_degree , TDelta old_delta) {
                            index_t dest = graph.edge_dest(edge);
                            TWeight weight = 0;

                            if (IsWeighted) {
                                weight = weight_datum.get_item(edge);
                            }

                            TDelta new_delta = (*graph_api_base)->DeltaMapper(old_delta, weight, out_degree);
                            TDelta prev_delta = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                            if ((*graph_api_base)->Filter(prev_delta, new_delta)) {
                                work_target.append_warp(dest);
                            }
                        }
                );
            }
        }
    }
}

#endif //GROUTE_WORK_KERNEL_H
