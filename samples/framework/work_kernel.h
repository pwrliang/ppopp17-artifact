//
// Created by liang on 2/19/18.
//

#ifndef GROUTE_WORK_KERNEL_H
#define GROUTE_WORK_KERNEL_H

namespace maiter {
    namespace kernel {
        template<typename TValue, typename TDelta>
        __global__ void GraphInit(groute::graphs::dev::CSRGraph dev_graph,
                                  maiter::IterateKernel<TValue, TDelta> **iterateKernel,
                                  groute::graphs::dev::GraphDatum <TValue> arr_value,
                                  groute::graphs::dev::GraphDatum <TDelta> arr_delta) {
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

                arr_value[node] = (*iterateKernel)->InitValue(node, out_degree);
                arr_delta[node] = (*iterateKernel)->InitDelta(node, out_degree);
            }
        };

        template<typename WorkSource,
                typename TAtomicFunc,
                typename TValue,
                typename TDelta>
        __global__ void GraphKernelTopology(groute::graphs::dev::CSRGraph dev_graph,
                                            WorkSource work_source,
                                            TAtomicFunc atomicFunc,
                                            maiter::IterateKernel<TValue, TDelta> **iterateKernel,
                                            groute::graphs::dev::GraphDatum <TValue> arr_value,
                                            groute::graphs::dev::GraphDatum <TDelta> arr_delta) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(arr_delta.get_item_ptr(node), 0);

                if (old_delta != (*iterateKernel)->IdentityElement()) {
                    arr_value[node] = (*iterateKernel)->accumulate(arr_value[node], old_delta);


                    index_t begin_edge = dev_graph.begin_edge(node),
                            end_edge = dev_graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta = (*iterateKernel)->g_func(old_delta, -1, out_degree);

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
                                              maiter::IterateKernel<TValue, TDelta> **iterateKernel,
                                              groute::graphs::dev::CSRGraph graph,
                                              groute::graphs::dev::GraphDatum <TValue> value_datum,
                                              groute::graphs::dev::GraphDatum <TDelta> delta_datum,
                                              groute::graphs::dev::GraphDatum <TWeight> weight_datum,
                                              bool IsWeighted = false) {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TDelta old_delta = atomicExch(delta_datum.get_item_ptr(node), 0);

                if (old_delta != (*iterateKernel)->IdentityElement()) {
                    value_datum[node] = (*iterateKernel)->accumulate(value_datum[node], old_delta);


                    index_t begin_edge = graph.begin_edge(node),
                            end_edge = graph.end_edge(node),
                            out_degree = end_edge - begin_edge;

                    if (out_degree > 0) {
                        TDelta new_delta;

                        if (!IsWeighted)
                            new_delta = (*iterateKernel)->g_func(old_delta, 0, out_degree);

                        for (index_t edge = begin_edge; edge < end_edge; edge++) {
                            index_t dest = graph.edge_dest(edge);
                            const float EPSLION = 0.01;

                            if (IsWeighted) {
                                new_delta = (*iterateKernel)->g_func(old_delta, weight_datum.get_item(edge),
                                                                     out_degree);
                            }

                            TDelta prev = atomicFunc(delta_datum.get_item_ptr(dest), new_delta);

                            if (prev < EPSLION && prev + new_delta > EPSLION) {
                                work_target.append_warp(dest);
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif //GROUTE_WORK_KERNEL_H
