// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>

#include <gflags/gflags.h>

#include <groute/device/cta_scheduler.cuh>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/distributed_worklist.cuh>
#include <groute/dwl/workers.cuh>

#include <utils/graphs/traversal.h>
#include <device_launch_parameters.h>

#include "sssp_common.h"

DECLARE_int32(source_node);
const distance_t INF = UINT_MAX;

namespace sssp {

    struct LocalData {
        index_t m_node;
        uint32_t m_iterated_round;

        __device__ __host__

        __forceinline__ LocalData(uint32_t iterated_round, index_t node) : m_iterated_round(iterated_round),
                                                                           m_node(node) {

        }
    };

    struct DistanceData {
        index_t node;
        distance_t distance;
        uint32_t iterated_round;
        uint32_t padding;

        __device__ __host__

        __forceinline__ DistanceData(uint32_t iterated_round, index_t node, distance_t distance) :
                iterated_round(iterated_round),
                node(node), distance(distance) {}

        __device__ __host__

        __forceinline__ DistanceData() : iterated_round(0), node(INF), distance(INF) {
            printf("call DistanceData default constructor\n");
        }
    };

    typedef LocalData local_work_t;
    typedef DistanceData remote_work_t;

    __global__ void
    SSSPMemsetKernel(index_t source_node, distance_t *distances, distance_t *delta, distance_t *last_delta,
                     int nnodes) {
        int tid = TID_1D;
        if (tid < nnodes) {
            distances[tid] = INF;
            delta[tid] = INF;
            last_delta[tid] = INF;

//            if (tid == source_node) {
//                delta[tid] = 0;
//                last_delta[tid] = 0;
//            }
        }
    }

    template<bool CTAScheduling = true>
    /// SSSP work with Collective Thread Array scheduling for exploiting nested parallelism 
    struct SSSPWork {
        template<
                typename WorkSource, typename WorkTarget,
                typename TGraph, typename TWeightDatum, typename TDistanceDatum>
        __device__ static void work(
                const uint32_t iterated_round,
                const WorkSource &work_source, WorkTarget &work_target,
                const TGraph &graph, TWeightDatum &edge_weights,
                TDistanceDatum &node_distances,
                TDistanceDatum &node_delta,
                TDistanceDatum &node_last_delta) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();
            uint32_t work_size_rup = round_up(work_size, blockDim.x) *
                                     blockDim.x; // we want all threads in active blocks to enter the loop

            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<distance_t> np_local = {0, 0, 0};

                if (i < work_size) {
                    index_t node = work_source.get_work(i).m_node;

                    distance_t old_value = node_distances[node];
                    printf("abc: iter:%u %u %u %u\n", iterated_round, node_distances[node], node_delta[node],
                           node_last_delta[node]);


                    distance_t old_delta;
                    if (iterated_round % 2)
                        old_delta = atomicExch(node_delta.get_item_ptr(node), INF);
                    else
                        old_delta = atomicExch(node_last_delta.get_item_ptr(node), INF);


                    distance_t new_value = min(old_value, old_delta);

                    printf("iter:%d node %d old_delta %d\n", iterated_round, node, old_delta);
                    printf("old value:%d new value:%d\n", old_value, new_value);

                    if (new_value != old_value) {
                        printf("node: %d old_delta:%d\n", node, old_delta);

                        node_distances[node] = new_value;
                        np_local.start = graph.begin_edge(node);
                        np_local.size = graph.end_edge(node) - np_local.start;
                        np_local.meta_data = old_delta;
                    }
                }

                groute::dev::CTAWorkScheduler<distance_t>::template schedule(
                        np_local,
                        [&iterated_round, &work_target, &graph, &edge_weights, &node_last_delta, &node_delta](
                                index_t edge,
                                index_t size,
                                distance_t old_delta) {
                            index_t dest = graph.edge_dest(edge);
                            distance_t weight = edge_weights.get_item(edge);
                            distance_t new_delta = old_delta + weight;

                            if (iterated_round % 2) {
                                if (new_delta < atomicMin(node_last_delta.get_item_ptr(dest), new_delta))
                                    work_target.append_work(DistanceData(iterated_round, dest, new_delta));
                            } else {
                                if (new_delta < atomicMin(node_delta.get_item_ptr(dest), new_delta))
                                    work_target.append_work(DistanceData(iterated_round, dest, new_delta));
                            }
                        }
                );
            }
        }
    };

    template<>
    /// SSSP work without CTA support
    struct SSSPWork<false> {
        template<
                typename WorkSource, typename WorkTarget,
                typename TGraph, typename TWeightDatum, typename TDistanceDatum>
        __device__ static void work(
                const uint32_t iterated_round,
                const WorkSource &work_source, WorkTarget &work_target,
                const TGraph &graph, TWeightDatum &edge_weights,
                TDistanceDatum &node_distances,
                TDistanceDatum &node_delta,
                TDistanceDatum &node_last_delta
        ) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i).m_node;
                distance_t old_value = node_distances.get_item(node);
                distance_t old_delta;

                if (iterated_round % 2)
                    old_delta = atomicExch(node_delta.get_item_ptr(node), INF);
                else
                    old_delta = atomicExch(node_last_delta.get_item_ptr(node), INF);
                distance_t new_value = min(old_value, old_delta);

                if (new_value != old_value) {
                    for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node);
                         edge < end_edge; ++edge) {
                        index_t dest = graph.edge_dest(edge);
                        distance_t weight = edge_weights.get_item(edge);
                        distance_t new_delta = old_delta + weight;

                        if (iterated_round % 2) {
                            if (new_delta < atomicMin(node_last_delta.get_item_ptr(node), new_delta)) {
                                work_target.append_work(DistanceData(iterated_round, dest, new_delta));
                            }
                        } else {
                            if (new_delta < atomicMin(node_delta.get_item_ptr(node), new_delta)) {
                                work_target.append_work(DistanceData(iterated_round, dest, new_delta));
                            }
                        }
                    }
                }
            }
        }
    };

    //DWCallbacks instance per GPU
    struct DWCallbacks {
    private:
        groute::graphs::dev::CSRGraphSeg m_graph_seg;
        groute::graphs::dev::GraphDatum<distance_t> m_distances_datum;
        groute::graphs::dev::GraphDatum<distance_t> m_delta_datum;
        groute::graphs::dev::GraphDatum<distance_t> m_last_delta_datum;

    public:
        template<typename...UnusedData>
        DWCallbacks(
                const groute::graphs::dev::CSRGraphSeg &graph_seg,
                const groute::graphs::dev::GraphDatumSeg<distance_t> &weights_datum,
                const groute::graphs::dev::GraphDatum<distance_t> &distances_datum,
                const groute::graphs::dev::GraphDatum<distance_t> &delta_datum,
                const groute::graphs::dev::GraphDatum<distance_t> &last_datum,
                UnusedData &... data)
                : m_graph_seg(graph_seg),
                  m_distances_datum(distances_datum),
                  m_delta_datum(delta_datum),
                  m_last_delta_datum(last_datum) {
        }

        DWCallbacks() {}

        __device__ __forceinline__

        groute::SplitFlags on_receive(const remote_work_t &work) {
            printf("call on_receive %d,%d,%d\n", work.iterated_round, work.node, work.distance);
            if (m_graph_seg.owns(work.node)) {
//                return groute::SF_Take;
                if (work.iterated_round % 2) {
                    return (work.distance < atomicMin(m_delta_datum.get_item_ptr(work.node), work.distance))
                           ? groute::SF_Take
                           : groute::SF_None; // Filter
                } else {
                    return (work.distance < atomicMin(m_last_delta_datum.get_item_ptr(work.node), work.distance))
                           ? groute::SF_Take
                           : groute::SF_None; // Filter
                }
            }

            return groute::SF_Pass;
        }

        __device__ __forceinline__

        bool should_defer(const local_work_t &work, const distance_t &global_threshold) {
            printf("iter:%u delta:%u\n", work.m_iterated_round, work.m_node);
            bool defer;
            if (work.m_iterated_round % 2)
                defer = m_delta_datum[work.m_node] > global_threshold;
            else
                defer = m_last_delta_datum[work.m_node] > global_threshold;
            if (defer)
                printf("defer\n");
            return defer;
        }

        __device__ __forceinline__

        groute::SplitFlags on_send(local_work_t work) {
            printf("call on_send\n");
            return (m_graph_seg.owns(work.m_node))
                   ? groute::SF_Take
                   : groute::SF_Pass;
        }

        __device__ __forceinline__

        remote_work_t pack(local_work_t work) {
            printf("iterated round:%d\n", work.m_iterated_round);
            return DistanceData(work.m_iterated_round, work.m_node, m_distances_datum.get_item(work.m_node));
        }

        __device__ __forceinline__

        local_work_t unpack(const remote_work_t &work) {
            return LocalData(work.iterated_round, work.node);
        }
    };

    struct Algo {
        static const char *NameLower() { return "sssp"; }

        static const char *Name() { return "SSSP"; }

        static void HostInit(
                utils::traversal::Context<sssp::Algo> &context,
                groute::graphs::multi::CSRGraphAllocator &graph_manager,
                groute::IDistributedWorklist<local_work_t, remote_work_t> &distributed_worklist) {
            // Get a valid source_node from flag 
            index_t source_node = min(max((index_t) 0, (index_t) FLAGS_source_node), context.host_graph.nnodes - 1);

            // Map to the (possibly new) partitioned vertex space
            source_node = graph_manager.GetGraphPartitioner()->ReverseLookup(source_node);

            assert(source_node == 0);

            // Host endpoint for sending initial work
            groute::Endpoint host = groute::Endpoint::HostEndpoint(0);

            // Report the initial work
            distributed_worklist.ReportInitialWork(1, host);

            std::vector<remote_work_t> initial_work;
            initial_work.push_back(remote_work_t(1, source_node, 0));
            distributed_worklist
                    .GetLink(host)
                    .Send(groute::Segment<remote_work_t>(&initial_work[0], 1), groute::Event());
            printf("Host Init finished\n");
        }

        template<typename TGraph, typename TWeightDatum, typename TDistanceDatum, typename...UnusedData>
        static void DeviceMemset(groute::Stream &stream, TGraph &graph, TWeightDatum &weights_datum,
                                 TDistanceDatum &distances_datum,
                                 TDistanceDatum &delta_datum,
                                 TDistanceDatum &last_datum, const UnusedData &... data) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, distances_datum.size);

            index_t s_node = 0;

            SSSPMemsetKernel << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                    s_node,
                            distances_datum.data_ptr,
                            delta_datum.data_ptr,
                            last_datum.data_ptr,
                            distances_datum.size);
        }

        template<typename TGraph, typename TWeightDatum, typename TDistanceDatum, typename...UnusedData>
        static void DeviceInit(
                groute::Endpoint endpoint, groute::Stream &stream,
                groute::IDistributedWorklist<local_work_t, remote_work_t> &distributed_worklist,
                groute::IDistributedWorklistPeer<local_work_t, remote_work_t, DWCallbacks> *peer,
                TGraph &graph, TWeightDatum &weights_datum,
                TDistanceDatum &distances_datum,
                TDistanceDatum &delta_datum,
                TDistanceDatum &last_datum,
                const UnusedData &... data) {
        }

        template<
                typename TGraphAllocator,
                typename TWeightDatum,
                typename TDistanceDatum,
                typename...UnusedData>
        static const std::vector<distance_t> &
        Gather(TGraphAllocator &graph_allocator,
               TWeightDatum &weights_datum,
               TDistanceDatum &distances_datum,
               TDistanceDatum &delta_datum,
               TDistanceDatum &last_delta_datum,
               UnusedData &... data) {
            //graph_allocator.GatherDatum(distances_datum);
            //graph_allocator.GatherDatum(delta_datum);


            return distances_datum.GetHostData();
        }

        template<
                typename TWeightDatum, typename TDistanceDatum, typename...UnusedData>
        static std::vector<distance_t>
        Host(groute::graphs::host::CSRGraph &graph,
             TWeightDatum &weights_datum,
             TDistanceDatum &distances_datum,
             TDistanceDatum &delta_datum,
             TDistanceDatum &last_delta_datum,
             UnusedData &... data) {
            return SSSPHostNaive(graph, weights_datum.GetHostDataPtr(),
                                 min(max((index_t) 0, (index_t) FLAGS_source_node), graph.nnodes - 1));
        }

        static int Output(const char *file, const std::vector<distance_t> &distances) {
            return SSSPOutput(file, distances);
        }

        static int CheckErrors(const std::vector<distance_t> &distances, const std::vector<distance_t> &regression) {
            return SSSPCheckErrors(distances, regression);
        }
    };

    using EdgeWeightDatumType = groute::graphs::multi::EdgeInputDatum<distance_t>;
    using NodeDistanceDatumType = groute::graphs::multi::NodeOutputGlobalDatum<distance_t>;

    template<bool IterationFusion = true, bool CTAScheduling = true>
    using FusedWorkerType = groute::FusedWorker<
            IterationFusion, local_work_t, remote_work_t, int, DWCallbacks, SSSPWork<CTAScheduling>,
            groute::graphs::dev::CSRGraphSeg, EdgeWeightDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType>;

    template<bool CTAScheduling = true>
    using WorkerType = groute::Worker<
            local_work_t, remote_work_t, DWCallbacks, SSSPWork<CTAScheduling>,
            groute::graphs::dev::CSRGraphSeg, EdgeWeightDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType, NodeDistanceDatumType::DeviceObjectType>;

    template<typename TWorker>
    using RunnerType = utils::traversal::Runner<
            Algo, TWorker, DWCallbacks, local_work_t, remote_work_t,
            EdgeWeightDatumType, NodeDistanceDatumType, NodeDistanceDatumType, NodeDistanceDatumType>;
}

template<typename TWorker>
bool TestSSSPAsyncMultiTemplate(int ngpus) {
    sssp::RunnerType<TWorker> runner;

    sssp::EdgeWeightDatumType edge_weights;
    sssp::NodeDistanceDatumType node_distances;
    sssp::NodeDistanceDatumType node_delta;
    sssp::NodeDistanceDatumType node_last_delta;

    printf("%u %u %u\n", node_distances, node_delta, node_last_delta);

    bool res = runner(ngpus, FLAGS_prio_delta, edge_weights, node_distances, node_delta, node_last_delta);

    return res;
}

//bool TestSSSPAsyncMultiOptimized(int ngpus) {
//    return FLAGS_cta_np
//           ? FLAGS_iteration_fusion
//             ? TestSSSPAsyncMultiTemplate<sssp::FusedWorkerType<true, true   >>(ngpus)
//             : TestSSSPAsyncMultiTemplate<sssp::FusedWorkerType<false, true  >>(ngpus)
//           : FLAGS_iteration_fusion
//             ? TestSSSPAsyncMultiTemplate<sssp::FusedWorkerType<true, false  >>(ngpus)
//             : TestSSSPAsyncMultiTemplate<sssp::FusedWorkerType<false, false >>(ngpus);
//}
//
//bool TestSSSPAsyncMulti(int ngpus) {
//    return FLAGS_cta_np
//           ? TestSSSPAsyncMultiTemplate<sssp::WorkerType<true  >>(ngpus)
//           : TestSSSPAsyncMultiTemplate<sssp::WorkerType<false >>(ngpus);
//}

bool TestSSSPSingle() {
    return TestSSSPAsyncMultiTemplate<sssp::FusedWorkerType<true, true>>(1);
}
