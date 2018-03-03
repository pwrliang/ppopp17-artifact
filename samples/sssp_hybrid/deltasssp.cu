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
#include <device_launch_parameters.h>
#include <gflags/gflags.h>

#include <groute/device/cta_scheduler.cuh>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/distributed_worklist.cuh>
#include <groute/dwl/workers.cuh>
#include <utils/cuda_utils.h>
#include <utils/graphs/traversal.h>
#include <utils/balancer.h>
#include "sssp_common.h"

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)
#define FILTER_THRESHOLD 0.0000000001
DECLARE_double(wl_alloc_factor_local);
DECLARE_int32(source_node);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);
DECLARE_int32(mode);
DECLARE_int32(source_node);
const distance_t INF = UINT_MAX;
namespace sssp_expr {
    const distance_t IDENTITY_ELEMENT = 9999999;

    struct Algo {
        static const char *Name() { return "SSSP"; }
    };

    template<
            template<typename> class WorkList,
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __device__ void SSSPAsync(
            const WorkList<index_t> &work_source,
            WorkList<index_t> &work_immediate_target,
            WorkList<index_t> &work_later_target,
            const distance_t priority_threshold,
            const TGraph &graph,
            const TWeightDatum &edge_weights,
            TDistanceDatum<distance_t> &node_distances,
            TDistanceDeltaDatum<distance_t> &node_distances_delta) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;


        uint32_t work_size = work_source.count();
        uint32_t work_size_rup =
                round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

        if (tid == 0)
            printf("work size:%d\n", work_size);

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<distance_t> np_local = {0, 0, 0};

            if (i < work_size) {
                index_t node = work_source.read(i);
                distance_t old_value = node_distances[node];
                distance_t old_delta = atomicExch(node_distances_delta.get_item_ptr(node), IDENTITY_ELEMENT);
                distance_t new_value = min(old_value, old_delta);

                if (new_value != old_value) {
                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;
                    np_local.meta_data = old_delta;
                }
            }

            groute::dev::CTAWorkScheduler<distance_t>::template schedule(
                    np_local,
                    [&work_immediate_target, &work_later_target, &priority_threshold, &graph, &edge_weights, &node_distances_delta](
                            index_t edge,
                            index_t size,
                            distance_t old_delta) {
                        index_t dest = graph.edge_dest(edge);
                        distance_t weight = edge_weights.get_item(edge);
                        distance_t new_delta = old_delta + weight;
                        distance_t before_update = atomicMin(node_distances_delta.get_item_ptr(dest), new_delta);

                        if (new_delta < before_update) {
                            if (new_delta < priority_threshold)
                                work_immediate_target.append_warp(dest);
                            else
                                work_later_target.append_warp(dest);
                        }
                    });
        }
    }

    template<
            typename WorkSource, template<typename> class WorkTarget,
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __global__ void SSSPSync(
            const WorkSource work_source,
            WorkTarget<index_t> work_immediate_target,
            WorkTarget<index_t> work_later_target,
            distance_t priority_threshold,
            index_t iteration,
            const TGraph graph,
            TWeightDatum edge_weights,
            TDistanceDatum<distance_t> node_distances,
            TDistanceDeltaDatum<distance_t> node_distances_delta,
            TDistanceDeltaDatum<distance_t> node_distances_last_delta) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            distance_t delta;

            if (iteration % 2 == 0)
                delta = atomicExch(node_distances_delta.get_item_ptr(node), IDENTITY_ELEMENT);
            else
                delta = atomicExch(node_distances_last_delta.get_item_ptr(node), IDENTITY_ELEMENT);


            if (delta < node_distances[node]) {
                node_distances[node] = delta;

                for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);
                    distance_t weight = edge_weights.get_item(edge);

                    distance_t new_delta = delta + weight;

                    distance_t before_update;

                    if (iteration % 2 == 0)
                        before_update = atomicMin(node_distances_last_delta.get_item_ptr(dest), new_delta);
                    else
                        before_update = atomicMin(node_distances_delta.get_item_ptr(dest), new_delta);

                    //If update the dest success, expand the worklist
                    if (new_delta < before_update) {
                        if (new_delta < priority_threshold)
                            work_immediate_target.append_warp(dest);
                        else {
                            work_later_target.append_warp(dest);
                            //printf("later\n");
                        }
                    }
                }
            }
        }
    }

    template<template<typename> class WorkList,
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __global__ void SSSPControl__Single__(cub::GridBarrier grid_barrier,
                                          WorkList<index_t> work_source,
                                          WorkList<index_t> work_immediate_target,
                                          WorkList<index_t> work_later_target,
                                          const distance_t priority_threshold,
                                          const TGraph graph,
                                          const TWeightDatum edge_weights,
                                          TDistanceDatum<distance_t> node_distances,
                                          TDistanceDeltaDatum<distance_t> node_distances_delta,
                                          TDistanceDeltaDatum<distance_t> node_distances_last_delta) {

        uint32_t tid = TID_1D;
        WorkList<index_t> *in_wl = &work_source;
        WorkList<index_t> *out_immediate_wl = &work_immediate_target;
        WorkList<index_t> *out_later_wl = &work_later_target;
        distance_t curr_threshold = priority_threshold;

        if (tid == 0) {
            printf("CALL SSSPControl__Single__\n");
        }
//
        int iteration = 0;
        while (in_wl->count() > 0) {
            SSSPAsync(*in_wl,
                      *out_immediate_wl,
                      *out_later_wl,
                      curr_threshold,
                      graph,
                      edge_weights,
                      node_distances,
                      node_distances_delta);

           // grid_barrier.Sync();

            if (tid == 0) {
                printf("INPUT %d IMMEDIATE %d LATER %d\n", in_wl->count(), out_immediate_wl->count(),
                       out_later_wl->count());
                in_wl->reset();
                iteration++;
            }

            if (out_immediate_wl->count() > 0) {
                WorkList<index_t> *tmp_wl = in_wl;
                in_wl = out_immediate_wl;
                out_immediate_wl = tmp_wl;
            } else {
                WorkList<index_t> *tmp_wl = in_wl;
                in_wl = out_later_wl;
                out_later_wl = tmp_wl;
            }

        }

        if (tid == 0) {
            printf("Total iterations: %d\n", iteration);
        }
    }


    template<template<typename> class DistanceDatum,
            template<typename> class DistanceDeltaDatum>
    __global__ void
    SSSPInit(index_t source, DistanceDatum<distance_t> distances, DistanceDeltaDatum<distance_t> delta_distances,
             DistanceDeltaDatum<distance_t> last_delta_distances,
             int nnodes) {
        int tid = GTID;
        if (tid < nnodes) {
            distances[tid] = IDENTITY_ELEMENT;
            last_delta_distances[tid] = IDENTITY_ELEMENT;
            delta_distances[tid] = (tid == source ? 0 : IDENTITY_ELEMENT);
        }
    }


    template<
            typename TGraph,
            template<typename> class TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    struct Problem {
        TGraph m_graph;
        TWeightDatum<distance_t> m_weights_datum;
        TDistanceDatum<distance_t> m_distances_datum;
        TDistanceDeltaDatum<distance_t> m_distances_delta_datum;
        TDistanceDeltaDatum<distance_t> m_distances_last_delta_datum;
        distance_t m_priority_threshold;
        int m_curr_threshold = 0;
    public:
        Problem(const TGraph &graph, const TWeightDatum<distance_t> &weights_datum,
                const TDistanceDatum<distance_t> &distances_datum,
                const TDistanceDeltaDatum<distance_t> &distances_delta_datum,
                const TDistanceDeltaDatum<distance_t> &distances_last_delta_datum,
                const distance_t priority_threshold) :
                m_graph(graph), m_weights_datum(weights_datum), m_distances_datum(distances_datum),
                m_distances_delta_datum(distances_delta_datum),
                m_distances_last_delta_datum(distances_last_delta_datum),
                m_priority_threshold(priority_threshold) {
        }

        void Init(groute::Queue<index_t> &in_wl, groute::Stream &stream) const {
            index_t source_node = min(max(0, FLAGS_source_node), m_graph.nnodes - 1);

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_distances_datum.size);

            Marker::MarkWorkitems(m_distances_datum.size, "SSSPInit");

            SSSPInit << < grid_dims, block_dims, 0, stream.cuda_stream >> > (source_node,
                    m_distances_datum, m_distances_delta_datum, m_distances_last_delta_datum, m_distances_datum.size);

            in_wl.AppendItemAsync(stream.cuda_stream, source_node); // add the first item to the worklist
        }
//
//        template<typename TWorklist>
//        void RelaxAsync(const groute::Segment<index_t> &work,
//                        TWorklist &output_immediate_worklist,
//                        TWorklist &output_later_worklist, groute::Stream &stream) {
//            if (work.Empty())return;
//            m_curr_threshold -= m_priority_threshold * 0.1;
//            dim3 grid_dims, block_dims;
//            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());
//            SSSPAsync << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
//                            output_immediate_worklist.DeviceObject(),
//                            output_later_worklist.DeviceObject(),
//                            m_curr_threshold,
//                            m_graph, m_weights_datum, m_distances_datum, m_distances_delta_datum);
//        }
//
//
//        template<typename TWorklist>
//        void RelaxSync(const groute::Segment<index_t> &work,
//                       TWorklist &output_immediate_worklist,
//                       TWorklist &output_later_worklist,
//                       groute::Stream &stream, int iteration) {
//            if (work.Empty())return;
//            m_curr_threshold += m_priority_threshold * 0.5;
//            dim3 grid_dims, block_dims;
//            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());
//            SSSPSync << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
//                            output_immediate_worklist.DeviceObject(),
//                            output_later_worklist.DeviceObject(),
//                            m_curr_threshold,
//                            iteration, m_graph, m_weights_datum, m_distances_datum, m_distances_delta_datum, m_distances_last_delta_datum);
//        }

    };
}

bool SSSPExpr() {
    typedef sssp_expr::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum> Problem;

    utils::traversal::Context<sssp_expr::Algo> context(1);
    context.configuration.verbose = FLAGS_verbose;
    context.configuration.trace = FLAGS_trace;
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    groute::graphs::single::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::single::NodeOutputDatum<distance_t> node_distances;
    groute::graphs::single::NodeOutputDatum<distance_t> node_delta_distances;
    groute::graphs::single::NodeOutputDatum<distance_t> node_last_delta_distances;

    dev_graph_allocator.AllocateDatumObjects(edge_weights, node_distances, node_delta_distances,
                                             node_last_delta_distances);

    context.SyncDevice(0);

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor_local;

    groute::Stream stream = context.CreateStream(0);

    groute::Queue<index_t> wl1(max_work_size, 0, "input queue");
    groute::Queue<index_t> wl2(max_work_size, 0, "output queue1");
    groute::Queue<index_t> wl3(max_work_size, 0, "output queue2");


    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    wl3.ResetAsync(stream.cuda_stream);
    stream.Sync();


    Problem problem(dev_graph_allocator.DeviceObject(), edge_weights.DeviceObject(), node_distances.DeviceObject(),
                    node_delta_distances.DeviceObject(), node_last_delta_distances.DeviceObject(), FLAGS_prio_delta);

    problem.Init(wl1, stream);
    stream.Sync();


    int occupancy_per_MP = FLAGS_grid_size;
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
//                                                  sssp_expr::SSSPControl__Single__<groute::dev::Queue<index_t>,
//                                                          groute::graphs::dev::CSRGraph,
//                                                          groute::graphs::dev::GraphDatum<distance_t>,
//                                                          groute::graphs::dev::GraphDatum<distance_t>,
//                                                          groute::graphs::dev::GraphDatum<distance_t>>,
//                                                  FLAGS_block_size, 0);

    cub::GridBarrierLifetime grid_barrier;

    grid_barrier.Setup(occupancy_per_MP);

    printf("grid size %d block size %d\n", FLAGS_grid_size, FLAGS_block_size);

    Stopwatch sw(true);

    sssp_expr::SSSPControl__Single__
            << < occupancy_per_MP, FLAGS_block_size, 0, stream.cuda_stream >> >
                                                        (grid_barrier,
                                                                wl1.DeviceObject(),
                                                                wl2.DeviceObject(),
                                                                wl3.DeviceObject(),
                                                                FLAGS_prio_delta,
                                                                dev_graph_allocator.DeviceObject(),
                                                                edge_weights.DeviceObject(),
                                                                node_distances.DeviceObject(),
                                                                node_delta_distances.DeviceObject(),
                                                                node_last_delta_distances.DeviceObject());
    stream.Sync();

//    while (!work_seg.Empty()) {
//        if (mode == 0) {
//            problem.RelaxSync(work_seg, *out_immediate_wl, *out_later_wl, stream, syncIteration++);
//        } else {
//            problem.RelaxAsync(work_seg, *out_immediate_wl, *out_later_wl, stream);
//        }
//
//        printf("After iteration: %d input: %lu immediate output: %u later output: %u\n", iteration++,
//               work_seg.GetSegmentSize(), out_immediate_wl->GetCount(stream),
//               out_later_wl->GetCount(stream));
//
//        in_wl->ResetAsync(stream);
//        stream.Sync();
//
//        if (out_immediate_wl->GetCount(stream) > 0)
//            std::swap(out_immediate_wl, in_wl);
//        else
//            std::swap(out_later_wl, in_wl);
//        work_seg = in_wl->GetSeg(stream);
//    }

    sw.stop();

    printf("sssp done:%f\n", sw.ms());
    if (FLAGS_output.size() > 0) {
        dev_graph_allocator.GatherDatum(node_distances);
        SSSPOutput(FLAGS_output.data(), node_distances.GetHostData());
    }
    return true;
}