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
DECLARE_bool(persist);
DECLARE_bool(cta_np);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);

namespace deltasssp {
    const distance_t IDENTITY_ELEMENT = 9999999;

    struct Algo {
        static const char *Name() { return "SSSP"; }
    };

    typedef index_t local_work_t;


    __inline__ __device__ uint32_t warpReduce(uint32_t localSum) {
        localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

        return localSum;
    }

    /// SSSP work without CTA support
    template<
            typename WorkSource, template<typename> class WorkTarget,
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __global__ void SSSP(
            const WorkSource work_source, WorkTarget<index_t> work_target,
            const TGraph graph, TWeightDatum edge_weights, TDistanceDatum<distance_t> node_distances,
            TDistanceDeltaDatum<distance_t> node_distances_delta) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
            index_t node = work_source.get_work(i);
            distance_t delta = atomicExch(node_distances_delta.get_item_ptr(node), IDENTITY_ELEMENT);


            if (delta < node_distances[node]) {
                node_distances[node] = delta;

                for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge) {
                    index_t dest = graph.edge_dest(edge);
                    distance_t weight = edge_weights.get_item(edge);

                    distance_t new_delta = delta + weight;

                    distance_t before_update = atomicMin(node_distances_delta.get_item_ptr(dest), new_delta);
                    distance_t after_update = node_distances_delta[dest];

                    //If update the dest success, expand the worklist
                    if (after_update <= new_delta) {
                        work_target.append_warp(dest);
                    }
                }
            }
        }
    }

    template<
            typename WorkSource, template<typename> class WorkTarget,
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __global__ void SSSPCTA(
            const WorkSource work_source, WorkTarget<index_t> work_target,
            const TGraph graph, TWeightDatum edge_weights, TDistanceDatum<distance_t> node_distances,
            TDistanceDeltaDatum<distance_t> node_distances_delta) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup =
                round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<distance_t> np_local = {0, 0, 0};

            if (i < work_size) {
                index_t node = work_source.get_work(i);
                distance_t delta = atomicExch(node_distances_delta.get_item_ptr(node), IDENTITY_ELEMENT);

                if (delta < node_distances[node]) {
                    node_distances[node] = delta;

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;
                    np_local.meta_data = delta;
                }
            }

            groute::dev::CTAWorkScheduler<distance_t>::template schedule(
                    np_local,
                    [&work_target, &graph, &edge_weights, &node_distances_delta](index_t edge, index_t size, distance_t delta) {
                        index_t dest = graph.edge_dest(edge);
                        distance_t weight = edge_weights.get_item(edge);

                        distance_t new_delta = delta + weight;

                        distance_t before_update = atomicMin(node_distances_delta.get_item_ptr(dest), new_delta);
                        distance_t after_update = node_distances_delta[dest];

                        //If update the dest success, expand the worklist
                        if (after_update <= new_delta) {
                            work_target.append_warp(dest);
                        }
                    }
            );
        }
    }

    template<
            typename TGraph, typename TWeightDatum,
            template<typename> class TDistanceDatum,
            template<typename> class TDistanceDeltaDatum>
    __global__ void SSSPToplog(
            index_t *lbounds, index_t *ubounds, int *running, distance_t *block_sum,
            const TGraph graph, TWeightDatum edge_weights, TDistanceDatum<distance_t> node_distances,
            TDistanceDeltaDatum<distance_t> node_distances_delta) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        index_t node_start = lbounds[blockIdx.x];
        index_t node_end = ubounds[blockIdx.x];
        uint32_t work_size = node_end - node_start;
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        extern __shared__ distance_t smem[];
        const int SMEMDIM = blockDim.x / warpSize;

        int current_round = 0;

        while (*running) {
            uint64_t local_sum = 0;

            for (uint32_t i = 0 + threadIdx.x; i < work_size_rup; i += blockDim.x) {
                groute::dev::np_local<distance_t> np_local = {0, 0, 0};

                if (i < work_size) {
                    index_t node = i + node_start;

                    distance_t delta = atomicExch(node_distances_delta.get_item_ptr(node), IDENTITY_ELEMENT);

                    if (delta != IDENTITY_ELEMENT)
                        local_sum += delta;

                    if (delta < node_distances[node]) {
                        node_distances[node] = delta;
                        np_local.start = graph.begin_edge(node);
                        np_local.size = graph.end_edge(node) - np_local.start;
                        np_local.meta_data = delta;
                    }
                }

                groute::dev::CTAWorkScheduler<distance_t>::template schedule(
                        np_local,
                        [&graph, &edge_weights, &node_distances_delta](index_t edge, index_t size, distance_t delta) {
                            index_t dest = graph.edge_dest(edge);
                            distance_t weight = edge_weights.get_item(edge);

                            distance_t new_delta = delta + weight;
                            distance_t before_update = atomicMin(node_distances_delta.get_item_ptr(dest), new_delta);
                        }
                );
            }

            if (current_round++ % 100000 == 0) {
//                if (tid == 0) {
//                    uint64_t dist_sum = 0;
//                    for (int node = 0; node < graph.nnodes; node++) {
//                        distance_t dist = node_distances_delta[node];
//                        if (dist != IDENTITY_ELEMENT) {
//                            dist_sum += dist;
//                        }
//                    }
//                }

                local_sum = warpReduce(local_sum);

                if (laneIdx == 0)
                    smem[warpIdx] = local_sum;
                __syncthreads();

                if (threadIdx.x < warpSize)
                    local_sum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;

                if (warpIdx == 0)
                    local_sum = warpReduce(local_sum);

                if (threadIdx.x == 0)
                    block_sum[blockIdx.x] = local_sum;

                if (tid == 0) {
                    uint64_t sum = 0;
                    for (int bid = 0; bid < gridDim.x; bid++) {
                        sum += block_sum[bid];
                    }
                    *running = (sum != 0);
                }
                current_round = 0;
            }
        }

    }

    template<template<typename> class DistanceDatum,
            template<typename> class DistanceDeltaDatum>
    __global__ void
    SSSPInit(index_t source, DistanceDatum<distance_t> distances, DistanceDeltaDatum<distance_t> delta_distances,
             int nnodes) {
        int tid = GTID;
        if (tid < nnodes) {
            distances[tid] = IDENTITY_ELEMENT;
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

    public:
        Problem(const TGraph &graph, const TWeightDatum<distance_t> &weights_datum,
                const TDistanceDatum<distance_t> &distances_datum,
                const TDistanceDeltaDatum<distance_t> &distances_delta_datum) :
                m_graph(graph), m_weights_datum(weights_datum), m_distances_datum(distances_datum),
                m_distances_delta_datum(distances_delta_datum) {
        }

        void Init(groute::Queue<index_t> &in_wl, groute::Stream &stream) const {
            index_t source_node = min(max(0, FLAGS_source_node), m_graph.nnodes - 1);

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_distances_datum.size);

            Marker::MarkWorkitems(m_distances_datum.size, "SSSPInit");

            SSSPInit << < grid_dims, block_dims, 0, stream.cuda_stream >> > (source_node,
                    m_distances_datum, m_distances_delta_datum, m_distances_datum.size);

            in_wl.AppendItemAsync(stream.cuda_stream, source_node); // add the first item to the worklist
        }

        template<typename TWorklist>
        void Relax(const groute::Segment<index_t> &work, TWorklist &output_worklist, groute::Stream &stream) {
            if (work.Empty())return;

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());
            if (FLAGS_cta_np) {
                SSSPCTA << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                        groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                                output_worklist.DeviceObject(), m_graph, m_weights_datum, m_distances_datum, m_distances_delta_datum);
            } else {
                SSSP << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                        groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                                output_worklist.DeviceObject(), m_graph, m_weights_datum, m_distances_datum, m_distances_delta_datum);
            }
        }

        void PersistSSSP(index_t blocksPerGrid, index_t *node_outdegrees, groute::Stream &stream) {
            utils::SharedValue<int> running;
            utils::SharedArray<distance_t> sum_buffer(blocksPerGrid);
            const int SMEMDIM = FLAGS_block_size / 32;

            running.set_val_H2D(1);
            GROUTE_CUDA_CHECK(cudaMemset(sum_buffer.dev_ptr, 0, sum_buffer.buffer_size * sizeof(distance_t)));


            utils::SharedArray<index_t> node_lbounds(FLAGS_grid_size);
            utils::SharedArray<index_t> node_ubounds(FLAGS_grid_size);

            groute::balanced_alloctor(m_graph.nnodes, node_outdegrees, blocksPerGrid, node_lbounds.host_ptr(),
                                      node_ubounds.host_ptr());
            node_lbounds.H2D();
            node_ubounds.H2D();
            SSSPToplog << < blocksPerGrid, FLAGS_block_size, SMEMDIM * sizeof(distance_t), stream.cuda_stream >> >
                                                                                           (node_lbounds.dev_ptr, node_ubounds.dev_ptr, running.dev_ptr, sum_buffer.dev_ptr,
                                                                                                   m_graph, m_weights_datum, m_distances_datum, m_distances_delta_datum);
        }
    };
}

bool MyTestSSSPSingle() {
    typedef deltasssp::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum> Problem;

    utils::traversal::Context<deltasssp::Algo> context(1);
    context.configuration.verbose = FLAGS_verbose;
    context.configuration.trace = FLAGS_trace;
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    groute::graphs::single::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::single::NodeOutputDatum<distance_t> node_distances;
    groute::graphs::single::NodeOutputDatum<distance_t> node_delta_distances;

    dev_graph_allocator.AllocateDatumObjects(edge_weights, node_distances, node_delta_distances);

    context.SyncDevice(0);

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor_local;

    groute::Stream stream = context.CreateStream(0);

    groute::Queue<index_t> wl1(max_work_size, 0, "input queue");
    groute::Queue<index_t> wl2(max_work_size, 0, "output queue");

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

//    IntervalRangeMarker algo_rng(context.host_graph.nedges, "begin");

    groute::Queue<index_t> *in_wl = &wl1, *out_wl = &wl2;


    Problem problem(dev_graph_allocator.DeviceObject(), edge_weights.DeviceObject(), node_distances.DeviceObject(),
                    node_delta_distances.DeviceObject());

    problem.Init(*in_wl, stream);
    stream.Sync();
    Stopwatch sw(true);


    if (FLAGS_persist) {
        index_t *node_outdegrees = new index_t[context.host_graph.nnodes];
        for (index_t node = 0; node < context.host_graph.nnodes; node++) {
            index_t out_degree = context.host_graph.end_edge(node) - context.host_graph.begin_edge(node);
            node_outdegrees[node] = out_degree;
        }
        problem.PersistSSSP(FLAGS_grid_size, node_outdegrees, stream);
        stream.Sync();
        delete[] node_outdegrees;
    } else {

        groute::Segment<index_t> work_seg;
        work_seg = in_wl->GetSeg(stream);


        int iteration = 0;

        while (!work_seg.Empty()) {
            problem.Relax(work_seg, *out_wl, stream);

            in_wl->ResetAsync(stream);
            stream.Sync();
            std::swap(out_wl, in_wl);
            work_seg = in_wl->GetSeg(stream);

            printf("iteration:%d worlist:%d\n", iteration++, work_seg.GetSegmentSize());
        }
    }
    sw.stop();

    printf("sssp done:%f\n", sw.ms());

    return true;
}