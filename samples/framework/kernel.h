//
// Created by liang on 2/5/18.
//

#ifndef GROUTE_KERNEL_H
#define GROUTE_KERNEL_H

#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <gflags/gflags.h>
#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/work_source.cuh>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <glog/logging.h>
#include "kernelbase.h"
#include "myatomics.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);

namespace maiter {
    template<typename TValue, typename TDelta>
    __global__ void GraphInit(groute::graphs::dev::CSRGraph dev_graph,
                              maiter::IterateKernel<TValue, TDelta> **iterateKernel,
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
                                        groute::graphs::dev::GraphDatum<TValue> arr_value,
                                        groute::graphs::dev::GraphDatum<TDelta> arr_delta) {
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
            typename TDelta>
    __global__ void GraphKernelDataDriven(groute::graphs::dev::CSRGraph dev_graph,
                                          WorkSource work_source,
                                          WorkTarget work_target,
                                          TAtomicFunc atomicFunc,
                                          maiter::IterateKernel<TValue, TDelta> **iterateKernel,
                                          groute::graphs::dev::GraphDatum<TValue> arr_value,
                                          groute::graphs::dev::GraphDatum<TDelta> arr_delta) {
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
                        TDelta prev = atomicFunc(arr_delta.get_item_ptr(dest), new_delta);
                        const float EPSLION = 0.01;
                        if (prev < EPSLION && prev + new_delta > EPSLION) {
                            work_target.append_warp(dest);
                        }
                    }
                }
            }
        }
    }

    template<typename TGraph>
    __global__ void traverseTest(TGraph graph) {
        for (int node = 0; node < 100; node++) {
            printf("%d %d\n", graph.begin_edge(node), graph.end_edge(node));
        }

    }

    template<typename TValue, typename TDelta>
    class MaiterKernel {

    private:
        groute::Stream m_stream;
        IterateKernel<TValue, TDelta> *iterateKernel;
        groute::graphs::single::NodeOutputDatum<TValue> m_arr_value;
        groute::graphs::single::NodeOutputDatum<TDelta> m_arr_delta;
        groute::graphs::single::CSRGraphAllocator *m_dev_graph_allocator;
        maiter::IterateKernel<TValue, TDelta> **m_dev_kernel;
        typedef groute::Queue<index_t> Worklist;

        template<typename WorkSource>
        void RelaxTopology(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            GraphKernelTopology << < grid_dims, block_dims >> >
                                                (m_dev_graph_allocator->DeviceObject(), work_source, MyAtomicAdd<TDelta>(), m_dev_kernel, m_arr_value, m_arr_delta);
        }

        template<typename WorkSource, typename WorkTarget>
        void RelaxDataDriven(const WorkSource &work_source, WorkTarget &work_target, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            GraphKernelDataDriven << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                (m_dev_graph_allocator->DeviceObject(),
                                                        work_source,
                                                        work_target.DeviceObject(),
                                                        MyAtomicAdd<TDelta>(), m_dev_kernel, m_arr_value.DeviceObject(), m_arr_delta.DeviceObject());
        }

    public:


        MaiterKernel() {
            utils::traversal::Context<maiter::Algo> context(1);

            m_dev_graph_allocator = new groute::graphs::single::CSRGraphAllocator(context.host_graph);

            context.SetDevice(0);

            m_dev_graph_allocator->AllocateDatumObjects(m_arr_value, m_arr_delta);

            m_stream = context.CreateStream(0);

            GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_kernel, sizeof(maiter::IterateKernel<TValue, TDelta> *)));
        }

        ~MaiterKernel() {
            delete m_dev_graph_allocator;
        }

        groute::Stream &getStream() {
            return m_stream;
        }

        maiter::IterateKernel<TValue, TDelta> **DeviceKernelObject() { return m_dev_kernel; };

        void InitValue() const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_dev_graph_allocator->DeviceObject().nnodes);

            Stopwatch sw(true);

            GraphInit<TValue, TDelta> << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                     (m_dev_graph_allocator->DeviceObject(), m_dev_kernel, m_arr_value.DeviceObject(), m_arr_delta.DeviceObject());
            m_stream.Sync();
            sw.stop();
            LOG(INFO) << "Graph Init " << sw.ms() << " ms";
        }

        void DataDriven() {

            const groute::graphs::dev::CSRGraph &dev_graph = m_dev_graph_allocator->DeviceObject();

            size_t max_work_size = dev_graph.nedges * FLAGS_wl_alloc_factor;

            if (FLAGS_wl_alloc_abs > 0)
                max_work_size = FLAGS_wl_alloc_abs;

            Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

            wl1.ResetAsync(m_stream.cuda_stream);
            wl2.ResetAsync(m_stream.cuda_stream);
            m_stream.Sync();

            Stopwatch sw(true);

            Worklist *in_wl = &wl1, *out_wl = &wl2;

            RelaxDataDriven(
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(), dev_graph.owned_nnodes()),
                    *in_wl, m_stream);

            groute::Segment<index_t> work_seg = in_wl->GetSeg(m_stream);

            int iteration = 0;
            while (work_seg.GetSegmentSize() > 0) {
                RelaxDataDriven(
                        groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(), work_seg.GetSegmentSize()),
                        *out_wl, m_stream);

                VLOG(1)
                << "Iteration: " << ++iteration << " In-Worklist: " << work_seg.GetSegmentSize() << " Out-Worklist: "
                << out_wl->GetCount(m_stream);

                work_seg = out_wl->GetSeg(m_stream);

                in_wl->ResetAsync(m_stream);

                std::swap(in_wl, out_wl);

            }
        }


    };
}
#endif //GROUTE_KERNEL_H
