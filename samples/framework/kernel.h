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

    //TAtomicFunc atomicFunc,
    template<typename WorkSource,
//            typename TAtomicFunc,
            typename TValue, typename TDelta>
    __global__ void GraphKernelTopology(groute::graphs::dev::CSRGraph dev_graph,
                                        WorkSource work_source,
                                        maiter::IterateKernel<TValue, TDelta> **iterateKernel,
//                                        TAtomicFunc atomicFunc,
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

                        //call atomicFunction
//                        atomicFunc(arr_delta.get_item_ptr(dest), new_delta);
                        atomicAdd(arr_delta.get_item_ptr(dest), new_delta);
                    }
                }
            }
        }
    }

    template<typename TValue, typename TDelta>
    class MaiterKernel {
    private:
        groute::Stream m_stream;
        IterateKernel<TValue, TDelta> *iterateKernel;
        groute::graphs::single::NodeOutputDatum<TValue> m_arr_value;
        groute::graphs::single::NodeOutputDatum<TDelta> m_arr_delta;
        groute::graphs::dev::CSRGraph m_dev_graph;
        maiter::IterateKernel<TValue, TDelta> **m_dev_kernel;

        template<typename WorkSource, typename WorkTarget>
        void Relax_Topology(const WorkSource &work_source, WorkTarget &work_target, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

//            GraphKernelTopology << < grid_dims, block_dims >> > (m_dev_graph, work_source, m_dev_kernel,
////                    atomicAdd,
//                    m_arr_value, m_arr_delta);
        }
    public:
        typedef groute::Queue<index_t> Worklist;

        MaiterKernel() {
            utils::traversal::Context<maiter::Algo> context(1);
            groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

            context.SetDevice(0);
            m_stream = context.CreateStream(0);

            dev_graph_allocator.AllocateDatumObjects(m_arr_value, m_arr_delta);
            m_dev_graph = dev_graph_allocator.DeviceObject();

            GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_kernel, sizeof(maiter::IterateKernel<TValue, TDelta> *)));
        }

        maiter::IterateKernel<TValue, TDelta> **DeviceKernelObject() { return m_dev_kernel; };

        void InitValue() const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_dev_graph.nnodes);

            Stopwatch sw(true);

            GraphInit<TValue, TDelta> << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                     (m_dev_graph, m_dev_kernel, m_arr_value.DeviceObject(), m_arr_delta.DeviceObject());
            m_stream.Sync();
            sw.stop();
            LOG(INFO) << "Graph Init " << sw.ms() << " ms";
        }

        void DataDriven() {

            size_t max_work_size = m_dev_graph.nedges * FLAGS_wl_alloc_factor;
            if (FLAGS_wl_alloc_abs > 0)
                max_work_size = FLAGS_wl_alloc_abs;

            Worklist wl1(max_work_size, 0, "input queue"), wl2(max_work_size, 0, "output queue");

            wl1.ResetAsync(m_stream.cuda_stream);
            wl2.ResetAsync(m_stream.cuda_stream);
            m_stream.Sync();

            Stopwatch sw(true);

            Worklist *in_wl = &wl1, *out_wl = &wl2;


            Relax_Topology(
                    groute::dev::WorkSourceRange<index_t>(m_dev_graph.owned_start_node(), m_dev_graph.owned_nnodes()),
                    *in_wl, m_stream);

            groute::Segment<index_t> work_seg = in_wl->GetSeg(m_stream);

            int iteration = 0;
            while (in_wl->GetCount(m_stream) > 0) {
                Relax_Topology(
                        groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(), work_seg.GetSegmentSize()),
                        *out_wl, m_stream);
                in_wl->ResetAsync(m_stream);
                work_seg = out_wl->GetSeg(m_stream);
                std::swap(in_wl, out_wl);

                VLOG(1)
                << "Iteration: " << ++iteration << " In-Worklist: " << in_wl->GetCount(m_stream) << " Out-Worklist: "
                << out_wl->GetCount(m_stream);
            }
        }


    };
}
#endif //GROUTE_KERNEL_H
