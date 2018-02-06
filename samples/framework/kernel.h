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

namespace maiter {
    template<V>
    __global__ void GraphInit(groute::graphs::dev::CSRGraph dev_graph, maiter::IterateKernel iterateKernel,
                                   V arr_value, D arr_delta) {
        unsigned TID = TID_1D;
        unsigned NTHREADS = TOTAL_THREADS_1D;

        index_t start_node = dev_graph.owned_start_node();
        index_t end_node = start_node + dev_graph.owned_nnodes();
        for (index_t node = start_node + tid;
             node < end_node; node += NTHREADS) {

            index_t
                    begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

            arr_value[node] = iterateKernel.InitValue(node, out_degree);
            arr_delta[node] = iterateKernel.InitDelta(node,out_degree);
        }
    };



    template<class K, class V, class D>
    class MaiterKernel {
    private:
        groute::Stream m_stream;
        IterateKernel<K, V, D> *iterateKernel;
        groute::graphs::single::NodeOutputDatum<V> m_arr_value;
        groute::graphs::single::NodeOutputDatum<D> m_arr_delta;
        groute::graphs::dev::CSRGraph m_dev_graph;
    public:
        MaiterKernel() {
            utils::traversal::Context<maiter::Algo> context(1);
            groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);


            context.SetDevice(0);
            m_stream = context.CreateStream(0);

            dev_graph_allocator.AllocateDatumObjects(m_arr_value, m_arr_delta);
            m_dev_graph = dev_graph_allocator.DeviceObject();
        }

        void InitValue(groute::Stream &stream) const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_dev_graph.nnodes);

            Stopwatch sw(true);

            GraphInit << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_dev_graph, m_arr_value);
            stream.Sync();
            sw.stop();
            LOG(google::INFO) << "Graph Init " << sw.ms() << " ms";
        }
    };
}
#endif //GROUTE_KERNEL_H
