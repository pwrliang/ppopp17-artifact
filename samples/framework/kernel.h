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
#include "myatomics.h"
#include "graph_api.h"
#include "work_kernel.h"
#include "resultsaver.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);

namespace maiter {

    template<typename TAPIImple, typename TValue, typename TDelta>
    __global__ void CreateAPIInstance(maiter::IterateKernel<TValue, TDelta> **baseFunc) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //*baseFunc = new MyIterateKernel();
            *baseFunc = new TAPIImple();
        }
    }

    template<typename TAPIImple,
            typename TValue,
            typename TDelta,
            typename TWeight=unsigned>
    class MaiterKernel {

    private:
        utils::traversal::Context<maiter::Algo> *m_context;
        groute::graphs::single::CSRGraphAllocator *m_graph_allocator;
        groute::graphs::single::NodeOutputDatum<TValue> m_arr_value;
        groute::graphs::single::NodeOutputDatum<TDelta> m_arr_delta;
        groute::graphs::single::EdgeInputDatum<TWeight> m_arr_weights;
        groute::Stream m_stream;

        IterateKernel<TValue, TDelta> *iterateKernel;
        maiter::IterateKernel<TValue, TDelta> **m_dev_kernel;
        bool m_is_weighted;

        typedef groute::Queue<index_t> Worklist;

        template<typename WorkSource>
        void RelaxTopology(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            maiter::kernel::GraphKernelTopology << < grid_dims, block_dims >> >
                                                                (m_graph_allocator->DeviceObject(), work_source, MyAtomicAdd<TDelta>(), m_dev_kernel, m_arr_value, m_arr_delta);
        }

        template<typename WorkSource, typename WorkTarget, typename TAtomicFunc>
        void RelaxDataDriven(const WorkSource &work_source, WorkTarget &work_target, TAtomicFunc atomic_func) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            maiter::kernel::GraphKernelDataDriven << < grid_dims, block_dims, 0, m_stream.cuda_stream >> > (work_source,
                    work_target.DeviceObject(),
                    atomic_func,
                    m_dev_kernel,
                    m_graph_allocator->DeviceObject(),
                    m_arr_value.DeviceObject(),
                    m_arr_delta.DeviceObject(),
                    m_arr_weights.DeviceObject(), m_is_weighted);
        }

    public:
        MaiterKernel(bool weighted = false) : m_is_weighted(weighted) {
            m_context = new utils::traversal::Context<maiter::Algo>(1);

            m_graph_allocator = new groute::graphs::single::CSRGraphAllocator(m_context->host_graph);

            m_context->SetDevice(0);

            m_graph_allocator->AllocateDatumObjects(m_arr_value, m_arr_delta);
            if (weighted)
                m_graph_allocator->AllocateDatum(m_arr_weights);

            m_stream = m_context->CreateStream(0);

            //create a function pointer which point to a APIs collection.
            GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_kernel, sizeof(maiter::IterateKernel<TValue, TDelta> *)));

            //Instance the Functions
            CreateAPIInstance<TAPIImple, TValue, TDelta> << < 1, 1, 0, m_stream.cuda_stream >> > (m_dev_kernel);

            m_stream.Sync();
        }

        ~MaiterKernel() {
            delete m_graph_allocator;
            delete m_context;
        }

        groute::Stream &getStream() {
            return m_stream;
        }

        maiter::IterateKernel<TValue, TDelta> **DeviceKernelObject() { return m_dev_kernel; };

        void InitValue() const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph_allocator->DeviceObject().nnodes);

            Stopwatch sw(true);

            maiter::kernel::GraphInit<TValue, TDelta> << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                                     (m_graph_allocator->DeviceObject(), m_dev_kernel, m_arr_value.DeviceObject(), m_arr_delta.DeviceObject());
            m_stream.Sync();
            sw.stop();
            LOG(INFO) << "Graph Init " << sw.ms() << " ms";
        }

        template<typename TAtomicFunc>
        void DataDriven(TAtomicFunc atomic_func) {

            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();

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
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                          dev_graph.owned_nnodes()),
                    *in_wl,
                    atomic_func);

            groute::Segment<index_t> work_seg = in_wl->GetSeg(m_stream);

            int iteration = 0;
            while (work_seg.GetSegmentSize() > 0) {
                RelaxDataDriven(
                        groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(),
                                                              work_seg.GetSegmentSize()),
                        *out_wl,
                        atomic_func);

                VLOG(1)
                << "Iteration: " << ++iteration << " In-Worklist: " << work_seg.GetSegmentSize() << " Out-Worklist: "
                << out_wl->GetCount(m_stream);

                work_seg = out_wl->GetSeg(m_stream);

                in_wl->ResetAsync(m_stream);

                std::swap(in_wl, out_wl);

            }

            sw.stop();

            VLOG(0) << "DataDriven Time: " << sw.ms() << " ms.";
        }


        bool SaveResult(const char *file, bool sort = false) {
            m_graph_allocator->GatherDatum(m_arr_value);
            return ResultOutput<float>(file, m_arr_value.GetHostData(), sort);
        }
    };
}
#endif //GROUTE_KERNEL_H
