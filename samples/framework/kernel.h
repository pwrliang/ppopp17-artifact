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

namespace gframe {

    template<typename TAPIImple, typename TValue, typename TDelta>
    __global__ void CreateAPIInstance(gframe::GraphAPIBase<TValue, TDelta> **graph_api_base,
                                      index_t nnodes, index_t nedges) {
        uint32_t tid = TID_1D;

        if (tid == 0) {
            *graph_api_base = new TAPIImple();
            (*graph_api_base)->GraphInfo.nnodes = nnodes;
            (*graph_api_base)->GraphInfo.nedges = nedges;
        }
    }


    template<typename TValue, typename TDelta>
    __global__ void DestroyAPIInstance(gframe::GraphAPIBase<TValue, TDelta> **graph_api_base) {
        uint32_t tid = TID_1D;

        if (tid == 0) {
            delete *graph_api_base;
        }
    };

    template<typename TAPIImple,
            typename TValue,
            typename TDelta,
            typename TWeight=unsigned>
    class GFrameKernel {

    private:
        utils::traversal::Context<gframe::Algo> *m_context;
        groute::graphs::single::CSRGraphAllocator *m_graph_allocator;
        groute::graphs::single::NodeOutputDatum<TValue> m_arr_value;
        groute::graphs::single::NodeOutputDatum<TDelta> m_arr_delta;
        groute::graphs::single::EdgeInputDatum<TWeight> m_arr_weights;
        groute::Stream m_stream;

        gframe::GraphAPIBase<TValue, TDelta> **m_graph_api_base;
        bool m_is_weighted;
        bool m_cta;

        typedef groute::Queue<index_t> Worklist;

        template<typename WorkSource>
        void RelaxTopology(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            gframe::kernel::GraphKernelTopology << < grid_dims, block_dims >> >
                                                                (m_graph_allocator->DeviceObject(), work_source, MyAtomicAdd<TDelta>(), m_graph_api_base, m_arr_value, m_arr_delta);
        }

        template<typename WorkSource, typename WorkTarget, typename TAtomicFunc>
        void RelaxDataDriven(const WorkSource &work_source, WorkTarget &work_target, TAtomicFunc atomic_func) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            if (m_cta) {
                gframe::kernel::GraphKernelDataDrivenCTA
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (work_source,
                                                               work_target.DeviceObject(),
                                                               atomic_func,
                                                               m_graph_api_base,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_arr_value.DeviceObject(),
                                                               m_arr_delta.DeviceObject(),
                                                               m_arr_weights.DeviceObject(),
                                                               m_is_weighted);
            } else {
                gframe::kernel::GraphKernelDataDriven
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (work_source,
                                                               work_target.DeviceObject(),
                                                               atomic_func,
                                                               m_graph_api_base,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_arr_value.DeviceObject(),
                                                               m_arr_delta.DeviceObject(),
                                                               m_arr_weights.DeviceObject(), m_is_weighted);
            }

        }

    public:
        GFrameKernel(bool weighted = false, bool cta = true) : m_is_weighted(weighted), m_cta(cta) {
            m_context = new utils::traversal::Context<gframe::Algo>(1);

            m_graph_allocator = new groute::graphs::single::CSRGraphAllocator(m_context->host_graph);

            m_context->SetDevice(0);

            m_graph_allocator->AllocateDatumObjects(m_arr_value, m_arr_delta);
            if (weighted)
                m_graph_allocator->AllocateDatum(m_arr_weights);

            m_stream = m_context->CreateStream(0);

            //create a function pointer which point to a APIs collection.
            GROUTE_CUDA_CHECK(cudaMalloc(&m_graph_api_base, sizeof(gframe::GraphAPIBase<TValue, TDelta> *)));

            //Instance the Functions
            CreateAPIInstance<TAPIImple, TValue, TDelta> << < 1, 1, 0, m_stream.cuda_stream >> >
                                                                       (m_graph_api_base, m_context->host_graph.nnodes, m_context->host_graph.nedges);

            m_stream.Sync();
        }

        ~GFrameKernel() {
            DestroyAPIInstance<TValue, TDelta> << < 1, 1, 0, m_stream.cuda_stream >> > (m_graph_api_base);
            m_stream.Sync();

            GROUTE_CUDA_CHECK(cudaFree(m_graph_api_base));
            delete m_graph_allocator;
            delete m_context;
            VLOG(1) << "Destroy Engine";
        }

        void InitValue() const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph_allocator->DeviceObject().nnodes);

            Stopwatch sw(true);

            gframe::kernel::GraphInit<TValue, TDelta> << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                                     (m_graph_allocator->DeviceObject(), m_graph_api_base, m_arr_value.DeviceObject(), m_arr_delta.DeviceObject());
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

            int iteration = 0;

            RelaxDataDriven(
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                          dev_graph.owned_nnodes()),
                    *in_wl,
                    atomic_func);


            groute::Segment<index_t> work_seg = in_wl->GetSeg(m_stream);

            VLOG(1)
            << "Iteration: " << ++iteration << " In-Worklist: " << dev_graph.owned_nnodes() << " Out-Worklist: "
            << work_seg.GetSegmentSize();


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
