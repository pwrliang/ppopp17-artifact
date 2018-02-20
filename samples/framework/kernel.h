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
#include "work_kernel.h"
#include "resultsaver.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);

namespace gframe {
    struct Algo {
        static const char *Name() { return "gframe Kernel"; }
    };

    template<typename TAPIImple,
            typename TAtomicFunc,
            typename TValue,
            typename TDelta,
            typename TWeight=unsigned>
    class GFrameKernel {

    private:
        utils::traversal::Context<gframe::Algo> *m_context;
        groute::graphs::single::CSRGraphAllocator *m_graph_allocator;
        groute::graphs::single::NodeOutputDatum<TValue> m_value_datum;
        groute::graphs::single::NodeOutputDatum<TDelta> m_delta_datum;
        groute::graphs::single::EdgeInputDatum<TWeight> m_weight_datum;
        groute::Stream m_stream;
        TAPIImple m_api_imple;
        TAtomicFunc m_atomic_func;
        bool m_is_weighted;
        bool m_cta;
        typedef groute::Queue<index_t> Worklist;

        template<typename WorkSource>
        void RelaxTopology(const WorkSource &work_source, groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            gframe::kernel::GraphKernelTopology << < grid_dims, block_dims >> >
                                                                (m_api_imple,
                                                                        m_graph_allocator->DeviceObject(),
                                                                        work_source,
                                                                        m_atomic_func,
                                                                        m_value_datum,
                                                                        m_delta_datum);
        }

        template<typename WorkSource, typename WorkTarget>
        void RelaxDataDriven(const WorkSource &work_source, WorkTarget &work_target) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());
            if (m_cta) {
                gframe::kernel::GraphKernelDataDrivenCTA
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               work_source,
                                                               work_target.DeviceObject(),
                                                               m_atomic_func,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(),
                                                               m_is_weighted);
            } else {
                gframe::kernel::GraphKernelDataDriven
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               work_source,
                                                               work_target.DeviceObject(),
                                                               m_atomic_func,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(), m_is_weighted);
            }
        }

        template<typename WorkSource>
        void RelaxTopologyDriven(const WorkSource &work_source) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            if (m_cta) {
                gframe::kernel::GraphKernelTopologyCTA
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_graph_allocator->DeviceObject(),
                                                               work_source,
                                                               m_atomic_func,
                                                               m_value_datum,
                                                               m_delta_datum,
                                                               m_weight_datum,
                                                               m_is_weighted);
            } else {
                gframe::kernel::GraphKernelTopology
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_graph_allocator->DeviceObject(),
                                                               work_source,
                                                               m_atomic_func,
                                                               m_value_datum,
                                                               m_delta_datum,
                                                               m_weight_datum,
                                                               m_is_weighted);
            }
        }


    public:
        GFrameKernel(TAPIImple api_imple, TAtomicFunc atomic_func, bool weighted = false, bool cta = true) :
                m_api_imple(api_imple), m_atomic_func(atomic_func), m_is_weighted(weighted), m_cta(cta) {
            m_context = new utils::traversal::Context<gframe::Algo>(1);

            m_graph_allocator = new groute::graphs::single::CSRGraphAllocator(m_context->host_graph);

            m_context->SetDevice(0);

            m_graph_allocator->AllocateDatumObjects(m_value_datum, m_delta_datum);
            if (weighted)
                m_graph_allocator->AllocateDatum(m_weight_datum);

            m_stream = m_context->CreateStream(0);
        }

        ~GFrameKernel() {
            delete m_graph_allocator;
            delete m_context;
            VLOG(1) << "Destroy Engine";
        }

        void InitValue() const {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph_allocator->DeviceObject().nnodes);

            Stopwatch sw(true);

            gframe::kernel::GraphInit<TAPIImple, TValue, TDelta> << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                                                (m_api_imple,
                                                                                                        m_graph_allocator->DeviceObject(), m_value_datum.DeviceObject(), m_delta_datum.DeviceObject());
            m_stream.Sync();
            sw.stop();
            LOG(INFO) << "Graph Init " << sw.ms() << " ms";
        }

        void DataDriven() {

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
                    *in_wl
            );


            groute::Segment<index_t> work_seg = in_wl->GetSeg(m_stream);

            VLOG(1)
            << "Iteration: " << ++iteration << " In-Worklist: " << dev_graph.owned_nnodes() << " Out-Worklist: "
            << work_seg.GetSegmentSize();


            while (work_seg.GetSegmentSize() > 0) {
                RelaxDataDriven(
                        groute::dev::WorkSourceArray<index_t>(work_seg.GetSegmentPtr(),
                                                              work_seg.GetSegmentSize()),
                        *out_wl
                );

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

//        template <typename TAtomicFunc>


        bool SaveResult(const char *file, bool sort = false) {
            m_graph_allocator->GatherDatum(m_value_datum);
            return ResultOutput<float>(file, m_value_datum.GetHostData(), sort);
        }
    };
}
#endif //GROUTE_KERNEL_H
