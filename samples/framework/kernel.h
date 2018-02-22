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
#include <utils/cuda_utils.h>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <glog/logging.h>
#include "myatomics.h"
#include "work_kernel.h"
#include "resultsaver.h"
#include "graph_common.h"

DECLARE_double(wl_alloc_factor);
DECLARE_uint64(wl_alloc_abs);
DECLARE_int32(max_iterations);
DECLARE_int32(grid_size);
DECLARE_int32(block_size);

namespace gframe {
    struct Algo {
        static const char *Name() { return "gframe Kernel"; }
    };

    template<typename TAPIImple,
            typename TAtomicFunc,
            typename TValue,
            typename TDelta,
            typename TWeight=unsigned>
    class GFrameEngine {
    public:
        enum EngineType {
            Engine_DataDriven, Engine_TopologyDriven
        };

        GFrameEngine(TAPIImple api_imple, TAtomicFunc atomic_func, EngineType engine_type, bool weighted = false, bool cta = true) :
                m_api_imple(api_imple), m_atomic_func(atomic_func), m_engine_type(engine_type), m_is_weighted(weighted), m_cta(cta) {

            m_context = new utils::traversal::Context<gframe::Algo>(1);

            m_graph_allocator = new groute::graphs::single::CSRGraphAllocator(m_context->host_graph);

            m_context->SetDevice(0);

            m_graph_allocator->AllocateDatumObjects(m_value_datum, m_delta_datum);
            if (m_is_weighted) m_graph_allocator->AllocateDatum(m_weight_datum);

            m_stream = m_context->CreateStream(0);

            m_mgpu_context = new mgpu::standard_context_t(true, m_stream.cuda_stream);

            m_api_imple.graphInfo.nnodes = m_context->host_graph.nnodes;
            m_api_imple.graphInfo.nedges = m_context->host_graph.nedges;

            size_t max_work_size = m_context->host_graph.nedges * FLAGS_wl_alloc_factor;

            if (FLAGS_wl_alloc_abs > 0)
                max_work_size = FLAGS_wl_alloc_abs;

            m_work_list1 = new Worklist(max_work_size, 0, "input queue");
            m_work_list2 = new Worklist(max_work_size, 0, "output queue");

            m_work_list1->ResetAsync(m_stream);
            m_work_list2->ResetAsync(m_stream);
            m_stream.Sync();


        }

        ~GFrameEngine() {
            delete m_work_list1;
            delete m_work_list2;
            delete m_mgpu_context;
            delete m_graph_allocator;
            delete m_context;

            VLOG(1) << "Destroy Engine";
        }

        void InitValue() const {
            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph_allocator->DeviceObject().nnodes);

            Stopwatch sw(true);

            if (m_engine_type == Engine_DataDriven) {
                gframe::kernel::InitWorklist << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                            (m_work_list1->DeviceObject(),
                                                                                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                                                                                          dev_graph.owned_nnodes()));
            }

            gframe::kernel::GraphInit << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                                     (m_api_imple,
                                                                             m_atomic_func,
                                                                             groute::dev::WorkSourceRange<index_t>
                                                                                     (dev_graph.owned_start_node(),
                                                                                      dev_graph.owned_nnodes()),
                                                                             dev_graph,
                                                                             m_value_datum.DeviceObject(),
                                                                             m_delta_datum.DeviceObject(),
                                                                             m_weight_datum.DeviceObject(),
                                                                             m_is_weighted);
            m_stream.Sync();
            sw.stop();
            LOG(INFO) << "Graph Init " << sw.ms() << " ms";
        }

        void Run() {
            if (m_engine_type == EngineType::Engine_DataDriven) {
#ifdef __OUTLINING__
                DataDrivenOutlining();
#else
                DataDriven();
#endif
            } else if (m_engine_type == EngineType::Engine_TopologyDriven) {
#ifdef __OUTLINING__
                TopologyDrivenOutlining();
#else
                TopologyDriven();
#endif
            }
        }


        bool SaveResult(const char *file, bool sort = false) {
            m_graph_allocator->GatherDatum(m_value_datum);
            return ResultOutput<float>(file, m_value_datum.GetHostData(), sort);
        }

    private:
        typedef groute::Queue<index_t> Worklist;

        mgpu::context_t *m_mgpu_context;
        utils::traversal::Context<gframe::Algo> *m_context;
        groute::Stream m_stream;
        groute::graphs::single::CSRGraphAllocator *m_graph_allocator;
        groute::graphs::single::NodeOutputDatum<TValue> m_value_datum;
        groute::graphs::single::NodeOutputDatum<TDelta> m_delta_datum;
        groute::graphs::single::EdgeInputDatum<TWeight> m_weight_datum;
        Worklist *m_work_list1, *m_work_list2;
        TAPIImple m_api_imple;
        TAtomicFunc m_atomic_func;
        EngineType m_engine_type;
        bool m_is_weighted;
        bool m_cta;

        void RelaxDataDriven(const groute::Queue<index_t> &work_source, groute::Queue<index_t> &work_target) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.GetCount(m_stream));
            if (m_cta) {
                gframe::kernel::GraphKernelDataDrivenCTA
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_atomic_func,
                                                               work_source.DeviceObject(),
                                                               work_target.DeviceObject(),
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(),
                                                               m_is_weighted);
            } else {
                gframe::kernel::GraphKernelDataDriven
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_atomic_func,
                                                               work_source.DeviceObject(),
                                                               work_target.DeviceObject(),
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(),
                                                               m_is_weighted);
            }
        }

        void RelaxTopologyDriven(const groute::dev::WorkSourceRange<index_t> &work_source) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            if (m_cta) {
                gframe::kernel::GraphKernelTopologyCTA
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_atomic_func,
                                                               work_source,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(),
                                                               m_is_weighted);
            } else {
                gframe::kernel::GraphKernelTopology
                        << < grid_dims, block_dims, 0, m_stream.cuda_stream >> >
                                                       (m_api_imple,
                                                               m_atomic_func,
                                                               work_source,
                                                               m_graph_allocator->DeviceObject(),
                                                               m_value_datum.DeviceObject(),
                                                               m_delta_datum.DeviceObject(),
                                                               m_weight_datum.DeviceObject(),
                                                               m_is_weighted);
            }
        }


        void DataDriven() {
            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();
            Worklist *in_wl = m_work_list1, *out_wl = m_work_list2;
            std::vector<index_t> host_init_wl;

            int iteration = 0;

            VLOG(1)
            << "Iteration: " << ++iteration << " In-Worklist: " << dev_graph.owned_nnodes() << " Out-Worklist: "
            << in_wl->GetCount(m_stream);

            Stopwatch sw(true);

            while (in_wl->GetCount(m_stream) > 0) {
                RelaxDataDriven(*in_wl, *out_wl);

                VLOG(0) << "Iteration: " << ++iteration << " In-Worklist: " << in_wl->GetCount(m_stream)
                        << " Out-Worklist: " << out_wl->GetCount(m_stream);

                if (iteration >= FLAGS_max_iterations) {
                    LOG(WARNING) << "Maximum iteration times reached";
                    break;
                }

                in_wl->ResetAsync(m_stream);

                std::swap(in_wl, out_wl);
            }

            sw.stop();

            VLOG(0) << "DataDriven Time: " << sw.ms() << " ms.";
        }

#ifdef __OUTLINING__

        void DataDrivenOutlining() {
            int grid_size = FLAGS_grid_size;
            int block_size = FLAGS_block_size;
            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();
            Worklist *in_wl = m_work_list1, *out_wl = m_work_list2;
            cub::GridBarrierLifetime grid_barrier;

            VLOG(1) << "Outlining Mode - Grid Size: " << grid_size << " Block Size:" << block_size;

            grid_barrier.Setup(grid_size);

            Stopwatch sw(true);

            gframe::kernel::KernelControllerDataDriven << < grid_size, block_size, 0, m_stream.cuda_stream >> >
                                                                                      (m_api_imple,
                                                                                              m_atomic_func,
                                                                                              in_wl->DeviceObject(),
                                                                                              out_wl->DeviceObject(),
                                                                                              dev_graph,
                                                                                              m_value_datum.DeviceObject(),
                                                                                              m_delta_datum.DeviceObject(),
                                                                                              m_weight_datum.DeviceObject(),
                                                                                              m_is_weighted,
                                                                                              m_cta,
                                                                                              FLAGS_max_iterations,
                                                                                              grid_barrier);
            m_stream.Sync();
            sw.stop();

            VLOG(0) << "DataDrivenOutlining Times: " << sw.ms() << " ms.";
        }

#endif

        void TopologyDriven() {
            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();
            groute::dev::WorkSourceRange<index_t> work_source(dev_graph.owned_start_node(), dev_graph.owned_nnodes());

            Stopwatch sw(true);

            for (int iteration = 0; iteration < FLAGS_max_iterations; iteration++) {
                RelaxTopologyDriven(groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(), dev_graph.owned_nnodes()));

                TValue accumulated_value = gframe::kernel::ConvergeCheck(m_api_imple,
                                                                         *m_mgpu_context,
                                                                         work_source,
                                                                         m_value_datum.DeviceObject());
                VLOG(0) << "Iteration: " << ++iteration << " current sum: " << accumulated_value;

                if (m_api_imple.IsConverge(accumulated_value)) {
                    VLOG(0) << "Convergence condition satisfied";
                    break;
                }

                if (iteration == FLAGS_max_iterations - 1) {
                    LOG(WARNING) << "Maximum iteration times reached";
                }
            }

            sw.stop();
            VLOG(0) << "TopologyDriven Time: " << sw.ms() << " ms.";
        }

#ifdef __OUTLINING__

        void TopologyDrivenOutlining() {
            int grid_size = FLAGS_grid_size;
            int block_size = FLAGS_block_size;
            const groute::graphs::dev::CSRGraph &dev_graph = m_graph_allocator->DeviceObject();
            cub::GridBarrierLifetime grid_barrier;
            utils::SharedArray<TValue> grid_buffer(grid_size);
            utils::SharedValue<int> running_flag;

            VLOG(1) << "Outlining Mode - Grid Size: " << grid_size << " Block Size:" << block_size;

            grid_barrier.Setup(grid_size);
            GROUTE_CUDA_CHECK(cudaMemset(grid_buffer.dev_ptr, 0, grid_buffer.buffer_size * sizeof(TValue)));
            running_flag.set_val_H2D(1);

            Stopwatch sw(true);

            gframe::kernel::KernelControllerTopologyDriven << < grid_size, block_size, grid_size * sizeof(TValue), m_stream.cuda_stream >> > (m_api_imple,
                    m_atomic_func,
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(), dev_graph.owned_nnodes()),
                    dev_graph,
                    m_value_datum.DeviceObject(),
                    m_delta_datum.DeviceObject(),
                    m_weight_datum.DeviceObject(),
                    m_is_weighted,
                    m_cta,
                    FLAGS_max_iterations,
                    grid_buffer.dev_ptr,
                    running_flag.dev_ptr,
                    grid_barrier);

            m_stream.Sync();
            sw.stop();

            VLOG(0) << "TopologyDrivenOutlining Times: " << sw.ms() << " ms.";
        }

#endif

    };
}
#endif //GROUTE_KERNEL_H
