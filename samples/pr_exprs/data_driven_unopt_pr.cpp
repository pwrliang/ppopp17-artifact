//
// Created by liang on 2/16/18.
//
#include <device_atomic_functions.hpp>
#include <host_defines.h>
#include <cstdint>
#include <groute/common.h>
#include <groute/device/queue.cuh>
#include <groute/device/cta_scheduler.cuh>
#include <device_launch_parameters.h>
#include "pr_common.h"

namespace data_driven_unopt_pr{
    template<
            typename WorkSource, typename WorkTarget,
            typename TGraph, typename ResidualDatum, typename RankDatum>
    __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
            const TGraph& graph, ResidualDatum& residual, RankDatum& current_ranks
    )
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x; // We need all threads in active blocks to enter the loop

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
        {
            groute::dev::np_local<rank_t> np_local = { 0, 0, 0.0 };

            if (i < work_size)
            {
                index_t node = work_source.get_work(i);
                current_ranks[node] = 1.0 - ALPHA;  // Initial rank

                np_local.start = graph.begin_edge(node);
                np_local.size = graph.end_edge(node) - np_local.start;

                if (np_local.size > 0) // Skip zero-degree nodes
                {
                    rank_t update = ((1.0 - ALPHA) * ALPHA) / np_local.size; // Initial update
                    np_local.meta_data = update;
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    np_local,
                    [&work_target, &graph, &residual](index_t edge, rank_t update)
                    {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        if (!graph.owns(dest) && prev == 0) // Push only remote nodes since we process all owned nodes at init step 2 anyhow
                        {
                            work_target.append_work(dest);
                        }
                    }
            );
        }
    }
}