//
// Created by liang on 2/3/18.
//

#ifndef GROUTE_MY_PR_BALANCED_H
#define GROUTE_MY_PR_BALANCED_H

#include <groute/device/cta_scheduler.cuh>
#include "pr_common.h"

namespace mypr {

    template<typename TMetaData>
    struct work_chunk {
        index_t start;
        index_t size;
        TMetaData meta_data;
    };

    template<typename TMetaData>
    struct tb_info {
        index_t owner;
        index_t start;
        index_t size;
        TMetaData meta_data;
    };

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            template<typename> class TWorkList>
    __global__ void PageRankDeviceBalanced__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            TWorkList<index_t> input_worklist, TWorkList<index_t> output_worklist) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        const int TB_SIZE = blockDim.x;
        const int WP_SIZE = warpSize;

        const int NP_TB_CROSSOVER = TB_SIZE;
        const int NP_WP_CROSSOVER = WP_SIZE;

        typedef struct tb_info<rank_t> tb_info_type;

        uint32_t work_size = input_worklist.count();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
        __shared__ tb_info_type tb_info_shared;

        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {


            work_chunk<rank_t> local_work = {0, 0, 0.0};
            if (i < work_size) {

                index_t node = input_worklist.read(i);

                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                if (res > 0) {
                    current_ranks[node] += res;

                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;

                    if (local_work.size > 0) {
                        rank_t update = ALPHA * res / local_work.size;

                        local_work.meta_data = update;
                    }
                }
            }

            //all threads in the block can reach here, empty thread(i>=work_size) have responsibility to do the work

            if (threadIdx.x == 0) {
                tb_info_shared.owner = TB_SIZE + 1;
            }

            __syncthreads();

            while (true) {
                if (local_work.size >= NP_TB_CROSSOVER) {
                    tb_info_shared.owner = threadIdx.x;
                }

                __syncthreads();

                //no hight-degree work found
                if (tb_info_shared.owner == TB_SIZE + 1) {
                    break;
                }

                // I'm the owner of thread block
                if (tb_info_shared.owner == threadIdx.x) {
                    tb_info_shared.start = local_work.start;
                    tb_info_shared.size = local_work.size;
                    tb_info_shared.meta_data = local_work.meta_data;

                    // mark this work has processed
                    local_work.start = 0;
                    local_work.size = 0;
                }

                __syncthreads();

                //all threads in block read from tb_info_shared to know the works
                index_t start = tb_info_shared.start;
                index_t size = tb_info_shared.size;
                rank_t meta_data = tb_info_shared.meta_data;

                if (tb_info_shared.owner == threadIdx.x) {
                    tb_info_shared.owner = TB_SIZE + 1;
                }


                // all threads in TB do the work
                // thread block stride style
                for (index_t edge = threadIdx.x; edge < size; edge += TB_SIZE) {
                    index_t dest = graph.edge_dest(edge + start);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), meta_data);

                    if (prev + meta_data > EPSILON && prev <= EPSILON) {
                        output_worklist.append_warp(dest);
                    }
                }

                __syncthreads();
            }

            const int lane_id = cub::LaneId();

            while (__any(local_work.size >= NP_WP_CROSSOVER)) {
                // filter threads which task size > WARP_SIZE
//                int mask = __ballot_sync(0xffffffff, local_work.size >= NP_WP_CROSSOVER ? 1 : 0);
                int mask = __ballot(local_work.size >= NP_WP_CROSSOVER ? 1 : 0);

                // elect smallest lane_id in filtered threads
                int leader = __ffs(mask) - 1;

                index_t start = cub::ShuffleIndex(local_work.start, leader);
                index_t size = cub::ShuffleIndex(local_work.size, leader);
                rank_t meta_data = cub::ShuffleIndex(local_work.meta_data, leader);

                // I'm the leader thread in warp
                if (leader == lane_id) {
                    local_work.start = 0;
                    local_work.size = 0;
                }

                // use all thread in warp to do the task
                for (int edge = lane_id; edge < size; edge += WP_SIZE) {
                    index_t dest = graph.edge_dest(start + edge);

                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), meta_data);

                    if (prev + meta_data > EPSILON && prev <= EPSILON) {
                        output_worklist.append_warp(dest);
                    }
                }
                //we don't need sync in warp, because they exec in lock-step
            }

            __syncthreads();

            //do all the task which size < WARP_SIZE
            for (int edge = 0; edge < local_work.size; edge++) {
                index_t dest = graph.edge_dest(local_work.start + edge);

                rank_t prev = atomicAdd(residual.get_item_ptr(dest), local_work.meta_data);

                if (prev + local_work.meta_data > EPSILON && prev <= EPSILON) {
                    output_worklist.append_warp(dest);
                }
            }
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            template<typename> class TWorkList>
    __global__ void PageRankDeviceBalanced1__Single__(
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
            TWorkList<index_t> input_worklist, TWorkList<index_t> output_worklist) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = input_worklist.count();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {

            groute::dev::np_local<rank_t> np_local = {0, 0, 0.0};

            if (i < work_size) {

                index_t node = input_worklist.read(i);

                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                if (res > 0) {
                    current_ranks[node] += res;

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;

                    if (np_local.size > 0) {
                        rank_t update = ALPHA * res / np_local.size;
                        np_local.meta_data = update;
                    }
                }
            }

            groute::dev::CTAWorkScheduler<rank_t>::template
            schedule(np_local,
                     [&output_worklist, &graph, &residual](index_t edge,
                                                           rank_t update) {
                         index_t dest = graph.edge_dest(edge);
                         rank_t prev = atomicAdd(
                                 residual.get_item_ptr(dest), update);

                         if (prev <= EPSILON &&
                             prev + update > EPSILON) {
                             output_worklist.append_warp(dest);
                         }
                     });
        }
    }
//    I give up to do this, because append_warp did it.......
//    template<
//            typename TGraph,
//            template<typename> class RankDatum,
//            template<typename> class ResidualDatum,
//            typename WorkSource,
//            template<typename> class TWorkList>
//    __global__ void PageRankKernelShfl__Single__(
//            TGraph graph,
//            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
//            WorkSource work_source, TWorkList<index_t> output_worklist) {
//        unsigned tid = TID_1D;
//        unsigned nthreads = TOTAL_THREADS_1D;
//
//        uint32_t work_size = work_source.get_size();
//
//
//        for (index_t i = 0 + tid; i < work_size; i += nthreads) {
//            index_t node = work_source.get_work(i);
//
//            rank_t res = atomicExch(residual.get_item_ptr(node), 0);
//
//            if (res == 0)continue;
//
//            current_ranks[node] += res;
//
//            index_t begin_edge = graph.begin_edge(node),
//                    end_edge = graph.end_edge(node),
//                    out_degree = end_edge - begin_edge;
//
//            if (out_degree == 0) continue;
//
//            rank_t update = ALPHA * res / out_degree;
//
//            for (index_t edge = begin_edge; edge < end_edge; ++edge) {
//                index_t dest = graph.edge_dest(edge);
//
//                rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
//
//                int add = prev + update > EPSILON && prev < EPSILON ?1:0;
//                if(add) {
//                    int mask = __ballot(add);
//                    int leader = __ffs(mask) - 1;
//
//                    //calculate the total number to add
//                    int val = add;
//                    for (int offset = 16; offset > 0; offset /= 2)
//                        val += __shfl_down_sync(0xffffffff, val, offset);
//
//                    int last_unused_pos;
//                    if (leader == cub::LaneId()) {
//                        last_unused_pos = atomicAdd(&output_worklist.len, val);
//                    }
//                    //broadcast last unused position
//                    last_unused_pos = cub::ShuffleIndex(last_unused_pos, leader);
//
//                    int m = (1<<cub::LaneId()) - 1;
//                    int relative_pos = __popc(mask & m);
//
//                    //relative_index is the position of '1' of mask
//                    output_worklist[last_len+relative_pos] = dest;
//                }
//
//
//                //output_worklist.append_warp(dest);
//
//            }
//        }
//    }
}

#endif //GROUTE_MY_PR_BALANCED_H
