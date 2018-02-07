//
// Created by liang on 2/5/18.
//
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
#include <utils/cuda_utils.h>
#include <device_launch_parameters.h>
#include <utils/graphs/traversal.h>
#include <glog/logging.h>
#include "find_block_size.cuh"

typedef float rank_t;
namespace cusha {
    struct Algo {

        static const char *Name() { return "PR"; }
    };

    struct shard_entry {
        index_t srcIdx;
        index_t dstIdx;
    };

    __global__ void CuSha_GShard(const index_t nnodes,
                                 const index_t nedges,
                                 const index_t nshards,
                                 const index_t N,
                                 const index_t *src_index,
                                 const index_t *dest_index,
                                 rank_t *src_value,
                                 rank_t *vertex_values,
                                 const index_t *out_degree,
                                 int *finished_processing,
                                 const index_t *shard_sizes_scan,
                                 const index_t *window_sizes_scans_vertical) {
        assert(blockIdx.x < nshards);
        index_t shard_offset = blockIdx.x * N;
        index_t shard_start_idx = shard_sizes_scan[blockIdx.x];
        index_t shard_end_idx = shard_sizes_scan[blockIdx.x + 1];
        rank_t *shard_vertex_values = vertex_values + shard_offset;
        extern __shared__ rank_t local_vertices[];

        for (index_t local_offset = threadIdx.x; local_offset < N; local_offset += blockDim.x) {
            index_t vertexIdx = shard_offset + local_offset;
            if (vertexIdx < nnodes)
                local_vertices[local_offset] = vertex_values[vertexIdx];
        }

        __syncthreads();
        for (index_t entry_idx = shard_start_idx + threadIdx.x;
             entry_idx < shard_end_idx; entry_idx += blockDim.x) {
            assert(entry_idx < nedges);
            rank_t new_delta = 0.85f * src_value[entry_idx] / out_degree[entry_idx];
            atomicAdd(&local_vertices[dest_index[entry_idx] - shard_offset],
                      new_delta);
        }

        index_t destRange0, destRange1;
        if (threadIdx.x == 0) {
            printf("%d - %d\n", shard_start_idx, shard_end_idx);
        }

        __syncthreads();
        int changed = false;
        for (index_t local_offset = threadIdx.x; local_offset < N; local_offset += blockDim.x) {
            int vertexIdx = shard_offset + local_offset;
            if (vertexIdx < nnodes) {
                if (fabs(local_vertices[local_offset] - vertex_values[vertexIdx]) > 0.001) {
                    changed = true;
                    vertex_values[vertexIdx] = local_vertices[local_offset];
                }
            }
        }

//        if (__syncthreads_or(changed)) {

        for (index_t target_shard_idx = threadIdx.x / warpSize;
             target_shard_idx < nshards;
             target_shard_idx += (blockDim.x / warpSize)) {
            index_t target_window_start_idx = window_sizes_scans_vertical[target_shard_idx * nshards + blockIdx.x];
            index_t target_window_end_idx = window_sizes_scans_vertical[target_shard_idx * nshards + blockIdx.x +
                                                                        1];
            //for each warp in block
            //threadIdx & warpSize - 1 == laneId
            for (index_t window_entry = target_window_start_idx + (threadIdx.x & (warpSize - 1));
                 window_entry < target_window_end_idx; window_entry += warpSize) {
                assert(window_entry >= 0 && window_entry < nedges);
                assert(src_index[window_entry] - shard_offset >= 0 && src_index[window_entry] - shard_offset < N);
                src_value[window_entry] = local_vertices[src_index[window_entry] - shard_offset];
            }
        }

        if (threadIdx.x == 0)
            *finished_processing = 1;
//        }
    }


}

bool RunCusha() {
    printf("Run Cusha\n");
    utils::traversal::Context<cusha::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(0);
    groute::Stream stream = context.CreateStream(0);
    printf("Init device\n");

    const int bSize = 512;
    const blockSize_N_pair bsizeNPair = find_proper_block_size(bSize, context.host_graph.nedges,
                                                               context.host_graph.nnodes);
    const index_t nshards = (context.host_graph.nnodes + bsizeNPair.N - 1) /
                            bsizeNPair.N; //N  The maximum number of vertices inside a shard.
    const index_t nnodes_aligned = nshards * bsizeNPair.N;

//    std::cout << "block size:" << bsizeNPair.blockSize;
//    std::cout << "shard size:" << nshards;
    printf("block size:%d\n", bsizeNPair.blockSize);
    printf("shard size:%d\n", nshards);
    printf(" The maximum number of vertices inside a shard:%d\n", bsizeNPair.N);

    std::vector<std::vector<cusha::shard_entry>> graphWindows(nshards * nshards, std::vector<cusha::shard_entry>(0));

    const index_t nnodes = context.host_graph.nnodes;
    const index_t nedges = context.host_graph.nedges;

    printf("nodes:%d edges:%d\n", nnodes, nedges);
    utils::SharedArray<rank_t> vertex_values(nnodes);
    utils::SharedArray<rank_t> src_values(nedges);
    utils::SharedArray<index_t> out_degrees(nedges);
    utils::SharedArray<index_t> src_index(nedges);
    utils::SharedArray<index_t> dst_index(nedges);
    utils::SharedArray<index_t> window_sizes_scans_vertical(nshards * nshards + 1);
    utils::SharedArray<index_t> shard_sizes_scans(nshards + 1);
    utils::SharedArray<index_t> concatenated_windows_sizes_scan(nshards + 1);

    for (index_t node = 0; node < context.host_graph.nnodes; node++) {
        index_t begin_edge = context.host_graph.begin_edge(node),
                end_edge = context.host_graph.end_edge(node);

        vertex_values.host_ptr()[node] = 1.0;

        for (index_t edge = begin_edge; edge < end_edge; edge++) {
            index_t dest = context.host_graph.edge_dest(edge);
            cusha::shard_entry shard_entry;
            shard_entry.srcIdx = node;
            shard_entry.dstIdx = dest;

            index_t belonging_shard_idx = shard_entry.dstIdx * nshards / nnodes;
            index_t belonging_window_idx = shard_entry.srcIdx * nshards / nnodes;
            assert (belonging_shard_idx * nshards + belonging_window_idx < nshards * nshards);
            graphWindows[belonging_shard_idx * nshards + belonging_window_idx].push_back(shard_entry);
        }
    }


    window_sizes_scans_vertical.host_ptr()[0] = 0;
    shard_sizes_scans.host_ptr()[0] = 0;
    concatenated_windows_sizes_scan.host_ptr()[0] = 0;

    index_t curr_edge_idx = 0;//current edge to process
    index_t curr_win_idx = 0;

    for (index_t shard_idx = 0; shard_idx < nshards; shard_idx++) {
        for (index_t win_idx = 0; win_idx < nshards; win_idx++) {
            std::vector<cusha::shard_entry> &window = graphWindows[shard_idx * nshards + win_idx];

            for (index_t entry_idx = 0; entry_idx < window.size(); entry_idx++) {
                index_t src = window[entry_idx].srcIdx;
                index_t dst = window[entry_idx].dstIdx;
                src_index.host_ptr()[curr_edge_idx] = src;
                dst_index.host_ptr()[curr_edge_idx] = dst;
                out_degrees.host_ptr()[curr_edge_idx] =
                        context.host_graph.begin_edge(src) - context.host_graph.end_edge(src);

//                    src_values.host_ptr()[curr_edge_idx]= init value for node, this can be done by node_values[src]
//                    out_degree.host_ptr()[curr_edge_idx] =
                curr_edge_idx++;
            }
            assert(window.size() >= 0 && window.size() < nshards);
            window_sizes_scans_vertical.host_ptr()[curr_win_idx + 1] =
                    window_sizes_scans_vertical.host_ptr()[curr_win_idx] + window.size();
            curr_win_idx++;
        }
        shard_sizes_scans.host_ptr()[shard_idx + 1] = curr_edge_idx;
    }

    for (int shard_idx = 0; shard_idx < nshards; shard_idx++) {
        printf("%d %d\n", shard_sizes_scans.host_ptr()[shard_idx], shard_sizes_scans.host_ptr()[shard_idx + 1]);
    }
    return true;

    utils::SharedValue<int> finished;

    int iter = 0;
    do {
        finished.set_val_H2D(0);
        cusha::CuSha_GShard << < nshards, bsizeNPair.blockSize, bsizeNPair.N * sizeof(rank_t) >> > (
                nnodes, nedges,
                        nshards, bsizeNPair.N, src_index.dev_ptr, dst_index.dev_ptr,
                        src_values.dev_ptr, vertex_values.dev_ptr, out_degrees.dev_ptr,
                        finished.dev_ptr, shard_sizes_scans.dev_ptr, window_sizes_scans_vertical.dev_ptr);
        cudaDeviceSynchronize();
        printf("iter:%d\n", iter++);
        break;
    } while (finished.get_val_D2H() == 1);

//        curr_edge_idx = 0;
//        for (index_t win_idx = 0; win_idx < nshards; win_idx++) {
//            for (index_t shard_idx = 0; shard_idx < nshards; shard_idx++) {
//                std::vector<shard_entry> &window = graphWindows[shard_idx * nshards + win_idx];
//                index_t in_win_moving = 0;
//
//                for (index_t entry_idx = 0; entry_idx < window.size(); entry_idx++) {
//                    src
//                }
//
//            }
//        }
}