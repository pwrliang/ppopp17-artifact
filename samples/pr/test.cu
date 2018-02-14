//
// Created by liang on 2/7/18.
//
#include <utils/stopwatch.h>
#include <groute/graphs/common.h>
#include <cstdio>
#include <groute/internal/cuda_utils.h>
#include <groute/graphs/csr_graph_align.h>
#include <utils/graphs/traversal.h>

namespace mytest {
    struct Algo {
        static const char *Name() { return "Test"; }
    };

    template<typename TGraph>
    __global__ void testTraverse(TGraph graph) {
        index_t nnodes = graph.nnodes;


        for (index_t node = 0; node < nnodes; node++) {
            index_t begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge,
                    aligned_begin_edge = graph.aligned_begin_edge(node),
                    aligned_end_edge = graph.aligned_end_edge(node);


            const index_t VEC_SIZE = 4;

            if (out_degree == 0) continue;

            index_t offset = begin_edge;
            index_t aligned_offset = aligned_begin_edge;

            while (end_edge - offset >= VEC_SIZE) {
                uint4 dest4 = graph.edge_dest4(aligned_offset);
                printf("%d %d %d %d ", dest4.x, dest4.y, dest4.z, dest4.w);

                aligned_offset++;
                offset += VEC_SIZE;
            }

            if (offset < end_edge) {
                uint4 last_trunk = graph.edge_dest4(aligned_offset);
                index_t rest_len = end_edge - offset;

                if (rest_len == 3) {
                    printf("%d %d %d ", last_trunk.x, last_trunk.y, last_trunk.z);
                } else if (rest_len == 2) {
                    printf("%d %d ", last_trunk.x, last_trunk.y);
                } else if (rest_len == 1) {
                    printf("%d", last_trunk.x);
                }
            }
//            for (index_t edge = aligned_begin_edge; edge < aligned_end_edge; edge++) {
//                printf("%d ", graph.edge_dest(edge));
//            }
            printf("\n");
        }
    }
}

void test() {

    utils::traversal::Context<mytest::Algo> context(1);
    groute::graphs::single::CSRGraphAllocatorAlign dev_graph_allocator(context.host_graph);

    mytest::testTraverse << < 1, 1 >> > (dev_graph_allocator.DeviceObject());
    return;
    uint32_t N = 1024 * 1024 * 10;
    index_t *host_data = static_cast<index_t *>(malloc(sizeof(index_t) * N));
    index_t *dev_data;

    GROUTE_CUDA_CHECK(cudaMalloc(&dev_data, sizeof(index_t) * N));

    Stopwatch sw(true);
    GROUTE_CUDA_CHECK(cudaMemcpy(dev_data, host_data, sizeof(index_t) * N, cudaMemcpyHostToDevice));
    sw.stop();

    printf("copy times:%f bandwidth:%f \n", sw.ms(), 1.0 * sizeof(index_t) * N / 1024 / 1024 / 1024 / (sw.ms() / 1000));

    sw.start();

    for (int i = 0; i < N / 128; i++) {
        GROUTE_CUDA_CHECK(cudaMemcpyAsync(dev_data + i, host_data + i, sizeof(index_t) * 128, cudaMemcpyHostToDevice));
    }
    cudaDeviceSynchronize();
    sw.stop();

    printf("copy times:%f bandwidth:%f \n", sw.ms(), 1.0 * sizeof(index_t) * N / 1024 / 1024 / 1024 / (sw.ms() / 1000));
}