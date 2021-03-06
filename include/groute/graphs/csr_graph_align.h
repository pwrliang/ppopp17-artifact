// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __GROUTE_GRAPHS_CSR_GRAPH_ALIGN_H
#define __GROUTE_GRAPHS_CSR_GRAPH_ALIGN_H

#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include <groute/context.h>
#include <groute/graphs/common.h>
#include <groute/graphs/csr_graph.h>

typedef struct __device_builtin__ __builtin_align__(32) {
    unsigned int a0, a1, a2, a3, a4, a5, a6, a7;
} uint8;
namespace groute {
    namespace graphs {
//        const unsigned int VEC_SIZE = 8;

        namespace dev // device objects
        {
            /*
            * @brief A single GPU graph object (a complete graph allocated at one GPU)
            */
            template <const unsigned int VEC_SIZE>
            struct CSRGraphAlign : public groute::graphs::dev::CSRGraph {
                CSRGraphAlign() {}

                __device__ __forceinline__ index_t aligned_begin_edge(index_t node) const {
                    return row_start[node] / VEC_SIZE;
                }


                __device__ __forceinline__ index_t aligned_end_edge(index_t node) const {
                    return row_start[node + 1] / VEC_SIZE;
                }

                __device__ __forceinline__ index_t end_edge4(index_t node) const {
                    if (CSRGraph::begin_edge(node) == CSRGraph::end_edge(node))
                        return CSRGraph::end_edge(node);

                    index_t end_edge = CSRGraph::end_edge(node);
                    uint4 last_trunk = edge_dest4(end_edge / VEC_SIZE - 1);

                    if (last_trunk.y == -1) end_edge--;
                    if (last_trunk.z == -1) end_edge--;
                    if (last_trunk.w == -1) end_edge--;

                    return end_edge;
                }

                __device__ __forceinline__ index_t end_edge8(index_t node) const {
                    if (CSRGraph::begin_edge(node) == CSRGraph::end_edge(node))
                        return CSRGraph::end_edge(node);

                    index_t end_edge = CSRGraph::end_edge(node);
                    uint8 last_trunk = edge_dest8(end_edge / VEC_SIZE - 1);

                    if (last_trunk.a1 == -1)
                        end_edge -= 7;
                    else if (last_trunk.a2 == -1)
                        end_edge -= 6;
                    else if (last_trunk.a3 == -1)
                        end_edge -= 5;
                    else if (last_trunk.a4 == -1)
                        end_edge -= 4;
                    else if (last_trunk.a5 == -1)
                        end_edge -= 3;
                    else if (last_trunk.a6 == -1)
                        end_edge -= 2;
                    else if (last_trunk.a7 == -1)
                        end_edge -= 1;

                    return end_edge;
                }

                __device__ __forceinline__ uint1 edge_dest1(index_t edge) const {
                    return reinterpret_cast<uint1 *>(edge_dst)[edge];
                }

                __device__ __forceinline__ uint2 edge_dest2(index_t edge) const {
                    return reinterpret_cast<uint2 *>(edge_dst)[edge];
                }

                __device__ __forceinline__ uint3 edge_dest3(index_t edge) const {
                    return reinterpret_cast<uint3 *>(edge_dst)[edge];
                }

                __device__ __forceinline__ uint4 edge_dest4(index_t edge) const {
                    return reinterpret_cast<uint4 *>(edge_dst)[edge];
                }

                __device__ __forceinline__ uint8 edge_dest8(index_t edge) const {
                    return reinterpret_cast<uint8 *>(edge_dst)[edge];
                }
//                __device__ __forceinline__ index_t out_degree(index_t node) const {
//                    index_t aligned_begin_edge = begin_edge(node),
//                            aligned_end_edge = end_edge(node),
//                            aligned_out_degree = aligned_end_edge - aligned_begin_edge;
//                    index_t blank = 0;
//
//                    if (aligned_out_degree > 0) {
//                        while (edge_dest(aligned_end_edge - 1 - blank) == -1)
//                            blank++;
//                    }
//
//                    return aligned_out_degree - blank;
//                }
            };
        }

        namespace single {
            /*
            * @brief A single GPU graph allocator (allocates a complete mirror graph at one GPU)
            */
            template <const unsigned int VEC_SIZE>
            struct CSRGraphAllocatorAlign {
                typedef dev::CSRGraphAlign<VEC_SIZE> DeviceObjectType;

            private:
                host::CSRGraph &m_origin_graph;
                dev::CSRGraphAlign<VEC_SIZE> m_dev_mirror;

            public:
                CSRGraphAllocatorAlign(host::CSRGraph &host_graph) :
                        m_origin_graph(host_graph) {
                    AllocateDevMirror();
                }

                ~CSRGraphAllocatorAlign() {
                    DeallocateDevMirror();
                }

                const dev::CSRGraphAlign<VEC_SIZE> &DeviceObject() const {
                    return m_dev_mirror;
                }

                void AllocateDatumObjects() {}

                template<typename TFirstGraphDatum, typename...TGraphDatum>
                void AllocateDatumObjects(TFirstGraphDatum &first_datum, TGraphDatum &... more_data) {
                    AllocateDatum(first_datum);
                    AllocateDatumObjects(more_data...);
                }

                template<typename TGraphDatum>
                void AllocateDatum(TGraphDatum &graph_datum) {
                    graph_datum.Allocate(m_origin_graph);
                }

                template<typename TGraphDatum>
                void GatherDatum(TGraphDatum &graph_datum) {
                    graph_datum.Gather(m_origin_graph);
                }

            private:
                void AllocateDevMirror() {
                    index_t nnodes, nedges;

                    m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                    m_dev_mirror.nedges = nedges = m_origin_graph.nedges;

                    index_t *row_start = new index_t[nnodes + 1];
                    row_start[0] = 0;

                    for (index_t node = 0; node < nnodes; node++) {
                        index_t begin_edge = m_origin_graph.row_start[node],
                                end_edge = m_origin_graph.row_start[node + 1],
                                out_degree = end_edge - begin_edge;
                        index_t out_degree_round_up =
                                (out_degree + VEC_SIZE - 1) / VEC_SIZE * VEC_SIZE;
                        row_start[node + 1] = row_start[node] + out_degree_round_up;
                    }

                    uint32_t aligned_nedges = row_start[nnodes];
                    index_t *host_edge_dst = new index_t[aligned_nedges];

                    for (index_t node = 0; node < nnodes; node++) {
                        index_t begin_edge = m_origin_graph.row_start[node],
                                end_edge = m_origin_graph.row_start[node + 1],
                                out_degree = end_edge - begin_edge;

                        index_t aligned_begin_edge = row_start[node],
                                aligned_end_edge = row_start[node + 1],
                                aligned_out_degree = aligned_end_edge - aligned_begin_edge;

                        index_t offset = 0;
                        for (int edge = begin_edge; edge < end_edge; edge++, offset++) {
                            index_t dest = m_origin_graph.edge_dest(edge);
                            host_edge_dst[aligned_begin_edge + offset] = dest;
                        }

                        while (offset + aligned_begin_edge < aligned_end_edge) {
                            host_edge_dst[aligned_begin_edge + offset] = -1;
                            offset++;
                        }
                    }

                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.row_start, (nnodes + 1) *
                                                                          sizeof(index_t))); // malloc and copy +1 for the row_start's extra cell
                    GROUTE_CUDA_CHECK(
                            cudaMemcpy(m_dev_mirror.row_start, row_start, (nnodes + 1) * sizeof(index_t),
                                       cudaMemcpyHostToDevice));


                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst, aligned_nedges * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.edge_dst, host_edge_dst, aligned_nedges * sizeof(index_t),
                                                 cudaMemcpyHostToDevice));
                    delete[] row_start;
                    delete[] host_edge_dst;
                }

                void DeallocateDevMirror() {
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.row_start));
                    GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.edge_dst));

                    m_dev_mirror.row_start = nullptr;
                    m_dev_mirror.edge_dst = nullptr;
                }
            };
        }
    }
}


#endif // __GROUTE_GRAPHS_CSR_GRAPH_H
