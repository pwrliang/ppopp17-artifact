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

#ifndef __GROUTE_CTA_SCHEDULER_H
#define __GROUTE_CTA_SCHEDULER_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>

#include <cub/util_ptx.cuh>

//#define NO_CTA_WARP_INTRINSICS
//#define NP_LAMBDA

namespace groute {
    namespace dev {

        typedef uint32_t index_type;

        template<typename TMetaData>
        struct tb_np_aligned {
            index_type owner;
            index_type start;
            index_type size;
            index_type start_aligned;
            index_type size_aligned;
            TMetaData meta_data;
        };

        /*
        * @brief A structure representing a scheduled chunk of work
        */
        template<typename TMetaData>
        struct np_local_aligned {
            index_type start; // work start
            index_type size; // work size
            index_type start_aligned; // equals start / 4 (or 8)
            index_type size_aligned; // how many uint4/8 used for neighbors
            TMetaData meta_data;
        };

        template<typename TMetaData, const unsigned int VEC_SIZE>
        struct CTAWorkSchedulerAligned {
            template<typename TWork>
            __device__ __forceinline__ static void schedule(np_local_aligned<TMetaData> &np_local, TWork work) {
                const int WP_SIZE = CUB_PTX_WARP_THREADS;
                const int TB_SIZE = blockDim.x;

                const int NP_WP_CROSSOVER = CUB_PTX_WARP_THREADS;
                const int NP_TB_CROSSOVER = blockDim.x;

                typedef tb_np_aligned<TMetaData> np_shared_type;

                __shared__ np_shared_type np_shared;

                if (threadIdx.x == 0) {
                    np_shared.owner = TB_SIZE + 1;
                }

                __syncthreads();

                //
                // First scheduler: processing high-degree work items using the entire block
                //

                //because multi-thread in the thread block can carry data more than thread block
                // so we use loop to elect next owner of thread block to do the task
                while (true) {
                    //multi-threads will overwrite owner field because of shared type np_shared
                    //maybe more threads in blocks's task more than block size, elect one thread to manage other threads in block
                    if (np_local.size >= NP_TB_CROSSOVER) //if neighbors more than thread block size
                    {
                        // 'Elect' one owner for the entire thread block
                        np_shared.owner = threadIdx.x;
                    }

                    __syncthreads();

                    if (np_shared.owner == TB_SIZE + 1) {
                        // No owner was elected, i.e. no high-degree work items remain
                        break;
                    }

                    if (np_shared.owner == threadIdx.x) {
                        // This thread is the owner
                        np_shared.start = np_local.start;
                        np_shared.size = np_local.size;
                        np_shared.start_aligned = np_local.start_aligned;
                        np_shared.size_aligned = np_local.size_aligned;
                        np_shared.meta_data = np_local.meta_data;

                        // Mark this work-item as processed for future schedulers
                        np_local.start = 0;
                        np_local.size = 0;
                        np_local.start_aligned = 0;
                        np_local.size_aligned = 0;
                    }

                    __syncthreads();// wait for last owner who done for his task

                    //threads in the block get the owner information
                    index_type start = np_shared.start;
                    index_type size = np_shared.size;
                    index_type start_aligned = np_shared.start_aligned;
                    index_type size_aligned = np_shared.size_aligned;
                    TMetaData meta_data = np_shared.meta_data;

                    //owner reset the flag
                    if (np_shared.owner == threadIdx.x) {
                        np_shared.owner = TB_SIZE + 1;
                    }

                    // Use all threads in thread block to execute individual work
                    for (int ii = threadIdx.x; ii < size_aligned; ii += TB_SIZE) // TB stride style
                    {
                        //work(ii, start_aligned + ii, size_aligned, size, meta_data);
//                        int real_size = size - (size_aligned - 1) * VEC_SIZE;
                        int real_size = ii == size_aligned - 1 ? size - (size_aligned - 1) * VEC_SIZE : VEC_SIZE;
                        work(start_aligned + ii, real_size, meta_data);
                    }

                    __syncthreads();
                }

                //
                // Second scheduler: tackle medium-degree work items using the warp
                //
                const int lane_id = cub::LaneId();

                // if any len of threads's tasks in warp more than warp size, then parallel do it
                while (__any(np_local.size >= NP_WP_CROSSOVER)) {

                    // Compete for work scheduling
                    int mask = __ballot(np_local.size >= NP_WP_CROSSOVER ? 1 : 0);
                    // Select a deterministic winner
                    // __ffs to get the smallest lane id
                    int leader = __ffs(mask) -
                                 1;   // Find the position of the least significant bit set to 1 in a 32 bit integer.

                    // Broadcast data from the leader
                    index_type start = cub::ShuffleIndex(np_local.start, leader);
                    index_type size = cub::ShuffleIndex(np_local.size, leader);
                    index_type start_aligned = cub::ShuffleIndex(np_local.start_aligned, leader);
                    index_type size_aligned = cub::ShuffleIndex(np_local.size_aligned, leader);
                    TMetaData meta_data = cub::ShuffleIndex(np_local.meta_data, leader);

                    if (leader == lane_id) {
                        // Mark this work-item as processed
                        np_local.start = 0;
                        np_local.size = 0;
                        np_local.start_aligned = 0;
                        np_local.size_aligned = 0;
                    }
                    // Use all threads in warp to execute individual work
                    for (int ii = lane_id; ii < size_aligned; ii += WP_SIZE) {
//                        work(ii, start_aligned + ii, size_aligned, size, meta_data);
                        int real_size = ii == size_aligned - 1 ? size - (size_aligned - 1) * VEC_SIZE : VEC_SIZE;
                        work(start_aligned + ii, real_size, meta_data);
                    }
                }

                __syncthreads();

                //
                // Third scheduler: tackle all work-items with size < 32 serially
                //
                // We did not implement the FG (Finegrained) scheduling for simplicity
                // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0

                for (int ii = 0; ii < np_local.size_aligned; ii++) {
//                    work(ii, np_local.start_aligned + ii, np_local.size_aligned, np_local.size, np_local.meta_data);
//                    int real_size = np_local.size - (np_local.size_aligned - 1) * VEC_SIZE;
                    int real_size = ii == np_local.size_aligned - 1 ? np_local.size - (np_local.size_aligned - 1) * VEC_SIZE : VEC_SIZE;
                    work(np_local.start_aligned + ii, real_size, np_local.meta_data);
                }
            }
        };

    }
}

#endif // __GROUTE_CTA_SCHEDULER_H
