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

        template<const int WARPS_PER_TB, typename TMetaData1, typename TMetaData2>
        struct warp_np {
            volatile index_type owner[WARPS_PER_TB];
            volatile index_type start[WARPS_PER_TB];
            volatile index_type size[WARPS_PER_TB];
            volatile TMetaData1 meta_data1[WARPS_PER_TB];
            volatile TMetaData2 meta_data2[WARPS_PER_TB];
        };

        template<typename TMetaData1, TMetaData2>
        struct tb_np {
            index_type owner;
            index_type start;
            index_type size;
            TMetaData1 meta_data1;
            TMetaData2 meta_data2;
        };

        struct empty_np {
        };


        template<typename ts_type, typename TTB, typename TWP, typename TFG = empty_np>
        union np_shared {
            // for scans
            ts_type temp_storage;

            // for tb-level np
            TTB tb;

            // for warp-level np
            TWP warp;

            TFG fg;
        };

        /*
        * @brief A structure representing a scheduled chunk of work
        */
        template<typename TMetaData1, typename TMetaData2>
        struct np_local {
            index_type size; // work size
            index_type start; // work start
            TMetaData1 meta_data1;
            TMetaData2 meta_data2;
        };

        template<typename TMetaData1, typename TMetaData2>
        struct CTAWorkScheduler {
            template<typename TWork>
            __device__ __forceinline__ static void schedule(np_local<TMetaData1, TMetaData2> &np_local, TWork work) {
                const int WP_SIZE = CUB_PTX_WARP_THREADS;
                const int TB_SIZE = blockDim.x;

                const int NP_WP_CROSSOVER = CUB_PTX_WARP_THREADS;
                const int NP_TB_CROSSOVER = blockDim.x;

#ifndef NO_CTA_WARP_INTRINSICS
                typedef union np_shared<empty_np, tb_np<TMetaData1, TMetaData2>, empty_np> np_shared_type;
#else
                typedef union np_shared<empty_np, tb_np<TMetaData1, TMetaData2>, warp_np<32, TMetaData1,TMetaData2>> np_shared_type; // 32 is max number of warps in block
#endif

                __shared__ np_shared_type np_shared;

                if (threadIdx.x == 0) {
                    np_shared.tb.owner = TB_SIZE + 1;
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
                        np_shared.tb.owner = threadIdx.x;
                    }

                    __syncthreads();

                    if (np_shared.tb.owner == TB_SIZE + 1) {
                        // No owner was elected, i.e. no high-degree work items remain

#ifndef NO_CTA_WARP_INTRINSICS
                        // No need to sync threads before moving on to WP scheduler
                        // because it does not use shared memory
#else
                        //have to sync, because we can not use tb or warp field concurrently
                        __syncthreads(); // Necessary do to the shared memory union used by both TB and WP schedulers
#endif
                        break;
                    }

                    if (np_shared.tb.owner == threadIdx.x) {
                        // This thread is the owner
                        np_shared.tb.start = np_local.start;
                        np_shared.tb.size = np_local.size;
                        np_shared.tb.meta_data1 = np_local.meta_data1;
                        np_shared.tb.meta_data2 = np_local.meta_data2;

                        // Mark this work-item as processed for future schedulers
                        np_local.start = 0;
                        np_local.size = 0;
                    }

                    __syncthreads();// wait for last owner who done for his task

                    //threads in the block get the owner information
                    index_type start = np_shared.tb.start;
                    index_type size = np_shared.tb.size;
                    TMetaData1 meta_data1 = np_shared.tb.meta_data1;
                    TMetaData2 meta_data2 = np_shared.tb.meta_data2;

                    //owner reset the flag
                    if (np_shared.tb.owner == threadIdx.x) {
                        np_shared.tb.owner = TB_SIZE + 1;
                    }

                    // Use all threads in thread block to execute individual work
                    for (int ii = threadIdx.x; ii < size; ii += TB_SIZE) // TB stride style
                    {
                        work(start + ii, size, meta_data1, meta_data2);
                    }

                    __syncthreads();
                }

                //
                // Second scheduler: tackle medium-degree work items using the warp
                //
#ifdef NO_CTA_WARP_INTRINSICS
                //            const int warp_id = cub::WarpId();
                            const int warp_id = threadIdx.x / warpSize;
#endif
                const int lane_id = cub::LaneId();

                // if any len of threads's tasks in warp more than warp size, then parallel do it
                while (__any(np_local.size >= NP_WP_CROSSOVER)) {

#ifndef NO_CTA_WARP_INTRINSICS
                    // Compete for work scheduling
                    int mask = __ballot(np_local.size >= NP_WP_CROSSOVER ? 1 : 0);
                    // Select a deterministic winner
                    // __ffs to get the smallest lane id
                    int leader = __ffs(mask) -
                                 1;   // Find the position of the least significant bit set to 1 in a 32 bit integer.

                    // Broadcast data from the leader
                    index_type start = cub::ShuffleIndex<32>(np_local.start, leader, 0xffffffff);
                    index_type size = cub::ShuffleIndex<32>(np_local.size, leader, 0xffffffff);
                    TMetaData1 meta_data1 = cub::ShuffleIndex<32>(np_local.meta_data1, leader, 0xffffffff);
                    TMetaData2 meta_data2 = cub::ShuffleIndex<32>(np_local.meta_data2, leader, 0xffffffff);

                    if (leader == lane_id) {
                        // Mark this work-item as processed
                        np_local.start = 0;
                        np_local.size = 0;
                    }
#else
                    if (np_local.size >= NP_WP_CROSSOVER)
                    {
                        // Again, race to select an owner for warp
                        np_shared.warp.owner[warp_id] = lane_id;
                    }
                    if (np_shared.warp.owner[warp_id] == lane_id)
                    {
                        // This thread is owner
                        np_shared.warp.start[warp_id] = np_local.start;
                        np_shared.warp.size[warp_id] = np_local.size;
                        np_shared.warp.meta_data1[warp_id] = np_local.meta_data1;
                        np_shared.warp.meta_data2[warp_id] = np_local.meta_data2;

                        // Mark this work-item as processed
                        np_local.start = 0;
                        np_local.size = 0;
                    }
                    __syncthreads();
                    index_type start = np_shared.warp.start[warp_id];
                    index_type size = np_shared.warp.size[warp_id];
                    TMetaData1 meta_data1 = np_shared.warp.meta_data1[warp_id];
                    TMetaData2 meta_data2 = np_shared.warp.meta_data2[warp_id];
#endif
                    // Use all threads in warp to execute individual work
                    for (int ii = lane_id; ii < size; ii += WP_SIZE) {
                        work(start + ii, size, meta_data1, meta_data2);
                    }
                }

                __syncthreads();

                //
                // Third scheduler: tackle all work-items with size < 32 serially
                //
                // We did not implement the FG (Finegrained) scheduling for simplicity
                // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0

                for (int ii = 0; ii < np_local.size; ii++) {
                    work(np_local.start + ii, np_local.size, np_local.meta_data1, np_local.meta_data2);
                }
            }
        };

    }
}

#endif // __GROUTE_CTA_SCHEDULER_H
