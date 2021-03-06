cmake_minimum_required(VERSION 2.8)
project(groute)

############## DEPENDENCIES ##############
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 7 REQUIRED)

#Dynamic Parallelism
link_directories(/usr/local/cuda/lib64)

set(LINKER_LIBS "")

# glog
include("cmake/FindGlog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GLOG_LIBRARIES})

# gflags
include("cmake/FindGFlags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GFLAGS_LIBRARIES})

#boost
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# gtest
#include("cmake/FindGTest.cmake")
#include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
#list(APPEND LINKER_LIBS ${GTEST_INCLUDE_DIRS})
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})
### GFlags
#add_subdirectory(deps/gflags)
#include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
#link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)
#
#### Glog
#add_subdirectory(deps/glog)
#include_directories(${PROJECT_BINARY_DIR}/deps/glog/include)
#link_directories(${PROJECT_BINARY_DIR}/deps/glog/lib)

# GTest
#find_package(GTest QUIET)
#if (NOT GTEST_FOUND)
#    add_subdirectory(deps/googletest/googletest)
#endif ()

# Try to find METIS
find_package(METIS QUIET)
if (NOT METIS_FOUND)
    # Otherwise, try to find a local copy
    if (EXISTS "${PROJECT_SOURCE_DIR}/metis/")
        set(GKLIB_PATH ${PROJECT_SOURCE_DIR}/metis/GKlib CACHE PATH "path to GKlib" FORCE)
        add_subdirectory(metis)
        set(METIS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/metis/include)
        set(METIS_LIBRARIES metis)
        set(METIS_FOUND "true")
    else ()
        message(WARNING "Compiling without METIS partitioning. To install METIS, run 'sudo apt-get install libmetis-dev' or manually download metis and extract to a subdirectory called 'metis'")

        set(METIS_INCLUDE_DIRS "")
        set(METIS_LIBRARIES "")

    endif ()
endif ()

if (METIS_FOUND)
    message("-- Found METIS: ${METIS_LIBRARIES}")
    add_definitions(-DHAVE_METIS)
endif ()


# CUB
include_directories(${PROJECT_SOURCE_DIR}/deps/cub)

############## BUILD ##############

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXTRA_LIBS pthread ${Boost_LIBRARIES} ${METIS_LIBRARIES} ${LINKER_LIBS} ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so -lcudadevrt)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11;--expt-extended-lambda)
else ()
    message("Release mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11;-rdc=true;--expt-extended-lambda)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DNDEBUG")

#include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${METIS_INCLUDE_DIRS})
#cuda_add_executable(cc src/utils/parser.cpp src/utils/utils.cpp
#                    samples/cc/cc_async.cu samples/cc/main.cpp)
#target_link_libraries(cc ${EXTRA_LIBS})

#cuda_add_executable(bfs src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
#                    samples/bfs/bfs_async.cu
#                    samples/bfs/bfs_host.cpp
#					samples/bfs/main.cpp)
#target_link_libraries(bfs ${EXTRA_LIBS})

#cuda_add_executable(sssp src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
#        samples/sssp/sssp_async.cu
#        samples/sssp/deltasssp.cu
#        samples/sssp/sssp_host.cpp
#        samples/sssp/main.cpp)
#target_link_libraries(sssp ${EXTRA_LIBS})

cuda_add_executable(sssp_expr src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
                samples/sssp_hybrid/deltasssp.cu
#        samples/sssp_hybrid/deltasssp_topology.cu
        #        samples/sssp_hybrid/sssp_async.cu
        samples/sssp_hybrid/sssp_host.cpp
        samples/sssp_hybrid/main.cpp)
target_link_libraries(sssp_expr ${EXTRA_LIBS})

#cuda_add_executable(pr src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
#        samples/pr/test.cu
#        samples/pr/my_pr_async.cu
#        samples/pr/deltapr.cu
#        samples/pr/outlined_pr_async.cu
#        samples/pr/persist_pr_async.cu
#        samples/pr/pr_async.cu
#        samples/pr/pr_host.cpp
#        samples/pr/aligned_outlined_pr_async.cu
#        samples/pr/main.cpp)
#target_link_libraries(pr ${EXTRA_LIBS})

#cuda_add_executable(gframe_pr src/utils/parser.cpp
#                    src/utils/utils.cpp
#                    src/groute/graphs/csr_graph.cpp
#                    samples/framework/entry.cu
#                    samples/framework/pagerank.cu)
#target_link_libraries(gframe_pr ${EXTRA_LIBS})

#cuda_add_executable(gframe_sssp src/utils/parser.cpp
#        src/utils/utils.cpp
#        src/groute/graphs/csr_graph.cpp
#        samples/framework/entry.cu
#        samples/framework/sssp.cu)
#target_link_libraries(gframe_sssp ${EXTRA_LIBS})

cuda_add_executable(exprs src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
        samples/exprs/pr_host.cpp
        samples/exprs/data_driven_unopt_pr.cu
        samples/exprs/topology_driven_unopt_pr.cu
        samples/exprs/hybrid_topodriven.cu
        samples/exprs/hybrid_datadriven.cu
        #        samples/exprs/dual.cu
        samples/exprs/main.cpp)
target_link_libraries(exprs ${EXTRA_LIBS})

#cuda_add_executable(pr_exprs src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
#        samples/pr_exprs/pr_host.cpp
#        samples/pr_exprs/data_driven_unopt_pr.cu
#        samples/pr_exprs/data_driven_append_warp.cu
#        samples/pr_exprs/data_driven_ctanp_pr.cu
#        samples/pr_exprs/data_driven_outlining_pr.cu
#        samples/pr_exprs/topology_driven_unopt_pr.cu
#        samples/pr_exprs/hybrid_driven.cu
#        samples/pr_exprs/main.cpp)
#target_link_libraries(pr_exprs ${EXTRA_LIBS})

#cuda_add_executable(cusha src/utils/parser.cpp src/utils/utils.cpp src/groute/graphs/csr_graph.cpp
#        samples/cusha/cusha.cu samples/cusha/main.cpp)
#target_link_libraries(cusha ${EXTRA_LIBS})
#cuda_add_executable(pbf samples/pbf/pbf_async.cu samples/pbf/main.cpp)
#target_link_libraries(pbf ${EXTRA_LIBS})


# Unit tests
#enable_testing()

#if(GTEST_FOUND)
#    message("gtest found")
#endif()
# GTest directory settings
#if(NOT GTEST_FOUND)
#  include_directories(
#    ${gtest_SOURCE_DIR}/include
#    ${gtest_SOURCE_DIR})
#  link_directories(${gtest_BINARY_DIR}/src)
#endif()

# Groute async component tests
#cuda_add_executable(async-tests test/async-tests/main.cpp
#                    test/async-tests/router_test.cu
#                    test/async-tests/dwl_test.cu
#                    test/async-tests/pc_queue_test.cu)
#
#target_link_libraries(async-tests gtest pthread dl ${EXTRA_LIBS})
#cuda_add_cublas_to_target(async-tests)

# Groute async component micro-benchmark tests
#cuda_add_executable(mb-tests test/micro-benchmarks/main.cpp
#                    test/micro-benchmarks/timed_kernel.cu test/micro-benchmarks/high_priority_copy.cu)
#
#target_link_libraries(mb-tests gtest pthread)
#cuda_add_cublas_to_target(mb-tests)
