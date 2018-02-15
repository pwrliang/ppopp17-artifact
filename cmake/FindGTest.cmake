# - Try to find GTEST
#
# The following variables are optionally searched for defaults
#  GTEST_ROOT_DIR:            Base directory where all GTEST components are found
#
# The following are set after configuration is done:
#  GTEST_FOUND
#  GTEST_INCLUDE_DIRS
#  GTEST_LIBRARIES
#  GTEST_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GTEST_ROOT_DIR "" CACHE PATH "Folder contains Google GTEST")

if(WIN32)
    find_path(GTEST_INCLUDE_DIR gtest/test.h
        PATHS ${GTEST_ROOT_DIR}/src/windows)
else()
    find_path(GTEST_INCLUDE_DIR gtest/test.h
        PATHS ${GTEST_ROOT_DIR})
endif()

if(MSVC)
    find_library(GTEST_LIBRARY_RELEASE libGTEST_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(GTEST_LIBRARY_DEBUG libGTEST_static
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES Debug)

    set(GTEST_LIBRARY optimized ${GTEST_LIBRARY_RELEASE} debug ${GTEST_LIBRARY_DEBUG})
else()
    find_library(GTEST_LIBRARY GTEST
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
endif()

find_package_handle_standard_args(GTEST DEFAULT_MSG GTEST_INCLUDE_DIR GTEST_LIBRARY)

if(GTEST_FOUND)
  set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
  set(GTEST_LIBRARIES ${GTEST_LIBRARY})
  message(STATUS "Found GTEST    (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARY})")
  mark_as_advanced(GTEST_ROOT_DIR GTEST_LIBRARY_RELEASE GTEST_LIBRARY_DEBUG
                                 GTEST_LIBRARY GTEST_INCLUDE_DIR)
endif()
