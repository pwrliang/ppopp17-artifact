# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2017.3.2/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2017.3.2/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liang/groute-dev

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liang/groute-dev/cmake-build-release

# Utility rule file for NightlyTest.

# Include the progress variables for this target.
include deps/gflags/CMakeFiles/NightlyTest.dir/progress.make

deps/gflags/CMakeFiles/NightlyTest:
	cd /home/liang/groute-dev/cmake-build-release/deps/gflags && /opt/clion-2017.3.2/bin/cmake/bin/ctest -D NightlyTest

NightlyTest: deps/gflags/CMakeFiles/NightlyTest
NightlyTest: deps/gflags/CMakeFiles/NightlyTest.dir/build.make

.PHONY : NightlyTest

# Rule to build all files generated by this target.
deps/gflags/CMakeFiles/NightlyTest.dir/build: NightlyTest

.PHONY : deps/gflags/CMakeFiles/NightlyTest.dir/build

deps/gflags/CMakeFiles/NightlyTest.dir/clean:
	cd /home/liang/groute-dev/cmake-build-release/deps/gflags && $(CMAKE_COMMAND) -P CMakeFiles/NightlyTest.dir/cmake_clean.cmake
.PHONY : deps/gflags/CMakeFiles/NightlyTest.dir/clean

deps/gflags/CMakeFiles/NightlyTest.dir/depend:
	cd /home/liang/groute-dev/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liang/groute-dev /home/liang/groute-dev/deps/gflags /home/liang/groute-dev/cmake-build-release /home/liang/groute-dev/cmake-build-release/deps/gflags /home/liang/groute-dev/cmake-build-release/deps/gflags/CMakeFiles/NightlyTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : deps/gflags/CMakeFiles/NightlyTest.dir/depend

