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

# Utility rule file for NightlyMemoryCheck.

# Include the progress variables for this target.
include deps/glog/CMakeFiles/NightlyMemoryCheck.dir/progress.make

deps/glog/CMakeFiles/NightlyMemoryCheck:
	cd /home/liang/groute-dev/cmake-build-release/deps/glog && /opt/clion-2017.3.2/bin/cmake/bin/ctest -D NightlyMemoryCheck

NightlyMemoryCheck: deps/glog/CMakeFiles/NightlyMemoryCheck
NightlyMemoryCheck: deps/glog/CMakeFiles/NightlyMemoryCheck.dir/build.make

.PHONY : NightlyMemoryCheck

# Rule to build all files generated by this target.
deps/glog/CMakeFiles/NightlyMemoryCheck.dir/build: NightlyMemoryCheck

.PHONY : deps/glog/CMakeFiles/NightlyMemoryCheck.dir/build

deps/glog/CMakeFiles/NightlyMemoryCheck.dir/clean:
	cd /home/liang/groute-dev/cmake-build-release/deps/glog && $(CMAKE_COMMAND) -P CMakeFiles/NightlyMemoryCheck.dir/cmake_clean.cmake
.PHONY : deps/glog/CMakeFiles/NightlyMemoryCheck.dir/clean

deps/glog/CMakeFiles/NightlyMemoryCheck.dir/depend:
	cd /home/liang/groute-dev/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liang/groute-dev /home/liang/groute-dev/deps/glog /home/liang/groute-dev/cmake-build-release /home/liang/groute-dev/cmake-build-release/deps/glog /home/liang/groute-dev/cmake-build-release/deps/glog/CMakeFiles/NightlyMemoryCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : deps/glog/CMakeFiles/NightlyMemoryCheck.dir/depend

