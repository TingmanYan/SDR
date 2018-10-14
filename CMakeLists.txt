cmake_minimum_required(VERSION 3.8)
project(fdr)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/fdr ${PROJECT_SOURCE_DIR}/GCoptimization ${PROJECT_SOURCE_DIR}/Superpixel)

link_directories(${PROJECT_BINARY_DIR}/libs)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(Superpixel)
add_subdirectory(GCoptimization)
add_subdirectory(fdr)

