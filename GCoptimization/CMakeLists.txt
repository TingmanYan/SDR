cmake_minimum_required(VERSION 3.7)
project(fdr)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_library(GCoptimization STATIC GCoptimization.cpp GCoptimization.h LinkedBlockList.cpp LinkedBlockList.h graph.cpp graph.h maxflow.cpp energy.h block.h)
