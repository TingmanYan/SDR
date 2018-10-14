cmake_minimum_required(VERSION 3.8)
project(fdr)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if(MSVC)
    set(OpenCV_DIR "C:/Program Files/opencv_3_4/opencv/build")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3.4 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 11)

add_executable(fdr main.cpp FastDR.cpp FastDR.h SurfaceFitting.cpp SurfaceFitting.h ArgsParser.h Evaluator.h Parameters.h TimeStamper.h Utilities.hpp)
target_link_libraries(fdr ${OpenCV_LIBS} Eigen3::Eigen GCoptimization Superpixel)
