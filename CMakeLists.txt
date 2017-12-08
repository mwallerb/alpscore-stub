cmake_minimum_required(VERSION 3.0)
project(alpscore-stub CXX)

find_package(ALPSCore REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(main 
               main.cc
	       )
target_link_libraries(main ${ALPSCore_LIBRARIES})

