cmake_minimum_required(VERSION 3.9)
project(opencv_mnist)

set(CMAKE_CXX_STANDARD 11)

add_executable(opencv_mnist main.cpp mnist.cpp mnist.h)

find_package(OPENCV REQUIRED)

target_link_libraries(opencv_mnist ${OpenCV_LIBS})
