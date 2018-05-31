# Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved.
cmake_minimum_required(VERSION 2.8)

project(yapd)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DYAPD_SHARED)

set(OCL_HEADERS_DIR ${CMAKE_BINARY_DIR}/cl_sources)
set(YAPD_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCL REQUIRED)

add_subdirectory(private)
add_subdirectory(servo)