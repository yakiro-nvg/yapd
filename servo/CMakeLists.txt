# Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved.
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${YAPD_ROOT}/inc)

add_executable(servo ${HEADERS} ${SOURCES})
target_link_libraries(servo yapd ${OpenCV_LIBRARIES})
set_property(TARGET servo PROPERTY CXX_STANDARD 11)

if(WIN32)
	target_link_libraries(servo wsock32 ws2_32)
endif()
