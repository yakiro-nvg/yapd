# Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved.
find_package(PythonInterp REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${YAPD_ROOT}/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OCL_HEADERS_DIR})

file(GLOB_RECURSE HEADERS ${YAPD_ROOT}/inc/*.h)
file(GLOB_RECURSE PRI_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE PRI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE OCL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/opencl/*.cl)

add_definitions(-DYAPD_EXPORT)

add_library(yapd SHARED ${HEADERS} ${PRI_HEADERS} ${PRI_SOURCES} ${OCL_SOURCES})
target_link_libraries(yapd ${OpenCL_LIBRARIES})

file(MAKE_DIRECTORY ${OCL_HEADERS_DIR})
foreach(OCL_SOURCE ${OCL_SOURCES})
	get_filename_component(OCL_FILENAME ${OCL_SOURCE} NAME)
	add_custom_command(
		OUTPUT ${OCL_HEADERS_DIR}/${OCL_FILENAME}.h
		COMMAND ${PYTHON_EXECUTABLE}
		ARGS ${CMAKE_CURRENT_SOURCE_DIR}/xxd.py ${OCL_SOURCE} ${OCL_HEADERS_DIR}/${OCL_FILENAME}.h
		COMMENT "Creating opencl header for ${OCL_FILENAME}")
	set(OCL_HEADERS ${OCL_HEADERS} ${OCL_HEADERS_DIR}/${OCL_FILENAME}.h)
endforeach()
add_custom_command(
	OUTPUT ${OCL_HEADERS_DIR}/color_consts.h
	COMMAND ${PYTHON_EXECUTABLE}
	ARGS ${CMAKE_CURRENT_SOURCE_DIR}/color_consts.py ${OCL_HEADERS_DIR}/color_consts.h
	COMMENT "Creating color constants header")

add_custom_target(
	yapd_prebuild DEPENDS
	${OCL_HEADERS}
	${OCL_HEADERS_DIR}/color_consts.h)
add_dependencies(yapd yapd_prebuild)
