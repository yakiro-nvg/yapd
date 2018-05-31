/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#ifdef _MSC_VER
#define YAPD_MSVC _MSC_VER
#define YAPD_STATIC_ASSERT(c) typedef char _astatic_assertion[(c) ? 1 : -1]
#define _CRT_SECURE_NO_WARNINGS
#elif defined(__clang__)
#define YAPD_CLANG (((__clang_major__)*100) + \
    (__clang_minor__*10) + \
     __clang_patchlevel__)
#define YAPD_STATIC_ASSERT(c) _Static_assert(c, "failed")
#elif defined(__GNUC__)
#define YAPD_GNUC (((__GNUC__)*100) + \
    (__GNUC_MINOR__*10) + \
     __GNUC_PATCHLEVEL__)
#define YAPD_STATIC_ASSERT(c) typedef char _astatic_assertion[(c) ? 1 : -1]
#else
#   error "unknown compiler"
#endif

#if defined(__WIN32__) || defined(_WIN32)
#define YAPD_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <WinSock2.h>
#elif defined(__APPLE_CC__)
#define YAPD_APPLE
#else
#define YAPD_LINUX
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__arm64__)
#define YAPD_64BIT
YAPD_STATIC_ASSERT(sizeof(void*) == 8);
#else
#define YAPD_32BIT
YAPD_STATIC_ASSERT(sizeof(void*) == 4);
#endif

#ifdef YAPD_MSVC
#define YAPD_INLINE __inline
#elif defined(YAPD_CLANG) || defined(YAPD_GNUC)
#define YAPD_INLINE inline
#endif

#ifndef TRUE
#define TRUE  1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL 0
#endif

#ifndef YAPD_SHARED
#define YAPD_API
#elif defined(YAPD_WINDOWS)
#    ifndef YAPD_EXPORT
#        define YAPD_API __declspec(dllimport)
#    else
#        define YAPD_API __declspec(dllexport)
#    endif
#else
#define YAPD_API
#endif

#define YAPD_UNUSED(x) ((void)x)

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef YAPD_APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define YAPD_STATIC_ARRAY_COUNT(arr) (sizeof(arr)/sizeof(arr[0]))

#define YAPD_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define YAPD_MAX(x, y) (((x) > (y)) ? (x) : (y))

// compare float with tolerance.
static YAPD_INLINE int
yapd_fuzzy_equals(float a, float b, float p)
{
    return (a - p) < b && (a + p) > b;
}

#define YAPD_DEFAULT_ALIGN 4