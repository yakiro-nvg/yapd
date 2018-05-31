/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API yapd_gpu_t
yapd_gpu_new();

YAPD_API cl_program
yapd_gpu_load_program(
    yapd_gpu_t* gpu, const char* source);

YAPD_API void
yapd_gpu_sync(
    yapd_gpu_t* gpu);

YAPD_API void
yapd_gpu_release(
    yapd_gpu_t* gpu);

#ifdef __cplusplus
} // extern "C"
#endif
