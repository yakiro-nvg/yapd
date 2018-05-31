/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API const yapd_channels_opts_t*
yapd_channels_default_opts();

YAPD_API yapd_channels_t
yapd_channels_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu,
    int cap_w, int cap_h, const yapd_channels_opts_t* opts);

YAPD_API void
yapd_channels_release(
    yapd_channels_t* c);

YAPD_API void
yapd_channels_prepare(
    yapd_channels_t* c, const yapd_size_t* sz,
    yapd_buffer_t* color, yapd_buffer_t* mag, yapd_buffer_t* hist);

YAPD_API void
yapd_channels_compute(
    yapd_channels_t* c, yapd_buffer_t* img,
    yapd_buffer_t* tmp, yapd_buffer_t* color,
    yapd_buffer_t* mag, yapd_buffer_t* hist);

#ifdef __cplusplus
} // extern "C"
#endif
