/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API const yapd_pyramid_opts_t*
yapd_pyramid_default_opts();

YAPD_API yapd_pyramid_t
yapd_pyramid_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu,
    yapd_channels_t* channels, const yapd_pyramid_opts_t* opts);

YAPD_API void
yapd_pyramid_release(
    yapd_pyramid_t* p);

YAPD_API void
yapd_pyramid_compute(
    yapd_pyramid_t* p, const yapd_mat_t* img,
    float lambda_color, float lambda_mag, float lambda_hist);

#ifdef __cplusplus
} // extern "C"
#endif
