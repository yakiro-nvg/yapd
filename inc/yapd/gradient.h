/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API void
yapd_gradient_mag32fc4(
    yapd_buffer_t* img, const yapd_size_t* sz,
    yapd_buffer_t* mag, yapd_buffer_t* angle);

YAPD_API void
yapd_gradient_mag_norm(
    yapd_buffer_t* mag, const yapd_size_t* sz, yapd_buffer_t* tmp,
    float norm_const, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_gradient_scale_angle(
    yapd_buffer_t* angle, const yapd_size_t* sz, float scale);

YAPD_API void
yapd_gradient_hist(
    yapd_buffer_t* hist, yapd_buffer_t* mag, yapd_buffer_t* angle,
    const yapd_size_t* sz, int bin_size, int num_orients);

#ifdef __cplusplus
} // extern "C"
#endif
