/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API yapd_buffer_t
yapd_buffer_create(
    yapd_gpu_t* gpu, int bytes);

YAPD_API yapd_buffer_t
yapd_buffer_readonly(
    yapd_gpu_t* gpu, int bytes);

YAPD_API yapd_buffer_t
yapd_buffer_writeonly(
    yapd_gpu_t* gpu, int bytes);

YAPD_API void
yapd_buffer_reserve(
    yapd_buffer_t* buf, int bytes);

YAPD_API void
yapd_buffer_release(
    yapd_buffer_t* buf);

YAPD_API void
yapd_buffer_upload(
    yapd_buffer_t* buf, const uint8_t* data, int bytes);

YAPD_API void
yapd_buffer_upload_2d(
    yapd_buffer_t* buf, const uint8_t* data, int bytes,
    const yapd_size_t* sz, int stride);

YAPD_API void
yapd_buffer_download_sync(
    yapd_buffer_t* buf, uint8_t* data, int bytes);

YAPD_API void
yapd_buffer_download(
    yapd_buffer_t* buf, uint8_t* data, int bytes);

YAPD_API void
yapd_buffer_luv_from_rgb8uc4(
    yapd_buffer_t* buf, yapd_buffer_t* rgb, const yapd_size_t* sz);

YAPD_API void
yapd_tri_filter(
    yapd_alloc_t a, void* aud, int r, float** filter, int* bytes);

YAPD_API void
yapd_buffer_conv_tri_cols32f(
    yapd_buffer_t* dst, yapd_buffer_t* src, int dst_off,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_conv_tri_rows32f(
    yapd_buffer_t* buf, int dst_off, int src_off,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_conv_tri32f(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_conv_tri32fc4(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_conv_tri32fc8(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_conv_tri32fc16(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter);

YAPD_API void
yapd_buffer_resample32f(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm);

YAPD_API void
yapd_buffer_resample32fc4(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm);

YAPD_API void
yapd_buffer_resample32fc8(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm);

#ifdef __cplusplus
} // extern "C"
#endif
