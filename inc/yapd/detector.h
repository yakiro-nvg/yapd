/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API yapd_detector_t
yapd_detector_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu);

YAPD_API void
yapd_detector_release(
    yapd_detector_t* d);

YAPD_INLINE int
yapd_detector_ready(
    yapd_detector_t* d)
{
    return d->num_weaks > 0;
}

YAPD_API void
yapd_detector_classifier(
    yapd_detector_t* d, int copy,
    int num_weaks, int depth, int shrink,
    const yapd_size_t* win_sz, const yapd_size_t* org_win,
    yapd_mat_t* thrs, yapd_mat_t* fids, yapd_mat_t* hs);

YAPD_API yapd_mat_t
yapd_detector_predict(
    yapd_alloc_t a, void* aud,
    yapd_detector_t* d, yapd_pyramid_t* p,
    int stride, float casc_thr);

#ifdef __cplusplus
} // extern "C"
#endif
