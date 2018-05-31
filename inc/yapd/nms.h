/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

YAPD_API void
yapd_nms(
    yapd_alloc_t a, void* aud, yapd_mat_t* bbs,
    float casc_thr, float overlap, int greedy);

#ifdef __cplusplus
} // extern "C"
#endif
