/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>

#ifdef __cplusplus
extern "C" {
#endif

static YAPD_INLINE yapd_alloc_t
yapd_alloc_null()
{
    yapd_alloc_t a;
    a.alloc = NULL;
    return a;
}

static YAPD_INLINE int
yapd_alloc_is_null(
    yapd_alloc_t* a)
{
    return a->alloc == NULL;
}

YAPD_API void
yapd_malloc_new(
    yapd_malloc_t* a);

YAPD_API void
yapd_malloc_release(
    yapd_malloc_t* a);

YAPD_API void
yapd_scratch_new(
    yapd_scratch_t* a, yapd_alloc_t backing, void* backing_ud, int cap);

YAPD_API void
yapd_scratch_release(
    yapd_scratch_t* a);

#ifdef __cplusplus
} // extern "C"
#endif
