/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>
#include <yapd/types.h>
#include <yapd/alloc.h>

#ifdef __cplusplus
extern "C" {
#endif

static YAPD_INLINE yapd_mat_t
yapd_mat_borrow(
    uint8_t* data, int w, int h, yapd_type_t type)
{
    yapd_mat_t m;
    m.a = yapd_alloc_null();
    m.aud = NULL;
    m.data = data;
    m.type = type;
    m.size.w = w;
    m.size.h = h;
    return m;
}

static YAPD_INLINE yapd_mat_t
yapd_mat_take(
    yapd_alloc_t a, void* aud,
    uint8_t* data, int w, int h, yapd_type_t type)
{
    yapd_mat_t m;
    m.a = a;
    m.aud = aud;
    m.data = data;
    m.type = type;
    m.size.w = w;
    m.size.h = h;
    return m;
}

static YAPD_INLINE yapd_mat_t
yapd_mat_new(
    yapd_alloc_t a, void* aud)
{
    yapd_mat_t m;
    m.a = a;
    m.aud = aud;
    m.data = NULL;
    return m;
}

static YAPD_INLINE void
yapd_mat_release(
    yapd_mat_t* mat)
{
    if (mat->data && !yapd_alloc_is_null(&mat->a)) {
        mat->a.dealloc(mat->aud, mat->data);
        mat->data = NULL;
        mat->size.w = 0;
        mat->size.h = 0;
    }
}

static YAPD_INLINE void
yapd_mat_create(
    yapd_mat_t* mat, int w, int h, yapd_type_t type)
{
    mat->size.w = w;
    mat->size.h = h;
    mat->type = type;
    if (w > 0 && h > 0) {
        mat->data = (uint8_t*)mat->a.alloc(
            mat->aud, w*h*yapd_type_size(type), YAPD_DEFAULT_ALIGN);
    }
}

static YAPD_INLINE int
yapd_mat_totals(
    const yapd_mat_t* mat)
{
    return mat->size.w*mat->size.h;
}

static YAPD_INLINE int
yapd_mat_bytes(
    const yapd_mat_t* mat)
{
    return yapd_mat_totals(mat)*yapd_type_size(mat->type);
}

#ifdef __cplusplus
} // extern "C"
#endif