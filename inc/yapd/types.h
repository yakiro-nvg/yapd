/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#pragma once

#include <yapd/platform.h>

// Allocator interface.
typedef struct yapd_alloc_s {
    void*(*alloc)(void* ud, int sz, int align);
    void(*dealloc)(void* ud, void* p);
} yapd_alloc_t;

typedef struct yapd_malloc_s {
    yapd_alloc_t aif;
    uint32_t allocated;
} yapd_malloc_t;

typedef struct yapd_scratch_s {
    yapd_alloc_t aif;
    yapd_alloc_t backing;
    void* backing_ud;
    uint8_t* begin;
    uint8_t* end;
    uint8_t* allocate;
    uint8_t* free;
} yapd_scratch_t;

typedef enum {
    YAPD_8U,
    YAPD_8UC4,
    YAPD_32S,
    YAPD_32F,
    YAPD_32FC4,
    YAPD_32FC8,
    YAPD_32FC16
} yapd_type_t;

static YAPD_INLINE int
yapd_type_channels(yapd_type_t type)
{
    switch (type) {
    case YAPD_8U:
        return 1;
    case YAPD_8UC4:
        return 4;
    case YAPD_32S:
        return 1;
    case YAPD_32F:
        return 1;
    case YAPD_32FC4:
        return 4;
    case YAPD_32FC8:
        return 8;
    case YAPD_32FC16:
        return 16;
    default:
        assert(!"bad type");
        return 0;
    }
}

static YAPD_INLINE int
yapd_type_depth(yapd_type_t type)
{
    switch (type) {
    case YAPD_8U:
        return 1;
    case YAPD_8UC4:
        return 1;
    case YAPD_32S:
        return 4;
    case YAPD_32F:
        return 4;
    case YAPD_32FC4:
        return 4;
    case YAPD_32FC8:
        return 4;
    case YAPD_32FC16:
        return 4;
    default:
        assert(!"bad type");
        return 0;
    }
}

static YAPD_INLINE int
yapd_type_size(yapd_type_t type)
{
    return yapd_type_channels(type)*yapd_type_depth(type);
}

typedef struct yapd_size_s {
    int w, h;
} yapd_size_t;

static YAPD_INLINE int
yapd_size_equals(
    const yapd_size_t* a, const yapd_size_t* b)
{
    return a->w == b->w && a->h == b->h;
}

typedef struct yapd_mat_s {
    yapd_alloc_t a;
    void* aud;
    uint8_t* data;
    yapd_type_t type;
    yapd_size_t size;
} yapd_mat_t;

typedef struct yapd_buffer_s {
    struct yapd_gpu_s* gpu;
    int bytes;
    int flags;
    cl_mem mem;
} yapd_buffer_t;

typedef struct yapd_gpu_color_ctx_s {
    cl_program program;
    cl_kernel luv_from_rgb8uc4;
    yapd_buffer_t ltable;
} yapd_gpu_color_ctx_t;

typedef struct yapd_gpu_resample_ctx_s {
    cl_program program;
    cl_kernel resample32f;
    cl_kernel resample32fc4;
    cl_kernel resample32fc8;
} yapd_gpu_resample_ctx_t;

typedef struct yapd_gpu_convolution_ctx_s {
    cl_program program;
    cl_kernel conv_tri_cols32f;
    cl_kernel conv_tri_rows32f;
    cl_kernel conv_tri32f;
    cl_kernel conv_tri32fc4;
    cl_kernel conv_tri32fc8;
    cl_kernel conv_tri32fc16;
} yapd_gpu_convolution_ctx_t;

typedef struct yapd_gpu_gradient_ctx_s {
    cl_program program;
    cl_kernel grad_mag32fc4;
    cl_kernel grad_mag_norm;
    cl_kernel grad_scale_angle;
    cl_kernel grad_hist;
} yapd_gpu_gradient_ctx_t;

typedef struct yapd_gpu_pyramid_ctx_s {
    cl_program program;
    cl_kernel pyramid_conpad;
} yapd_gpu_pyramid_ctx_t;

typedef struct yapd_gpu_detector_ctx_s {
    cl_program program;
    cl_kernel early_reject;
    cl_kernel early_scan;
    cl_kernel early_prefix_sum;
    cl_kernel early_bbs;
    cl_kernel predict;
    cl_kernel predict_sum;
} yapd_gpu_detector_ctx_t;

typedef struct yapd_gpu_s {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev_ids[1];
    yapd_gpu_color_ctx_t color;
    yapd_gpu_resample_ctx_t resample;
    yapd_gpu_convolution_ctx_t convolution;
    yapd_gpu_gradient_ctx_t gradient;
    yapd_gpu_pyramid_ctx_t pyramid;
    yapd_gpu_detector_ctx_t detector;
} yapd_gpu_t;

typedef struct yapd_channels_opts_s {
    int shrink;
    struct color_s {
        int smooth;
    } color;
    struct grad_mag_s {
        int norm_radius;
        float norm_const;
    } grad_mag;
    struct grad_hist_s {
        int bin_size;
        int num_orients;
    } grad_hist;
} yapd_channels_opts_t;

typedef struct yapd_channels_s {
    yapd_alloc_t a;
    void* aud;
    yapd_gpu_t* gpu;
    yapd_channels_opts_t opts;
    float* smooth_filter_host;
    yapd_buffer_t smooth_filter;
    float* mag_norm_filter_host;
    yapd_buffer_t mag_norm_filter;
    yapd_size_t capacity;
    yapd_size_t crop_sz;
    yapd_size_t hist_sz;
    yapd_size_t data_sz;
    yapd_buffer_t mag;
    yapd_buffer_t angle;
    yapd_buffer_t hist;
} yapd_channels_t;

typedef cl_float16 yapd_feature_t;
enum { YAPD_FEATURE_CHANNELS = sizeof(yapd_feature_t) / sizeof(float) };

typedef struct yapd_pyramid_opts_s {
    yapd_size_t pad;
    yapd_size_t min_ds;
    int num_approx;
    int per_oct;
    int oct_up;
    int smooth;
} yapd_pyramid_opts_t;

typedef struct yapd_pyramid_s {
    yapd_alloc_t a;
    void* aud;
    yapd_gpu_t* gpu;
    yapd_channels_t* channels;
    yapd_pyramid_opts_t opts;
    float* smooth_filter_host;
    yapd_buffer_t smooth_filter;
    yapd_size_t last_sz;
    int cap_scales;
    int num_scales;
    int* approxes;
    float* scales;
    float* scalesw;
    float* scalesh;
    yapd_size_t* data_sz;
    yapd_buffer_t* data;
    yapd_buffer_t tmp;
    yapd_buffer_t img;
    yapd_buffer_t small;
    yapd_buffer_t color;
    yapd_buffer_t mag;
    yapd_buffer_t hist;
    yapd_buffer_t apx_color;
    yapd_buffer_t apx_mag;
    yapd_buffer_t apx_hist;
} yapd_pyramid_t;

typedef struct yapd_detector_s {
    yapd_alloc_t a;
    void* aud;
    yapd_gpu_t* gpu;
    int dirty;
    int num_weaks;
    int depth;
    yapd_size_t win_sz;
    yapd_size_t org_win;
    int shrink;
    yapd_mat_t thrs_host;
    yapd_mat_t fids_host;
    yapd_mat_t hs_host;
    yapd_buffer_t thrs;
    yapd_buffer_t fids;
    yapd_buffer_t hs;
    int num_scales;
    yapd_size_t* sizes;
    int** cids_host;
    yapd_buffer_t* cids;
    yapd_buffer_t dsz;
    yapd_buffer_t out;
    yapd_buffer_t idx;
    yapd_buffer_t len;
    yapd_buffer_t sum;
    yapd_buffer_t bbs;
    yapd_buffer_t hss;
    yapd_buffer_t tmp;
} yapd_detector_t;

enum { YAPD_DETECTOR_TREE_NODES = 8 };