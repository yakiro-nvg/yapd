/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/pyramid.h>

#include <yapd/gpu.h>
#include <yapd/buffer.h>
#include <yapd/channels.h>
#include <pyramid.cl.h>

enum { APX_REAL = -1 };

static void
get_scales(
    yapd_pyramid_t* p, int w, int h)
{
    static const float stbl[] = {
        0.0000f,
        0.0100f, 0.0200f, 0.0300f, 0.0400f, 0.0500f,
        0.0600f, 0.0700f, 0.0800f, 0.0900f, 0.1000f,
        0.1100f, 0.1200f, 0.1300f, 0.1400f, 0.1500f,
        0.1600f, 0.1700f, 0.1800f, 0.1900f, 0.2000f,
        0.2100f, 0.2200f, 0.2300f, 0.2400f, 0.2500f,
        0.2600f, 0.2700f, 0.2800f, 0.2900f, 0.3000f,
        0.3100f, 0.3200f, 0.3300f, 0.3400f, 0.3500f,
        0.3600f, 0.3700f, 0.3800f, 0.3900f, 0.4000f,
        0.4100f, 0.4200f, 0.4300f, 0.4400f, 0.4500f,
        0.4600f, 0.4700f, 0.4800f, 0.4900f, 0.5000f,
        0.5100f, 0.5200f, 0.5300f, 0.5400f, 0.5500f,
        0.5600f, 0.5700f, 0.5800f, 0.5900f, 0.6000f,
        0.6100f, 0.6200f, 0.6300f, 0.6400f, 0.6500f,
        0.6600f, 0.6700f, 0.6800f, 0.6900f, 0.7000f,
        0.7100f, 0.7200f, 0.7300f, 0.7400f, 0.7500f,
        0.7600f, 0.7700f, 0.7800f, 0.7900f, 0.8000f,
        0.8100f, 0.8200f, 0.8300f, 0.8400f, 0.8500f,
        0.8600f, 0.8700f, 0.8800f, 0.8900f, 0.9000f,
        0.9100f, 0.9200f, 0.9300f, 0.9400f, 0.9500f,
        0.9600f, 0.9700f, 0.9800f, 0.9900f, 1.0000f
    };
    float d0, d1;
    int i, j, nxt_oct;
    const float shrink = (float)p->channels->opts.shrink;
    const int num_approx = p->opts.num_approx;
    const int per_oct = p->opts.per_oct;
    const int oct_up = p->opts.oct_up;
    const int min_ds_w = p->opts.min_ds.w;
    const int min_ds_h = p->opts.min_ds.h;
    p->num_scales = (int)(per_oct*
        (oct_up + log2f(
            YAPD_MIN(w / (float)min_ds_w, h / (float)min_ds_h))) + 1);
    if (p->num_scales > p->cap_scales) { // not enough capacity
        p->a.dealloc(p->aud, p->approxes);
        p->approxes = (int*)p->a.alloc(
            p->aud, sizeof(int)*p->num_scales, YAPD_DEFAULT_ALIGN);

        p->a.dealloc(p->aud, p->scales);
        p->scales = (float*)p->a.alloc(
            p->aud, sizeof(float)*p->num_scales, YAPD_DEFAULT_ALIGN);

        p->a.dealloc(p->aud, p->scalesw);
        p->scalesw = (float*)p->a.alloc(
            p->aud, sizeof(float)*p->num_scales, YAPD_DEFAULT_ALIGN);

        p->a.dealloc(p->aud, p->scalesh);
        p->scalesh = (float*)p->a.alloc(
            p->aud, sizeof(float)*p->num_scales, YAPD_DEFAULT_ALIGN);

        p->a.dealloc(p->aud, p->data_sz);
        p->data_sz = (yapd_size_t*)p->a.alloc(
            p->aud, sizeof(yapd_size_t)*p->num_scales, YAPD_DEFAULT_ALIGN);
        for (i = 0; i < p->cap_scales; ++i) {
            yapd_buffer_release(p->data + i);
        }
        p->a.dealloc(p->aud, p->data);
        p->data = (yapd_buffer_t*)p->a.alloc(
            p->aud, sizeof(yapd_buffer_t)*p->num_scales, YAPD_DEFAULT_ALIGN);
        for (i = 0; i < p->num_scales; ++i) {
            p->data[i] = yapd_buffer_create(p->gpu, 0);
        }
        p->cap_scales = p->num_scales;
    }
    for (i = 0; i < p->num_scales; ++i) {
        p->scales[i] = powf(2, -(float)i / per_oct + oct_up);
    }
    if (w < h) {
        d0 = (float)w;
        d1 = (float)h;
    } else {
        d0 = (float)h;
        d1 = (float)w;
    }
    // set each scale s such that max(abs(round(sz*s / shrink)*shrink - sz*s))
    // is minimized without changing the smaller dim of sz(tricky algebra)
    for (i = 0; i < p->num_scales; ++i) {
        const float s = p->scales[i];
        // TODO: https://github.com/pdollar/toolbox/issues/26
        const float s0 = (rintf(d0*s / shrink)*shrink - 0.25f*shrink) / d0;
        const float s1 = (rintf(d0*s / shrink)*shrink + 0.25f*shrink) / d0;
        float min, x;
        for (j = 0; j < YAPD_STATIC_ARRAY_COUNT(stbl); ++j) {
            const float ss = stbl[i] * (s1 - s0) + s0;
            float es0 = d0 * ss; es0 = fabsf(es0 - rintf(es0 / shrink)*shrink);
            float es1 = d1 * ss; es1 = fabsf(es1 - rintf(es1 / shrink)*shrink);
            const float es = YAPD_MAX(es0, es1);
            if (j == 0 || es < min) {
                x = ss;
                min = es;
            }
        }
        p->scales[i] = x;
    }
    // remove redundant
    for (i = p->num_scales - 1; i > 0; --i) {
        if (yapd_fuzzy_equals(p->scales[i], p->scales[i - 1], 0.001f)) {
            p->scales[i - 1] = p->scales[--p->num_scales];
        }
    }
    // bubble sort
    while (TRUE) {
        int done = TRUE;
        for (i = p->num_scales; i > 0; --i) {
            if (p->scales[i - 1] < p->scales[i]) {
                const float tmp = p->scales[i];
                p->scales[i] = p->scales[i - 1];
                p->scales[i - 1] = tmp;
                done = FALSE;
            }
        }
        if (done) break;
    }
    // exact
    for (i = 0; i < p->num_scales; ++i) {
        const float s = p->scales[i];
        p->scalesw[i] = rintf(w*s / shrink)*shrink / w;
        p->scalesh[i] = rintf(h*s / shrink)*shrink / h;
    }
    // real/approx
    memset(p->approxes, 0, sizeof(int)*p->num_scales);
    for (i = 0; i < p->num_scales; i += num_approx + 1) {
        p->approxes[i] = APX_REAL;
    }
    for (i = 0; i < p->num_scales; ++i) {
        if (p->approxes[i] == APX_REAL) {
            j = i; nxt_oct = j + num_approx / 2 + num_approx % 2;
        } else {
            p->approxes[i] = i <= nxt_oct ? j : (j + num_approx + 1);
            if (p->approxes[i] >= p->num_scales) {
                p->approxes[i] = j;
            }
        }
    }
}

static void
reserve_buffers(
    yapd_pyramid_t* p, int w, int h)
{
    assert(w > 0 && h > 0);
    yapd_buffer_reserve(&p->tmp, sizeof(cl_float4)*w*h*3);
    yapd_buffer_reserve(&p->img, sizeof(cl_float4)*w*h*3);
    yapd_buffer_reserve(&p->small, sizeof(cl_float4)*w*h*3);
}

static void
compute_real(
    yapd_pyramid_t* p, const yapd_size_t* sz, float s)
{
    yapd_size_t small_sz;
    yapd_buffer_t *small;
    const int shrink = p->channels->opts.shrink;
    small_sz.w = ((int)rintf(sz->w*s / shrink))*shrink;
    small_sz.h = ((int)rintf(sz->h*s / shrink))*shrink;
    if (yapd_size_equals(&small_sz, sz)) {
        small = &p->img;
    } else {
        small = &p->small;
        yapd_buffer_resample32fc4(small, &small_sz, &p->img, sz, 1.0f);
    }
    yapd_channels_prepare(
        p->channels, &small_sz, &p->color, &p->mag, &p->hist);
    yapd_channels_compute(
        p->channels, small, &p->tmp, &p->color, &p->mag, &p->hist);
}

static void
conpad(
    yapd_buffer_t* dst, const yapd_size_t* sz, const yapd_size_t* pad_sz,
    yapd_buffer_t* color, yapd_buffer_t* mag, yapd_buffer_t* hist)
{
    cl_int err;
    yapd_gpu_t* gpu = dst->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { pad_sz->w, pad_sz->h, 1 };
    const yapd_size_t pad = {
        (pad_sz->w - sz->w) / 2,
        (pad_sz->h - sz->h) / 2
    };
    yapd_gpu_pyramid_ctx_t* c = &gpu->pyramid;

    YAPD_STATIC_ASSERT(sizeof(yapd_feature_t) == sizeof(cl_float16));

    assert(gpu == color->gpu && gpu == mag->gpu && gpu == hist->gpu);
    assert(dst->bytes >= sizeof(cl_float16)*pad_sz->w*pad_sz->h);
    assert(color->bytes >= sizeof(cl_float4)*sz->w*sz->h);
    assert(mag->bytes >= sizeof(float)*sz->w*sz->h);
    assert(hist->bytes >= sizeof(cl_float8)*sz->w*sz->h);

    err = clSetKernelArg(c->pyramid_conpad, 0, sizeof(cl_mem), &dst->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->pyramid_conpad, 1, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->pyramid_conpad, 2, sizeof(cl_int2), &pad);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->pyramid_conpad, 3, sizeof(cl_mem), &color->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->pyramid_conpad, 4, sizeof(cl_mem), &mag->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->pyramid_conpad, 5, sizeof(cl_mem), &hist->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->pyramid_conpad, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static const yapd_pyramid_opts_t default_opts = {
    .pad = { 0, 0 },
    .min_ds = { 16, 16 },
    .num_approx = 0,
    .per_oct = 8,
    .oct_up = 0,
    .smooth = 0
};

void
yapd_gpu_setup_pyramid(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_pyramid_ctx_t* p = &gpu->pyramid;
    p->program = yapd_gpu_load_program(gpu, pyramid_cl);
    p->pyramid_conpad = clCreateKernel(p->program, "pyramid_conpad", &err);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_pyramid(
    yapd_gpu_t* gpu)
{
    yapd_gpu_pyramid_ctx_t* p = &gpu->pyramid;
    clReleaseKernel(p->pyramid_conpad);
    clReleaseProgram(p->program);
}

const yapd_pyramid_opts_t*
yapd_pyramid_default_opts()
{
    return &default_opts;
}

yapd_pyramid_t
yapd_pyramid_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu,
    yapd_channels_t* channels, const yapd_pyramid_opts_t* opts)
{
    yapd_pyramid_t p;
    int bytes;

    p.a= a;
    p.aud = aud;
    p.gpu = gpu;
    p.channels = channels;
    if (!opts) opts = &default_opts;
    p.opts = *opts;

    p.last_sz.w = 0;
    p.last_sz.h = 0;
    p.cap_scales = 0;
    p.approxes = NULL;
    p.scales = NULL;
    p.scalesw = NULL;
    p.scalesh = NULL;
    p.data_sz = NULL;
    p.data = NULL;
    p.tmp = yapd_buffer_create(gpu, 0);
    p.img = yapd_buffer_create(gpu, 0);
    p.small = yapd_buffer_create(gpu, 0);
    p.color = yapd_buffer_create(gpu, 0);
    p.mag = yapd_buffer_create(gpu, 0);
    p.hist = yapd_buffer_create(gpu, 0);
    p.apx_color = yapd_buffer_create(gpu, 0);
    p.apx_mag = yapd_buffer_create(gpu, 0);
    p.apx_hist = yapd_buffer_create(gpu, 0);

    assert(p.opts.num_approx == -1 || p.opts.num_approx >= 0);
    if (p.opts.num_approx == -1) {
        p.opts.num_approx = p.opts.per_oct - 1;
    }

    yapd_tri_filter(
        a, aud, opts->smooth, &p.smooth_filter_host, &bytes);
    p.smooth_filter = yapd_buffer_readonly(gpu, bytes);
    yapd_buffer_upload(
        &p.smooth_filter, (uint8_t*)p.smooth_filter_host, bytes);

    return p;
}

void
yapd_pyramid_release(
    yapd_pyramid_t* p)
{
    int i;

    p->a.dealloc(p->aud, p->smooth_filter_host);
    p->smooth_filter_host = NULL;
    yapd_buffer_release(&p->smooth_filter);

    yapd_buffer_release(&p->tmp);
    yapd_buffer_release(&p->img);
    yapd_buffer_release(&p->color);
    yapd_buffer_release(&p->mag);
    yapd_buffer_release(&p->hist);
    yapd_buffer_release(&p->apx_color);
    yapd_buffer_release(&p->apx_mag);
    yapd_buffer_release(&p->apx_hist);
    if (p->cap_scales > 0) {
        p->a.dealloc(p->aud, p->approxes);
        p->approxes = NULL;
        p->a.dealloc(p->aud, p->scales);
        p->scales = NULL;
        p->a.dealloc(p->aud, p->scalesw);
        p->scalesw = NULL;
        p->a.dealloc(p->aud, p->scalesh);
        p->scalesh = NULL;
        p->a.dealloc(p->aud, p->data_sz);
        p->data_sz = NULL;
        for (i = 0; i < p->cap_scales; ++i) {
            yapd_buffer_release(p->data + i);
        }
        p->a.dealloc(p->aud, p->data);
        p->data = NULL;
        p->cap_scales = 0;
    }
}

void
yapd_pyramid_compute(
    yapd_pyramid_t* p, const yapd_mat_t* img,
    float lambda_color, float lambda_mag, float lambda_hist)
{
    int i, lr = -1;
    yapd_buffer_t *color, *mag, *hist;
    yapd_size_t small_sz, lr_sz, pad_sz;
    const int shrink = p->channels->opts.shrink;
    assert(img->size.w > 0 && img->size.h > 0 && img->type == YAPD_8UC4);
    fesetround(FE_TONEAREST);
    // prepare resources
    if (!yapd_size_equals(&img->size, &p->last_sz)) {
        get_scales(p, img->size.w, img->size.h);
        reserve_buffers(p, img->size.w, img->size.h);
        p->last_sz = img->size;
    }
    // convert color
    yapd_buffer_upload_2d(
        &p->tmp, img->data, sizeof(cl_uchar4)*img->size.w*img->size.h,
        &img->size, sizeof(cl_uchar4)*img->size.w);
    yapd_buffer_luv_from_rgb8uc4(
        &p->img, &p->tmp, &img->size);
    // compute pyramid
    for (i = 0; i < p->num_scales; ++i) {
        const float s = p->scales[i];
        if (p->approxes[i] == APX_REAL) {
            if (lr != i) {
                compute_real(p, &img->size, s);
                p->data_sz[i] = p->channels->data_sz;
                lr_sz = p->channels->data_sz;
                lr = i;
            }
            color = &p->color;
            mag = &p->mag;
            hist = &p->hist;
        } else { // approximated
            int small_totals;
            const int real = p->approxes[i];
            float ratio, rs = p->scales[real];
            small_sz.w = (int)rintf(img->size.w*s / shrink);
            small_sz.h = (int)rintf(img->size.h*s / shrink);
            small_totals = small_sz.w*small_sz.h;
            if (lr != real) {
                compute_real(p, &img->size, p->scales[real]);
                p->data_sz[real] = p->channels->data_sz;
                lr_sz = p->channels->data_sz;
                lr = real;
            }
            p->data_sz[i] = small_sz;
            ratio = powf(s / rs, -lambda_color);
            yapd_buffer_reserve(
                &p->apx_color, sizeof(cl_float4)*small_totals);
            yapd_buffer_resample32fc4(
                &p->apx_color, &small_sz, &p->color, &lr_sz, ratio);
            ratio = powf(s / rs, -lambda_mag);
            yapd_buffer_reserve(
                &p->apx_mag, sizeof(float)*small_totals);
            yapd_buffer_resample32f(
                &p->apx_mag, &small_sz, &p->mag, &lr_sz, ratio);
            ratio = powf(s / rs, -lambda_hist);
            yapd_buffer_reserve(
                &p->apx_hist, sizeof(cl_float8)*small_totals);
            yapd_buffer_resample32fc8(
                &p->apx_hist, &small_sz, &p->hist, &lr_sz , ratio);
            color = &p->apx_color;
            mag = &p->apx_mag;
            hist = &p->apx_hist;
        }
        // concat and pad to single float16 vector
        pad_sz.w = p->data_sz[i].w + (p->opts.pad.w / shrink) * 2;
        pad_sz.h = p->data_sz[i].h + (p->opts.pad.h / shrink) * 2;
        yapd_buffer_reserve(
            p->data + i, sizeof(yapd_feature_t)*pad_sz.w*pad_sz.h);
        conpad(p->data + i, p->data_sz + i, &pad_sz, color, mag, hist);
        if (p->opts.smooth > 0) {
            yapd_buffer_reserve(
                &p->tmp, sizeof(yapd_feature_t)*pad_sz.w*pad_sz.h);
            YAPD_STATIC_ASSERT(sizeof(yapd_feature_t) == sizeof(cl_float16));
            yapd_buffer_conv_tri32fc16(
                p->data + i, &p->tmp, &pad_sz,
                p->opts.smooth, &p->smooth_filter);
        }
        p->data_sz[i] = pad_sz;
    }
}