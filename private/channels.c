/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/channels.h>

#include <yapd/matrix.h>
#include <yapd/buffer.h>
#include <yapd/gradient.h>

static void
release_buffers(
    yapd_channels_t* c)
{
    yapd_buffer_release(&c->mag);
    yapd_buffer_release(&c->angle);
    yapd_buffer_release(&c->hist);
    c->capacity.w = 0;
    c->capacity.h = 0;
}

static void
reserve_buffers(
    yapd_channels_t* c, int w, int h)
{
    assert(w > 0 && h > 0);
    // crop imgage so divisible by shrink
    int shrink = c->opts.shrink;
    c->crop_sz.w = w - (w % shrink);
    c->crop_sz.h = h - (h % shrink);
    c->data_sz.w = c->crop_sz.w / shrink;
    c->data_sz.h = c->crop_sz.h / shrink;
    c->hist_sz.w = c->crop_sz.w / c->opts.grad_hist.bin_size;
    c->hist_sz.h = c->crop_sz.h / c->opts.grad_hist.bin_size;
    assert(c->crop_sz.w % c->opts.grad_hist.bin_size == 0);
    assert(c->crop_sz.h % c->opts.grad_hist.bin_size == 0);
    if (c->capacity.w >= c->crop_sz.w && c->capacity.h >= c->crop_sz.h) {
        return; // enough capacity
    }
    release_buffers(c);
    c->capacity = c->crop_sz;
    c->mag = yapd_buffer_create(
        c->gpu, sizeof(float)*c->crop_sz.w*c->crop_sz.h);
    c->angle = yapd_buffer_create(
        c->gpu, sizeof(float)*c->crop_sz.w*c->crop_sz.h);
    c->hist = yapd_buffer_create(
        c->gpu, sizeof(cl_float8)*c->hist_sz.w*c->hist_sz.h);
}

static const yapd_channels_opts_t default_opts = {
    .shrink = 4,
    .color = {
        .smooth = 1
    },
    .grad_mag = {
        .norm_radius = 5,
        .norm_const = 0.005f
    },
    .grad_hist = {
        .bin_size = 0,
        .num_orients = 6
    }
};

YAPD_API const yapd_channels_opts_t*
yapd_channels_default_opts()
{
    return &default_opts;
}


yapd_channels_t
yapd_channels_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu,
    int cap_w, int cap_h, const yapd_channels_opts_t* opts)
{
    yapd_channels_t c;
    int bytes;

    c.a = a;
    c.aud = aud;
    c.gpu = gpu;
    if (!opts) opts = &default_opts;
    c.opts = *opts;

    c.capacity.w = 0;
    c.capacity.h = 0;
    c.mag = yapd_buffer_create(gpu, 0);
    c.angle = yapd_buffer_create(gpu, 0);
    c.hist = yapd_buffer_create(gpu, 0);

    assert(c.opts.grad_hist.num_orients <= 8);
    if (c.opts.grad_hist.bin_size == 0) {
        c.opts.grad_hist.bin_size = c.opts.shrink;
    }

    yapd_tri_filter(
        a, aud, opts->color.smooth, &c.smooth_filter_host, &bytes);
    c.smooth_filter = yapd_buffer_readonly(gpu, bytes);
    yapd_buffer_upload(
        &c.smooth_filter, (uint8_t*)c.smooth_filter_host, bytes);

    yapd_tri_filter(
        a, aud, opts->grad_mag.norm_radius,
        &c.mag_norm_filter_host, &bytes);
    c.mag_norm_filter = yapd_buffer_readonly(gpu, bytes);
    yapd_buffer_upload(
        &c.mag_norm_filter, (uint8_t*)c.mag_norm_filter_host, bytes);

    reserve_buffers(&c, cap_w, cap_h);

    return c;
}

void
yapd_channels_release(
    yapd_channels_t* c)
{
    release_buffers(c);

    c->a.dealloc(c->aud, c->smooth_filter_host);
    c->smooth_filter_host = NULL;
    yapd_buffer_release(&c->smooth_filter);

    c->a.dealloc(c->aud, c->mag_norm_filter_host);
    c->mag_norm_filter_host = NULL;
    yapd_buffer_release(&c->mag_norm_filter);
}

void
yapd_channels_prepare(
    yapd_channels_t* c, const yapd_size_t* sz,
    yapd_buffer_t* color, yapd_buffer_t* mag, yapd_buffer_t* hist)
{
    assert(sz->w > 0 && sz->h > 0);
    reserve_buffers(c, sz->w, sz->h);
    yapd_buffer_reserve(
        color, sizeof(cl_float4)*c->data_sz.w*c->data_sz.h);
    yapd_buffer_reserve(
        mag, sizeof(float)*c->data_sz.w*c->data_sz.h);
    yapd_buffer_reserve(
        hist, sizeof(cl_float8)*c->data_sz.w*c->data_sz.h);
}

void
yapd_channels_compute(
    yapd_channels_t* c, yapd_buffer_t* img,
    yapd_buffer_t* tmp, yapd_buffer_t* color,
    yapd_buffer_t* mag, yapd_buffer_t* hist)
{
    yapd_buffer_t* ohist =
        yapd_size_equals(&c->hist_sz, &c->data_sz) ? hist : &c->hist;
    // presmooth
    yapd_buffer_conv_tri32fc4(
        img, tmp, &c->crop_sz, c->opts.color.smooth, &c->smooth_filter);
    yapd_buffer_resample32fc4(
        color, &c->data_sz, img, &c->crop_sz, 1.0f);
    // gradient magnitude
    yapd_gradient_mag32fc4(
        img, &c->crop_sz, &c->mag, &c->angle);
    yapd_gradient_mag_norm(
        &c->mag, &c->crop_sz, tmp, c->opts.grad_mag.norm_const,
        c->opts.grad_mag.norm_radius, &c->mag_norm_filter);
    yapd_buffer_resample32f(
        mag, &c->data_sz, &c->mag, &c->crop_sz, 1.0f);
    // gradient histogram
    yapd_gradient_scale_angle(
        &c->angle, &c->crop_sz, c->opts.grad_hist.num_orients / CL_M_PI_F);
    yapd_gradient_hist(
        ohist, &c->mag, &c->angle, &c->hist_sz,
        c->opts.grad_hist.bin_size, c->opts.grad_hist.num_orients);
    if (ohist != hist) {
        yapd_buffer_resample32fc8(
            hist, &c->data_sz, ohist, &c->hist_sz, 1.0f);
    }
}