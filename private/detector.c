/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/detector.h>

#include <yapd/gpu.h>
#include <yapd/matrix.h>
#include <yapd/buffer.h>
#include <yapd/nms.h>
#include <detector.cl.h>

static void
compute_cids(
    yapd_alloc_t a, void* aud,
    const yapd_size_t* win_sz, int shrink,
    const yapd_size_t* sz, int** cids_host, yapd_buffer_t* cids)
{
    int x, y, z, m = 0;
    const int mw = win_sz->w / shrink;
    const int mh = win_sz->h / shrink;
    const int nftrs = mw * mh * YAPD_FEATURE_CHANNELS;
    const int bytes = nftrs * sizeof(int);
    a.dealloc(aud, *cids_host);
    *cids_host = (int*)a.alloc(aud, bytes, YAPD_DEFAULT_ALIGN);
    for (z = 0; z < YAPD_FEATURE_CHANNELS; ++z) {
        for (x = 0; x < mw; ++x) {
            for (y = 0; y < mh; ++y) {
                (*cids_host)[m++] = (y*sz->w + x)*YAPD_FEATURE_CHANNELS + z;
            }
        }
    }
    yapd_buffer_reserve(cids, bytes);
    yapd_buffer_upload(cids, (uint8_t*)*cids_host, bytes);
}

static void
release_cids(
    yapd_detector_t* d)
{
    int i;
    d->a.dealloc(d->aud, d->sizes);
    d->sizes = NULL;
    for (i = 0; i < d->num_scales; ++i) {
        d->a.dealloc(d->aud, d->cids_host[i]);
        yapd_buffer_release(d->cids + i);
    }
    d->a.dealloc(d->aud, d->cids_host);
    d->cids_host = NULL;
    d->a.dealloc(d->aud, d->cids);
    d->cids = NULL;
}

static void
alloc(
    yapd_detector_t* d, int num_scales)
{
    int i;
    release_cids(d);
    d->num_scales = num_scales;
    d->sizes = (yapd_size_t*)d->a.alloc(
        d->aud, sizeof(yapd_size_t)*num_scales, YAPD_DEFAULT_ALIGN);
    d->cids_host = (int**)d->a.alloc(
        d->aud, sizeof(int*)*num_scales, YAPD_DEFAULT_ALIGN);
    d->cids = (yapd_buffer_t*)d->a.alloc(
        d->aud, sizeof(yapd_buffer_t)*num_scales, YAPD_DEFAULT_ALIGN);
    for (i = 0; i < num_scales; ++i) {
        d->sizes[i].w = -1;
        d->sizes[i].h = -1;
        d->cids_host[i] = NULL;
        d->cids[i] = yapd_buffer_readonly(d->gpu, 0);
    }
}

static void
reserve(
    yapd_detector_t* d, const yapd_size_t* sizes, int num_scales)
{
    int i;
    if (d->num_scales != num_scales) {
        alloc(d, num_scales);
    }
    for (i = 0; i < num_scales; ++i) {
        if (!yapd_size_equals(d->sizes + i, sizes + i)) {
            compute_cids(
                d->a, d->aud, &d->win_sz, d->shrink,
                sizes + i, d->cids_host + i, d->cids + i);
            d->sizes[i] = sizes[i];
        }
    }
}

static void
output_dims(
    yapd_size_t* dims, int shrink, int stride,
    const yapd_size_t* win_sz, const yapd_size_t* sz)
{
    dims->w = (int)ceilf(
        (sz->w*shrink - win_sz->w + 1) / (float)stride);
    dims->h = (int)ceilf(
        (sz->h*shrink - win_sz->h + 1) / (float)stride);
}

static void
early_reject(
    yapd_gpu_t* gpu, yapd_buffer_t* chns, yapd_buffer_t* cids,
    int depth, int to_org, int out_off, int org_w, float casc_thr,
    const yapd_size_t* dims, yapd_buffer_t* out, yapd_buffer_t* idx,
    yapd_buffer_t* thrs, yapd_buffer_t* hs, yapd_buffer_t* fids)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { dims->w, dims->h, 1 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    assert(out->bytes >= (dims->w*dims->h + out_off) * sizeof(float));
    assert(idx->bytes >= (dims->w*dims->h + out_off) * sizeof(int));

    err = clSetKernelArg(dc->early_reject, 0, sizeof(int), &depth);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 1, sizeof(int), &to_org);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 2, sizeof(int), &org_w);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 3, sizeof(int), &dims->w);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 4, sizeof(float), &casc_thr);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 5, sizeof(cl_mem), &thrs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 6, sizeof(cl_mem), &hs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 7, sizeof(cl_mem), &fids->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 8, sizeof(cl_mem), &chns->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 9, sizeof(cl_mem), &cids->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 10, sizeof(cl_mem), &out->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 11, sizeof(cl_mem), &idx->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_reject, 12, sizeof(int), &out_off);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->early_reject, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static void
early_scan(
    yapd_gpu_t* gpu, yapd_buffer_t* tmp, yapd_buffer_t* idx,
    int len_off, int out_off, const yapd_size_t* dims, yapd_buffer_t* len)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { dims->h, 0, 0 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    assert(idx->bytes >= (dims->w*dims->h + out_off) * sizeof(float));
    assert(len->bytes >= (dims->h + len_off) * sizeof(int));

    err = clSetKernelArg(dc->early_scan, 0, sizeof(cl_mem), &tmp->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_scan, 1, sizeof(int), &dims->w);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_scan, 2, sizeof(int), &out_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_scan, 3, sizeof(cl_mem), &idx->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_scan, 4, sizeof(cl_mem), &len->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_scan, 5, sizeof(int), &len_off);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->early_scan, 1, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static void
early_prefix_sum(
    yapd_gpu_t* gpu, int num_scales,
    yapd_buffer_t* len, yapd_buffer_t* sum, yapd_buffer_t* dsz)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { num_scales, 0, 0 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    err = clSetKernelArg(dc->early_prefix_sum, 0, sizeof(cl_mem), &len->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_prefix_sum, 1, sizeof(cl_mem), &sum->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_prefix_sum, 2, sizeof(cl_mem), &dsz->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->early_prefix_sum,
        1, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static void
early_bbs(
    yapd_gpu_t* gpu, float casc_thr,
    int bbs_off, int sum_off, int off, int num_scales,
    const yapd_size_t* dims, yapd_buffer_t* out,
    yapd_buffer_t* idx, yapd_buffer_t* bbs, yapd_buffer_t* sum)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { dims->w, dims->h, 1 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    err = clSetKernelArg(dc->early_bbs, 0, sizeof(int), &dims->w);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 1, sizeof(float), &casc_thr);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 2, sizeof(int), &bbs_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 3, sizeof(int), &sum_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 4, sizeof(int), &off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 5, sizeof(cl_mem), &out->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 6, sizeof(cl_mem), &idx->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 7, sizeof(cl_mem), &bbs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->early_bbs, 8, sizeof(cl_mem), &sum->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->early_bbs,
        2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static void
predict(
    yapd_gpu_t* gpu, yapd_buffer_t* chns, yapd_buffer_t* cids,
    int depth, int to_org, int bbs_off, int hss_off, int org_w, float casc_thr,
    int num_weaks, int bbs_sz, yapd_buffer_t* bbs, yapd_buffer_t* hss,
    yapd_buffer_t* thrs, yapd_buffer_t* hs, yapd_buffer_t* fids)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { bbs_sz, num_weaks, 1 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    assert(bbs_sz > 0);
    assert(bbs->bytes >= bbs_sz * sizeof(float) * 5);

    err = clSetKernelArg(dc->predict, 0, sizeof(int), &num_weaks);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 1, sizeof(int), &depth);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 2, sizeof(int), &to_org);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 3, sizeof(int), &org_w);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 4, sizeof(float), &casc_thr);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 5, sizeof(int), &bbs_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 6, sizeof(int), &hss_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 7, sizeof(cl_mem), &thrs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 8, sizeof(cl_mem), &hs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 9, sizeof(cl_mem), &fids->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 10, sizeof(cl_mem), &chns->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 11, sizeof(cl_mem), &cids->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 12, sizeof(cl_mem), &bbs->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict, 13, sizeof(cl_mem), &hss->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->predict, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

static void
predict_sum(
    yapd_gpu_t* gpu, int num_weaks, int bbs_sz, float casc_thr,
    yapd_buffer_t* hss, yapd_buffer_t* bss)
{
    cl_int err;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { bbs_sz, 0, 0 };
    yapd_gpu_detector_ctx_t* dc = &gpu->detector;

    err = clSetKernelArg(dc->predict_sum, 0, sizeof(int), &num_weaks);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict_sum, 1, sizeof(float), &casc_thr);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict_sum, 2, sizeof(cl_mem), &hss->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(dc->predict_sum, 3, sizeof(cl_mem), &bss->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, dc->predict_sum,
        1, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_setup_detector(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_detector_ctx_t* d = &gpu->detector;
    d->program = yapd_gpu_load_program(gpu, detector_cl);
    d->early_reject = clCreateKernel(
        d->program, "detector_early_reject", &err);
    assert(err == CL_SUCCESS);
    d->early_scan = clCreateKernel(
        d->program, "detector_early_scan", &err);
    assert(err == CL_SUCCESS);
    d->early_prefix_sum = clCreateKernel(
        d->program, "detector_early_prefix_sum", &err);
    assert(err == CL_SUCCESS);
    d->early_bbs = clCreateKernel(
        d->program, "detector_early_bbs", &err);
    assert(err == CL_SUCCESS);
    d->predict = clCreateKernel(
        d->program, "detector_predict", &err);
    assert(err == CL_SUCCESS);
    d->predict_sum = clCreateKernel(
        d->program, "detector_predict_sum", &err);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_detector(
    yapd_gpu_t* gpu)
{
    yapd_gpu_detector_ctx_t* d = &gpu->detector;
    clReleaseKernel(d->early_reject);
    clReleaseKernel(d->early_scan);
    clReleaseKernel(d->early_prefix_sum);
    clReleaseKernel(d->early_bbs);
    clReleaseKernel(d->predict);
    clReleaseKernel(d->predict_sum);
    clReleaseProgram(d->program);
}

yapd_detector_t
yapd_detector_new(
    yapd_alloc_t a, void* aud, yapd_gpu_t* gpu)
{
    yapd_detector_t d;

    d.a = a;
    d.aud = aud;
    d.gpu = gpu;

    d.dirty = TRUE;
    d.num_weaks = 0;
    d.depth = 0;
    d.win_sz.w = 0;
    d.win_sz.h = 0;
    d.org_win.w = 0;
    d.org_win.h = 0;
    d.shrink = 0;

    d.thrs_host = yapd_mat_new(a, aud);
    d.fids_host = yapd_mat_new(a, aud);
    d.hs_host = yapd_mat_new(a, aud);
    d.thrs = yapd_buffer_readonly(gpu, 0);
    d.fids = yapd_buffer_readonly(gpu, 0);
    d.hs = yapd_buffer_readonly(gpu, 0);

    d.num_scales = 0;
    d.sizes = NULL;
    d.cids_host = NULL;
    d.cids = NULL;

    d.dsz = yapd_buffer_create(gpu, 0);
    d.out = yapd_buffer_create(gpu, 0);
    d.idx = yapd_buffer_create(gpu, 0);
    d.len = yapd_buffer_create(gpu, 0);
    d.sum = yapd_buffer_create(gpu, 0);
    d.bbs = yapd_buffer_create(gpu, 0);
    d.hss = yapd_buffer_create(gpu, 0);
    d.tmp = yapd_buffer_create(gpu, 0);

    return d;
}

void
yapd_detector_release(
    yapd_detector_t* d)
{
    release_cids(d);

    yapd_buffer_release(&d->dsz);
    yapd_buffer_release(&d->out);
    yapd_buffer_release(&d->idx);
    yapd_buffer_release(&d->len);
    yapd_buffer_release(&d->sum);
    yapd_buffer_release(&d->bbs);
    yapd_buffer_release(&d->hss);
    yapd_buffer_release(&d->tmp);

    d->win_sz.w = 0;
    d->win_sz.h = 0;
    yapd_buffer_release(&d->thrs);
    yapd_buffer_release(&d->fids);
    yapd_buffer_release(&d->hs);
    yapd_mat_release(&d->thrs_host);
    yapd_mat_release(&d->fids_host);
    yapd_mat_release(&d->hs_host);
}

void
yapd_detector_classifier(
    yapd_detector_t* d, int copy,
    int num_weaks, int depth, int shrink,
    const yapd_size_t* win_sz, const yapd_size_t* org_win,
    yapd_mat_t* thrs, yapd_mat_t* fids, yapd_mat_t* hs)
{
    assert(num_weaks > 0);
    assert(depth > 0 && depth < 4);
    assert(shrink > 0);
    assert(win_sz->w > 0 && win_sz->h > 0);
    assert(org_win->w > 0 && org_win->h > 0);
    assert(thrs->type == YAPD_32F);
    assert(fids->type == YAPD_32S);
    assert(hs->type == YAPD_32F);
    assert(thrs->size.h == num_weaks && thrs->size.w == 8);
    assert(fids->size.h == num_weaks && fids->size.w == 8);
    assert(hs->size.h == num_weaks && hs->size.w == 8);

    d->num_weaks = num_weaks;
    d->shrink = shrink;
    d->depth = depth;
    d->win_sz = *win_sz;
    d->org_win = *org_win;

    yapd_mat_release(&d->thrs_host);
    yapd_mat_release(&d->fids_host);
    yapd_mat_release(&d->hs_host);

    if (copy) {
        yapd_mat_create(
            &d->thrs_host, thrs->size.w, thrs->size.h, thrs->type);
        memcpy(d->thrs_host.data, thrs->data, yapd_mat_bytes(thrs));
        yapd_mat_create(
            &d->fids_host, fids->size.w, fids->size.h, fids->type);
        memcpy(d->fids_host.data, fids->data, yapd_mat_bytes(fids));
        yapd_mat_create(
            &d->hs_host, hs->size.w, hs->size.h, hs->type);
        memcpy(d->hs_host.data, hs->data, yapd_mat_bytes(hs));
    } else {
        d->thrs_host = yapd_mat_borrow(
            thrs->data, thrs->size.w, thrs->size.h, thrs->type);
        d->fids_host = yapd_mat_borrow(
            fids->data, fids->size.w, fids->size.h, fids->type);
        d->hs_host = yapd_mat_borrow(
            hs->data, hs->size.w, hs->size.h, hs->type);
    }

    yapd_buffer_reserve(
        &d->thrs, yapd_mat_bytes(&d->thrs_host));
    yapd_buffer_upload(
        &d->thrs, d->thrs_host.data, yapd_mat_bytes(&d->thrs_host));
    yapd_buffer_reserve(
        &d->fids, yapd_mat_bytes(&d->fids_host));
    yapd_buffer_upload(
        &d->fids, d->fids_host.data, yapd_mat_bytes(&d->fids_host));
    yapd_buffer_reserve(
        &d->hs, yapd_mat_bytes(&d->hs_host));
    yapd_buffer_upload(
        &d->hs, d->hs_host.data, yapd_mat_bytes(&d->hs_host));

    d->dirty = TRUE;
}

yapd_mat_t
yapd_detector_predict(
    yapd_alloc_t a, void* aud,
    yapd_detector_t* d, yapd_pyramid_t* p,
    int stride, float casc_thr)
{
    yapd_mat_t r;
    float *bbs, *b;
    int *dsz, *len, len_off, off, bbs_off, hss_off, bbs_sz;
    int i, j, k, dsz_bytes, out_bytes, idx_bytes, lens, bbs_bytes, hss_bytes;
    const float win_pad_w = (d->win_sz.w - d->org_win.w) / 2.0f;
    const float win_pad_h = (d->win_sz.h - d->org_win.h) / 2.0f;
    const float shift_x = win_pad_w - p->opts.pad.w;
    const float shift_y = win_pad_h - p->opts.pad.h;
    assert(d->num_weaks > 0);

    if (d->dirty) {
        d->dirty = FALSE;
        alloc(d, p->num_scales);
    }
    reserve(d, p->data_sz, p->num_scales);

    // prepare output buffer
    out_bytes = 0, idx_bytes = 0; lens = 0;
    dsz_bytes = p->num_scales * sizeof(int) * 2;
    dsz = (int*)a.alloc(aud, dsz_bytes, YAPD_DEFAULT_ALIGN);
    for (i = 0; i < p->num_scales; ++i) {
        yapd_size_t dims;
        output_dims(&dims, d->shrink, stride, &d->win_sz, p->data_sz + i);
        assert(dims.w > 0 && dims.h > 0);
        out_bytes += dims.w * dims.h * sizeof(float);
        idx_bytes += dims.w * dims.h * sizeof(int);
        lens += dims.h;
        if (i == 0) {
            dsz[0] = dims.h;
            dsz[1] = 0;
        } else {
            dsz[i * 2] = dims.h;
            dsz[i * 2 + 1] = dsz[(i - 1) * 2] + dsz[(i - 1) * 2 + 1];
        }
    }
    yapd_buffer_reserve(&d->dsz, dsz_bytes);
    yapd_buffer_upload(&d->dsz, (uint8_t*)dsz, dsz_bytes);
    yapd_buffer_reserve(&d->out, out_bytes);
    yapd_buffer_reserve(&d->idx, idx_bytes);
    yapd_buffer_reserve(&d->len, lens * sizeof(int));
    yapd_buffer_reserve(&d->sum, lens * sizeof(int));
    yapd_buffer_reserve(&d->tmp, idx_bytes);

    // early cascade rejection
    len_off = 0; off = 0;
    len = (int*)a.alloc(aud, lens * sizeof(int), YAPD_DEFAULT_ALIGN);
    for (i = 0; i < p->num_scales; ++i) {
        yapd_size_t dims;
        output_dims(&dims, d->shrink, stride, &d->win_sz, p->data_sz + i);
        early_reject(
            d->gpu, p->data + i, d->cids + i, d->depth,
            stride/d->shrink, off, p->data_sz[i].w, casc_thr,
            &dims, &d->out, &d->tmp, &d->thrs, &d->hs, &d->fids);
        early_scan(
            d->gpu, &d->tmp, &d->idx, len_off, off, &dims, &d->len);
        len_off += dims.h; off += dims.w * dims.h;
    }
    early_prefix_sum(d->gpu, p->num_scales, &d->len, &d->sum, &d->dsz);
    yapd_buffer_download_sync(&d->len, (uint8_t*)len, lens * sizeof(int));
    for (i = 0, j = 0, bbs_sz = 0; i < p->num_scales; ++i) {
        const int h = dsz[i * 2];
        int* const l = len + j; j += h;
        for (k = 1; k < h; ++k) {
            // inclusive prefix sum
            l[k] = l[k] + l[k - 1];
        }
        bbs_sz += l[k - 1];
    }
    bbs_bytes = bbs_sz * sizeof(float) * 5;
    yapd_buffer_reserve(&d->bbs, bbs_bytes);
    hss_bytes = bbs_sz * sizeof(float) * d->num_weaks;
    yapd_buffer_reserve(&d->hss, hss_bytes);
    len_off = 0; off = 0; j = 0;  bbs_off = 0;
    for (i = 0; i < p->num_scales; ++i) {
        yapd_size_t dims;
        output_dims(&dims, d->shrink, stride, &d->win_sz, p->data_sz + i);
        early_bbs(
            d->gpu, casc_thr, bbs_off, len_off, off,
            p->num_scales, &dims, &d->out, &d->idx, &d->bbs, &d->sum);
        j += dsz[i * 2]; bbs_off += len[j - 1] * 5;
        len_off += dims.h; off += dims.w * dims.h;
    }

    // predict the remainings
#if 1
    j = 0; bbs_off = 0; hss_off = 0;
    for (i = 0; i < p->num_scales; ++i) {
        j += dsz[i * 2];
        if (len[j - 1] == 0) continue;
        predict(
            d->gpu, p->data + i, d->cids + i, d->depth,
            stride / d->shrink, bbs_off, hss_off, p->data_sz[i].w,
            casc_thr, d->num_weaks, len[j - 1], &d->bbs, &d->hss,
            &d->thrs, &d->hs, &d->fids);
        bbs_off += len[j - 1] * 5;
        hss_off += len[j - 1] * d->num_weaks;
    }
    predict_sum(d->gpu, d->num_weaks, bbs_sz, casc_thr, &d->hss, &d->bbs);

    // convert to bounding boxes
    bbs = (float*)a.alloc(aud, bbs_bytes, YAPD_DEFAULT_ALIGN);
    yapd_buffer_download_sync(&d->bbs, (uint8_t*)bbs, bbs_bytes);
    for (i = 0, j = 0, b = bbs; i < p->num_scales; ++i) {
        j += dsz[i * 2];
        for (k = 0; k < len[j - 1]; ++k) {
            b[0] = (b[0] * stride + shift_x) / p->scalesw[i];
            b[1] = (b[1] * stride + shift_y) / p->scalesh[i];
            b[2] = d->org_win.w / p->scales[i];
            b[3] = d->org_win.h / p->scales[i];
            b += 5;
        }
    }
    r = yapd_mat_take(a, aud, (uint8_t*)bbs, 5, bbs_sz, YAPD_32F);
    yapd_nms(a, aud, &r, 30, 0.65f, TRUE);
#else
    r = yapd_mat_new(a, aud);
#endif
    a.dealloc(aud, dsz);
    a.dealloc(aud, len);
    return r;
}
