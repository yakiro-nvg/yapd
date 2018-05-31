/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/buffer.h>

#include <yapd/gpu.h>
#include <convolution.cl.h>

static void
conv_tri32f(
    int pixel_sz, cl_kernel kernel,
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    cl_int err;
    yapd_gpu_t* gpu = img->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    cl_int2 dir;

    assert(gpu == tmp->gpu && gpu == filter->gpu);
    assert(img->bytes >= pixel_sz * sz->w*sz->h);
    assert(tmp->bytes >= pixel_sz * sz->w*sz->h);
    assert(filter->bytes >= (2 * r + 1) * sizeof(float));

    err = clSetKernelArg(kernel, 0, sizeof(int), &r);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);

    // convolution each columns
    dir.s0 = 0; dir.s1 = 1;
    err = clSetKernelArg(kernel, 1, sizeof(cl_int2), &dir);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &tmp->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &img->mem);
    assert(err == CL_SUCCESS);
    err = clEnqueueNDRangeKernel(
        gpu->queue, kernel, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    // convolution each rows
    dir.s0 = 1; dir.s1 = 0;
    err = clSetKernelArg(kernel, 1, sizeof(cl_int2), &dir);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &img->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &tmp->mem);
    assert(err == CL_SUCCESS);
    err = clEnqueueNDRangeKernel(
        gpu->queue, kernel, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_setup_convolution(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_convolution_ctx_t* c = &gpu->convolution;
    c->program = yapd_gpu_load_program(gpu, convolution_cl);
    c->conv_tri_cols32f = clCreateKernel(c->program, "conv_tri_cols32f", &err);
    assert(err == CL_SUCCESS);
    c->conv_tri_rows32f = clCreateKernel(c->program, "conv_tri_rows32f", &err);
    assert(err == CL_SUCCESS);
    c->conv_tri32f = clCreateKernel(c->program, "conv_tri32f", &err);
    assert(err == CL_SUCCESS);
    c->conv_tri32fc4 = clCreateKernel(c->program, "conv_tri32fc4", &err);
    assert(err == CL_SUCCESS);
    c->conv_tri32fc8 = clCreateKernel(c->program, "conv_tri32fc8", &err);
    assert(err == CL_SUCCESS);
    c->conv_tri32fc16 = clCreateKernel(c->program, "conv_tri32fc16", &err);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_convolution(
    yapd_gpu_t* gpu)
{
    yapd_gpu_convolution_ctx_t* c = &gpu->convolution;
    clReleaseKernel(c->conv_tri_cols32f);
    clReleaseKernel(c->conv_tri_rows32f);
    clReleaseKernel(c->conv_tri32f);
    clReleaseKernel(c->conv_tri32fc4);
    clReleaseKernel(c->conv_tri32fc8);
    clReleaseKernel(c->conv_tri32fc16);
    clReleaseProgram(c->program);
}

void
yapd_tri_filter(
    yapd_alloc_t a, void* aud, int r, float** filter, int* bytes)
{
    float nrm = 0, v;
    int sz = 2 * r + 1, i;
    *bytes = sz * sizeof(float);
    *filter = (float*)a.alloc(aud, *bytes, YAPD_DEFAULT_ALIGN);
    for (i = 0; i < r; ++i) {
        v = (float)i + 1;
        nrm += 2 * v;
        (*filter)[i] = (*filter)[sz - i - 1] = v;
    }
    v = (float)r + 1;
    (*filter)[r] = v;
    nrm += v;
    nrm = 1.0f / nrm;
    for (i = 0; i < sz; ++i) {
        (*filter)[i] *= nrm;
    }
}

void
yapd_buffer_conv_tri_cols32f(
    yapd_buffer_t* dst, yapd_buffer_t* src, int dst_off,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    cl_int err;
    yapd_gpu_t* gpu = dst->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    yapd_gpu_convolution_ctx_t* c = &gpu->convolution;

    assert(gpu == src->gpu && gpu == filter->gpu);
    assert(dst->bytes >= sizeof(float)*(dst_off + sz->w*sz->h));
    assert(src->bytes >= sizeof(float)*sz->w*sz->h);
    assert(filter->bytes >= (2 * r + 1) * sizeof(float));

    err = clSetKernelArg(c->conv_tri_cols32f, 0, sizeof(int), &r);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_cols32f, 1, sizeof(cl_mem), &filter->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_cols32f, 2, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_cols32f, 3, sizeof(cl_mem), &dst->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_cols32f, 4, sizeof(int), &dst_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_cols32f, 5, sizeof(cl_mem), &src->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->conv_tri_cols32f, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_buffer_conv_tri_rows32f(
    yapd_buffer_t* buf, int dst_off, int src_off,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    cl_int err;
    yapd_gpu_t* gpu = buf->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    int hi_off = dst_off > src_off ? dst_off : src_off;
    int lo_off = dst_off < src_off ? dst_off : src_off;
    yapd_gpu_convolution_ctx_t* c = &gpu->convolution;

    assert(gpu == filter->gpu);
    assert(buf->bytes >= sizeof(float)*(hi_off + sz->w*sz->h));
    assert(lo_off + sz->w*sz->h <= hi_off);
    assert(filter->bytes >= (2 * r + 1) * sizeof(float));

    err = clSetKernelArg(c->conv_tri_rows32f, 0, sizeof(int), &r);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_rows32f, 1, sizeof(cl_mem), &filter->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_rows32f, 2, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_rows32f, 3, sizeof(cl_mem), &buf->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_rows32f, 4, sizeof(int), &dst_off);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->conv_tri_rows32f, 5, sizeof(int), &src_off);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->conv_tri_rows32f, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_buffer_conv_tri32f(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    conv_tri32f(
        sizeof(float), img->gpu->convolution.conv_tri32f,
        img, tmp, sz, r, filter);
}

void
yapd_buffer_conv_tri32fc4(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    conv_tri32f(
        sizeof(cl_float4), img->gpu->convolution.conv_tri32fc4,
        img, tmp, sz, r, filter);
}

void
yapd_buffer_conv_tri32fc8(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    conv_tri32f(
        sizeof(cl_float8), img->gpu->convolution.conv_tri32fc8,
        img, tmp, sz, r, filter);
}

void
yapd_buffer_conv_tri32fc16(
    yapd_buffer_t* img, yapd_buffer_t* tmp,
    const yapd_size_t* sz, int r, yapd_buffer_t* filter)
{
    conv_tri32f(
        sizeof(cl_float16), img->gpu->convolution.conv_tri32fc16,
        img, tmp, sz, r, filter);
}