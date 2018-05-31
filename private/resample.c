/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/buffer.h>

#include <yapd/gpu.h>
#include <resample.cl.h>

static void
resample32f(
    int pixel_sz, cl_kernel kernel,
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm)
{
    cl_int err;
    yapd_gpu_t* gpu = dst->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { dst_sz->w, dst_sz->h, 1 };

    assert(gpu == src->gpu);
    assert(dst->bytes >= pixel_sz * dst_sz->w*dst_sz->h);
    assert(src->bytes >= pixel_sz * src_sz->w*src_sz->h);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dst->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 1, sizeof(cl_int2), dst_sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &src->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int2), src_sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 4, sizeof(float), &norm);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, kernel, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_setup_resample(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_resample_ctx_t* c = &gpu->resample;
    c->program = yapd_gpu_load_program(gpu, resample_cl);
    c->resample32f = clCreateKernel(c->program, "resample32f", &err);
    assert(err == CL_SUCCESS);
    c->resample32fc4 = clCreateKernel(c->program, "resample32fc4", &err);
    assert(err == CL_SUCCESS);
    c->resample32fc8 = clCreateKernel(c->program, "resample32fc8", &err);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_resample(
    yapd_gpu_t* gpu)
{
    yapd_gpu_resample_ctx_t* c = &gpu->resample;
    clReleaseKernel(c->resample32f);
    clReleaseKernel(c->resample32fc4);
    clReleaseKernel(c->resample32fc8);
    clReleaseProgram(c->program);
}

void
yapd_buffer_resample32f(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm)
{
    resample32f(
        sizeof(float), dst->gpu->resample.resample32f,
        dst, dst_sz, src, src_sz, norm);
}

void
yapd_buffer_resample32fc4(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm)
{
    resample32f(
        sizeof(cl_float4), dst->gpu->resample.resample32fc4,
        dst, dst_sz, src, src_sz, norm);
}

void
yapd_buffer_resample32fc8(
    yapd_buffer_t* dst, const yapd_size_t* dst_sz,
    yapd_buffer_t* src, const yapd_size_t* src_sz, float norm)
{
    resample32f(
        sizeof(cl_float8), dst->gpu->resample.resample32fc8,
        dst, dst_sz, src, src_sz, norm);
}