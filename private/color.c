/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/buffer.h>

#include <yapd/gpu.h>
#include <color.cl.h>
#include <color_consts.h>

void
yapd_gpu_setup_color(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_color_ctx_t* c = &gpu->color;
    c->program = yapd_gpu_load_program(gpu, color_cl);
    c->luv_from_rgb8uc4 = clCreateKernel(c->program, "luv_from_rgb8uc4", &err);
    assert(err == CL_SUCCESS);

    gpu->color.ltable = yapd_buffer_readonly(gpu, sizeof(color_ltable));
    yapd_buffer_upload(
        &gpu->color.ltable, (uint8_t*)color_ltable, sizeof(color_ltable));

    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 0, sizeof(float), &color_consts.minu);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 1, sizeof(float), &color_consts.minv);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 2, sizeof(float), &color_consts.un);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 3, sizeof(float), &color_consts.vn);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 4, sizeof(cl_float4), &color_consts.mr);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 5, sizeof(cl_float4), &color_consts.mg);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 6, sizeof(cl_float4), &color_consts.mb);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(
        c->luv_from_rgb8uc4, 7, sizeof(cl_mem), &c->ltable.mem);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_color(
    yapd_gpu_t* gpu)
{
    yapd_gpu_color_ctx_t* c = &gpu->color;
    yapd_buffer_release(&c->ltable);
    clReleaseKernel(c->luv_from_rgb8uc4);
    clReleaseProgram(c->program);
}

void
yapd_buffer_luv_from_rgb8uc4(
    yapd_buffer_t* dst, yapd_buffer_t* src, const yapd_size_t* sz)
{
    cl_int err;
    yapd_gpu_t* gpu = dst->gpu;
    size_t size[] = { sz->w*sz->h, 0, 0 };
    yapd_gpu_color_ctx_t* c = &gpu->color;

    assert(gpu == src->gpu);
    assert(dst->bytes >= sizeof(cl_float4)*sz->w*sz->h);
    assert(src->bytes >= sizeof(cl_uchar4)*sz->w*sz->h);

    err = clSetKernelArg(c->luv_from_rgb8uc4, 8, sizeof(cl_mem), &dst->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->luv_from_rgb8uc4, 9, sizeof(cl_mem), &src->mem);
    assert(err == CL_SUCCESS);
    err = clEnqueueNDRangeKernel(
        gpu->queue, c->luv_from_rgb8uc4, 1, NULL, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}