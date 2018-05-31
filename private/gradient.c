/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/buffer.h>

#include <yapd/gpu.h>
#include <gradient.cl.h>

void
yapd_gpu_setup_gradient(
    yapd_gpu_t* gpu)
{
    cl_int err;
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;
    c->program = yapd_gpu_load_program(gpu, gradient_cl);
    c->grad_mag32fc4 = clCreateKernel(c->program, "grad_mag32fc4", &err);
    assert(err == CL_SUCCESS);
    c->grad_mag_norm = clCreateKernel(c->program, "grad_mag_norm", &err);
    assert(err == CL_SUCCESS);
    c->grad_scale_angle = clCreateKernel(c->program, "grad_scale_angle", &err);
    assert(err == CL_SUCCESS);
    c->grad_hist = clCreateKernel(c->program, "grad_hist", &err);
    assert(err == CL_SUCCESS);
}

void
yapd_gpu_release_gradient(
    yapd_gpu_t* gpu)
{
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;
    clReleaseKernel(c->grad_mag32fc4);
    clReleaseKernel(c->grad_mag_norm);
    clReleaseKernel(c->grad_scale_angle);
    clReleaseKernel(c->grad_hist);
    clReleaseProgram(c->program);
}

void
yapd_gradient_mag32fc4(
    yapd_buffer_t* img, const yapd_size_t* sz,
    yapd_buffer_t* mag, yapd_buffer_t* angle)
{
    cl_int err;
    yapd_gpu_t* gpu = img->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;

    assert(gpu == mag->gpu && gpu == angle->gpu);
    assert(img->bytes >= sizeof(cl_float4)*sz->w*sz->h);
    assert(mag->bytes >= sizeof(float)*sz->w*sz->h);
    assert(angle->bytes >= sizeof(float)*sz->w*sz->h);

    err = clSetKernelArg(c->grad_mag32fc4, 0, sizeof(cl_mem), &img->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag32fc4, 1, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag32fc4, 2, sizeof(cl_mem), &mag->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag32fc4, 3, sizeof(cl_mem), &angle->mem);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->grad_mag32fc4, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

YAPD_API void
yapd_gradient_mag_norm(
    yapd_buffer_t* mag, const yapd_size_t* sz, yapd_buffer_t* tmp,
    float norm_const, int r, yapd_buffer_t* filter)
{
    cl_int err;
    yapd_gpu_t* gpu = mag->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;

    int off = sz->w*sz->h;
    yapd_buffer_conv_tri_cols32f(tmp, mag, off, sz, r, filter);
    yapd_buffer_conv_tri_rows32f(tmp, 0, off, sz, r, filter);

    err = clSetKernelArg(c->grad_mag_norm, 0, sizeof(cl_mem), &mag->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag_norm, 1, sizeof(cl_mem), &tmp->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag_norm, 2, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_mag_norm, 3, sizeof(float), &norm_const);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->grad_mag_norm, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_gradient_scale_angle(
    yapd_buffer_t* angle, const yapd_size_t* sz, float scale)
{
    cl_int err;
    yapd_gpu_t* gpu = angle->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;

    assert(angle->bytes >= sizeof(float)*sz->w*sz->h);

    err = clSetKernelArg(c->grad_scale_angle, 0, sizeof(cl_mem), &angle->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_scale_angle, 1, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_scale_angle, 2, sizeof(float), &scale);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->grad_scale_angle, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_gradient_hist(
    yapd_buffer_t* hist, yapd_buffer_t* mag, yapd_buffer_t* angle,
    const yapd_size_t* sz, int bin_size, int num_orients)
{
    cl_int err;
    yapd_gpu_t* gpu = hist->gpu;
    size_t offset[] = { 0, 0, 0 };
    size_t size[] = { sz->w, sz->h, 1 };
    yapd_gpu_gradient_ctx_t* c = &gpu->gradient;

    assert(gpu == mag->gpu && gpu == angle->gpu);
    assert(hist->bytes >= sizeof(cl_float8)*sz->w*sz->h);
    assert(mag->bytes >= sizeof(float)*sz->w*bin_size*sz->h*bin_size);
    assert(angle->bytes >= sizeof(float)*sz->w*bin_size*sz->h*bin_size);

    err = clSetKernelArg(c->grad_hist, 0, sizeof(cl_mem), &hist->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_hist, 1, sizeof(cl_mem), &mag->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_hist, 2, sizeof(cl_mem), &angle->mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_hist, 3, sizeof(cl_int2), sz);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_hist, 4, sizeof(int), &bin_size);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(c->grad_hist, 5, sizeof(int), &num_orients);
    assert(err == CL_SUCCESS);

    err = clEnqueueNDRangeKernel(
        gpu->queue, c->grad_hist, 2, offset, size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}