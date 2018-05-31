/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/gpu.h>

// TODO: build log
#include <stdio.h>
#include <malloc.h>

#define YAPD_DEVICE_TYPE CL_DEVICE_TYPE_GPU

extern void
yapd_gpu_setup_color(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_color(
    yapd_gpu_t* gpu);

extern void
yapd_gpu_setup_resample(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_resample(
    yapd_gpu_t* gpu);

extern void
yapd_gpu_setup_convolution(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_convolution(
    yapd_gpu_t* gpu);

extern void
yapd_gpu_setup_gradient(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_gradient(
    yapd_gpu_t* gpu);

extern void
yapd_gpu_setup_pyramid(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_pyramid(
    yapd_gpu_t* gpu);

extern void
yapd_gpu_setup_detector(
    yapd_gpu_t* gpu);
extern void
yapd_gpu_release_detector(
    yapd_gpu_t* gpu);

static void
create(yapd_gpu_t* gpu)
{
    cl_int err;
    cl_uint count = 0;
    cl_platform_id platform_ids[1];
    cl_context_properties ctx_props[4];
    err = clGetPlatformIDs(0, NULL, &count);
    assert(err == CL_SUCCESS);
    assert(count > 0);
    err = clGetPlatformIDs(1, platform_ids, NULL);
    assert(err == CL_SUCCESS);
    err = clGetDeviceIDs(
        platform_ids[0], YAPD_DEVICE_TYPE, 0, NULL, &count);
    assert(err == CL_SUCCESS);
    assert(count > 0);
    err = clGetDeviceIDs(
        platform_ids[0], YAPD_DEVICE_TYPE, 1, gpu->dev_ids, NULL);
    assert(err == CL_SUCCESS);
    ctx_props[0] = CL_CONTEXT_PLATFORM;
    ctx_props[1] = (cl_context_properties)platform_ids[0];
    ctx_props[2] = 0;
    ctx_props[3] = 0;
    gpu->ctx = clCreateContext(ctx_props, 1, gpu->dev_ids, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    gpu->queue = clCreateCommandQueue(gpu->ctx, gpu->dev_ids[0], 0, &err);
    assert(err == CL_SUCCESS);
}

yapd_gpu_t
yapd_gpu_new()
{
    yapd_gpu_t gpu;
    create(&gpu);
    yapd_gpu_setup_color(&gpu);
    yapd_gpu_setup_resample(&gpu);
    yapd_gpu_setup_convolution(&gpu);
    yapd_gpu_setup_gradient(&gpu);
    yapd_gpu_setup_pyramid(&gpu);
    yapd_gpu_setup_detector(&gpu);
    return gpu;
}

cl_program
yapd_gpu_load_program(
    yapd_gpu_t* gpu, const char* source)
{
    cl_int err;
    cl_program p;
    const char* strings[] = { source };
    p = clCreateProgramWithSource(gpu->ctx, 1, strings, NULL, &err);
    assert(err == CL_SUCCESS);
    err = clBuildProgram(p, 1, gpu->dev_ids, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        err = clGetProgramBuildInfo(
            p, gpu->dev_ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        assert(err == CL_SUCCESS);
        char* log = (char*)malloc(len);
        err = clGetProgramBuildInfo(
            p, gpu->dev_ids[0], CL_PROGRAM_BUILD_LOG, len, log, NULL);
        assert(err == CL_SUCCESS);
        printf("%s\n", log);
        free(log);
        assert(!"failed to build program");
    }
    return p;
}

void
yapd_gpu_sync(
    yapd_gpu_t* gpu)
{
    clFinish(gpu->queue);
}

void
yapd_gpu_release(
    yapd_gpu_t* gpu)
{
    yapd_gpu_release_color(gpu);
    yapd_gpu_release_resample(gpu);
    yapd_gpu_release_convolution(gpu);
    yapd_gpu_release_gradient(gpu);
    yapd_gpu_release_pyramid(gpu);
    yapd_gpu_release_detector(gpu);
    clReleaseCommandQueue(gpu->queue);
    clReleaseContext(gpu->ctx);
}