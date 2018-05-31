/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/buffer.h>

static YAPD_INLINE yapd_buffer_t
buffer_create(
    yapd_gpu_t* gpu, int bytes, int flags)
{
    cl_int err;
    yapd_buffer_t b;
    b.gpu = gpu;
    b.bytes = bytes;
    b.flags = flags;
    if (bytes > 0) {
        b.mem = clCreateBuffer(gpu->ctx, flags, bytes, NULL, &err);
        assert(err == CL_SUCCESS);
    } else {
        b.mem = 0;
    }
    return b;
}

yapd_buffer_t
yapd_buffer_create(
    yapd_gpu_t* gpu, int bytes)
{
    return buffer_create(gpu, bytes, CL_MEM_READ_WRITE);
}

yapd_buffer_t
yapd_buffer_readonly(
    yapd_gpu_t* gpu, int bytes)
{
    return buffer_create(gpu, bytes, CL_MEM_READ_ONLY);
}

yapd_buffer_t
yapd_buffer_writeonly(
    yapd_gpu_t* gpu, int bytes)
{
    return buffer_create(gpu, bytes, CL_MEM_WRITE_ONLY);
}

void
yapd_buffer_reserve(
    yapd_buffer_t* buf, int bytes)
{
    if (buf->bytes < bytes) {
        yapd_buffer_release(buf);
        *buf = buffer_create(buf->gpu, bytes, buf->flags);
    }
}

void
yapd_buffer_release(
    yapd_buffer_t* buf)
{
    if (buf->gpu && buf->bytes > 0) {
        clReleaseMemObject(buf->mem);
        buf->mem = 0;
        buf->bytes = 0;
    }
}

void
yapd_buffer_upload(
    yapd_buffer_t* buf, const uint8_t* data, int bytes)
{
    cl_int err;
    if (bytes == 0) return;
    err = clEnqueueWriteBuffer(
        buf->gpu->queue, buf->mem, CL_FALSE, 0, bytes, data, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_buffer_upload_2d(
    yapd_buffer_t* buf, const uint8_t* data, int bytes,
    const yapd_size_t* sz, int stride)
{
    cl_int err;
    if (bytes == 0) return;
    int pixel_sz = bytes / (sz->w*sz->h);
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { sz->w*pixel_sz, sz->h, 1 };
    err = clEnqueueWriteBufferRect(
        buf->gpu->queue, buf->mem, CL_FALSE,
        origin, origin, region,
        0, 0, stride, 0, data, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void
yapd_buffer_download_sync(
    yapd_buffer_t* buf, uint8_t* data, int bytes)
{
    cl_int err;
    if (bytes == 0) return;
    err = clEnqueueReadBuffer(
        buf->gpu->queue, buf->mem, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

YAPD_API void
yapd_buffer_download(
    yapd_buffer_t* buf, uint8_t* data, int bytes)
{
    cl_int err;
    if (bytes == 0) return;
    err = clEnqueueReadBuffer(
        buf->gpu->queue, buf->mem, CL_FALSE, 0, bytes, data, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}