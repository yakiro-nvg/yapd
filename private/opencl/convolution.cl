/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

int2 border(const int2 pos, const int2 sz)
{
    return (int2)(
        // symmetric padding
        pos.x < 0 ? -pos.x : (pos.x < sz.s0 ? pos.x : (2*sz.s0 - pos.x - 2)),
        pos.y < 0 ? -pos.y : (pos.y < sz.s1 ? pos.y : (2*sz.s1 - pos.y - 2)));
}

__kernel void conv_tri_cols32f(
    const int r, __constant float* filter, const int2 sz,
    __global float* dst, const int dst_off, __global float* src)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float sum = (float)0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + (int2)(0, 1)*i, sz);
        sum += src[p.y*sz.s0 + p.x]*filter[i + r];
    }
    dst[dst_off + pos.y*sz.s0 + pos.x] = sum;
}

__kernel void conv_tri_rows32f(
    const int r, __constant float* filter, const int2 sz,
    __global float* buf, const int dst_off, const int src_off)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float sum = (float)0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + (int2)(1, 0)*i, sz);
        sum += buf[src_off + p.y*sz.s0 + p.x]*filter[i + r];
    }
    buf[dst_off + pos.y*sz.s0 + pos.x] = sum;
}

__kernel void conv_tri32f(
    const int r, const int2 dir, __constant float* filter,
    const int2 sz, __global float* dst, __global float* src)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + dir*i, sz);
        sum += src[p.y*sz.s0 + p.x]*filter[i + r];
    }
    dst[pos.y*sz.s0 + pos.x] = sum;
}

__kernel void conv_tri32fc4(
    const int r, const int2 dir, __constant float* filter,
    const int2 sz, __global float4* dst, __global float4* src)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float4 sum = (float4)0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + dir*i, sz);
        sum += src[p.y*sz.s0 + p.x]*filter[i + r];
    }
    dst[pos.y*sz.s0 + pos.x] = sum;
}

__kernel void conv_tri32fc8(
    const int r, const int2 dir, __constant float* filter,
    const int2 sz, __global float8* dst, __global float8* src)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float8 sum = (float8)0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + dir*i, sz);
        sum += src[p.y*sz.s0 + p.x]*filter[i + r];
    }
    dst[pos.y*sz.s0 + pos.x] = sum;
}

__kernel void conv_tri32fc16(
    const int r, const int2 dir, __constant float* filter,
    const int2 sz, __global float16* dst, __global float16* src)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float16 sum = (float16)0.0f;
    for (int i = -r; i <= r; ++i) {
        const float f = filter[i + r];
        int2 p = border(pos + dir*i, sz);
        sum += src[p.y*sz.s0 + p.x]*filter[i + r];
    }
    dst[pos.y*sz.s0 + pos.x] = sum;
}
