/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

int pixel_idx(const int w, const int x, const int y)
{
    return y*w + x;
}

__kernel void resample32f(
    __global float* dst, int2 dst_sz,
	__global float* src, int2 src_sz, float norm)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const float gx = x / (float)dst_sz.s0 * (src_sz.s0 - 1);
    const float gy = y / (float)dst_sz.s1 * (src_sz.s1 - 1);
    const int gxi = (int)gx;
    const int gyi = (int)gy;
    const float c00 = src[pixel_idx(src_sz.s0, gxi, gyi)];
    const float c10 = src[pixel_idx(src_sz.s0, gxi + 1, gyi)];
    const float c01 = src[pixel_idx(src_sz.s0, gxi, gyi + 1)];
    const float c11 = src[pixel_idx(src_sz.s0, gxi + 1, gyi + 1)];
    const float tx = gx - gxi;
    const float ty = gy - gyi;
    dst[pixel_idx(dst_sz.s0, x, y)] =
        mix(mix(c00, c10, tx), mix(c01, c11, tx), ty)*norm;
}

__kernel void resample32fc4(
    __global float4* dst, int2 dst_sz,
	__global float4* src, int2 src_sz, float norm)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const float gx = x / (float)dst_sz.s0 * (src_sz.s0 - 1);
    const float gy = y / (float)dst_sz.s1 * (src_sz.s1 - 1);
    const int gxi = (int)gx;
    const int gyi = (int)gy;
    const float4 c00 = src[pixel_idx(src_sz.s0, gxi, gyi)];
    const float4 c10 = src[pixel_idx(src_sz.s0, gxi + 1, gyi)];
    const float4 c01 = src[pixel_idx(src_sz.s0, gxi, gyi + 1)];
    const float4 c11 = src[pixel_idx(src_sz.s0, gxi + 1, gyi + 1)];
    const float tx = gx - gxi;
    const float ty = gy - gyi;
    dst[pixel_idx(dst_sz.s0, x, y)] =
        mix(mix(c00, c10, tx), mix(c01, c11, tx), ty)*norm;
}

__kernel void resample32fc8(
    __global float8* dst, int2 dst_sz,
	__global float8* src, int2 src_sz, float norm)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const float gx = x / (float)dst_sz.s0 * (src_sz.s0 - 1);
    const float gy = y / (float)dst_sz.s1 * (src_sz.s1 - 1);
    const int gxi = (int)gx;
    const int gyi = (int)gy;
    const float8 c00 = src[pixel_idx(src_sz.s0, gxi, gyi)];
    const float8 c10 = src[pixel_idx(src_sz.s0, gxi + 1, gyi)];
    const float8 c01 = src[pixel_idx(src_sz.s0, gxi, gyi + 1)];
    const float8 c11 = src[pixel_idx(src_sz.s0, gxi + 1, gyi + 1)];
    const float tx = gx - gxi;
    const float ty = gy - gyi;
    dst[pixel_idx(dst_sz.s0, x, y)] =
        mix(mix(c00, c10, tx), mix(c01, c11, tx), ty)*norm;
}
