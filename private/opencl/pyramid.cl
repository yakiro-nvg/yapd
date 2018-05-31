/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

int pixel_idx(const int w, const int2 pos)
{
    return pos.y*w + pos.x;
}

__kernel void pyramid_conpad(
    __global float16* dst,
    int2 sz, int2 pad,
    __global float4* color,
    __global float* mag,
    __global float8* hist)
{
    const int2 dst_pos = { get_global_id(0), get_global_id(1) };
    const int2 org_pos = dst_pos - pad;
    const int2 nrm_pos = clamp(org_pos, (int2)0.0f, sz - (int2)1.0f);
    const int dst_idx = pixel_idx(sz.s0 + 2*pad.s0, dst_pos);
    const int nrm_idx = pixel_idx(sz.s0, nrm_pos);
    float16 out = (float16)0.0f;
    out.s0123 = color[nrm_idx];
    if (nrm_pos.s0 == org_pos.s0 && nrm_pos.s1 == org_pos.s1) {
        out.s4 = mag[nrm_idx];
        out.s89abcdef = hist[nrm_idx];
    }
    dst[dst_idx] = out;
}