/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

__kernel void luv_from_rgb8uc4(
    const float minu, const float minv, const float un, const float vn,
    const float4 mr, const float4 mg, const float4 mb, __constant float* ltable,
    __global float4* dst, __global uchar4* src)
{
    const int i = get_global_id(0);
    const uchar4 src_color = src[i];
    const float r = src_color.s0;
    const float g = src_color.s1;
    const float b = src_color.s2;
    float x = mr.s0*r + mg.s0*g + mb.s0*b;
    float y = mr.s1*r + mg.s1*g + mb.s1*b;
    float z = mr.s2*r + mg.s2*g + mb.s2*b;
    float l = ltable[(int)(y*1024)];
    z = 1.0f / (x + 15.0f*y + 3.0f*z + 1e-35f);
    dst[i] = (float4)(
        l,
        l*(13.0f*4.0f*x*z - 13.0f*un) - minu,
        l*(13.0f*9.0f*y*z - 13.0f*vn) - minv,
        0.0f);
}
