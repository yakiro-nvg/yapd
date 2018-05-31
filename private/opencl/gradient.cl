/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

int2 border(const int x, const int y, const int2 sz)
{
    return (int2)(
        // symmetric padding
        x < 0 ? -x : (x < sz.s0 ? x : (2*sz.s0 - x - 2)),
        y < 0 ? -y : (y < sz.s1 ? y : (2*sz.s1 - y - 2)));
}

int pixel_idx(const int w, const int2 pos)
{
    return pos.y*w + pos.x;
}

__kernel void grad_mag32fc4(
    __global float4* img, const int2 sz,
	__global float* mag, __global float* angle)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int idx = pixel_idx(sz.s0, (int2)(x, y));
    const float4 l = img[pixel_idx(sz.s0, border(x - 1, y, sz))];
    const float4 r = img[pixel_idx(sz.s0, border(x + 1, y, sz))];
    const float4 t = img[pixel_idx(sz.s0, border(x, y - 1, sz))];
    const float4 b = img[pixel_idx(sz.s0, border(x, y + 1, sz))];
    const float dx[3] = {
        (r.s0 - l.s0)*0.5f,
        (r.s1 - l.s1)*0.5f,
        (r.s2 - l.s2)*0.5f
    };
    const float dy[3] = {
        (b.s0 - t.s0)*0.5f,
        (b.s1 - t.s1)*0.5f,
        (b.s2 - t.s2)*0.5
    };
    const float mag2[3] = { // squared magnitude
        dx[0]*dx[0] + dy[0]*dy[0],
        dx[1]*dx[1] + dy[1]*dy[1],
        dx[2]*dx[2] + dy[2]*dy[2]
    };
    int max_idx = mag2[0] > mag2[1] ? 0 : 1;
    max_idx = mag2[max_idx] > mag2[2] ? max_idx : 2;
    mag[idx] = native_sqrt(mag2[max_idx]);
    const float o = mag[idx] == 0 ? 0 : atan2(dy[max_idx], dx[max_idx]);
    angle[idx] = o < 0 ? M_PI_F + o : o;
}

__kernel void grad_mag_norm(
    __global float* mag, __global float* nrm,
	const int2 sz, const float nrm_const)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int idx = pixel_idx(sz.s0, (int2)(x, y));
    mag[idx] = mag[idx] / (nrm[idx] + nrm_const);
}

__kernel void grad_scale_angle(
    __global float* angle, const int2 sz, const float scale)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int idx = pixel_idx(sz.s0, (int2)(x, y));
    angle[idx] = angle[idx]*scale;
}

__kernel void grad_hist(
    __global float8* hist, __global float* mag, __global float* angle,
    const int2 sz, const int bin_size, const int num_orients)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    const int2 sz0 = sz*bin_size;
    const int2 pos0 = pos*bin_size;
    const float nrm = 1.0f / (bin_size*bin_size);
    float h[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    for (int y = 0; y < bin_size; ++y) {
        for (int x = 0; x < bin_size; ++x) {
            const int idx = pixel_idx(sz0.s0, pos0 + (int2)(x, y));
            const float o = angle[idx];
            const int io0 = (int)o;
            const float od = o - io0;
            const int o0 = io0 % num_orients;
            const int o1 = (o0 + 1) % num_orients;
            const float m = mag[idx]*nrm;
            const float m1 = od*m;
            h[o0] += m - m1;
            h[o1] += m1;
        }
    }
    hist[pixel_idx(sz.s0, pos)] = (float8)(
        h[0], h[1], h[2], h[3],
        h[4], h[5], h[6], h[7]
    );
}
