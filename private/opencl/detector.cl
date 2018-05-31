/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */

#define TREE_NODES 8
#define EARLY_WEAKS 32

void get_child(
    __global float* thrs,
    __global int* fids,
    __global float* chns,
    __global int* cids,
    const int off,
    int* k0, int* k)
{
    float ftr = chns[cids[fids[*k]]];
    *k = ftr < thrs[*k] ? 1 : 2;
    *k0 = *k += (*k0)*2; *k += off;
}

__kernel void detector_early_reject(
    const int depth,
    const int to_org,
    const int org_w,
    const int out_w,
    const float casc_thr,
    __global float* thrs,
    __global float* hs,
    __global int* fids,
    __global float* chns,
    __global int* cids,
    __global float* out,
    __global int* tmp,
    const int off)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    const int2 org_pos = pos*to_org;
    const int chns_off = (org_pos.y*org_w + org_pos.x)*16;
    const int out_idx = off + pos.y*out_w + pos.x;
    float h = 0.0f;
    for (int t = 0; t < EARLY_WEAKS; ++t) {
        const int off = t*TREE_NODES;
        int k = off, k0 = 0;
        for (int i = 0; i < depth; ++i) {
            get_child(
                thrs, fids, chns + chns_off,
                cids, off, &k0, &k);
        }
        h += hs[k]; if (h <= casc_thr) break;
    }
    out[out_idx] = h;
    tmp[out_idx] = h > casc_thr;
}

__kernel void detector_early_scan(
    __global int* tmp, const int w, const int off,
    __global int* idx, __global int* len, const int len_off)
{
    const int y = get_global_id(0);
    __global int* i = tmp + off + y*w; int a = i[0];
    __global int* o = idx + off + y*w; o[0] = 0;
    for (int j = 1; j < w; ++j) {
        a += i[j]; o[j] = i[j - 1] + o[j - 1];
    }
    len[len_off + y] = a > 0 ? (o[w - 1] + (i[w - 1] == 1)) : 0;
}

__kernel void detector_early_prefix_sum(
    __global int* len, __global int* sum, __constant int2* dsz)
{
    const int s = get_global_id(0);
    const int h = dsz[s].s0;
    __global int* la = len + dsz[s].s1;
    __global int* sa = sum + dsz[s].s1; sa[0] = 0;
    for (int i = 1; i < h; ++i) {
        sa[i] = la[i - 1] + sa[i - 1];
    }
}

__kernel void detector_early_bbs(
    const int out_w, const float casc_thr,
    const int bbs_off, const int sum_off, const int off,
    __global float* out, __global int* idx,
    __global float* bbs, __global int* sum)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    const int out_idx = off + pos.y*out_w + pos.x;
    const float h = out[out_idx]; if (h <= casc_thr) return;
    const int bbs_idx = bbs_off + (sum[sum_off + pos.y] + idx[out_idx])*5;
    bbs[bbs_idx + 0] = pos.x; bbs[bbs_idx + 1] = pos.y;
    bbs[bbs_idx + 4] = h;
}

__kernel void detector_predict(
    const int num_weaks,
    const int depth,
    const int to_org,
    const int org_w,
    const float casc_thr,
    const int bbs_off,
    const int hss_off,
    __global float* thrs,
    __global float* hs,
    __global int* fids,
    __global float* chns,
    __global int* cids,
    __global float* bbs,
    __global float* hss)
{
    __global float* b = bbs + bbs_off + get_global_id(0)*5;
    __global float* h = hss + hss_off + get_global_id(0)*num_weaks;
    const int2 pos = { b[0], b[1] };
    const int2 org_pos = pos*to_org;
    const int chns_off = (org_pos.y*org_w + org_pos.x)*16;
    const int t = get_global_id(1);
    const int off = t*TREE_NODES;
    int k = off, k0 = 0;
    for (int i = 0; i < depth; ++i) {
        get_child(
            thrs, fids, chns + chns_off,
            cids, off, &k0, &k);
    }
    h[t] = hs[k];
}

__kernel void detector_predict_sum(
    const int num_weaks, const float casc_thr,
    __global float* hss, __global float* bss)
{
    __global float* h = hss + get_global_id(0)*num_weaks;
    __global float* b = bss + get_global_id(0)*5; b[4] = h[0];
    for (int i = 1; i < num_weaks; ++i) {
        b[4] += h[i]; if (b[4] <= -1) break;
    }
}