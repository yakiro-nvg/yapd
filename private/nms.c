/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/nms.h>

void
yapd_nms(
    yapd_alloc_t a, void* aud, yapd_mat_t* bbs,
    float casc_thr, float overlap, int greedy)
{
    int* kp;
    int i, j;
    float *as, *xs, *xe, *ys, *ye;
    assert(bbs->size.w == 5);

    // filter negatives
    for (i = bbs->size.h - 1; i >= 0; --i) {
        float* l = (float*)bbs->data + i * 5;
        if (l[4] <= casc_thr) {
            float* r = (float*)bbs->data + --bbs->size.h * 5;
            if (l == r) continue;
            l[0] = r[0]; l[1] = r[1]; l[2] = r[2]; l[3] = r[3]; l[4] = r[4];
        }
    }

    // bubble sort
    while (TRUE) {
        int done = TRUE;
        for (i = bbs->size.h; i > 0; --i) {
            float* const l = (float*)bbs->data + (i - 1) * 5;
            float* const r = (float*)bbs->data + (i + 0) * 5;
            if (l[4] < r[4]) {
                const float t[] = { l[0], l[1], l[2], l[3], l[4] };
                l[0] = r[0]; l[1] = r[1]; l[2] = r[2]; l[3] = r[3]; l[4] = r[4];
                r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3]; r[4] = t[4];
                done = FALSE;
            }
        }
        if (done) break;
    }

    kp = (int*)a.alloc(
        aud, sizeof(int)*bbs->size.h, YAPD_DEFAULT_ALIGN);
    as = (float*)a.alloc(
        aud, sizeof(float)*bbs->size.h, YAPD_DEFAULT_ALIGN);
    xs = (float*)a.alloc(
        aud, sizeof(float)*bbs->size.h, YAPD_DEFAULT_ALIGN);
    xe = (float*)a.alloc(
        aud, sizeof(float)*bbs->size.h, YAPD_DEFAULT_ALIGN);
    ys = (float*)a.alloc(
        aud, sizeof(float)*bbs->size.h, YAPD_DEFAULT_ALIGN);
    ye = (float*)a.alloc(
        aud, sizeof(float)*bbs->size.h, YAPD_DEFAULT_ALIGN);

    // non maximal suspression
    for (i = 0; i < bbs->size.h; ++i) {
        float* const b = (float*)bbs->data + i * 5;
        kp[i] = TRUE;
        as[i] = b[2] * b[3];
        xs[i] = b[0];
        xe[i] = b[0] + b[2];
        ys[i] = b[1];
        ye[i] = b[1] + b[3];
    }
    for (i = 0; i < bbs->size.h; ++i) {
        if (greedy && !kp[i]) continue;
        for (j = i + 1; j < bbs->size.h; ++j) {
            if (!kp[j]) continue;
            const float iw = YAPD_MIN(xe[i], xe[j]) - YAPD_MAX(xs[i], xs[j]);
            if (iw <= 0) continue;
            const float ih = YAPD_MIN(ye[i], ye[j]) - YAPD_MAX(ys[i], ys[j]);
            if (ih <= 0) continue;
            const float o = iw * ih;
            const float u = YAPD_MIN(as[i], as[j]);
            if (o / u > overlap) kp[j] = FALSE;
        }
    }

    // filter results
    for (i = bbs->size.h - 1; i >= 0; --i) {
        if (!kp[i]) {
            float* l = (float*)bbs->data + i * 5;
            float* r = (float*)bbs->data + --bbs->size.h * 5;
            if (l != r) {
                l[0] = r[0]; l[1] = r[1]; l[2] = r[2]; l[3] = r[3]; l[4] = r[4];
            }
        }
    }

    a.dealloc(aud, kp);
    a.dealloc(aud, as);
    a.dealloc(aud, xs);
    a.dealloc(aud, xe);
    a.dealloc(aud, ys);
    a.dealloc(aud, ye);
}