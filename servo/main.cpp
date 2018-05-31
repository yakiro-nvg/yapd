/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/platform.h>
#include <opencv/cv.hpp>

#include <yapd/alloc.h>
#include <yapd/gpu.h>
#include <yapd/matrix.h>
#include <yapd/buffer.h>
#include <yapd/channels.h>
#include <yapd/pyramid.h>
#include <yapd/detector.h>

#define WBY_UINT_PTR size_t
#include "wby.h"
#include "cmp.h"
#include <thread>
#include <chrono>
#include <stdarg.h>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

enum { REQUEST_BUFF_SZ = 2048 };
enum { IO_BUFF_SZ = 8192 };
enum { INIT_CAP_W = 1024 };
enum { INIT_CAP_H = 768 };
enum { SCRATCH_CAPACITY = INIT_CAP_W * INIT_CAP_H * 16 * 16};

struct servo_t
{
    bool running;
    yapd_malloc_t amalloc;
    yapd_scratch_t ascratch;
    struct wby_server wby;
    void* wby_buff;
    yapd_gpu_t gpu;
    yapd_buffer_t color;
    yapd_buffer_t mag;
    yapd_buffer_t hist;
    yapd_channels_t channels;
    yapd_pyramid_t pyramid;
    yapd_pyramid_t pyramid_real;
    yapd_detector_t detector;
    HOGDescriptor hogsvm;
    vector<Rect> hogsvm_detections;
    vector<double> hogsvm_weights;
};

static void error(const char* fmt, ...)
{
    va_list args;
    char buf[512];
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    throw std::logic_error(buf);
}

static const struct wby_header CORS_HEADERS[] = {
    { "Access-Control-Allow-Origin",
      "*"
    },
    { "Access-Control-Allow-Methods",
      "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    },
    { "Access-Control-Allow-Headers",
      "Connection, Content-Type"
    },
    { "Access-Control-Max-Age",
      "600"
    }
};

static const struct wby_header MSGP_HEADERS[] = {
    { "Content-Type",
      "application/x-msgpack"
    },
    { "Cache-Control",
      "no-cache"
    },
    { "Access-Control-Allow-Origin",
      "*"
    },
    { "Access-Control-Allow-Methods",
      "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    },
    { "Access-Control-Allow-Headers",
      "Connection, Content-Type"
    },
    { "Access-Control-Max-Age",
      "600"
    }
};

template <int SZ>
static void
write(
    cmp_ctx_t& cmp, const char (&str)[SZ])
{
    cmp_write_str(&cmp, str, SZ - 1);
}

static void
log(const char* msg)
{
    printf("%s\n", msg);
}

static int
simple_response(
    struct wby_con* con, int status)
{
    wby_response_begin(
        con, status, 0,
        CORS_HEADERS, (int)YAPD_STATIC_ARRAY_COUNT(CORS_HEADERS));
    wby_response_end(con);
    return 0;
}

static void
msgp_response_begin(
    struct wby_con* con, int status, int content_length)
{
    wby_response_begin(
        con, status, content_length,
        MSGP_HEADERS, (int)YAPD_STATIC_ARRAY_COUNT(MSGP_HEADERS));
}

static int
cvtype(yapd_type_t type)
{
    switch (type) {
    case YAPD_8U:
        return CV_8U;
    case YAPD_8UC4:
        return CV_8UC4;
    case YAPD_32S:
        return CV_32S;
    case YAPD_32F:
        return CV_32F;
    case YAPD_32FC4:
        return CV_32FC4;
    case YAPD_32FC8:
        return CV_32FC(8);
    case YAPD_32FC16:
        return CV_32FC(16);
    default:
        error("bad type");
        return 0;
    }
}

static yapd_type_t
yapd_type(int type)
{
    switch (type) {
    case CV_8U:
        return YAPD_8U;
    case CV_8UC4:
        return YAPD_8UC4;
    case CV_32S:
        return YAPD_32S;
    case CV_32F:
        return YAPD_32F;
    case CV_32FC4:
        return YAPD_32FC4;
    case CV_32FC(8):
        return YAPD_32FC8;
    case CV_32FC(16):
        return YAPD_32FC16;
    default:
        error("bad type");
        return YAPD_8U;
    }
}

static void
write(
    cmp_ctx_t& cmp, struct wby_con* con, const yapd_mat_t& mat)
{
    if (mat.data == NULL) {
        int rows = 0;
        cmp_write_bin_marker(&cmp, sizeof(int) * 3);
        wby_write(con, &rows, sizeof(int));
        wby_write(con, &rows, sizeof(int)); // pad
        wby_write(con, &rows, sizeof(int)); // pad
    } else {
        const int rows = mat.size.h;
        const int cols = mat.size.w;
        const int type = cvtype(mat.type);
        const int bytes = yapd_mat_bytes(&mat);
        cmp_write_bin_marker(&cmp, bytes + sizeof(int) * 3);
        wby_write(con, &rows, sizeof(int));
        wby_write(con, &cols, sizeof(int));
        wby_write(con, &type, sizeof(int));
        wby_write(con, mat.data, bytes);
    }
}

static bool
read(
    struct wby_con* con, int sz, yapd_mat_t& mat)
{
    if (sz < sizeof(int) * 3) return false;
    int rows; wby_read(con, &rows, sizeof(int));
    if (rows == 0) {
        yapd_mat_release(&mat);
        return true;
    }
    int cols; wby_read(con, &cols, sizeof(int));
    int type; wby_read(con, &type, sizeof(int));
    yapd_mat_create(&mat, cols, rows, yapd_type(type));
    const int bytes = yapd_mat_bytes(&mat);
    if (sz != bytes + sizeof(int) * 3) return false;
    wby_read(con, mat.data, bytes);
    return true;
}

static void
write_buf(
    yapd_alloc_t a, void* aud, yapd_type_t type,
    cmp_ctx_t& cmp, struct wby_con* con,
    yapd_buffer_t& buf, const yapd_size_t& sz)
{
    yapd_mat_t mat = yapd_mat_new(a, aud);
    yapd_mat_create(&mat, sz.w, sz.h, type);
    yapd_buffer_download_sync(&buf, mat.data, yapd_mat_bytes(&mat));
    write(cmp, con, mat);
    yapd_mat_release(&mat);
}

static void
write_buf(
    cmp_ctx_t& cmp, struct wby_con* con, const float* buf, int count)
{
    yapd_mat_t mat = yapd_mat_borrow((uint8_t*)buf, count, 1, YAPD_32F);
    write(cmp, con, mat);
}

static size_t
wby_writer(
    cmp_ctx_t* ctx, const void* data, size_t count)
{
    struct wby_con* con = (struct wby_con*)ctx->buf;
    wby_write(con, data, count);
    return count;
}

static bool
wby_reader(
    cmp_ctx_t* ctx, void* data, size_t limit)
{
    struct wby_con* con = (struct wby_con*)ctx->buf;
    wby_read(con, data, limit);
    return true;
}

static bool
query_float(
    struct wby_con* con, const char* name, float& v)
{
    char var[128];
    int var_len = 0;
    if (con->request.query_params) {
        var_len = wby_find_query_var(
            con->request.query_params, name, var, sizeof(var));
    }
    return var_len > 0 && sscanf(var, "%f", &v) == 1;
}

static bool
query_int(
    struct wby_con* con, const char* name, int& v)
{
    char var[128];
    int var_len = 0;
    if (con->request.query_params) {
        var_len = wby_find_query_var(
            con->request.query_params, name, var, sizeof(var));
    }
    return var_len > 0 && sscanf(var, "%d", &v) == 1;
}

#define BAD_IFN(exp) if (!(exp)) { r = simple_response(con, 400); break; }
#define CONFLICT_IFN(exp) if (!(exp)) { r = simple_response(con, 409); break; }

static int
handle_stop(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        sv.running = false;
        log("Shutting down");
        return simple_response(con, 204);
    }
    return simple_response(con, 405);
}

static int
handle_compute(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src) &&
                (src.type == YAPD_8UC4 || src.type == YAPD_32FC4));
            yapd_channels_t& c = sv.channels;
            yapd_pyramid_t& p = sv.pyramid;
            yapd_channels_prepare(
                &c, &src.size, &sv.color, &sv.mag, &sv.hist);
            const yapd_size_t& crop_sz = c.crop_sz;
            const int crop_totals = crop_sz.w*crop_sz.h;
            yapd_buffer_reserve(&p.tmp, sizeof(cl_float4)*crop_totals);
            yapd_buffer_reserve(&p.img, sizeof(cl_float4)*crop_totals);
            if (src.type == YAPD_8UC4) {
                yapd_buffer_upload_2d(
                    &p.tmp, src.data, sizeof(cl_uchar4)*crop_totals,
                    &crop_sz, sizeof(cl_uchar4)*src.size.w);
                yapd_buffer_luv_from_rgb8uc4(
                    &p.img, &p.tmp, &crop_sz);
            } else {
                yapd_buffer_upload_2d(
                    &p.img, src.data, sizeof(cl_float4)*crop_totals,
                    &crop_sz, sizeof(cl_float4)*src.size.w);
            }
            yapd_channels_compute(
                &c, &p.img, &p.tmp, &sv.color, &sv.mag, &sv.hist);
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            cmp_write_map(&cmp, 5);
            write(cmp, "width");
            cmp_write_integer(&cmp, c.data_sz.w);
            write(cmp, "height");
            cmp_write_integer(&cmp, c.data_sz.h);
            write(cmp, "color");
            write_buf(
                sv.ascratch.aif, &sv.ascratch, YAPD_32FC4,
                cmp, con, sv.color, c.data_sz);
            write(cmp, "mag");
            write_buf(
                sv.ascratch.aif, &sv.ascratch, YAPD_32F,
                cmp, con, sv.mag, c.data_sz);
            write(cmp, "hist");
            write_buf(
                sv.ascratch.aif, &sv.ascratch, YAPD_32FC8,
                cmp, con, sv.hist, c.data_sz);
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_pyramid(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src) &&
                src.type == YAPD_8UC4);
            float lambda_color, lambda_mag, lambda_hist;
            BAD_IFN(
                query_float(con, "lambda_color", lambda_color) &&
                query_float(con, "lambda_mag", lambda_mag) &&
                query_float(con, "lambda_hist", lambda_hist));
            yapd_pyramid_t& p = sv.pyramid;
            yapd_pyramid_compute(
                &p, &src, lambda_color, lambda_mag, lambda_hist);
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            cmp_write_map(&cmp, 5);
            write(cmp, "num_scales");
            cmp_write_integer(&cmp, p.num_scales);
            write(cmp, "scales");
            write_buf(cmp, con, p.scales, p.num_scales);
            write(cmp, "scalesw");
            write_buf(cmp, con, p.scalesw, p.num_scales);
            write(cmp, "scalesh");
            write_buf(cmp, con, p.scalesh, p.num_scales);
            write(cmp, "data");
            cmp_write_array(&cmp, (uint32_t)p.num_scales);
            YAPD_STATIC_ASSERT(sizeof(yapd_feature_t) == sizeof(cl_float16));
            for (int i = 0; i < p.num_scales; ++i) {
                write_buf(
                    sv.ascratch.aif, &sv.ascratch, YAPD_32FC16,
                    cmp, con, p.data[i], p.data_sz[i]);
            }
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_pyramid_real(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src) &&
                src.type == YAPD_8UC4);
            yapd_pyramid_t& p = sv.pyramid_real;
            yapd_pyramid_compute(&p, &src, 0.0f, 0.0f, 0.0f);
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            cmp_write_map(&cmp, 5);
            write(cmp, "num_scales");
            cmp_write_integer(&cmp, p.num_scales);
            write(cmp, "scales");
            write_buf(cmp, con, p.scales, p.num_scales);
            write(cmp, "scalesw");
            write_buf(cmp, con, p.scalesw, p.num_scales);
            write(cmp, "scalesh");
            write_buf(cmp, con, p.scalesh, p.num_scales);
            write(cmp, "data");
            cmp_write_array(&cmp, (uint32_t)p.num_scales);
            for (int i = 0; i < p.num_scales; ++i) {
                write_buf(
                    sv.ascratch.aif, &sv.ascratch, YAPD_32FC16,
                    cmp, con, p.data[i], p.data_sz[i]);
            }
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_classifier(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "PUT") == 0) {
        int r = 0;
        cmp_ctx_t in;
        cmp_init(&in, con, &wby_reader, NULL, NULL);
        char key[256];
        uint32_t map_sz, bin_sz;
        int64_t win_sz_w, win_sz_h;
        int64_t org_win_w, org_win_h;
        int64_t num_weaks, depth, shrink;
        yapd_mat_t thrs = yapd_mat_new(
            sv.ascratch.aif, &sv.ascratch);
        yapd_mat_t hs = yapd_mat_new(
            sv.ascratch.aif, &sv.ascratch);
        yapd_mat_t fids = yapd_mat_new(
            sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(cmp_read_map(&in, &map_sz));
            for (uint32_t i = 0; i < map_sz; ++i) {
                uint32_t key_sz = sizeof(key);
                BAD_IFN(cmp_read_str(&in, key, &key_sz));
                if (strcmp(key, "num_weaks") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &num_weaks) &&
                        num_weaks > 0);
                } else if (strcmp(key, "depth") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &depth) &&
                        depth > 0 && depth < 4);
                } else if (strcmp(key, "shrink") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &shrink) &&
                        shrink > 0);
                } else if (strcmp(key, "win_sz_w") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &win_sz_w) &&
                        win_sz_w > 0);
                } else if (strcmp(key, "win_sz_h") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &win_sz_h) &&
                        win_sz_h > 0);
                } else if (strcmp(key, "org_win_w") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &org_win_w) &&
                        org_win_w > 0);
                } else if (strcmp(key, "org_win_h") == 0) {
                    BAD_IFN(
                        cmp_read_integer(&in, &org_win_h) &&
                        org_win_h > 0);
                } else if (strcmp(key, "thrs") == 0) {
                    BAD_IFN(
                        cmp_read_bin_size(&in, &bin_sz) &&
                        read(con, bin_sz, thrs) &&
                        thrs.type == YAPD_32F);
                } else if (strcmp(key, "fids") == 0) {
                    BAD_IFN(
                        cmp_read_bin_size(&in, &bin_sz) &&
                        read(con, bin_sz, fids) &&
                        fids.type == YAPD_32S);
                } else if (strcmp(key, "hs") == 0) {
                    BAD_IFN(
                        cmp_read_bin_size(&in, &bin_sz) &&
                        read(con, bin_sz, hs) &&
                        hs.type == YAPD_32F);
                }
            }
            BAD_IFN(
                thrs.size.h == num_weaks &&
                thrs.size.w == YAPD_DETECTOR_TREE_NODES &&
                fids.size.h == num_weaks &&
                fids.size.w == YAPD_DETECTOR_TREE_NODES &&
                hs.size.h == num_weaks &&
                hs.size.w == YAPD_DETECTOR_TREE_NODES);
            const yapd_size_t win_sz = { (int)win_sz_w, (int)win_sz_h };
            const yapd_size_t org_win = { (int)org_win_w, (int)org_win_h };
            yapd_detector_classifier(
                &sv.detector, TRUE, (int)num_weaks, (int)depth,
                (int)shrink, &win_sz, &org_win, &thrs, &fids, &hs);
            r = simple_response(con, 204);
        } while (FALSE);
        yapd_mat_release(&thrs);
        yapd_mat_release(&hs);
        yapd_mat_release(&fids);
        return r;
    } else if (strcmp(con->request.method, "GET") == 0) {
        yapd_detector_t& d = sv.detector;
        const yapd_size_t sz = { 8, d.num_weaks };
        cmp_ctx_t cmp;
        cmp_init(&cmp, con, NULL, NULL, &wby_writer);
        msgp_response_begin(con, 200, -1);
        cmp_write_map(&cmp, 10);
        write(cmp, "num_weaks");
        cmp_write_integer(&cmp, d.num_weaks);
        write(cmp, "depth");
        cmp_write_integer(&cmp, d.depth);
        write(cmp, "shrink");
        cmp_write_integer(&cmp, d.shrink);
        write(cmp, "win_sz_w");
        cmp_write_integer(&cmp, d.win_sz.w);
        write(cmp, "win_sz_h");
        cmp_write_integer(&cmp, d.win_sz.h);
        write(cmp, "org_win_w");
        cmp_write_integer(&cmp, d.org_win.w);
        write(cmp, "org_win_h");
        cmp_write_integer(&cmp, d.org_win.h);
        write(cmp, "thrs");
        write_buf(
            sv.ascratch.aif, &sv.ascratch, YAPD_32F,
            cmp, con, d.thrs, sz);
        write(cmp, "fids");
        write_buf(
            sv.ascratch.aif, &sv.ascratch, YAPD_32S,
            cmp, con, d.fids, sz);
        write(cmp, "hs");
        write_buf(
            sv.ascratch.aif, &sv.ascratch, YAPD_32F,
            cmp, con, d.hs, sz);
        wby_response_end(con);
        return 0;
    }
    return simple_response(con, 405);
}

static int
handle_detect(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src) &&
                src.type == YAPD_8UC4);
            int stride;
            BAD_IFN(
                query_int(con, "stride", stride) &&
                stride > 0);
            float casc_thr, lambda_color, lambda_mag, lambda_hist;
            BAD_IFN(
                query_float(con, "casc_thr", casc_thr) &&
                query_float(con, "lambda_color", lambda_color) &&
                query_float(con, "lambda_mag", lambda_mag) &&
                query_float(con, "lambda_hist", lambda_hist));
            yapd_detector_t& d = sv.detector;
            CONFLICT_IFN(yapd_detector_ready(&d));
            yapd_pyramid_t& p = sv.pyramid;
            yapd_pyramid_compute(
                &p, &src, lambda_color, lambda_mag, lambda_hist);
            yapd_mat_t bbs = yapd_detector_predict(
                sv.ascratch.aif, &sv.ascratch, &d, &p, stride, casc_thr);
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            write(cmp, con, bbs);
            yapd_mat_release(&bbs);
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_benchmark(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src) &&
                src.type == YAPD_8UC4);
            int num_frames, stride;
            BAD_IFN(
                query_int(con, "num_frames", num_frames) &&
                num_frames > 0);
            BAD_IFN(
                query_int(con, "stride", stride) &&
                stride > 0);
            float casc_thr, lambda_color, lambda_mag, lambda_hist;
            BAD_IFN(
                query_float(con, "casc_thr", casc_thr) &&
                query_float(con, "lambda_color", lambda_color) &&
                query_float(con, "lambda_mag", lambda_mag) &&
                query_float(con, "lambda_hist", lambda_hist));
            yapd_detector_t& d = sv.detector;
            CONFLICT_IFN(yapd_detector_ready(&d));
            yapd_pyramid_t& p = sv.pyramid;
            auto t0 = chrono::high_resolution_clock::now();
            for (int i = 0; i < num_frames; ++i) {
                yapd_pyramid_compute(
                    &p, &src, lambda_color, lambda_mag, lambda_hist);
                yapd_mat_t bbs = yapd_detector_predict(
                    sv.ascratch.aif, &sv.ascratch, &d, &p, stride, casc_thr);
                yapd_mat_release(&bbs);
            }
            auto delta = chrono::high_resolution_clock::now() - t0;
            auto fps = 1000000000 / (delta / num_frames).count();
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            cmp_write_map(&cmp, 1);
            write(cmp, "fps");
            cmp_write_integer(&cmp, (int)fps);
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_demo(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        do {
            int stride;
            BAD_IFN(
                query_int(con, "stride", stride) &&
                stride > 0);
            float casc_thr, lambda_color, lambda_mag, lambda_hist;
            BAD_IFN(
                query_float(con, "casc_thr", casc_thr) &&
                query_float(con, "lambda_color", lambda_color) &&
                query_float(con, "lambda_mag", lambda_mag) &&
                query_float(con, "lambda_hist", lambda_hist));
            yapd_detector_t& d = sv.detector;
            CONFLICT_IFN(yapd_detector_ready(&d));
            r = simple_response(con, 204);
            yapd_pyramid_t& p = sv.pyramid;
            Mat f, rgba;
            float fps_ring[10] = { 0 };
            enum { FPS_RING_SZ = YAPD_STATIC_ARRAY_COUNT(fps_ring) };
            int fps_ring_idx = 0;
            VideoCapture cap("C:\\Users\\Giang\\Desktop\\bk3\\test_video\\V006.avi");
            const char* win_name = "Pedestrian Detector";
            while (TRUE) {
                cap >> f;
                if (waitKey(1) == 'q' || f.empty()) {
                    destroyWindow(win_name);
                    break;
                }
                assert(f.type() == CV_8UC3);
                cvtColor(f, rgba, COLOR_BGR2RGBA);
                yapd_mat_t src = yapd_mat_borrow(
                    rgba.data, f.cols, f.rows, YAPD_8UC4);
                auto t0 = chrono::high_resolution_clock::now();
                yapd_pyramid_compute(
                    &p, &src, lambda_color, lambda_mag, lambda_hist);
                yapd_mat_t bbs = yapd_detector_predict(
                    sv.ascratch.aif, &sv.ascratch, &d, &p, stride, casc_thr);
                auto delta = chrono::high_resolution_clock::now() - t0;
                const float* bbsf = (const float*)bbs.data;
                for (int i = 0; i < bbs.size.h; ++i) {
                    const int x = (int)bbsf[i * 5 + 0];
                    const int y = (int)bbsf[i * 5 + 1];
                    const int w = (int)bbsf[i * 5 + 2];
                    const int h = (int)bbsf[i * 5 + 3];
                    const int s = (int)bbsf[i * 5 + 4];
                    if (s < 50) continue;
                    Scalar color = Scalar(0, YAPD_MIN(s/50, 1.0f)*255, 0);
                    rectangle(f, Rect(x, y, w, h), color);
                }
                fps_ring[fps_ring_idx++ % FPS_RING_SZ] =
                    1000000000.0f / delta.count();
                const float fps = (float)accumulate(
                    begin(fps_ring), end(fps_ring), 0.0f) / FPS_RING_SZ;
                char fps_label[128];
                snprintf(
                    fps_label, sizeof(fps_label),
                    "Detection FPS: %u", (uint32_t)fps);
                putText(
                    f, fps_label, Point2f(20, 20),
                    FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255));
                imshow(win_name, f);
                yapd_mat_release(&bbs);
            }
        } while (FALSE);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_hogsvm(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src));
            Mat cvsrc(src.size.h, src.size.w, cvtype(src.type), src.data);
            cvtColor(cvsrc, cvsrc, COLOR_RGBA2BGR);
            sv.hogsvm.detectMultiScale(
                cvsrc, sv.hogsvm_detections, sv.hogsvm_weights);
            yapd_mat_t bbs = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
            yapd_mat_create(
                &bbs, 5, (int)sv.hogsvm_detections.size(), YAPD_32F);
            for (size_t i = 0; i < sv.hogsvm_detections.size(); ++i) {
                Rect& box = sv.hogsvm_detections[i];
                float* b = (float*)bbs.data + i * 5;
                b[0] = (float)box.x;
                b[1] = (float)box.y;
                b[2] = (float)box.width;
                b[3] = (float)box.height;
                b[4] = (float)sv.hogsvm_weights[i];
            }
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            write(cmp, con, bbs);
            wby_response_end(con);
            yapd_mat_release(&bbs);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_hogsvm_benchmark(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        yapd_mat_t src = yapd_mat_new(sv.ascratch.aif, &sv.ascratch);
        do {
            BAD_IFN(
                read(con, con->request.content_length, src));
            int num_frames;
            BAD_IFN(
                query_int(con, "num_frames", num_frames) &&
                num_frames > 0);
            Mat cvsrc(src.size.h, src.size.w, cvtype(src.type), src.data);
            cvtColor(cvsrc, cvsrc, COLOR_RGBA2BGR);
            auto t0 = chrono::high_resolution_clock::now();
            for (int i = 0; i < num_frames; ++i) {
                sv.hogsvm.detectMultiScale(
                    cvsrc, sv.hogsvm_detections, sv.hogsvm_weights);
            }
            auto delta = chrono::high_resolution_clock::now() - t0;
            auto fps = 1000000000 / (delta / num_frames).count();
            cmp_ctx_t cmp;
            cmp_init(&cmp, con, NULL, NULL, &wby_writer);
            msgp_response_begin(con, 200, -1);
            cmp_write_map(&cmp, 1);
            write(cmp, "fps");
            cmp_write_integer(&cmp, (int)fps);
            wby_response_end(con);
        } while (FALSE);
        yapd_mat_release(&src);
        return r;
    }
    return simple_response(con, 405);
}

static int
handle_hogsvm_demo(
    servo_t& sv, struct wby_con* con)
{
    if (strcmp(con->request.method, "POST") == 0) {
        int r = 0;
        do {
            r = simple_response(con, 204);
            Mat f, rgba;
            VideoCapture cap("C:\\Users\\Giang\\Desktop\\test_video\\V006.avi");
            const char* win_name = "Pedestrian Detector";
            while (TRUE) {
                cap >> f;
                if (waitKey(1) == 'q' || f.empty()) {
                    destroyWindow(win_name);
                    break;
                }
                auto t0 = chrono::high_resolution_clock::now();
                sv.hogsvm.detectMultiScale(
                    f, sv.hogsvm_detections, sv.hogsvm_weights);
                auto delta = chrono::high_resolution_clock::now() - t0;
                for (size_t j = 0; j < sv.hogsvm_detections.size(); j++) {
                    double weight = sv.hogsvm_weights[j];
                    Scalar color = Scalar(0, weight * weight * 200, 0);
                    const Rect& box = sv.hogsvm_detections[j];
                    rectangle(f, box, color);
                }
                auto fps = 1000000000 / delta.count();
                char fps_label[128];
                snprintf(
                    fps_label, sizeof(fps_label),
                    "Detection FPS: %u", (uint32_t)fps);
                putText(
                    f, fps_label, Point2f(20, 20),
                    FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255));
                imshow(win_name, f);
            }
        } while (FALSE);
        return r;
    }
    return simple_response(con, 405);
}

static int
dispatch(
    struct wby_con* con, void* ud)
{
    if (strcmp(con->request.method, "OPTIONS") == 0) {
        return simple_response(con, 200);
    } else {
        servo_t& sv = *(servo_t*)ud;
        const char* uri = con->request.uri;
        if (strcmp(uri, "/stop") == 0) {
            return handle_stop(sv, con);
        }
        if (strcmp(uri, "/compute") == 0) {
            return handle_compute(sv, con);
        }
        if (strcmp(uri, "/pyramid") == 0) {
            return handle_pyramid(sv, con);
        }
        if (strcmp(uri, "/pyramid_real") == 0) {
            return handle_pyramid_real(sv, con);
        }
        if (strcmp(uri, "/classifier") == 0) {
            return handle_classifier(sv, con);
        }
        if (strcmp(uri, "/detect") == 0) {
            return handle_detect(sv, con);
        }
        if (strcmp(uri, "/benchmark") == 0) {
            return handle_benchmark(sv, con);
        }
        if (strcmp(uri, "/demo") == 0) {
            return handle_demo(sv, con);
        }
        if (strcmp(uri, "/hogsvm") == 0) {
            return handle_hogsvm(sv, con);
        }
        if (strcmp(uri, "/hogsvm_benchmark") == 0) {
            return handle_hogsvm_benchmark(sv, con);
        }
        if (strcmp(uri, "/hogsvm_demo") == 0) {
            return handle_hogsvm_demo(sv, con);
        }
        return 1;
    }
}

static void
config_callback(
    struct wby_config& cfg)
{
    cfg.dispatch = &dispatch;
    cfg.ws_connect = NULL;
    cfg.ws_connected = NULL;
    cfg.ws_frame = NULL;
    cfg.ws_closed = NULL;
}

static void
config(
    struct wby_config& cfg, void* ud,
    const char* address, uint16_t port, int max_connections)
{
    cfg.userdata = ud;
    cfg.address = address;
    cfg.port = port;
    cfg.connection_max = (unsigned int)max_connections;
    cfg.request_buffer_size = REQUEST_BUFF_SZ;
    cfg.io_buffer_size = IO_BUFF_SZ;
    config_callback(cfg);
}

static void
servo_init(
    servo_t& self, const char* address, uint16_t port, int max_connections)
{
    self.running = true;
    yapd_malloc_new(&self.amalloc);
    yapd_scratch_new(
        &self.ascratch, self.amalloc.aif, &self.amalloc, SCRATCH_CAPACITY);

    wby_size wby_buff_sz;
    struct wby_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    config(cfg, &self, address, port, max_connections);
    wby_init(&self.wby, &cfg, &wby_buff_sz);
    self.wby_buff = self.amalloc.aif.alloc(
        &self.amalloc, (int)wby_buff_sz, YAPD_DEFAULT_ALIGN);
    if (wby_start(&self.wby, self.wby_buff) != 0) {
        error("failed to start webby server");
    }

    self.gpu = yapd_gpu_new();
    self.color = yapd_buffer_create(&self.gpu, 0);
    self.mag = yapd_buffer_create(&self.gpu, 0);
    self.hist = yapd_buffer_create(&self.gpu, 0);

    self.channels = yapd_channels_new(
        self.amalloc.aif, &self.amalloc,
        &self.gpu, INIT_CAP_W, INIT_CAP_H, NULL);

    yapd_pyramid_opts_t opts = *yapd_pyramid_default_opts();
    opts.pad.w = 12;
    opts.pad.h = 16;
    opts.min_ds.w = 41;
    opts.min_ds.h = 100;
    opts.num_approx = -1;
    opts.smooth = 1;
    self.pyramid = yapd_pyramid_new(
        self.amalloc.aif, &self.amalloc, &self.gpu, &self.channels, &opts);
    self.pyramid_real = yapd_pyramid_new(
        self.amalloc.aif, &self.amalloc, &self.gpu, &self.channels, NULL);

    self.detector = yapd_detector_new(
        self.amalloc.aif, &self.amalloc, &self.gpu);

#if 0
    self.hogsvm.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
#endif
}

static void
servo_cleanup(
    servo_t& self)
{
    yapd_gpu_sync(&self.gpu);
    yapd_buffer_release(&self.color);
    yapd_buffer_release(&self.mag);
    yapd_buffer_release(&self.hist);
    yapd_channels_release(&self.channels);
    yapd_pyramid_release(&self.pyramid);
    yapd_pyramid_release(&self.pyramid_real);
    yapd_detector_release(&self.detector);
    yapd_gpu_release(&self.gpu);
    if (self.wby_buff != NULL) {
        wby_stop(&self.wby);
        self.amalloc.aif.dealloc(&self.amalloc, self.wby_buff);
        self.wby_buff = NULL;
    }
    yapd_scratch_release(&self.ascratch);
    yapd_malloc_release(&self.amalloc);
}

int main()
{
#ifdef YAPD_WINDOWS
    WSADATA wsa_data;
    int ws_err = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (ws_err != 0) {
        error("WSAStartup failed %d", ws_err);
    }
#endif
    servo_t sv;
    servo_init(sv, "0.0.0.0", 41991, 100);
    log("Servo started at port 41991");
    while (sv.running) {
        wby_update(&sv.wby, FALSE);
        this_thread::sleep_for(chrono::milliseconds(1));
    }
    log("Bye bye");
    servo_cleanup(sv);
#ifdef YAPD_WINDOWS
    WSACleanup();
#endif
    return 0;
}
