/* Copyright (c) 2018 Giang "Yakiro" Nguyen. All rights reserved. */
#include <yapd/alloc.h>

// borrowed from: https://bitbucket.org/bitsquid/foundation/

// header stored at the beginning of a memory allocation
// to indicate the size of the allocated data
typedef struct header_s { uint32_t size; } header_t;

// if we need to align the memory allocation we pad the
// header with this value after storing the size
const uint32_t HEADER_PAD_VALUE = 0xffffffffu;

static YAPD_INLINE void*
align_forward(
    void* p, uint32_t align)
{
    uintptr_t pi = (uintptr_t)p;
    const uint32_t mod = pi % align;
    if (mod) pi += align - mod;
    return (void*)pi;
}

static YAPD_INLINE uint8_t*
data_pointer(
    header_t* header, int align)
{
    return (uint8_t*)align_forward(header + 1, align);
}

static YAPD_INLINE header_t*
header(
    void* data)
{
    uint32_t* p = (uint32_t*)data;
    while (p[-1] == HEADER_PAD_VALUE) --p;
    return ((header_t*)p) - 1;
}

static YAPD_INLINE void
fill(
    header_t* header, void* data, uint32_t size)
{
    uint32_t* p = (uint32_t*)(header + 1);
    header->size = size;
    while ((void*)p < data) *p++ = HEADER_PAD_VALUE;
}

// returns the size to allocate from malloc() for a given size and align
static YAPD_INLINE uint32_t
size_with_padding(
    uint32_t size, uint32_t align)
{
    return size + align + sizeof(header_t);
}

static void*
malloc_alloc(
    void* ud, int sz, int align)
{
    yapd_malloc_t* a = (yapd_malloc_t*)ud;
    const uint32_t ts = size_with_padding(sz, align);
    header_t* h = (header_t*)malloc(ts);
    void* p = data_pointer(h, align);
    fill(h, p, ts);
    a->allocated += ts;
    return p;
}

static void
malloc_dealloc(
    void* ud, void* p)
{
    if (!p) return;
    else {
        yapd_malloc_t* a = (yapd_malloc_t*)ud;
        header_t* h = header(p);
        a->allocated -= h->size;
        free(h);
    }
}

static YAPD_INLINE int
in_use(
    yapd_scratch_t* s, void* p)
{
    uint8_t* pb = (uint8_t*)p;
    if (s->free == s->allocate) return FALSE;
    if (s->allocate > s->free) {
        return pb >= s->free && pb < s->allocate;
    } else {
        return pb >= s->free || pb < s->allocate;
    }
}

static void*
scratch_alloc(
    void* ud, int sz, int align)
{
    uint8_t* data;
    yapd_scratch_t* a = (yapd_scratch_t*)ud;
    uint8_t* p = a->allocate;
    header_t* h = (header_t*)p;
    assert(align % 4 == 0);
    sz = ((sz + 3) / 4) * 4;
    data = data_pointer(h, align);
    p = data + sz;
    if (p > a->end) {
        // wrap around to the beginning
        h->size = (uint32_t)((a->end - (uint8_t*)h) | 0x80000000u);
        p = a->begin;
        h = (header_t*)p;
        data = data_pointer(h, align);
        p = data + sz;
    }
    if (in_use(a, p)) {
        return a->backing.alloc(a->backing_ud, sz, align);
    }
    fill(h, data, (uint32_t)(p - (uint8_t*)h));
    a->allocate = p;
    return data;
}

static void
scratch_dealloc(
    void* ud, void* p)
{
    yapd_scratch_t* a = (yapd_scratch_t*)ud;
    uint8_t* pb = (uint8_t*)p;
    if (!p) return;
    if (pb < a->begin || pb >= a->end) {
        a->backing.dealloc(a->backing_ud, p);
    } else {
        // mark this slot as free
        header_t* h = header(p);
        assert((h->size & 0x80000000u) == 0);
        h->size = h->size | 0x80000000u;
        // advance the free pointer past all free slots
        while (a->free != a->allocate) {
            header_t* h = (header_t*)a->free;
            if ((h->size & 0x80000000u) == 0) break;
            a->free += h->size & 0x7fffffffu;
            if (a->free == a->end) a->free = a->begin;
        }
    }
}

void
yapd_malloc_new(
    yapd_malloc_t* a)
{
    a->allocated = 0;
    a->aif.alloc = &malloc_alloc;
    a->aif.dealloc = &malloc_dealloc;
}

void
yapd_malloc_release(
    yapd_malloc_t* a)
{
    assert(a->allocated == 0);
    memset(a, 0, sizeof(yapd_malloc_t));
}

void
yapd_scratch_new(
    yapd_scratch_t* a, yapd_alloc_t backing, void* backing_ud, int cap)
{
    a->aif.alloc = &scratch_alloc;
    a->aif.dealloc = &scratch_dealloc;
    a->backing = backing;
    a->backing_ud = backing_ud;
    a->begin = (uint8_t*)backing.alloc(backing_ud, cap, YAPD_DEFAULT_ALIGN);
    a->end = a->begin + cap;
    a->allocate = a->begin;
    a->free = a->begin;
}

void
yapd_scratch_release(
    yapd_scratch_t* a)
{
    assert(a->free == a->allocate);
    a->backing.dealloc(a->backing_ud, a->begin);
    memset(a, 0, sizeof(yapd_scratch_t));
}