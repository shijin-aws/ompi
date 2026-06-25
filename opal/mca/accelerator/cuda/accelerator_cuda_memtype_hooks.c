/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights reserved.
 * $COPYRIGHT$
 *
 * CUDA memory allocation/free interception hooks.
 * Intercepts all CUDA allocation/free APIs (driver + runtime) to proactively
 * populate the memtype cache. Matches UCX's UCM coverage.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "opal_config.h"
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

/*
 * We intentionally do NOT include <cuda.h> here to avoid macro conflicts
 * (cuda.h defines cuMemAlloc -> cuMemAlloc_v2 etc.).
 * Instead we use opaque types and forward-declare what we need.
 */
typedef unsigned long long CUdeviceptr_t;
typedef int CUresult_t;
typedef void* CUstream_t;
typedef void* CUmemoryPool_t;
typedef unsigned long long CUmemGenericAllocationHandle_t;

#define CUDA_SUCCESS_VAL 0

#include "accelerator_cuda_memtype_cache.h"

/* ========================================================================
 * Real function pointers — resolved lazily via dlsym
 * ======================================================================== */

static void* (*pdlsym)(void*, const char*) = NULL;

static void *get_real(const char *name)
{
    void *sym = dlsym(RTLD_NEXT, name);
    if (!sym)
        sym = dlsym(RTLD_DEFAULT, name);
    return sym;
}

/* Driver API alloc */
static CUresult_t (*real_cuMemAlloc)(CUdeviceptr_t*, size_t) = NULL;
static CUresult_t (*real_cuMemAlloc_v2)(CUdeviceptr_t*, size_t) = NULL;
static CUresult_t (*real_cuMemAllocManaged)(CUdeviceptr_t*, size_t, unsigned int) = NULL;
static CUresult_t (*real_cuMemAllocPitch)(CUdeviceptr_t*, size_t*, size_t, size_t, unsigned int) = NULL;
static CUresult_t (*real_cuMemAllocPitch_v2)(CUdeviceptr_t*, size_t*, size_t, size_t, unsigned int) = NULL;
static CUresult_t (*real_cuMemAllocAsync)(CUdeviceptr_t*, size_t, CUstream_t) = NULL;
static CUresult_t (*real_cuMemAllocFromPoolAsync)(CUdeviceptr_t*, size_t, CUmemoryPool_t, CUstream_t) = NULL;
static CUresult_t (*real_cuMemMap)(CUdeviceptr_t, size_t, size_t, CUmemGenericAllocationHandle_t, unsigned long long) = NULL;

/* Driver API free */
static CUresult_t (*real_cuMemFree)(CUdeviceptr_t) = NULL;
static CUresult_t (*real_cuMemFree_v2)(CUdeviceptr_t) = NULL;
static CUresult_t (*real_cuMemFreeHost)(void*) = NULL;
static CUresult_t (*real_cuMemFreeHost_v2)(void*) = NULL;
static CUresult_t (*real_cuMemFreeAsync)(CUdeviceptr_t, CUstream_t) = NULL;
static CUresult_t (*real_cuMemUnmap)(CUdeviceptr_t, size_t) = NULL;

/* Runtime API alloc */
static int (*real_cudaMalloc)(void**, size_t) = NULL;
static int (*real_cudaMallocManaged)(void**, size_t, unsigned int) = NULL;
static int (*real_cudaMallocPitch)(void**, size_t*, size_t, size_t) = NULL;
static int (*real_cudaMallocAsync)(void**, size_t, void*) = NULL;
static int (*real_cudaMallocFromPoolAsync)(void**, size_t, void*, void*) = NULL;

/* Runtime API free */
static int (*real_cudaFree)(void*) = NULL;
static int (*real_cudaFreeAsync)(void*, void*) = NULL;
static int (*real_cudaFreeHost)(void*) = NULL;

static int resolved = 0;

static void resolve_all(void)
{
    if (resolved) return;
    real_cuMemAlloc = get_real("cuMemAlloc");
    real_cuMemAlloc_v2 = get_real("cuMemAlloc_v2");
    real_cuMemAllocManaged = get_real("cuMemAllocManaged");
    real_cuMemAllocPitch = get_real("cuMemAllocPitch");
    real_cuMemAllocPitch_v2 = get_real("cuMemAllocPitch_v2");
    real_cuMemAllocAsync = get_real("cuMemAllocAsync");
    real_cuMemAllocFromPoolAsync = get_real("cuMemAllocFromPoolAsync");
    real_cuMemMap = get_real("cuMemMap");
    real_cuMemFree = get_real("cuMemFree");
    real_cuMemFree_v2 = get_real("cuMemFree_v2");
    real_cuMemFreeHost = get_real("cuMemFreeHost");
    real_cuMemFreeHost_v2 = get_real("cuMemFreeHost_v2");
    real_cuMemFreeAsync = get_real("cuMemFreeAsync");
    real_cuMemUnmap = get_real("cuMemUnmap");
    real_cudaMalloc = get_real("cudaMalloc");
    real_cudaMallocManaged = get_real("cudaMallocManaged");
    real_cudaMallocPitch = get_real("cudaMallocPitch");
    real_cudaMallocAsync = get_real("cudaMallocAsync");
    real_cudaMallocFromPoolAsync = get_real("cudaMallocFromPoolAsync");
    real_cudaFree = get_real("cudaFree");
    real_cudaFreeAsync = get_real("cudaFreeAsync");
    real_cudaFreeHost = get_real("cudaFreeHost");
    resolved = 1;
}

static int get_device(void)
{
    /* Minimal device query without including cuda.h */
    static int (*fn)(void*) = NULL;
    int dev = 0;
    if (!fn) fn = get_real("cuCtxGetDevice");
    if (fn && fn(&dev) == 0) return dev;
    return 0;
}

/* ========================================================================
 * CUDA Driver API — Alloc interposition
 * ======================================================================== */

CUresult_t cuMemAlloc(CUdeviceptr_t *dptr, size_t s)
{
    resolve_all();
    if (!real_cuMemAlloc) return 1;
    CUresult_t r = real_cuMemAlloc(dptr, s);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, s, get_device());
    return r;
}

CUresult_t cuMemAlloc_v2(CUdeviceptr_t *dptr, size_t s)
{
    resolve_all();
    if (!real_cuMemAlloc_v2) return 1;
    CUresult_t r = real_cuMemAlloc_v2(dptr, s);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, s, get_device());
    return r;
}

CUresult_t cuMemAllocManaged(CUdeviceptr_t *dptr, size_t s, unsigned int flags)
{
    resolve_all();
    if (!real_cuMemAllocManaged) return 1;
    CUresult_t r = real_cuMemAllocManaged(dptr, s, flags);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, s, get_device());
    return r;
}

CUresult_t cuMemAllocPitch(CUdeviceptr_t *dptr, size_t *pitch,
                           size_t w, size_t h, unsigned int elem)
{
    resolve_all();
    if (!real_cuMemAllocPitch) return 1;
    CUresult_t r = real_cuMemAllocPitch(dptr, pitch, w, h, elem);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, (*pitch)*h, get_device());
    return r;
}

CUresult_t cuMemAllocPitch_v2(CUdeviceptr_t *dptr, size_t *pitch,
                              size_t w, size_t h, unsigned int elem)
{
    resolve_all();
    if (!real_cuMemAllocPitch_v2) return 1;
    CUresult_t r = real_cuMemAllocPitch_v2(dptr, pitch, w, h, elem);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, (*pitch)*h, get_device());
    return r;
}

CUresult_t cuMemAllocAsync(CUdeviceptr_t *dptr, size_t s, CUstream_t stream)
{
    resolve_all();
    if (!real_cuMemAllocAsync) return 1;
    CUresult_t r = real_cuMemAllocAsync(dptr, s, stream);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, s, get_device());
    return r;
}

CUresult_t cuMemAllocFromPoolAsync(CUdeviceptr_t *dptr, size_t s,
                                   CUmemoryPool_t pool, CUstream_t stream)
{
    resolve_all();
    if (!real_cuMemAllocFromPoolAsync) return 1;
    CUresult_t r = real_cuMemAllocFromPoolAsync(dptr, s, pool, stream);
    if (r == 0 && dptr && *dptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)*dptr, s, get_device());
    return r;
}

CUresult_t cuMemMap(CUdeviceptr_t ptr, size_t size, size_t offset,
                    CUmemGenericAllocationHandle_t handle, unsigned long long flags)
{
    resolve_all();
    if (!real_cuMemMap) return 1;
    CUresult_t r = real_cuMemMap(ptr, size, offset, handle, flags);
    if (r == 0 && ptr)
        opal_accelerator_cuda_memtype_cache_insert((void*)ptr, size, get_device());
    return r;
}

/* ========================================================================
 * CUDA Driver API — Free interposition
 * ======================================================================== */

CUresult_t cuMemFree(CUdeviceptr_t dptr)
{
    resolve_all();
    if (dptr) opal_accelerator_cuda_memtype_cache_remove((void*)dptr);
    if (!real_cuMemFree) return 1;
    return real_cuMemFree(dptr);
}

CUresult_t cuMemFree_v2(CUdeviceptr_t dptr)
{
    resolve_all();
    if (dptr) opal_accelerator_cuda_memtype_cache_remove((void*)dptr);
    if (!real_cuMemFree_v2) return 1;
    return real_cuMemFree_v2(dptr);
}

CUresult_t cuMemFreeHost(void *p)
{
    resolve_all();
    if (p) opal_accelerator_cuda_memtype_cache_remove(p);
    if (!real_cuMemFreeHost) return 1;
    return real_cuMemFreeHost(p);
}

CUresult_t cuMemFreeHost_v2(void *p)
{
    resolve_all();
    if (p) opal_accelerator_cuda_memtype_cache_remove(p);
    if (!real_cuMemFreeHost_v2) return 1;
    return real_cuMemFreeHost_v2(p);
}

CUresult_t cuMemFreeAsync(CUdeviceptr_t dptr, CUstream_t stream)
{
    resolve_all();
    if (dptr) opal_accelerator_cuda_memtype_cache_remove((void*)dptr);
    if (!real_cuMemFreeAsync) return 1;
    return real_cuMemFreeAsync(dptr, stream);
}

CUresult_t cuMemUnmap(CUdeviceptr_t ptr, size_t size)
{
    resolve_all();
    if (ptr) opal_accelerator_cuda_memtype_cache_remove((void*)ptr);
    if (!real_cuMemUnmap) return 1;
    return real_cuMemUnmap(ptr, size);
}

/* ========================================================================
 * CUDA Runtime API — Alloc interposition
 * ======================================================================== */

int cudaMalloc(void **devPtr, size_t size)
{
    resolve_all();
    if (!real_cudaMalloc) return 1;
    int r = real_cudaMalloc(devPtr, size);
    if (r == 0 && devPtr && *devPtr)
        opal_accelerator_cuda_memtype_cache_insert(*devPtr, size, get_device());
    return r;
}

int cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    resolve_all();
    if (!real_cudaMallocManaged) return 1;
    int r = real_cudaMallocManaged(devPtr, size, flags);
    if (r == 0 && devPtr && *devPtr)
        opal_accelerator_cuda_memtype_cache_insert(*devPtr, size, get_device());
    return r;
}

int cudaMallocPitch(void **devPtr, size_t *pitch, size_t w, size_t h)
{
    resolve_all();
    if (!real_cudaMallocPitch) return 1;
    int r = real_cudaMallocPitch(devPtr, pitch, w, h);
    if (r == 0 && devPtr && *devPtr)
        opal_accelerator_cuda_memtype_cache_insert(*devPtr, (*pitch)*h, get_device());
    return r;
}

int cudaMallocAsync(void **devPtr, size_t size, void *stream)
{
    resolve_all();
    if (!real_cudaMallocAsync) return 1;
    int r = real_cudaMallocAsync(devPtr, size, stream);
    if (r == 0 && devPtr && *devPtr)
        opal_accelerator_cuda_memtype_cache_insert(*devPtr, size, get_device());
    return r;
}

int cudaMallocFromPoolAsync(void **devPtr, size_t size, void *pool, void *stream)
{
    resolve_all();
    if (!real_cudaMallocFromPoolAsync) return 1;
    int r = real_cudaMallocFromPoolAsync(devPtr, size, pool, stream);
    if (r == 0 && devPtr && *devPtr)
        opal_accelerator_cuda_memtype_cache_insert(*devPtr, size, get_device());
    return r;
}

/* ========================================================================
 * CUDA Runtime API — Free interposition
 * ======================================================================== */

int cudaFree(void *devPtr)
{
    resolve_all();
    if (devPtr) opal_accelerator_cuda_memtype_cache_remove(devPtr);
    if (!real_cudaFree) return 1;
    return real_cudaFree(devPtr);
}

int cudaFreeAsync(void *devPtr, void *stream)
{
    resolve_all();
    if (devPtr) opal_accelerator_cuda_memtype_cache_remove(devPtr);
    if (!real_cudaFreeAsync) return 1;
    return real_cudaFreeAsync(devPtr, stream);
}

int cudaFreeHost(void *ptr)
{
    resolve_all();
    if (ptr) opal_accelerator_cuda_memtype_cache_remove(ptr);
    if (!real_cudaFreeHost) return 1;
    return real_cudaFreeHost(ptr);
}

/* ========================================================================
 * Init / Fini
 * ======================================================================== */

static void __attribute__((constructor)) opal_accelerator_cuda_memtype_hooks_constructor(void)
{
    resolve_all();
}

void opal_accelerator_cuda_memtype_hooks_init(void)
{
    resolve_all();
}

void opal_accelerator_cuda_memtype_hooks_fini(void)
{
    resolved = 0;
}
