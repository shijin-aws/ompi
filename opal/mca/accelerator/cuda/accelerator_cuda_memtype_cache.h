/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights reserved.
 * $COPYRIGHT$
 *
 * Accelerator memory type cache.
 * Caches pointer→memory_type classification to avoid expensive driver
 * calls (e.g., cuPointerGetAttributes) on every MPI message.
 * Populated proactively via alloc/free interception hooks.
 */

#ifndef ACCELERATOR_CUDA_MEMTYPE_CACHE_H
#define ACCELERATOR_CUDA_MEMTYPE_CACHE_H

#include "opal_config.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#define MEMTYPE_CACHE_MAX_ENTRIES 64

typedef struct {
    uintptr_t start;
    uintptr_t end;       /* start + length */
    int dev_id;
    uint64_t flags;      /* MCA_ACCELERATOR_FLAGS_* */
} opal_accelerator_memtype_cache_entry_t;

typedef struct {
    opal_accelerator_memtype_cache_entry_t entries[MEMTYPE_CACHE_MAX_ENTRIES];
    int count;
    pthread_spinlock_t lock;
} opal_accelerator_memtype_cache_t;

/* Global instance */
extern opal_accelerator_memtype_cache_t opal_accelerator_cuda_memtype_cache;

/* Initialize the cache */
void opal_accelerator_cuda_memtype_cache_init(void);

/* Finalize the cache */
void opal_accelerator_cuda_memtype_cache_fini(void);

/*
 * Insert an allocation into the cache.
 * Called from CUDA alloc hook after successful allocation.
 */
void opal_accelerator_cuda_memtype_cache_insert(void *addr, size_t length, int dev_id);

/*
 * Remove an allocation from the cache.
 * Called from CUDA free hook before actual free.
 */
void opal_accelerator_cuda_memtype_cache_remove(void *addr);

/*
 * Lookup an address in the cache.
 * Returns 1 if found (device memory), 0 if not found.
 * On hit, sets *dev_id and *flags.
 */
int opal_accelerator_cuda_memtype_cache_lookup(const void *addr, int *dev_id, uint64_t *flags);

#endif /* ACCELERATOR_CUDA_MEMTYPE_CACHE_H */
