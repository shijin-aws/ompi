/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights reserved.
 * $COPYRIGHT$
 *
 * Accelerator memory type cache implementation.
 * Sorted array of address ranges with binary search for O(log n) lookup.
 */

#include "accelerator_cuda_memtype_cache.h"
#include <string.h>

opal_accelerator_memtype_cache_t opal_accelerator_cuda_memtype_cache;

void opal_accelerator_cuda_memtype_cache_init(void)
{
    opal_accelerator_cuda_memtype_cache.count = 0;
    pthread_spin_init(&opal_accelerator_cuda_memtype_cache.lock, PTHREAD_PROCESS_PRIVATE);
}

void opal_accelerator_cuda_memtype_cache_fini(void)
{
    pthread_spin_destroy(&opal_accelerator_cuda_memtype_cache.lock);
}

/*
 * Binary search: find the entry containing addr, or the insertion point.
 * Returns index of entry where entry.start <= addr < entry.end, or -1.
 */
static int memtype_cache_find(opal_accelerator_memtype_cache_t *cache, uintptr_t addr)
{
    int lo = 0, hi = cache->count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (addr < cache->entries[mid].start) {
            hi = mid - 1;
        } else if (addr >= cache->entries[mid].end) {
            lo = mid + 1;
        } else {
            return mid; /* hit: start <= addr < end */
        }
    }
    return -1;
}

/*
 * Find insertion index (first entry with start > addr).
 */
static int memtype_cache_insert_idx(opal_accelerator_memtype_cache_t *cache, uintptr_t addr)
{
    int lo = 0, hi = cache->count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cache->entries[mid].start <= addr) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void opal_accelerator_cuda_memtype_cache_insert(void *addr, size_t length, int dev_id)
{
    opal_accelerator_memtype_cache_t *cache = &opal_accelerator_cuda_memtype_cache;

    pthread_spin_lock(&cache->lock);

    if (cache->count >= MEMTYPE_CACHE_MAX_ENTRIES) {
        /* Cache full — drop oldest (first) entry to make room */
        memmove(&cache->entries[0], &cache->entries[1],
                (cache->count - 1) * sizeof(cache->entries[0]));
        cache->count--;
    }

    int idx = memtype_cache_insert_idx(cache, (uintptr_t)addr);

    /* Shift entries to make room */
    if (idx < cache->count) {
        memmove(&cache->entries[idx + 1], &cache->entries[idx],
                (cache->count - idx) * sizeof(cache->entries[0]));
    }

    cache->entries[idx].start = (uintptr_t)addr;
    cache->entries[idx].end = (uintptr_t)addr + length;
    cache->entries[idx].dev_id = dev_id;
    cache->entries[idx].flags = 0;
    cache->count++;

    pthread_spin_unlock(&cache->lock);
}

void opal_accelerator_cuda_memtype_cache_remove(void *addr)
{
    opal_accelerator_memtype_cache_t *cache = &opal_accelerator_cuda_memtype_cache;

    pthread_spin_lock(&cache->lock);

    int idx = memtype_cache_find(cache, (uintptr_t)addr);
    if (idx >= 0) {
        /* Remove by shifting */
        if (idx < cache->count - 1) {
            memmove(&cache->entries[idx], &cache->entries[idx + 1],
                    (cache->count - idx - 1) * sizeof(cache->entries[0]));
        }
        cache->count--;
    }

    pthread_spin_unlock(&cache->lock);
}

int opal_accelerator_cuda_memtype_cache_lookup(const void *addr, int *dev_id, uint64_t *flags)
{
    opal_accelerator_memtype_cache_t *cache = &opal_accelerator_cuda_memtype_cache;
    int result;

    pthread_spin_lock(&cache->lock);

    int idx = memtype_cache_find(cache, (uintptr_t)addr);
    if (idx >= 0) {
        *dev_id = cache->entries[idx].dev_id;
        *flags = cache->entries[idx].flags;
        result = 1;
    } else {
        result = 0;
    }

    pthread_spin_unlock(&cache->lock);
    return result;
}
