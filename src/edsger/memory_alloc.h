#ifndef MEMORY_ALLOC_H
#define MEMORY_ALLOC_H

/*
 * Cross-Platform Memory Allocation Wrapper
 *
 * Provides a unified interface for memory allocation with platform-specific
 * optimizations:
 *
 * Windows (USE_LARGE_PAGES defined):
 *   - Tries Large Pages (2MB pages) to reduce TLB misses
 *   - Falls back to VirtualAlloc with VirtualLock
 *   - Final fallback to standard malloc
 *
 * Linux/macOS:
 *   - Uses madvise() with MADV_HUGEPAGE hint for transparent huge pages
 *   - Falls back to standard malloc
 *
 * Memory Statistics:
 *   Tracks allocation method used (for debugging and performance analysis)
 */

#include <stdlib.h>
#include <string.h>

// Memory allocation statistics structure
typedef struct {
    int used_large_pages;      // 1 if Large Pages were used (Windows)
    int used_virtual_lock;     // 1 if VirtualLock was used (Windows)
    int used_madvise_hugepage; // 1 if madvise HUGEPAGE was used (Linux)
    size_t allocated_size;     // Actual allocated size
    void* ptr;                 // Allocated pointer
} MemoryStats;

// ============================================================================
// Windows Implementation
// ============================================================================

#ifdef _WIN32

#include "win_memory.h"
#include <malloc.h>

static void* platform_malloc(size_t size, MemoryStats* stats) {
    void* ptr = NULL;

    // Initialize stats
    memset(stats, 0, sizeof(MemoryStats));
    stats->allocated_size = size;

#ifdef USE_LARGE_PAGES
    // Strategy 1: Try Large Pages allocation
    ptr = win_alloc_large_pages(size, &stats->used_large_pages);

    if (ptr != NULL) {
        stats->ptr = ptr;
        return ptr;
    }

    // Strategy 2: Try VirtualAlloc with VirtualLock
    ptr = win_alloc_virtual(size, &stats->used_virtual_lock);

    if (ptr != NULL) {
        stats->ptr = ptr;
        return ptr;
    }
#endif // USE_LARGE_PAGES

    // Strategy 3: Fallback to aligned malloc (64-byte alignment for cache lines)
    ptr = _aligned_malloc(size, 64);

    if (ptr != NULL) {
        stats->ptr = ptr;
    }

    return ptr;
}

static void platform_free(void* ptr, MemoryStats* stats) {
    if (ptr == NULL) {
        return;
    }

#ifdef USE_LARGE_PAGES
    // Free Large Pages or VirtualAlloc memory
    if (stats->used_large_pages || stats->used_virtual_lock) {
        win_free_virtual(ptr, stats->allocated_size, stats->used_virtual_lock);
        return;
    }
#endif // USE_LARGE_PAGES

    // Free aligned malloc memory
    _aligned_free(ptr);
}

// ============================================================================
// Linux/macOS Implementation
// ============================================================================

#else // Unix-like systems (Linux, macOS, etc.)

#include <sys/mman.h>
#include <unistd.h>

// Linux: Try to use transparent huge pages via madvise
// macOS: madvise exists but doesn't support MADV_HUGEPAGE
static void* platform_malloc(size_t size, MemoryStats* stats) {
    void* ptr = NULL;

    // Initialize stats
    memset(stats, 0, sizeof(MemoryStats));
    stats->allocated_size = size;

#ifdef USE_LARGE_PAGES

    #ifdef __linux__
    // Linux: Use mmap for large allocations to enable transparent huge pages
    if (size >= 2 * 1024 * 1024) {  // Only for allocations >= 2MB
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr != MAP_FAILED) {
            // Advise kernel to use huge pages if available
            // MADV_HUGEPAGE is available since Linux 2.6.38
            #ifdef MADV_HUGEPAGE
            if (madvise(ptr, size, MADV_HUGEPAGE) == 0) {
                stats->used_madvise_hugepage = 1;
            }
            #endif

            stats->ptr = ptr;
            return ptr;
        }
    }
    #endif // __linux__

#endif // USE_LARGE_PAGES

    // Fallback: standard malloc with alignment attempt
    // Use posix_memalign for 64-byte cache line alignment
    if (posix_memalign(&ptr, 64, size) == 0) {
        stats->ptr = ptr;
        return ptr;
    }

    // Final fallback: regular malloc
    ptr = malloc(size);
    stats->ptr = ptr;
    return ptr;
}

static void platform_free(void* ptr, MemoryStats* stats) {
    if (ptr == NULL) {
        return;
    }

#ifdef USE_LARGE_PAGES
    #ifdef __linux__
    // Free mmap-allocated memory
    if (stats->used_madvise_hugepage) {
        munmap(ptr, stats->allocated_size);
        return;
    }
    #endif
#endif // USE_LARGE_PAGES

    // Free standard malloc/posix_memalign memory
    free(ptr);
}

#endif // _WIN32

// ============================================================================
// Debug Information
// ============================================================================

#ifdef DEBUG_MEMORY_ALLOC
#include <stdio.h>

static void print_memory_stats(const char* array_name, MemoryStats* stats) {
    fprintf(stderr, "\n=== Memory Allocation Stats for %s ===\n", array_name);
    fprintf(stderr, "Size: %zu bytes (%.2f MB)\n",
            stats->allocated_size,
            stats->allocated_size / (1024.0 * 1024.0));
    fprintf(stderr, "Pointer: %p\n", stats->ptr);

#ifdef _WIN32
    if (stats->used_large_pages) {
        fprintf(stderr, "Method: Windows Large Pages (2MB pages)\n");
    } else if (stats->used_virtual_lock) {
        fprintf(stderr, "Method: VirtualAlloc + VirtualLock\n");
    } else {
        fprintf(stderr, "Method: Aligned malloc (fallback)\n");
    }
#else
    if (stats->used_madvise_hugepage) {
        fprintf(stderr, "Method: mmap + madvise(MADV_HUGEPAGE)\n");
    } else {
        fprintf(stderr, "Method: malloc/posix_memalign (fallback)\n");
    }
#endif

    fprintf(stderr, "======================================\n\n");
}
#endif // DEBUG_MEMORY_ALLOC

#endif // MEMORY_ALLOC_H
