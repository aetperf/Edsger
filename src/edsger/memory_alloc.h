/*
 * Memory allocation with Transparent Huge Pages (THP) support.
 *
 * On Linux, uses mmap() with madvise(MADV_HUGEPAGE) for large allocations
 * to reduce TLB misses. Falls back to standard malloc for small allocations.
 *
 * On Windows, uses VirtualAlloc with MEM_LARGE_PAGES when available.
 *
 * Usage:
 *   void* ptr = thp_alloc(size);
 *   thp_free(ptr, size);
 *
 * Build with USE_LARGE_PAGES defined to enable THP support.
 */

#ifndef MEMORY_ALLOC_H
#define MEMORY_ALLOC_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/* Threshold for using huge pages (2MB) */
#define THP_THRESHOLD (2 * 1024 * 1024)

/* Debug output - uncomment to enable */
/* #define DEBUG_MEMORY_ALLOC */

#ifdef USE_LARGE_PAGES

#ifdef _WIN32
/* Windows implementation */
#include <windows.h>

static int windows_large_pages_enabled = -1;  /* -1 = unknown, 0 = no, 1 = yes */

static int try_enable_large_pages(void) {
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    BOOL result;

    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                          &hToken)) {
        return 0;
    }

    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
        CloseHandle(hToken);
        return 0;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    result = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL);
    CloseHandle(hToken);

    if (!result || GetLastError() == ERROR_NOT_ALL_ASSIGNED) {
        return 0;
    }

    return 1;
}

static inline void* thp_alloc(size_t size) {
    void* ptr = NULL;
    SIZE_T large_page_size;

    if (windows_large_pages_enabled == -1) {
        windows_large_pages_enabled = try_enable_large_pages();
    }

    if (windows_large_pages_enabled && size >= THP_THRESHOLD) {
        large_page_size = GetLargePageMinimum();
        if (large_page_size > 0) {
            /* Round up to large page size */
            size_t aligned_size = (size + large_page_size - 1) & ~(large_page_size - 1);
            ptr = VirtualAlloc(NULL, aligned_size,
                               MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
#ifdef DEBUG_MEMORY_ALLOC
            if (ptr) {
                fprintf(stderr, "[THP] Windows Large Pages: allocated %zu bytes\n", aligned_size);
            }
#endif
        }
    }

    if (!ptr) {
        /* Fallback: try VirtualAlloc without large pages */
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (ptr) {
            VirtualLock(ptr, size);  /* Try to lock in memory */
#ifdef DEBUG_MEMORY_ALLOC
            fprintf(stderr, "[THP] Windows VirtualAlloc+Lock: allocated %zu bytes\n", size);
#endif
        }
    }

    if (!ptr) {
        /* Final fallback: malloc */
        ptr = malloc(size);
#ifdef DEBUG_MEMORY_ALLOC
        fprintf(stderr, "[THP] Windows malloc fallback: allocated %zu bytes\n", size);
#endif
    }

    return ptr;
}

static inline void thp_free(void* ptr, size_t size) {
    if (ptr) {
        /* Try VirtualFree first (works for both large pages and regular VirtualAlloc) */
        if (!VirtualFree(ptr, 0, MEM_RELEASE)) {
            /* If VirtualFree fails, it was allocated with malloc */
            free(ptr);
        }
    }
}

#else
/* Linux/macOS implementation */
#include <sys/mman.h>
#include <string.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

/* MADV_HUGEPAGE may not be defined on all systems */
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

static inline void* thp_alloc(size_t size) {
    void* ptr;

    if (size >= THP_THRESHOLD) {
        /* Use mmap for large allocations to enable THP */
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr != MAP_FAILED) {
            /* Hint to kernel to use huge pages */
            madvise(ptr, size, MADV_HUGEPAGE);
#ifdef DEBUG_MEMORY_ALLOC
            fprintf(stderr, "[THP] mmap+MADV_HUGEPAGE: allocated %zu bytes (%.2f MB)\n",
                    size, (double)size / (1024 * 1024));
#endif
            return ptr;
        }
        /* mmap failed, fall through to malloc */
    }

    /* Small allocation or mmap failed: use malloc */
    ptr = malloc(size);
#ifdef DEBUG_MEMORY_ALLOC
    if (ptr) {
        fprintf(stderr, "[THP] malloc: allocated %zu bytes\n", size);
    }
#endif
    return ptr;
}

static inline void thp_free(void* ptr, size_t size) {
    if (ptr) {
        if (size >= THP_THRESHOLD) {
            munmap(ptr, size);
        } else {
            free(ptr);
        }
    }
}

#endif /* _WIN32 */

#else /* USE_LARGE_PAGES not defined */

/* Standard allocation without THP support */
static inline void* thp_alloc(size_t size) {
    return malloc(size);
}

static inline void thp_free(void* ptr, size_t size) {
    (void)size;  /* unused */
    free(ptr);
}

#endif /* USE_LARGE_PAGES */

#endif /* MEMORY_ALLOC_H */
