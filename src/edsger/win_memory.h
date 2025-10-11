#ifndef WIN_MEMORY_H
#define WIN_MEMORY_H

/*
 * Windows Large Pages Memory Allocation
 *
 * Provides Large Pages support for Windows to reduce TLB misses and improve
 * memory access performance for large graph algorithms.
 *
 * Requirements:
 * - Windows 7 or later
 * - SeLockMemoryPrivilege for the user account (on Windows < 10)
 * - Administrator privileges to grant the privilege
 *
 * Setup Instructions:
 * 1. Run: secpol.msc
 * 2. Navigate to: Local Policies -> User Rights Assignment
 * 3. Find: "Lock pages in memory"
 * 4. Add your user account
 * 5. Restart or re-login
 */

#ifdef _WIN32

#include <windows.h>
#include <stdio.h>

// Windows Large Page allocation with privilege checking
static void* win_alloc_large_pages(size_t size, int* used_large_pages) {
    void* ptr = NULL;
    SIZE_T large_page_min;
    SIZE_T alloc_size;
    HANDLE token = NULL;
    TOKEN_PRIVILEGES tp;
    LUID luid;

    *used_large_pages = 0;

    // Step 1: Try to enable SeLockMemoryPrivilege
    if (OpenProcessToken(GetCurrentProcess(),
                         TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                         &token)) {

        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Luid = luid;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

            // Attempt to adjust privileges (may fail without admin rights)
            AdjustTokenPrivileges(token, FALSE, &tp, 0, NULL, NULL);
        }

        CloseHandle(token);
    }

    // Step 2: Get the large page minimum size (typically 2MB on x64)
    large_page_min = GetLargePageMinimum();

    if (large_page_min == 0) {
        // Large pages not supported or privilege not granted
        return NULL;
    }

    // Step 3: Align size to large page boundary
    alloc_size = ((size + large_page_min - 1) / large_page_min) * large_page_min;

    // Step 4: Attempt allocation with MEM_LARGE_PAGES
    ptr = VirtualAlloc(NULL,
                       alloc_size,
                       MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                       PAGE_READWRITE);

    if (ptr != NULL) {
        *used_large_pages = 1;

        // Optional: Display info for debugging
        #ifdef DEBUG_MEMORY_ALLOC
        fprintf(stderr,
                "Large Pages: Allocated %zu bytes (aligned to %zu) at %p\n",
                alloc_size, large_page_min, ptr);
        #endif
    }

    return ptr;
}

// VirtualAlloc fallback (without MEM_LARGE_PAGES, but with optimizations)
static void* win_alloc_virtual(size_t size, int* used_virtual_lock) {
    void* ptr = NULL;
    *used_virtual_lock = 0;

    // Allocate with MEM_COMMIT for immediate physical memory backing
    ptr = VirtualAlloc(NULL,
                       size,
                       MEM_COMMIT | MEM_RESERVE,
                       PAGE_READWRITE);

    if (ptr != NULL) {
        // Try to lock pages in memory to prevent swapping
        // This may fail without privilege, but we continue anyway
        if (VirtualLock(ptr, size)) {
            *used_virtual_lock = 1;

            #ifdef DEBUG_MEMORY_ALLOC
            fprintf(stderr,
                    "VirtualAlloc: Allocated and locked %zu bytes at %p\n",
                    size, ptr);
            #endif
        } else {
            #ifdef DEBUG_MEMORY_ALLOC
            fprintf(stderr,
                    "VirtualAlloc: Allocated %zu bytes at %p (lock failed, continuing)\n",
                    size, ptr);
            #endif
        }
    }

    return ptr;
}

// Free function for VirtualAlloc-based allocations
static void win_free_virtual(void* ptr, size_t size, int was_locked) {
    if (ptr != NULL) {
        if (was_locked) {
            VirtualUnlock(ptr, size);
        }
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}

#endif // _WIN32

#endif // WIN_MEMORY_H
