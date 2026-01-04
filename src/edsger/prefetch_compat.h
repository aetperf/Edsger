#ifndef PREFETCH_COMPAT_H
#define PREFETCH_COMPAT_H

// Cross-platform memory prefetching and branch prediction compatibility header

// ============================================================================
// Branch Prediction Hints
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
    // GCC and Clang support __builtin_expect
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    // MSVC doesn't have __builtin_expect
    // Use __assume for specific cases, otherwise no-op
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

// ============================================================================
// Memory Prefetching
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    // ARM64 platforms - use ARM-specific prefetch
    // hint=3 is highest temporal locality (like T0)
    #define prefetch_hint(addr, hint) __builtin_prefetch((const void*)(addr), 0, 3)
    #define PREFETCH_T0 0
    #define PREFETCH_T1 0
    #define PREFETCH_T2 0
    #define PREFETCH_NTA 0
    // Convenience macros for ARM (all map to same instruction with locality hint)
    #define prefetch_t0(addr) __builtin_prefetch((const void*)(addr), 0, 3)
    #define prefetch_t1(addr) __builtin_prefetch((const void*)(addr), 0, 2)
    #define prefetch_t2(addr) __builtin_prefetch((const void*)(addr), 0, 1)
    #define prefetch_nta(addr) __builtin_prefetch((const void*)(addr), 0, 0)
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 platforms - use SSE intrinsics
    #include <xmmintrin.h>
    #define prefetch_hint(addr, hint) _mm_prefetch((const char*)(addr), hint)
    #define PREFETCH_T0 _MM_HINT_T0
    #define PREFETCH_T1 _MM_HINT_T1
    #define PREFETCH_T2 _MM_HINT_T2
    #define PREFETCH_NTA _MM_HINT_NTA
    // Convenience macros for x86/x64
    #define prefetch_t0(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define prefetch_t1(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
    #define prefetch_t2(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T2)
    #define prefetch_nta(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#else
    // Other platforms - no-op (compile time optimization will remove calls)
    #define prefetch_hint(addr, hint) ((void)0)
    #define PREFETCH_T0 0
    #define PREFETCH_T1 0
    #define PREFETCH_T2 0
    #define PREFETCH_NTA 0
    #define prefetch_t0(addr) ((void)0)
    #define prefetch_t1(addr) ((void)0)
    #define prefetch_t2(addr) ((void)0)
    #define prefetch_nta(addr) ((void)0)
#endif

#endif // PREFETCH_COMPAT_H