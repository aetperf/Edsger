#ifndef PREFETCH_COMPAT_H
#define PREFETCH_COMPAT_H

// Cross-platform memory prefetching compatibility header

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    // ARM64 platforms - use ARM-specific prefetch
    #define prefetch_hint(addr, hint) __builtin_prefetch((const void*)(addr), 0, 3)
    #define PREFETCH_T0 0
    #define PREFETCH_T1 0
    // ARM doesn't differentiate cache levels in builtin_prefetch, so use same hint
    #define prefetch_graph_data(addr) __builtin_prefetch((const void*)(addr), 0, 3)
    #define prefetch_pqueue_data(addr) __builtin_prefetch((const void*)(addr), 0, 3)
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 platforms - use SSE intrinsics
    #include <xmmintrin.h>
    #define prefetch_hint(addr, hint) _mm_prefetch((const char*)(addr), hint)
    #define PREFETCH_T0 _MM_HINT_T0
    #define PREFETCH_T1 _MM_HINT_T1
    
    // Specialized prefetch macros for different data access patterns
    // Graph data: sequential access, use T1 (L2+L3 cache) to avoid L1 pollution
    #define prefetch_graph_data(addr) prefetch_hint((const char*)(addr), PREFETCH_T1)
    // Priority queue: random access, use T0 (L1+L2+L3 cache) for fastest access
    #define prefetch_pqueue_data(addr) prefetch_hint((const char*)(addr), PREFETCH_T0)
#else
    // Other platforms - no-op (compile time optimization will remove calls)
    #define prefetch_hint(addr, hint) ((void)0)
    #define PREFETCH_T0 0
    #define PREFETCH_T1 0
    #define prefetch_graph_data(addr) ((void)0)
    #define prefetch_pqueue_data(addr) ((void)0)
#endif

#endif // PREFETCH_COMPAT_H