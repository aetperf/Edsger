#ifndef PREFETCH_COMPAT_H
#define PREFETCH_COMPAT_H

#ifdef __GNUC__
    #define prefetch_hint(addr, locality) __builtin_prefetch((addr), 0, (locality))
    #define PREFETCH_T0 3
    #define PREFETCH_T1 2
    #define PREFETCH_T2 1
    #define PREFETCH_NTA 0
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define prefetch_hint(addr, locality) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define PREFETCH_T0 _MM_HINT_T0
    #define PREFETCH_T1 _MM_HINT_T1
    #define PREFETCH_T2 _MM_HINT_T2
    #define PREFETCH_NTA _MM_HINT_NTA
#else
    #define prefetch_hint(addr, locality) ((void)0)
    #define PREFETCH_T0 0
    #define PREFETCH_T1 0
    #define PREFETCH_T2 0
    #define PREFETCH_NTA 0
#endif

#endif