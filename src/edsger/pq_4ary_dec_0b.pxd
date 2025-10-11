""" Priority queue based on a minimum binary heap. Header file.
"""

cimport numpy as cnp

from edsger.commons cimport DTYPE_t, ElementState


cdef struct Element:
    DTYPE_t key
    ElementState state
    size_t node_idx

# Forward declare MemoryStats from memory_alloc.h
cdef extern from "memory_alloc.h":
    ctypedef struct MemoryStats:
        int used_large_pages
        int used_virtual_lock
        int used_madvise_hugepage
        size_t allocated_size
        void* ptr

cdef struct PriorityQueue:
    size_t length  # number of elements in the array
    size_t size  # number of elements in the heap
    size_t* A  # array storing the binary tree
    Element* Elements  # array storing the elements
    MemoryStats stats_A  # memory stats for A array
    MemoryStats stats_Elements  # memory stats for Elements array

cdef void init_pqueue(PriorityQueue*, size_t, size_t) noexcept nogil
cdef void free_pqueue(PriorityQueue*) noexcept nogil
cdef void insert(PriorityQueue*, size_t, DTYPE_t) noexcept nogil
cdef DTYPE_t peek(PriorityQueue*) noexcept nogil
cdef size_t extract_min(PriorityQueue*) noexcept nogil
cdef bint is_empty(PriorityQueue*) noexcept nogil
cdef void decrease_key(PriorityQueue*, size_t, DTYPE_t) noexcept nogil
cdef cnp.ndarray copy_keys_to_numpy(PriorityQueue*, size_t)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
