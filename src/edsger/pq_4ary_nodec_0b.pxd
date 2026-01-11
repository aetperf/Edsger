""" Priority queue based on a minimum 4-ary heap. Header file.

No decrease-key operation - simplified for lazy Dijkstra.
- nodec: no decrease-key support
- 0b: indices are zero-based
"""

cimport numpy as cnp

from edsger.commons cimport DTYPE_t, ElementState


cdef struct Element:
    DTYPE_t key
    ElementState state
    # NO node_idx field - not needed without decrease-key


cdef struct PriorityQueue:
    size_t length  # number of elements in the array (heap capacity)
    size_t size  # number of elements in the heap
    size_t element_count  # total number of elements (for deallocation)
    size_t* A  # array storing the binary tree
    Element* Elements  # array storing the elements


# Simplified API - no decrease_key
cdef void init_pqueue(PriorityQueue*, size_t, size_t) noexcept nogil
cdef void free_pqueue(PriorityQueue*) noexcept nogil
cdef void insert(PriorityQueue*, size_t, DTYPE_t) noexcept nogil
cdef DTYPE_t peek(PriorityQueue*) noexcept nogil
cdef size_t extract_min(PriorityQueue*) noexcept nogil
cdef bint is_empty(PriorityQueue*) noexcept nogil
cdef cnp.ndarray copy_keys_to_numpy(PriorityQueue*, size_t)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
