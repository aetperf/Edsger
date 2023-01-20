""" Priority queue based on a minimum binary heap. Header file.

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

cimport numpy as cnp

from edsger.commons cimport DTYPE_t, ElementState


cdef struct Element:
    DTYPE_t key
    ElementState state
    size_t node_idx

cdef struct PriorityQueue:
    size_t length  # number of elements in the array
    size_t size  # number of elements in the heap
    size_t* A  # array storing the binary tree
    Element* Elements  # array storing the elements

cdef void init_pqueue(PriorityQueue*, size_t, size_t) nogil
cdef void free_pqueue(PriorityQueue*) nogil
cdef void insert(PriorityQueue*, size_t, DTYPE_t) nogil
cdef DTYPE_t peek(PriorityQueue*) nogil
cdef size_t extract_min(PriorityQueue*) nogil
cdef bint is_empty(PriorityQueue*) nogil
cdef void decrease_key(PriorityQueue*, size_t, DTYPE_t) nogil
cdef cnp.ndarray copy_keys_to_numpy(PriorityQueue*, size_t)
