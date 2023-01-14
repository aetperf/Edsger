
cimport numpy as cnp

from edsger.commons import DTYPE_t

cdef struct Edges:
    cnp.uint32_t tail_vertex_idx
    cnp.uint32_t head_vertex_idx
    DTYPE_t c_a
    DTYPE_t f_a
    # DTYPE_t h_a
    # DTYPE_t p_a



