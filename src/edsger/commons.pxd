cimport numpy as cnp

cdef DTYPE
ctypedef cnp.float64_t DTYPE_t
cdef DTYPE_t DTYPE_INF

cdef enum ElementState:
   SCANNED
   NOT_IN_HEAP
   IN_HEAP