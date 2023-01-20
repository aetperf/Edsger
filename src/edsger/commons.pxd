""" 
Common definitions.

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t
cdef DTYPE_t DTYPE_INF

cdef enum ElementState:
   SCANNED
   UNLABELED
   LABELED