# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
An implementation of Dijkstra's algorithm, with decrease-key operation.

The code is in dijkstra_core.pxi, shared with dijkstra_nodec.pyx, and is
compiled here with LAZY = 0.

cpdef functions:

- compute_sssp
    Compute single-source shortest path (from one vertex to all vertices). Does
    not return predecessors.
- compute_sssp_w_path
    Compute single-source shortest path (from one vertex to all vertices).
    Compute predecessors.
- compute_sssp_early_termination
    Compute single-source shortest path with early termination when target nodes
    are reached. Does not return predecessors.
- compute_sssp_w_path_early_termination
    Compute single-source shortest path with early termination when target nodes
    are reached. Compute predecessors.
- compute_stsp
    Compute single-target shortest path (from all vertices to one vertex). Does
    not return successors.
- compute_stsp_w_path
    Compute single-target shortest path (from all vertices to one vertex).
    Compute successors.
- compute_stsp_early_termination
    Compute single-target shortest path with early termination when target nodes
    are reached. Does not return successors.
- compute_stsp_w_path_early_termination
    Compute single-target shortest path with early termination when target nodes
    are reached. Compute successors.
"""

# priority queue with decrease-key
cdef enum:
    LAZY = 0

include "dijkstra_core.pxi"
