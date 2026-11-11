# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Lazy Dijkstra's algorithm without decrease-key operation.

This implementation uses a priority queue without decrease-key operation
(pq_4ary_inline). Instead, when a shorter path is found, a new entry is
inserted into the heap with the better value. Stale entries (whose key is
larger than the path length of their vertex) are skipped during extraction.

The code is in dijkstra_core.pxi, shared with dijkstra.pyx, and is compiled
here with LAZY = 1.

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

# priority queue without decrease-key: stale entries are skipped
cdef enum:
    LAZY = 1

include "dijkstra_core.pxi"
