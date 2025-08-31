"""
An implementation of the Bellman-Ford algorithm.

cpdef functions:

- compute_bf_sssp
    Compute single-source shortest path (from one vertex to all vertices). Does
    not return predecessors. Supports negative weights.
- compute_bf_sssp_w_path
    Compute single-source shortest path (from one vertex to all vertices).
    Compute predecessors. Supports negative weights.
- compute_bf_stsp
    Compute single-target shortest path (from all vertices to one vertex). Does
    not return successors. Supports negative weights.
- compute_bf_stsp_w_path
    Compute single-target shortest path (from all vertices to one vertex).
    Compute successors. Supports negative weights.
- detect_negative_cycle
    Detect negative cycles in the graph.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

cimport numpy as cnp
import numpy as np

from edsger.commons cimport DTYPE_INF, DTYPE_t
from edsger.commons import DTYPE_PY


cpdef cnp.ndarray compute_bf_sssp(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        int source_vert_idx,
        int vertex_count):
    """
    Compute single-source shortest path using Bellman-Ford algorithm.

    From one vertex to all vertices. Does not return predecessors.
    Supports negative edge weights.

    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, v
        DTYPE_t tail_vert_val, new_dist
        cnp.ndarray[DTYPE_t, ndim=1] dist = np.full(
            vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        size_t source = <size_t>source_vert_idx
        bint changed

    # Initialize source distance
    dist[source] = 0.0

    with nogil:
        # Relax edges V-1 times
        for v in range(<size_t>(vertex_count - 1)):
            changed = False

            # Iterate through all vertices
            for tail_vert_idx in range(<size_t>vertex_count):
                tail_vert_val = dist[tail_vert_idx]

                # Skip if vertex is unreachable
                if tail_vert_val == DTYPE_INF:
                    continue

                # Relax edges from this vertex
                for idx in range(<size_t>csr_indptr[tail_vert_idx],
                                 <size_t>csr_indptr[tail_vert_idx + 1]):

                    head_vert_idx = <size_t>csr_indices[idx]
                    new_dist = tail_vert_val + csr_data[idx]

                    if dist[head_vert_idx] > new_dist:
                        dist[head_vert_idx] = new_dist
                        changed = True

            # Early termination if no changes
            if not changed:
                break

    return dist


cpdef cnp.ndarray compute_bf_sssp_w_path(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        cnp.uint32_t[::1] predecessor,
        int source_vert_idx,
        int vertex_count):
    """
    Compute single-source shortest path using Bellman-Ford algorithm.

    From one vertex to all vertices. Compute predecessors.
    Supports negative edge weights.

    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format
    predecessor : cnp.uint32_t[::1]
        array of indices, one for each vertex of the graph. Each vertex'
        entry contains the index of its predecessor in a path from the
        source, through the graph.
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, v
        DTYPE_t tail_vert_val, new_dist
        cnp.ndarray[DTYPE_t, ndim=1] dist = np.full(
            vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        size_t source = <size_t>source_vert_idx
        bint changed

    # Initialize source distance
    dist[source] = 0.0

    with nogil:
        # Relax edges V-1 times
        for v in range(<size_t>(vertex_count - 1)):
            changed = False

            # Iterate through all vertices
            for tail_vert_idx in range(<size_t>vertex_count):
                tail_vert_val = dist[tail_vert_idx]

                # Skip if vertex is unreachable
                if tail_vert_val == DTYPE_INF:
                    continue

                # Relax edges from this vertex
                for idx in range(<size_t>csr_indptr[tail_vert_idx],
                                 <size_t>csr_indptr[tail_vert_idx + 1]):

                    head_vert_idx = <size_t>csr_indices[idx]
                    new_dist = tail_vert_val + csr_data[idx]

                    if dist[head_vert_idx] > new_dist:
                        dist[head_vert_idx] = new_dist
                        predecessor[head_vert_idx] = tail_vert_idx
                        changed = True

            # Early termination if no changes
            if not changed:
                break

    return dist


cpdef cnp.ndarray compute_bf_stsp(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        int target_vert_idx,
        int vertex_count):
    """
    Compute single-target shortest path using Bellman-Ford algorithm.

    From all vertices to one vertex. Does not return successors.
    Supports negative edge weights.

    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        indices in the CSC format
    csc_indptr : cnp.uint32_t[::1]
        pointers in the CSC format
    csc_data : DTYPE_t[::1]
        data (edge weights) in the CSC format
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, v
        DTYPE_t head_vert_val, new_dist
        cnp.ndarray[DTYPE_t, ndim=1] dist = np.full(
            vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        size_t target = <size_t>target_vert_idx
        bint changed

    # Initialize target distance
    dist[target] = 0.0

    with nogil:
        # Relax edges V-1 times
        for v in range(<size_t>(vertex_count - 1)):
            changed = False

            # Iterate through all vertices (reversed for incoming edges)
            for head_vert_idx in range(<size_t>vertex_count):
                head_vert_val = dist[head_vert_idx]

                # Skip if vertex is unreachable
                if head_vert_val == DTYPE_INF:
                    continue

                # Relax incoming edges to this vertex
                for idx in range(<size_t>csc_indptr[head_vert_idx],
                                 <size_t>csc_indptr[head_vert_idx + 1]):

                    tail_vert_idx = <size_t>csc_indices[idx]
                    new_dist = head_vert_val + csc_data[idx]

                    if dist[tail_vert_idx] > new_dist:
                        dist[tail_vert_idx] = new_dist
                        changed = True

            # Early termination if no changes
            if not changed:
                break

    return dist


cpdef cnp.ndarray compute_bf_stsp_w_path(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        cnp.uint32_t[::1] successor,
        int target_vert_idx,
        int vertex_count):
    """
    Compute single-target shortest path using Bellman-Ford algorithm.

    From all vertices to one vertex. Compute successors.
    Supports negative edge weights.

    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        indices in the CSC format
    csc_indptr : cnp.uint32_t[::1]
        pointers in the CSC format
    csc_data : DTYPE_t[::1]
        data (edge weights) in the CSC format
    successor : cnp.uint32_t[::1]
        array of indices, one for each vertex of the graph. Each vertex'
        entry contains the index of its successor in a path to the
        target, through the graph.
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, v
        DTYPE_t head_vert_val, new_dist
        cnp.ndarray[DTYPE_t, ndim=1] dist = np.full(
            vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        size_t target = <size_t>target_vert_idx
        bint changed

    # Initialize target distance
    dist[target] = 0.0

    with nogil:
        # Relax edges V-1 times
        for v in range(<size_t>(vertex_count - 1)):
            changed = False

            # Iterate through all vertices (reversed for incoming edges)
            for head_vert_idx in range(<size_t>vertex_count):
                head_vert_val = dist[head_vert_idx]

                # Skip if vertex is unreachable
                if head_vert_val == DTYPE_INF:
                    continue

                # Relax incoming edges to this vertex
                for idx in range(<size_t>csc_indptr[head_vert_idx],
                                 <size_t>csc_indptr[head_vert_idx + 1]):

                    tail_vert_idx = <size_t>csc_indices[idx]
                    new_dist = head_vert_val + csc_data[idx]

                    if dist[tail_vert_idx] > new_dist:
                        dist[tail_vert_idx] = new_dist
                        successor[tail_vert_idx] = head_vert_idx
                        changed = True

            # Early termination if no changes
            if not changed:
                break

    return dist


cpdef bint detect_negative_cycle(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        DTYPE_t[::1] dist_matrix,
        int vertex_count):
    """
    Detect negative cycles using one more iteration of edge relaxation.

    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format
    dist_matrix : DTYPE_t[::1]
        current distance matrix from Bellman-Ford algorithm
    vertex_count : int
        vertex count

    Returns
    -------
    has_negative_cycle : bool
        True if negative cycle detected, False otherwise
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, new_dist
        bint has_negative_cycle = False

    with nogil:
        # One more iteration to detect negative cycles
        for tail_vert_idx in range(<size_t>vertex_count):
            tail_vert_val = dist_matrix[tail_vert_idx]

            # Skip if vertex is unreachable
            if tail_vert_val == DTYPE_INF:
                continue

            # Check edges from this vertex
            for idx in range(<size_t>csr_indptr[tail_vert_idx],
                             <size_t>csr_indptr[tail_vert_idx + 1]):

                head_vert_idx = <size_t>csr_indices[idx]
                new_dist = tail_vert_val + csr_data[idx]

                # If we can still relax, there's a negative cycle
                if dist_matrix[head_vert_idx] > new_dist:
                    has_negative_cycle = True
                    break

            if has_negative_cycle:
                break

    return has_negative_cycle


cpdef bint detect_negative_cycle_csc(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        DTYPE_t[::1] stsp_dist,
        int vertex_count):
    """
    Detect negative cycles using CSC format and STSP distances.

    For STSP (Single-Target Shortest Path):
    - stsp_dist[u] = distance FROM vertex u TO target vertex
    - Edge (u→v) can be relaxed if: stsp_dist[u] > stsp_dist[v] + weight(u→v)

    This function performs one additional iteration to check if any edge
    can still be relaxed, which indicates the presence of a negative cycle.

    Parameters
    ----------
    csc_indptr : cnp.uint32_t[::1]
        Pointers in the CSC format (incoming edges by destination)
    csc_indices : cnp.uint32_t[::1]
        Indices in the CSC format (source vertices)
    csc_data : DTYPE_t[::1]
        Data (edge weights) in the CSC format
    stsp_dist : DTYPE_t[::1]
        Current distance array from STSP algorithm (distances TO target)
    vertex_count : int
        Total number of vertices in the graph

    Returns
    -------
    has_negative_cycle : bool
        True if negative cycle detected, False otherwise
    """

    cdef:
        size_t head_vert_idx, tail_vert_idx, idx
        DTYPE_t new_dist
        bint has_negative_cycle = False

    with nogil:
        # Iterate over destination vertices (CSC organization)
        for head_vert_idx in range(<size_t>vertex_count):
            # Skip unreachable vertices
            if stsp_dist[head_vert_idx] == DTYPE_INF:
                continue

            # Check all incoming edges to this vertex
            for idx in range(<size_t>csc_indptr[head_vert_idx],
                             <size_t>csc_indptr[head_vert_idx + 1]):

                tail_vert_idx = <size_t>csc_indices[idx]  # Source of edge

                # STSP relaxation: can we improve distance FROM tail TO target?
                new_dist = stsp_dist[head_vert_idx] + csc_data[idx]

                # If we can still relax this edge, negative cycle exists
                if stsp_dist[tail_vert_idx] > new_dist:
                    has_negative_cycle = True
                    break

            if has_negative_cycle:
                break

    return has_negative_cycle


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #

cdef generate_negative_edge_network_csr():
    """
    Generate a network with negative edges (no negative cycle) in CSR format.

    Graph structure:
    0 -> 1 (weight: 1)
    0 -> 2 (weight: 4)
    1 -> 2 (weight: -2)
    1 -> 3 (weight: 5)
    2 -> 3 (weight: 1)
    3 -> 4 (weight: 3)

    This network has 6 edges and 5 vertices.
    """

    csr_indptr = np.array([0, 2, 4, 5, 6, 6], dtype=np.uint32)
    csr_indices = np.array([1, 2, 2, 3, 3, 4], dtype=np.uint32)
    csr_data = np.array([1., 4., -2., 5., 1., 3.], dtype=DTYPE_PY)

    return csr_indptr, csr_indices, csr_data


cdef generate_negative_cycle_network_csr():
    """
    Generate a network with a negative cycle in CSR format.

    Graph structure:
    0 -> 1 (weight: 1)
    1 -> 2 (weight: -2)
    2 -> 0 (weight: -1)
    2 -> 3 (weight: 1)

    Cycle 0->1->2->0 has total weight -2 (negative cycle)
    This network has 4 edges and 4 vertices.
    """

    csr_indptr = np.array([0, 1, 2, 4, 4], dtype=np.uint32)
    csr_indices = np.array([1, 2, 0, 3], dtype=np.uint32)
    csr_data = np.array([1., -2., -1., 1.], dtype=DTYPE_PY)

    return csr_indptr, csr_indices, csr_data


cpdef test_bf_negative_edges():
    """
    Test Bellman-Ford with negative edges (no negative cycle).
    """

    csr_indptr, csr_indices, csr_data = generate_negative_edge_network_csr()

    # from vertex 0
    path_lengths = compute_bf_sssp(csr_indptr, csr_indices, csr_data, 0, 5)
    path_lengths_ref = np.array([0., 1., -1., 0., 3.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # Test negative cycle detection (should return False)
    has_cycle = detect_negative_cycle(
        csr_indptr, csr_indices, csr_data, path_lengths, 5)
    assert not has_cycle


cpdef test_bf_negative_cycle():
    """
    Test Bellman-Ford negative cycle detection.
    """

    csr_indptr, csr_indices, csr_data = generate_negative_cycle_network_csr()

    # from vertex 0
    path_lengths = compute_bf_sssp(csr_indptr, csr_indices, csr_data, 0, 4)

    # Test negative cycle detection (should return True)
    has_cycle = detect_negative_cycle(
        csr_indptr, csr_indices, csr_data, path_lengths, 4)
    assert has_cycle


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
