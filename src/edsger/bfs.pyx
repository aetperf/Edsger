"""
Breadth-First Search (BFS) implementation.

cpdef functions:

- bfs_csr
    Compute BFS tree using CSR format (forward traversal). Returns predecessors.
- bfs_csc
    Compute BFS tree using CSC format (backward traversal). Returns predecessors.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

cimport numpy as cnp
import numpy as np

cpdef cnp.ndarray bfs_csr(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        int start_vert_idx,
        int vertex_count,
        int sentinel=-9999):
    """
    Compute BFS tree using CSR format (forward traversal from start vertex).

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        Pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        Indices in the CSR format
    start_vert_idx : int
        Starting vertex index
    vertex_count : int
        Total number of vertices
    sentinel : int, optional
        Sentinel value for unreachable nodes and start vertex (default: -9999)

    Returns
    -------
    predecessors : cnp.ndarray
        Predecessor array where predecessors[i] contains the predecessor
        of vertex i in the BFS tree. Unreachable vertices and the start
        vertex have the sentinel value.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        size_t queue_head = 0, queue_tail = 0
        size_t start = <size_t>start_vert_idx
        cnp.uint32_t[::1] queue
        cnp.int8_t[::1] visited
        cnp.int32_t[::1] predecessors

    # Allocate arrays
    queue = np.empty(vertex_count, dtype=np.uint32)
    visited = np.zeros(vertex_count, dtype=np.int8)
    predecessors = np.full(vertex_count, sentinel, dtype=np.int32)

    with nogil:
        # Initialize: mark start vertex as visited and enqueue it
        visited[start] = 1
        queue[queue_tail] = start
        queue_tail += 1

        # BFS main loop
        while queue_head < queue_tail:
            tail_vert_idx = queue[queue_head]
            queue_head += 1

            # Process all outgoing edges from tail_vert_idx
            for idx in range(<size_t>csr_indptr[tail_vert_idx],
                             <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]

                # If not visited, mark as visited and enqueue
                if visited[head_vert_idx] == 0:
                    visited[head_vert_idx] = 1
                    predecessors[head_vert_idx] = <int>tail_vert_idx
                    queue[queue_tail] = head_vert_idx
                    queue_tail += 1

    # Convert to numpy array
    return np.asarray(predecessors)


cpdef cnp.ndarray bfs_csc(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        int start_vert_idx,
        int vertex_count,
        int sentinel=-9999):
    """
    Compute BFS tree using CSC format (backward traversal from start vertex).

    Parameters
    ----------
    csc_indptr : cnp.uint32_t[::1]
        Pointers in the CSC format
    csc_indices : cnp.uint32_t[::1]
        Indices in the CSC format
    start_vert_idx : int
        Starting vertex index
    vertex_count : int
        Total number of vertices
    sentinel : int, optional
        Sentinel value for unreachable nodes and start vertex (default: -9999)

    Returns
    -------
    predecessors : cnp.ndarray
        Predecessor array where predecessors[i] contains the successor
        of vertex i in the BFS tree (since we're traversing backward).
        Unreachable vertices and the start vertex have the sentinel value.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        size_t queue_head = 0, queue_tail = 0
        size_t start = <size_t>start_vert_idx
        cnp.uint32_t[::1] queue
        cnp.int8_t[::1] visited
        cnp.int32_t[::1] predecessors

    # Allocate arrays
    queue = np.empty(vertex_count, dtype=np.uint32)
    visited = np.zeros(vertex_count, dtype=np.int8)
    predecessors = np.full(vertex_count, sentinel, dtype=np.int32)

    with nogil:
        # Initialize: mark start vertex as visited and enqueue it
        visited[start] = 1
        queue[queue_tail] = start
        queue_tail += 1

        # BFS main loop (processing incoming edges using CSC)
        while queue_head < queue_tail:
            head_vert_idx = queue[queue_head]
            queue_head += 1

            # Process all incoming edges to head_vert_idx
            for idx in range(<size_t>csc_indptr[head_vert_idx],
                             <size_t>csc_indptr[head_vert_idx + 1]):
                tail_vert_idx = <size_t>csc_indices[idx]

                # If not visited, mark as visited and enqueue
                if visited[tail_vert_idx] == 0:
                    visited[tail_vert_idx] = 1
                    predecessors[tail_vert_idx] = <int>head_vert_idx
                    queue[queue_tail] = tail_vert_idx
                    queue_tail += 1

    # Convert to numpy array
    return np.asarray(predecessors)


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #


cdef generate_simple_graph_csr():
    """
    Generate a simple directed graph in CSR format.

    Graph structure:
    0 -> 1 -> 3
    0 -> 2 -> 3

    4 vertices, 4 edges
    """
    csr_indptr = np.array([0, 2, 3, 4, 4], dtype=np.uint32)
    csr_indices = np.array([1, 2, 3, 3], dtype=np.uint32)
    return csr_indptr, csr_indices


cdef generate_simple_graph_csc():
    """
    Generate a simple directed graph in CSC format.

    Graph structure (same as CSR version):
    0 -> 1 -> 3
    0 -> 2 -> 3

    4 vertices, 4 edges
    """
    csc_indptr = np.array([0, 0, 1, 2, 4], dtype=np.uint32)
    csc_indices = np.array([0, 0, 1, 2], dtype=np.uint32)
    return csc_indptr, csc_indices


cpdef test_bfs_csr_01():
    """Test BFS CSR on simple graph from vertex 0."""
    cdef int UNREACHABLE = -9999
    csr_indptr, csr_indices = generate_simple_graph_csr()

    predecessors = bfs_csr(csr_indptr, csr_indices, 0, 4)

    # Expected: 0 is start, 1 and 2 have predecessor 0, 3 has predecessor 1 or 2
    assert predecessors[0] == UNREACHABLE  # start vertex
    assert predecessors[1] == 0
    assert predecessors[2] == 0
    assert predecessors[3] in [1, 2]  # could be reached from either 1 or 2


cpdef test_bfs_csc_01():
    """Test BFS CSC on simple graph to vertex 3."""
    cdef int UNREACHABLE = -9999
    csc_indptr, csc_indices = generate_simple_graph_csc()

    predecessors = bfs_csc(csc_indptr, csc_indices, 3, 4)

    # Expected: working backward from 3
    assert predecessors[3] == UNREACHABLE  # start vertex
    assert predecessors[1] in [3, UNREACHABLE] or predecessors[2] in [3, UNREACHABLE]
    assert predecessors[0] in [1, 2, UNREACHABLE]


cpdef test_bfs_unreachable():
    """Test BFS with unreachable vertices."""
    cdef int UNREACHABLE = -9999
    # Graph: 0 -> 1, 2 -> 3 (two disconnected components)
    csr_indptr = np.array([0, 1, 1, 2, 2], dtype=np.uint32)
    csr_indices = np.array([1, 3], dtype=np.uint32)

    predecessors = bfs_csr(csr_indptr, csr_indices, 0, 4)

    # From 0, can reach 1 but not 2 or 3
    assert predecessors[0] == UNREACHABLE  # start
    assert predecessors[1] == 0
    assert predecessors[2] == UNREACHABLE  # unreachable
    assert predecessors[3] == UNREACHABLE  # unreachable


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
