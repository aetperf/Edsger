# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Depth-First Search (DFS) implementation.

The search is iterative, with an explicit stack: its depth is not limited by
the call stack, so that it can run on graphs with any number of vertices.

The neighbours of a vertex are visited in the order in which they are stored in
the compressed format. A vertex is discovered when the search first reaches it,
and the search then goes on from its first undiscovered neighbour: this is the
order of the recursive depth-first search, also the one of
`scipy.sparse.csgraph.depth_first_order`.

cpdef functions:

- dfs_csr
    Compute DFS tree using CSR format (forward traversal). Returns the
    predecessors and the discovery order.
- dfs_csc
    Compute DFS tree using CSC format (backward traversal). Returns the
    successors and the discovery order.
"""

cimport numpy as cnp

from edsger.commons cimport check_graph_arguments
import numpy as np


cdef size_t _dfs(
        const cnp.uint32_t[::1] indptr,
        const cnp.uint32_t[::1] indices,
        size_t start,
        cnp.uint32_t[::1] stack_vert,
        cnp.uint32_t[::1] stack_edge,
        cnp.int8_t[::1] visited,
        cnp.int32_t[::1] links,
        cnp.uint32_t[::1] order) noexcept nogil:
    """
    Iterative depth-first search from the start vertex, on a graph in CSR or
    CSC format.

    For each vertex on the stack, `stack_edge` holds the position of the next
    edge to scan in `indices`, so that each edge is scanned only once.

    Returns the number of discovered vertices, including the start vertex,
    which are stored in `order` in discovery order.
    """
    cdef:
        size_t vert_idx, next_vert_idx, edge_idx, edge_end
        size_t top = 1
        size_t order_count = 1

    visited[start] = 1
    order[0] = <cnp.uint32_t>start
    stack_vert[0] = <cnp.uint32_t>start
    stack_edge[0] = indptr[start]

    while top > 0:
        vert_idx = <size_t>stack_vert[top - 1]
        edge_idx = <size_t>stack_edge[top - 1]
        edge_end = <size_t>indptr[vert_idx + 1]

        # first undiscovered neighbour of the vertex on top of the stack
        while edge_idx < edge_end:
            next_vert_idx = <size_t>indices[edge_idx]
            edge_idx += 1
            if visited[next_vert_idx] == 0:
                break
        else:
            # all the neighbours are discovered: backtrack
            top -= 1
            continue

        # resume from the next edge when the search comes back to this vertex
        stack_edge[top - 1] = <cnp.uint32_t>edge_idx

        visited[next_vert_idx] = 1
        links[next_vert_idx] = <cnp.int32_t>vert_idx
        order[order_count] = <cnp.uint32_t>next_vert_idx
        order_count += 1
        stack_vert[top] = <cnp.uint32_t>next_vert_idx
        stack_edge[top] = indptr[next_vert_idx]
        top += 1

    return order_count


cdef tuple _dfs_tree(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        int start_vert_idx,
        int vertex_count,
        int sentinel):
    """Check the arguments, allocate the arrays and run the search."""
    cdef:
        size_t order_count
        cnp.uint32_t[::1] stack_vert
        cnp.uint32_t[::1] stack_edge
        cnp.int8_t[::1] visited
        cnp.int32_t[::1] links
        cnp.uint32_t[::1] order

    # bounds checking is disabled in this module
    check_graph_arguments(indptr, indices, -1, start_vert_idx, vertex_count)

    stack_vert = np.empty(vertex_count, dtype=np.uint32)
    stack_edge = np.empty(vertex_count, dtype=np.uint32)
    visited = np.zeros(vertex_count, dtype=np.int8)
    links = np.full(vertex_count, sentinel, dtype=np.int32)
    order = np.empty(vertex_count, dtype=np.uint32)

    with nogil:
        order_count = _dfs(
            indptr, indices, <size_t>start_vert_idx,
            stack_vert, stack_edge, visited, links, order)

    return np.asarray(links), np.asarray(order)[:order_count].copy()


cpdef tuple dfs_csr(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        int start_vert_idx,
        int vertex_count,
        int sentinel=-9999):
    """
    Compute DFS tree using CSR format (forward traversal from start vertex).

    The outgoing edges of a vertex are scanned in the order of `csr_indices`.

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
        Predecessor array (int32) where predecessors[i] contains the
        predecessor of vertex i in the DFS tree. Unreachable vertices and the
        start vertex have the sentinel value.
    order : cnp.ndarray
        The reachable vertices (uint32), in discovery order (preorder),
        starting with the start vertex.
    """
    return _dfs_tree(
        csr_indptr, csr_indices, start_vert_idx, vertex_count, sentinel)


cpdef tuple dfs_csc(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        int start_vert_idx,
        int vertex_count,
        int sentinel=-9999):
    """
    Compute DFS tree using CSC format (backward traversal from start vertex).

    The incoming edges of a vertex are scanned in the order of `csc_indices`.

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
    successors : cnp.ndarray
        Array (int32) where successors[i] contains the successor of vertex i
        in the DFS tree (since we're traversing backward). Vertices that
        cannot reach the start vertex and the start vertex have the sentinel
        value.
    order : cnp.ndarray
        The vertices that can reach the start vertex (uint32), in discovery
        order (preorder), starting with the start vertex.
    """
    return _dfs_tree(
        csc_indptr, csc_indices, start_vert_idx, vertex_count, sentinel)


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #


cpdef test_dfs_csr_01():
    """Test DFS CSR on a small graph: 0 -> 1 -> 3, 0 -> 2 -> 3."""
    cdef int UNREACHABLE = -9999
    csr_indptr = np.array([0, 2, 3, 4, 4], dtype=np.uint32)
    csr_indices = np.array([1, 2, 3, 3], dtype=np.uint32)

    predecessors, order = dfs_csr(csr_indptr, csr_indices, 0, 4)

    # 3 is discovered from 1, before 2
    assert list(predecessors) == [UNREACHABLE, 0, 0, 1]
    assert list(order) == [0, 1, 3, 2]


cpdef test_dfs_csc_01():
    """Test DFS CSC on the same graph, to vertex 3."""
    cdef int UNREACHABLE = -9999
    csc_indptr = np.array([0, 0, 1, 2, 4], dtype=np.uint32)
    csc_indices = np.array([0, 0, 1, 2], dtype=np.uint32)

    successors, order = dfs_csc(csc_indptr, csc_indices, 3, 4)

    assert list(successors) == [1, 3, 3, UNREACHABLE]
    assert list(order) == [3, 1, 0, 2]


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
