# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Connected components of a directed graph.

Both functions take the graph in CSR format and return a component label per
vertex, with the components numbered from 0 in the order of their smallest
vertex index: vertex 0 is in component 0, the smallest vertex that is not in
component 0 is in component 1, and so on.

cpdef functions:

- weak_components_csr
    Weakly connected components (the edge directions are ignored), with a
    union-find structure.
- strong_components_csr
    Strongly connected components, with Pearce's algorithm, iterative.
"""

cimport numpy as cnp

from edsger.commons cimport check_graph_arguments
import numpy as np

cdef cnp.uint32_t UNSET = 0xFFFFFFFF


cdef inline size_t _find(cnp.uint32_t[::1] parent, size_t vert_idx) noexcept nogil:
    """Root of the tree of a vertex in the union-find forest, with path halving."""
    while <size_t>parent[vert_idx] != vert_idx:
        parent[vert_idx] = parent[parent[vert_idx]]
        vert_idx = <size_t>parent[vert_idx]
    return vert_idx


cpdef tuple weak_components_csr(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        int vertex_count):
    """
    Compute the weakly connected components of a directed graph.

    Two vertices are in the same weakly connected component if they are
    connected by a path when the edge directions are ignored. The edges are
    merged in a union-find forest, where the root of each tree is its smallest
    vertex.

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        Pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        Indices in the CSR format
    vertex_count : int
        Total number of vertices

    Returns
    -------
    labels : cnp.ndarray
        Component label of each vertex (uint32), numbered from 0 in the order
        of the smallest vertex of each component.
    n_components : int
        Number of components.
    """

    cdef:
        size_t vert_idx, edge_idx, root, other_root
        size_t n_vertices = <size_t>vertex_count
        size_t n_components = 0
        cnp.uint32_t[::1] parent
        cnp.uint32_t[::1] labels

    # bounds checking is disabled in this module
    check_graph_arguments(csr_indptr, csr_indices, -1, 0, vertex_count)

    parent = np.arange(vertex_count, dtype=np.uint32)
    labels = np.empty(vertex_count, dtype=np.uint32)

    with nogil:
        for vert_idx in range(n_vertices):
            root = _find(parent, vert_idx)
            for edge_idx in range(<size_t>csr_indptr[vert_idx],
                                  <size_t>csr_indptr[vert_idx + 1]):
                other_root = _find(parent, <size_t>csr_indices[edge_idx])
                if other_root == root:
                    continue
                # the root of the merged tree is the smaller one
                if other_root < root:
                    parent[root] = <cnp.uint32_t>other_root
                    root = other_root
                else:
                    parent[other_root] = <cnp.uint32_t>root

        # each root is the smallest vertex of its component, and comes before
        # the other vertices of its component
        for vert_idx in range(n_vertices):
            root = _find(parent, vert_idx)
            if root == vert_idx:
                labels[vert_idx] = <cnp.uint32_t>n_components
                n_components += 1
            else:
                labels[vert_idx] = labels[root]

    return np.asarray(labels), <int>n_components


cpdef tuple strong_components_csr(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        int vertex_count):
    """
    Compute the strongly connected components of a directed graph.

    Two vertices are in the same strongly connected component if there is a
    path from each one to the other. This is Pearce's algorithm (D. J. Pearce,
    "A space-efficient algorithm for finding strongly connected components",
    Information Processing Letters, 2016), a variant of Tarjan's algorithm,
    with an explicit stack instead of recursion.

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        Pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        Indices in the CSR format
    vertex_count : int
        Total number of vertices

    Returns
    -------
    labels : cnp.ndarray
        Component label of each vertex (uint32), numbered from 0 in the order
        of the smallest vertex of each component.
    n_components : int
        Number of components.
    """

    cdef:
        size_t start_vert_idx, vert_idx, next_vert_idx, parent_vert_idx
        size_t edge_idx, edge_end, label_idx
        size_t n_vertices = <size_t>vertex_count
        # visit index of the next discovered vertex, starting from 1: 0 means
        # undiscovered
        size_t index = 1
        # label of the next completed component, decreasing from n_vertices:
        # always larger than the visit index of the vertices in progress
        size_t component = n_vertices
        size_t n_components, new_label = 0
        # depth-first search stack in stack[:dfs_top], stack of the visited
        # vertices waiting for their component in stack[scc_top:]: a vertex is
        # in one of them at most
        size_t dfs_top, scc_top = n_vertices
        bint descended
        cnp.uint32_t[::1] rindex
        cnp.uint32_t[::1] stack
        cnp.uint32_t[::1] stack_edge
        cnp.int8_t[::1] is_root

    # bounds checking is disabled in this module
    check_graph_arguments(csr_indptr, csr_indices, -1, 0, vertex_count)

    rindex = np.zeros(vertex_count, dtype=np.uint32)
    stack = np.empty(vertex_count, dtype=np.uint32)
    stack_edge = np.empty(vertex_count, dtype=np.uint32)
    is_root = np.empty(vertex_count, dtype=np.int8)

    with nogil:
        for start_vert_idx in range(n_vertices):
            if rindex[start_vert_idx] != 0:
                continue

            stack[0] = <cnp.uint32_t>start_vert_idx
            stack_edge[0] = csr_indptr[start_vert_idx]
            dfs_top = 1
            rindex[start_vert_idx] = <cnp.uint32_t>index
            index += 1
            is_root[start_vert_idx] = 1

            while dfs_top > 0:
                vert_idx = <size_t>stack[dfs_top - 1]
                edge_idx = <size_t>stack_edge[dfs_top - 1]
                edge_end = <size_t>csr_indptr[vert_idx + 1]

                descended = False
                while edge_idx < edge_end:
                    next_vert_idx = <size_t>csr_indices[edge_idx]
                    edge_idx += 1
                    if rindex[next_vert_idx] == 0:
                        # visit the neighbour, then resume from the next edge
                        stack_edge[dfs_top - 1] = <cnp.uint32_t>edge_idx
                        stack[dfs_top] = <cnp.uint32_t>next_vert_idx
                        stack_edge[dfs_top] = csr_indptr[next_vert_idx]
                        dfs_top += 1
                        rindex[next_vert_idx] = <cnp.uint32_t>index
                        index += 1
                        is_root[next_vert_idx] = 1
                        descended = True
                        break
                    if rindex[next_vert_idx] < rindex[vert_idx]:
                        rindex[vert_idx] = rindex[next_vert_idx]
                        is_root[vert_idx] = 0
                if descended:
                    continue

                # all the edges of the vertex are scanned
                dfs_top -= 1
                if is_root[vert_idx]:
                    # the vertex and the vertices visited after it that are
                    # still waiting form a component
                    index -= 1
                    while (scc_top < n_vertices
                           and rindex[vert_idx] <= rindex[stack[scc_top]]):
                        rindex[stack[scc_top]] = <cnp.uint32_t>component
                        scc_top += 1
                        index -= 1
                    rindex[vert_idx] = <cnp.uint32_t>component
                    component -= 1
                else:
                    scc_top -= 1
                    stack[scc_top] = <cnp.uint32_t>vert_idx

                # back to the parent vertex
                if dfs_top > 0:
                    parent_vert_idx = <size_t>stack[dfs_top - 1]
                    if rindex[vert_idx] < rindex[parent_vert_idx]:
                        rindex[parent_vert_idx] = rindex[vert_idx]
                        is_root[parent_vert_idx] = 0

        # the component labels are in [component + 1, n_vertices]: renumber
        # them in the order of the smallest vertex of each component, in place
        n_components = n_vertices - component
        for label_idx in range(n_components):
            stack_edge[label_idx] = UNSET
        for vert_idx in range(n_vertices):
            label_idx = <size_t>rindex[vert_idx] - component - 1
            if stack_edge[label_idx] == UNSET:
                stack_edge[label_idx] = <cnp.uint32_t>new_label
                new_label += 1
            rindex[vert_idx] = stack_edge[label_idx]

    return np.asarray(rindex), <int>n_components


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #


cpdef test_components_01():
    """Test both kinds of components on a small graph.

    0 -> 1 -> 2 -> 0, 2 -> 3, 4 -> 3, 5 isolated
    """
    csr_indptr = np.array([0, 1, 2, 4, 4, 5, 5], dtype=np.uint32)
    csr_indices = np.array([1, 2, 0, 3, 3], dtype=np.uint32)

    labels, n_components = weak_components_csr(csr_indptr, csr_indices, 6)
    assert n_components == 2
    assert list(labels) == [0, 0, 0, 0, 0, 1]

    labels, n_components = strong_components_csr(csr_indptr, csr_indices, 6)
    assert n_components == 4
    assert list(labels) == [0, 0, 0, 1, 2, 3]


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
