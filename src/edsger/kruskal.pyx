# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Minimum spanning forest: Kruskal's algorithm.

cpdef functions:

- kruskal
    Select the edges of a minimum spanning forest, given the edges sorted by
    weight. Returns the positions of the selected edges.
"""

cimport numpy as cnp

import numpy as np


cdef inline cnp.uint32_t _find(cnp.uint32_t[::1] parent, cnp.uint32_t x) noexcept nogil:
    """Root of the set of x, with path halving."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


cpdef cnp.ndarray kruskal(
        cnp.uint32_t[::1] tails,
        cnp.uint32_t[::1] heads,
        cnp.int64_t[::1] order,
        int vertex_count):
    """
    Kruskal's algorithm, with a union-find structure (path halving and union
    by size).

    The edges are seen as undirected: the edge (u, v) joins u and v. Self-loops
    are never selected. Among the edges of equal weight, the first in `order`
    is selected first.

    Parameters
    ----------
    tails : cnp.uint32_t[::1]
        Tail vertex of each edge.
    heads : cnp.uint32_t[::1]
        Head vertex of each edge.
    order : cnp.int64_t[::1]
        Positions of the edges by increasing weight, e.g. from a stable argsort
        of the weights. It must be a permutation of 0..edge_count - 1, or a
        subset of it.
    vertex_count : int
        Number of vertices: the vertex indices are smaller than this.

    Returns
    -------
    selected : cnp.ndarray
        Positions (int64) of the edges of the minimum spanning forest, in the
        order in which they were selected, that is by increasing weight.
    """
    cdef:
        Py_ssize_t edge_count = tails.shape[0]
        Py_ssize_t order_count = order.shape[0]
        Py_ssize_t i, selected_count = 0, max_selected
        cnp.int64_t e
        cnp.uint32_t root_t, root_h, vc
        bint bad_vertex = False, bad_order = False
        cnp.uint32_t[::1] parent
        cnp.uint32_t[::1] size
        cnp.int64_t[::1] selected

    # bounds checking is disabled in this module
    if vertex_count < 0:
        raise ValueError(f"vertex_count should be non-negative, got {vertex_count}")
    if heads.shape[0] != edge_count:
        raise ValueError(
            f"tails and heads should have the same length, got {edge_count} "
            f"and {heads.shape[0]}"
        )
    if order_count > edge_count:
        raise ValueError(
            f"order has {order_count} elements, more than the {edge_count} edges"
        )
    vc = <cnp.uint32_t>vertex_count
    with nogil:
        for i in range(edge_count):
            bad_vertex |= (tails[i] >= vc) | (heads[i] >= vc)
        for i in range(order_count):
            bad_order |= (order[i] < 0) | (order[i] >= edge_count)
    if bad_vertex:
        raise ValueError(
            f"the vertex indices should be smaller than vertex_count {vertex_count}"
        )
    if bad_order:
        raise ValueError(
            f"the edge positions in order should be in [0, {edge_count})"
        )

    parent = np.arange(vertex_count, dtype=np.uint32)
    size = np.ones(vertex_count, dtype=np.uint32)
    max_selected = vertex_count - 1 if vertex_count > 0 else 0
    selected = np.empty(max_selected, dtype=np.int64)

    with nogil:
        for i in range(order_count):
            if selected_count == max_selected:
                break
            e = order[i]
            root_t = _find(parent, tails[e])
            root_h = _find(parent, heads[e])
            if root_t == root_h:
                continue
            # union by size: the smaller tree goes under the larger one
            if size[root_t] < size[root_h]:
                root_t, root_h = root_h, root_t
            parent[root_h] = root_t
            size[root_t] += size[root_h]
            selected[selected_count] = e
            selected_count += 1

    return np.asarray(selected)[:selected_count].copy()


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
