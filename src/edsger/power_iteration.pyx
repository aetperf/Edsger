# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Power iterations for the centrality measures: PageRank, HITS and Katz.

The graph is given in CSC format (pull-based iterations: each vertex sums the
values of its predecessors) and, for HITS, also in CSR format.

The iteration count is returned with the result: it is -1 if the iterations
did not converge within `max_iter` iterations.

cpdef functions:

- pagerank_csc
    PageRank by power iteration, with the graph in CSC format.
- hits_csr_csc
    HITS hubs and authorities by power iteration, with the graph in CSR and
    CSC formats.
- katz_csc
    Katz centrality by power iteration, with the graph in CSC format.
"""

cimport numpy as cnp

import numpy as np

# above this L1 change between two iterations, the iterations are diverging
cdef double DIVERGENCE = 1.0e300


cdef int _check_compressed(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        cnp.float64_t[::1] data,
        bint weighted,
        Py_ssize_t vertex_count) except -1:
    """
    Check a graph in CSR or CSC format, in linear time: bounds checking is
    disabled in this module, and the indices are used as array indices. This
    is cheaper than one iteration.
    """
    cdef:
        Py_ssize_t i, edge_count = indices.shape[0]
        bint unordered = False, out_of_bounds = False

    if vertex_count < 1 or indptr.shape[0] != vertex_count + 1:
        raise ValueError(
            f"the pointer array should have vertex_count + 1 = {vertex_count + 1} "
            f"elements, got {indptr.shape[0]}"
        )
    if indptr[0] != 0 or indptr[vertex_count] != edge_count:
        raise ValueError(
            f"the pointers should start at 0 and end at the edge count {edge_count}"
        )
    if weighted and data.shape[0] != edge_count:
        raise ValueError(
            f"the edge data array has {data.shape[0]} elements, the index array has "
            f"{edge_count} elements"
        )
    with nogil:
        for i in range(vertex_count):
            unordered |= indptr[i] > indptr[i + 1]
        for i in range(edge_count):
            out_of_bounds |= indices[i] >= vertex_count
    if unordered:
        raise ValueError("the pointers should be non-decreasing")
    if out_of_bounds:
        raise ValueError(
            f"the indices should be smaller than the vertex count {vertex_count}"
        )
    return 0


cdef int _check_vector(
        cnp.float64_t[::1] vector, str name, Py_ssize_t vertex_count) except -1:
    if vector.shape[0] != vertex_count:
        raise ValueError(
            f"{name} should have one element per vertex: {vertex_count}, got "
            f"{vector.shape[0]}"
        )
    return 0


cdef int _check_iteration_arguments(double tol, int max_iter) except -1:
    if not tol > 0.0:
        raise ValueError(f"tol should be positive, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter should be at least 1, got {max_iter}")
    return 0


cpdef tuple pagerank_csc(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        cnp.float64_t[::1] csc_data,
        bint weighted,
        cnp.float64_t[::1] inv_out_weight,
        cnp.float64_t[::1] teleport,
        cnp.float64_t[::1] dangling_weights,
        cnp.float64_t[::1] x0,
        double damping,
        double tol,
        int max_iter):
    """
    PageRank by power iteration, pull-based on the CSC format:

        x[v] <- damping * (sum_{u -> v} x[u] w(u, v) / W(u)
                           + dangling_sum * dangling_weights[v])
                + (1 - damping) * teleport[v]

    where W(u) is the out-weight of u and dangling_sum the sum of x over the
    dangling vertices: the vertices with inv_out_weight = 0.

    Parameters
    ----------
    csc_indptr, csc_indices : cnp.uint32_t[::1]
        Graph in CSC format: the predecessors of each vertex.
    csc_data : cnp.float64_t[::1]
        Edge weights in CSC format, ignored (and possibly empty) if not
        weighted.
    weighted : bint
        Whether to use the edge weights, or 1 for all the edges.
    inv_out_weight : cnp.float64_t[::1]
        Inverse of the out-weight (out-degree if not weighted) of each vertex,
        0 for the dangling vertices.
    teleport : cnp.float64_t[::1]
        Teleportation (personalization) distribution.
    dangling_weights : cnp.float64_t[::1]
        Distribution over which the value of the dangling vertices is spread.
    x0 : cnp.float64_t[::1]
        Starting vector.
    damping : double
        Damping factor, in [0, 1].
    tol : double
        The iterations stop when the L1 norm of the change of x is below tol.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple[cnp.ndarray, int]
        The PageRank vector and the iteration count, -1 if not converged.
    """
    cdef:
        Py_ssize_t vertex_count = csc_indptr.shape[0] - 1
        Py_ssize_t u, v, k
        int it, n_iter = -1
        double s, dangling_sum, err, d
        double one_minus_damping = 1.0 - damping
        cnp.float64_t[::1] x
        cnp.float64_t[::1] x_new
        cnp.float64_t[::1] y
        cnp.float64_t[::1] tmp

    # bounds checking is disabled in this module
    _check_compressed(csc_indptr, csc_indices, csc_data, weighted, vertex_count)
    _check_vector(inv_out_weight, "inv_out_weight", vertex_count)
    _check_vector(teleport, "teleport", vertex_count)
    _check_vector(dangling_weights, "dangling_weights", vertex_count)
    _check_vector(x0, "x0", vertex_count)
    _check_iteration_arguments(tol, max_iter)
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"damping should be in [0, 1], got {damping}")

    x = np.array(x0, dtype=np.float64, copy=True)
    x_new = np.empty(vertex_count, dtype=np.float64)
    y = np.empty(vertex_count, dtype=np.float64)

    with nogil:
        for it in range(max_iter):
            # contribution of each vertex to its successors
            dangling_sum = 0.0
            for u in range(vertex_count):
                y[u] = x[u] * inv_out_weight[u]
                if inv_out_weight[u] == 0.0:
                    dangling_sum = dangling_sum + x[u]
            # pull from the predecessors
            err = 0.0
            for v in range(vertex_count):
                s = 0.0
                if weighted:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + y[csc_indices[k]] * csc_data[k]
                else:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + y[csc_indices[k]]
                s = (damping * (s + dangling_sum * dangling_weights[v])
                     + one_minus_damping * teleport[v])
                d = s - x[v]
                err = err + (d if d >= 0.0 else -d)
                x_new[v] = s
            tmp = x
            x = x_new
            x_new = tmp
            if err < tol:
                n_iter = it + 1
                break
            if err > DIVERGENCE:
                break

    return np.asarray(x), n_iter


cdef inline void _scale_by_max(cnp.float64_t[::1] x) noexcept nogil:
    """Divide x by its largest element, if it is positive."""
    cdef:
        Py_ssize_t i, n = x.shape[0]
        double m = 0.0
    for i in range(n):
        if x[i] > m:
            m = x[i]
    if m > 0.0:
        m = 1.0 / m
        for i in range(n):
            x[i] = x[i] * m


cpdef tuple hits_csr_csc(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        cnp.float64_t[::1] csr_data,
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        cnp.float64_t[::1] csc_data,
        bint weighted,
        cnp.float64_t[::1] h0,
        double tol,
        int max_iter):
    """
    HITS hubs and authorities by power iteration:

        a[v] <- sum_{u -> v} h[u] w(u, v)    (pull on the CSC format)
        h[u] <- sum_{u -> v} a[v] w(u, v)    (pull on the CSR format)

    then h and a are divided by their largest element. The iterations stop
    when the L1 norm of the change of h is below tol.

    Parameters
    ----------
    csr_indptr, csr_indices : cnp.uint32_t[::1]
        Graph in CSR format: the successors of each vertex.
    csr_data : cnp.float64_t[::1]
        Edge weights in CSR format, ignored (and possibly empty) if not
        weighted.
    csc_indptr, csc_indices : cnp.uint32_t[::1]
        The same graph in CSC format: the predecessors of each vertex.
    csc_data : cnp.float64_t[::1]
        Edge weights in CSC format, ignored (and possibly empty) if not
        weighted.
    weighted : bint
        Whether to use the edge weights, or 1 for all the edges.
    h0 : cnp.float64_t[::1]
        Starting hub vector.
    tol : double
        Tolerance on the L1 norm of the change of h.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple[cnp.ndarray, cnp.ndarray, int]
        The hub and authority vectors, each scaled so that its largest element
        is 1, and the iteration count, -1 if not converged.
    """
    cdef:
        Py_ssize_t vertex_count = csr_indptr.shape[0] - 1
        Py_ssize_t u, v, k
        int it, n_iter = -1
        double s, err, d
        cnp.float64_t[::1] h
        cnp.float64_t[::1] h_last
        cnp.float64_t[::1] a
        cnp.float64_t[::1] tmp

    # bounds checking is disabled in this module
    _check_compressed(csr_indptr, csr_indices, csr_data, weighted, vertex_count)
    _check_compressed(csc_indptr, csc_indices, csc_data, weighted, vertex_count)
    if csr_indices.shape[0] != csc_indices.shape[0]:
        raise ValueError("the CSR and CSC formats should have the same edge count")
    _check_vector(h0, "h0", vertex_count)
    _check_iteration_arguments(tol, max_iter)

    h = np.array(h0, dtype=np.float64, copy=True)
    h_last = np.empty(vertex_count, dtype=np.float64)
    a = np.empty(vertex_count, dtype=np.float64)

    with nogil:
        for it in range(max_iter):
            tmp = h_last
            h_last = h
            h = tmp
            # authorities: pull the hub values of the predecessors
            for v in range(vertex_count):
                s = 0.0
                if weighted:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + h_last[csc_indices[k]] * csc_data[k]
                else:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + h_last[csc_indices[k]]
                a[v] = s
            # hubs: pull the authority values of the successors
            for u in range(vertex_count):
                s = 0.0
                if weighted:
                    for k in range(<Py_ssize_t>csr_indptr[u],
                                   <Py_ssize_t>csr_indptr[u + 1]):
                        s = s + a[csr_indices[k]] * csr_data[k]
                else:
                    for k in range(<Py_ssize_t>csr_indptr[u],
                                   <Py_ssize_t>csr_indptr[u + 1]):
                        s = s + a[csr_indices[k]]
                h[u] = s
            _scale_by_max(h)
            _scale_by_max(a)
            err = 0.0
            for u in range(vertex_count):
                d = h[u] - h_last[u]
                err = err + (d if d >= 0.0 else -d)
            if err < tol:
                n_iter = it + 1
                break

    return np.asarray(h), np.asarray(a), n_iter


cpdef tuple katz_csc(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        cnp.float64_t[::1] csc_data,
        bint weighted,
        cnp.float64_t[::1] beta,
        double alpha,
        double tol,
        int max_iter):
    """
    Katz centrality by power iteration, pull-based on the CSC format, starting
    from 0:

        x[v] <- alpha * sum_{u -> v} x[u] w(u, v) + beta[v]

    The iterations stop when the L1 norm of the change of x is below tol. They
    converge if alpha is smaller than the inverse of the spectral radius of
    the adjacency matrix, and diverge otherwise.

    Parameters
    ----------
    csc_indptr, csc_indices : cnp.uint32_t[::1]
        Graph in CSC format: the predecessors of each vertex.
    csc_data : cnp.float64_t[::1]
        Edge weights in CSC format, ignored (and possibly empty) if not
        weighted.
    weighted : bint
        Whether to use the edge weights, or 1 for all the edges.
    beta : cnp.float64_t[::1]
        Constant term of each vertex.
    alpha : double
        Attenuation factor.
    tol : double
        Tolerance on the L1 norm of the change of x.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    tuple[cnp.ndarray, int]
        The Katz centrality, not normalized, and the iteration count, -1 if not
        converged (or diverging).
    """
    cdef:
        Py_ssize_t vertex_count = csc_indptr.shape[0] - 1
        Py_ssize_t v, k
        int it, n_iter = -1
        double s, err, d
        cnp.float64_t[::1] x
        cnp.float64_t[::1] x_new
        cnp.float64_t[::1] tmp

    # bounds checking is disabled in this module
    _check_compressed(csc_indptr, csc_indices, csc_data, weighted, vertex_count)
    _check_vector(beta, "beta", vertex_count)
    _check_iteration_arguments(tol, max_iter)

    x = np.zeros(vertex_count, dtype=np.float64)
    x_new = np.empty(vertex_count, dtype=np.float64)

    with nogil:
        for it in range(max_iter):
            err = 0.0
            for v in range(vertex_count):
                s = 0.0
                if weighted:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + x[csc_indices[k]] * csc_data[k]
                else:
                    for k in range(<Py_ssize_t>csc_indptr[v],
                                   <Py_ssize_t>csc_indptr[v + 1]):
                        s = s + x[csc_indices[k]]
                s = alpha * s + beta[v]
                d = s - x[v]
                err = err + (d if d >= 0.0 else -d)
                x_new[v] = s
            tmp = x
            x = x_new
            x_new = tmp
            if err < tol:
                n_iter = it + 1
                break
            if err > DIVERGENCE:
                break

    return np.asarray(x), n_iter


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
