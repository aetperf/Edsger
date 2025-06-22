"""
Forward and reverse star representations of networks.


cpdef functions:

- convert_graph_to_csr_uint32
    Convert an edge dataframe in COO format into CSR format, with uint32
    data.
- convert_graph_to_csc_uint32
    Convert an edge dataframe in COO format into CSC format, with uint32
    data.
- convert_graph_to_csr_float64
    Convert an edge dataframe in COO format into CSR format, with float64
    data.
- convert_graph_to_csc_float64
    Convert an edge dataframe in COO format into CSC format, with float64
    data.


cdef functions:

- _coo_to_csr_uint32
- _coo_to_csc_uint32
- _coo_to_csr_float64
- _coo_to_csc_float64
"""

import numpy as np
cimport numpy as cnp


cpdef convert_graph_to_csr_uint32(edges, tail, head, data, vertex_count):
    """
    Convert an edge dataframe in COO format into CSR format, with uint32
    data.

    Parameters
    ----------
    edges : pandas.core.frame.DataFrame
        The edges dataframe.
    tail : str
        The column name in the edges dataframe for the tail vertex index.
    head : str
        The column name in the edges dataframe for the head vertex index.
    data : str
        The column name in the edges dataframe for the int edge attribute.
    vertex_count : int
        The vertex count in the given network edges.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    fs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    edge_count = len(edges)
    fs_indices = np.empty(edge_count, dtype=np.uint32)
    fs_data = np.empty(edge_count, dtype=np.uint32)

    _coo_to_csr_uint32(
        edges[tail].values.astype(np.uint32),
        edges[head].values.astype(np.uint32),
        edges[data].values.astype(np.uint32),
        fs_indptr,
        fs_indices,
        fs_data,
    )

    return fs_indptr, fs_indices, fs_data


cpdef convert_graph_to_csc_uint32(edges, tail, head, data, vertex_count):
    """
    Convert an edge dataframe in COO format into CSC format, with uint32
    data.

    Parameters
    ----------
    edges : pandas.core.frame.DataFrame
        The edges dataframe.
    tail : str
        The column name in the edges dataframe for the tail vertex index.
    head : str
        The column name in the edges dataframe for the head vertex index.
    data : str
        The column name in the edges dataframe for the int edge attribute.
    vertex_count : int
        The vertex count in the given network edges.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    rs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    edge_count = len(edges)
    rs_indices = np.empty(edge_count, dtype=np.uint32)
    rs_data = np.empty(edge_count, dtype=np.uint32)

    _coo_to_csc_uint32(
        edges[tail].values.astype(np.uint32),
        edges[head].values.astype(np.uint32),
        edges[data].values.astype(np.uint32),
        rs_indptr,
        rs_indices,
        rs_data,
    )

    return rs_indptr, rs_indices, rs_data


cpdef convert_graph_to_csr_float64(edges, tail, head, data, vertex_count):
    """
    Convert an edge dataframe in COO format into CSR format, with float64
    data.

    Parameters
    ----------
    edges : pandas.core.frame.DataFrame
        The edges dataframe.
    tail : str
        The column name in the edges dataframe for the tail vertex index.
    head : str
        The column name in the edges dataframe for the head vertex index.
    data : str
        The column name in the edges dataframe for the real edge attribute.
    vertex_count : int
        The vertex count in the given network edges.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    fs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    edge_count = len(edges)
    fs_indices = np.empty(edge_count, dtype=np.uint32)
    fs_data = np.empty(edge_count, dtype=np.float64)

    _coo_to_csr_float64(
        edges[tail].values.astype(np.uint32),
        edges[head].values.astype(np.uint32),
        edges[data].values.astype(np.float64),
        fs_indptr,
        fs_indices,
        fs_data,
    )

    return fs_indptr, fs_indices, fs_data


cpdef convert_graph_to_csc_float64(edges, tail, head, data, vertex_count):
    """
    Convert an edge dataframe in COO format into CSC format, with float64
    data.

    Parameters
    ----------
    edges : pandas.core.frame.DataFrame
        The edges dataframe.
    tail : str
        The column name in the edges dataframe for the tail vertex index.
    head : str
        The column name in the edges dataframe for the head vertex index.
    data : str
        The column name in the edges dataframe for the real edge attribute.
    vertex_count : int
        The vertex count in the given network edges.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    rs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    edge_count = len(edges)
    rs_indices = np.empty(edge_count, dtype=np.uint32)
    rs_data = np.empty(edge_count, dtype=np.float64)

    _coo_to_csc_float64(
        edges[tail].values.astype(np.uint32),
        edges[head].values.astype(np.uint32),
        edges[data].values.astype(np.float64),
        rs_indptr,
        rs_indices,
        rs_data,
    )

    return rs_indptr, rs_indices, rs_data


cdef void _coo_to_csr_uint32(
        cnp.uint32_t [::1] Ai,
        cnp.uint32_t [::1] Aj,
        cnp.uint32_t [::1] Ax,
        cnp.uint32_t [::1] Bp,
        cnp.uint32_t [::1] Bj,
        cnp.uint32_t [::1] Bx) noexcept nogil:

    cdef:
        size_t i, row, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bj.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Ai[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_vert] = <cnp.uint32_t>n_edge

    for i in range(n_edge):
        row = <size_t>Ai[i]
        dest = <size_t>Bp[row]
        Bj[dest] = Aj[i]
        Bx[dest] = Ax[i]
        Bp[row] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cdef void _coo_to_csc_uint32(
        cnp.uint32_t [::1] Ai,
        cnp.uint32_t [::1] Aj,
        cnp.uint32_t [::1] Ax,
        cnp.uint32_t [::1] Bp,
        cnp.uint32_t [::1] Bi,
        cnp.uint32_t [::1] Bx) nogil:

    cdef:
        size_t i, col, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bi.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Aj[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[<size_t>n_vert] = <cnp.uint32_t>n_edge

    for i in range(n_edge):
        col = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef void _coo_to_csr_float64(
        cnp.uint32_t  [::1] Ai,
        cnp.uint32_t  [::1] Aj,
        cnp.float64_t [::1] Ax,
        cnp.uint32_t  [::1] Bp,
        cnp.uint32_t  [::1] Bj,
        cnp.float64_t [::1] Bx) noexcept nogil:

    cdef:
        size_t i, row, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bj.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Ai[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_vert] = <cnp.uint32_t>n_edge

    for i in range(n_edge):
        row = <size_t>Ai[i]
        dest = <size_t>Bp[row]
        Bj[dest] = Aj[i]
        Bx[dest] = Ax[i]
        Bp[row] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef void _coo_to_csc_float64(
        cnp.uint32_t  [::1] Ai,
        cnp.uint32_t  [::1] Aj,
        cnp.float64_t [::1] Ax,
        cnp.uint32_t  [::1] Bp,
        cnp.uint32_t  [::1] Bi,
        cnp.float64_t [::1] Bx) noexcept nogil:

    cdef:
        size_t i, col, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bi.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Aj[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[<size_t>n_vert] = <cnp.uint32_t>n_edge

    for i in range(n_edge):
        col = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
