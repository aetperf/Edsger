""" Forward and reverse star representations of networks.
"""

import numpy as np
cimport numpy as cnp


cpdef convert_graph_to_csr_uint32(edges_df, head, tail, data, vertex_count, edge_count):

    fs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    fs_indices = np.empty(edge_count, dtype=np.uint32)
    fs_data = np.empty(edge_count, dtype=np.uint32)

    coo_tocsr_uint32(
        edges_df[tail].values.astype(np.uint32),
        edges_df[head].values.astype(np.uint32),
        edges_df[data].values.astype(np.uint32),
        fs_indptr,
        fs_indices,
        fs_data,
    )

    return fs_indptr, fs_indices, fs_data


cdef void coo_tocsr_uint32(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.uint32_t [::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bj,
    cnp.uint32_t [::1] Bx) nogil:    
    """ Convert to Forward star representation : COO to CSR sparse format.

        Data vector is of uint32 type.
    """

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
        row  = <size_t>Ai[i]
        dest = <size_t>Bp[row]
        Bj[dest] = Aj[i]
        Bx[dest] = Ax[i]
        Bp[row] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef convert_graph_to_csc_uint32(edges_df, tail, head, data, vertex_count, edge_count):

    rs_indptr = np.zeros(
        vertex_count + 1, dtype=np.uint32
    )  # make sure it is filled with zeros
    rs_indices = np.empty(edge_count, dtype=np.uint32)
    rs_data = np.empty(edge_count, dtype=np.uint32)

    coo_tocsc_uint32(
        edges_df[tail].values.astype(np.uint32),
        edges_df[head].values.astype(np.uint32),
        edges_df[data].values.astype(np.uint32),
        rs_indptr,
        rs_indices,
        rs_data,
    )

    return rs_indptr, rs_indices, rs_data


cdef void coo_tocsc_uint32(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.uint32_t [::1] Ax,   
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bi,
    cnp.uint32_t [::1] Bx) nogil:
    """ Convert to Reverse star representation : COO to CSC sparse format.

        Data vector is of uint32 type.
    """

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
        col  = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef void coo_tocsr_float64(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.float64_t[::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bj,
    cnp.float64_t[::1] Bx) nogil:
    """ Convert to Forward star representation : COO to CSR sparse format.

        Data vector is of float64 type.
    """

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
        row  = <size_t>Ai[i]
        dest = <size_t>Bp[row]
        Bj[dest] = Aj[i]
        Bx[dest] = Ax[i]
        Bp[row] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef void coo_tocsc_float64(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.float64_t[::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bi,
    cnp.float64_t[::1] Bx) nogil:
    """ Convert to Reverse star representation : COO to CSC sparse format.

        Data vector is of float64 type.
    """

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
        col  = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp