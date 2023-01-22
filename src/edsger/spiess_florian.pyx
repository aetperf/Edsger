""" An implementation of Spiess and Florian's hyperpath generating algorithm.

Spiess, H. and Florian, M. (1989). Optimal strategies: A new assignment model 
for transit networks. Transportation Research Part B 23(2), 83-102.
    
author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np
cimport numpy as cnp

from edsger.commons import DTYPE_PY, DTYPE_INF_PY
from edsger.commons cimport (
    DTYPE_INF, UNLABELED, SCANNED, DTYPE_t, ElementState)
cimport edsger.pq_bin_dec_0b as pq  # priority queue


cpdef void compute_SF_in(
    cnp.uint32_t[::1] csc_indptr,  
    cnp.uint32_t[::1] csc_indices, 
    cnp.uint32_t[::1] csc_edge_idx,
    DTYPE_t[::1] c_a,
    DTYPE_t[::1] f_a_vec,
    cnp.uint32_t[::1] tail_indices,
    int vertex_count,
    list orig_vert_indices,
    int dest_vert_index,
    list volumes
):

    cdef:
        int edge_count = tail_indices.shape[0]
        pq.PriorityQueue pqueue
        ElementState edge_state
        size_t edge_idx, min_edge_idx, tail_vert_idx
        DTYPE_t u_j_c_a, u_i, f_i, beta, u_i_new

    # vertex properties
    u_i_vec = DTYPE_INF_PY * np.ones(vertex_count, dtype=DTYPE_PY)  # vertex least travel time
    f_i_vec = np.zeros(vertex_count, dtype=DTYPE_PY)  # vertex frequency (inverse of the maximum delay)
    v_i_vec = np.zeros(vertex_count, dtype=DTYPE_PY)  # vertex volume
    
    # edge properties
    c_a_vec = np.zeros(edge_count, dtype=DTYPE_PY)    # uncongested edge travel time
    h_a_vec = np.zeros(edge_count, dtype=bool)    # edge belonging to hyperpath

    u_i_vec[<size_t>dest_vert_index] = 0.0
    for i, vert_idx in enumerate(orig_vert_indices):
        v_i_vec[vert_idx] = volumes[i]

    #     DTYPE_t ujpca, ujpca_new  # u_j + c_a
    #     DTYPE_t ui, fi, fa, beta
    #     size_t min_edge_idx
    #     size_t tail_vert_index
    #     size_t edge_idx
    #     ElementState edge_state
    #     DTYPE_t edge_val

    # first pass #
    #------------#

    # initialization of the heap elements 
    # all nodes have INFINITY key and UNLABELED state
    pq.init_pqueue(&pqueue, <size_t>edge_count, <size_t>edge_count)

    # only the incoming edges of the target vertex are inserted into the 
    # priority queue
    for edge_idx in range(<size_t>csc_indptr[<size_t>dest_vert_index], 
        <size_t>csc_indptr[<size_t>(dest_vert_index + 1)]):
        pq.insert(&pqueue, edge_idx, c_a_vec[edge_idx])

    # first pass
    while pqueue.size > 0:

        min_edge_idx = pq.extract_min(&pqueue)
        u_j_c_a = pqueue.Elements[min_edge_idx].key
        tail_vert_idx = <size_t>tail_indices[min_edge_idx]
        u_i = u_i_vec[tail_vert_idx]
        u_i_new = u_i

        if u_i >= u_j_c_a:

            f_i = f_i_vec[tail_vert_idx]

            # compute the beta coefficient
            if (u_i < DTYPE_INF) | (f_i > 0.0):

                beta = f_i * u_i

            else:

                beta = 1.0

            # update u_i
            f_a = f_a_vec[min_edge_idx]
            u_i_new = (beta + f_a * u_j_c_a) / (f_i + f_a)
            u_i_vec[tail_vert_idx] = u_i_new

            # update f_i
            f_i_vec[tail_vert_idx] = f_i + f_a

            # add the edge to hyperpath
            h_a_vec[min_edge_idx] = 1

        # loop on incoming edges
        for edge_idx in range(<size_t>csc_indptr[tail_vert_idx], 
            <size_t>csc_indptr[tail_vert_idx + 1]):

            edge_state = pqueue.Elements[edge_idx].state

            if edge_state != SCANNED:

                u_j_c_a = u_i_new + c_a[edge_idx]
                if edge_state == UNLABELED:
                    pq.insert(&pqueue, edge_idx, u_j_c_a)
                elif (pqueue.Elements[edge_idx].key > u_j_c_a):
                    pq.decrease_key(&pqueue, edge_idx, u_j_c_a)

    pq.free_pqueue(&pqueue)

    # second pass #
