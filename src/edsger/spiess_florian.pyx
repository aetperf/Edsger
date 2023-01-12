""" An implementation of Spiess and Florian's hyperpath generating algorithm.

Spiess, H. and Florian, M. (1989). Optimal strategies: A new assignment model 
for transit networks. Transportation Research Part B 23(2), 83-102.
    
author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import nuympy as np
cimport numpy as cnp

from edsger.commons cimport (
    DTYPE_INF, UNLABELED, SCANNED, DTYPE, DTYPE_t, ElementState)
cimport edsger.pq_bin_dec_0b as pq  # priority queue


cdef void compute_SF(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_indices,
    DTYPE_t[::1] csc_trav_time,
    DTYPE_t[::1] csc_freq,
    cnp.uint32_t[::1] tail_indices,
    int target_vert_idx,
    int vertex_count,
    int edge_count
):

    cdef:
        # vertex properties
        np.ndarray u_i = np.zeros(vertex_count, dtype=DTYPE)  # vertex least travel time
        np.ndarray f_i = np.zeros(vertex_count, dtype=DTYPE)  # vertex frequency (inverse of the maximum delay)
        # np.ndarray v_i = np.zeros(vertex_count, dtype=DTYPE)  # vertex volume
        # edge properties
        np.ndarray c_a = np.zeros(edge_count, dtype=DTYPE)    # uncongested edge travel time
        np.ndarray f_a = np.zeros(edge_count, dtype=DTYPE)    # edge frequency (inverse of the maximum delay)
        np.ndarray h_a = np.zeros(edge_count, dtype=bint)     # edge belonging to hyperpath

        size_t target = <size_t>target_vert_idx
        pq.PriorityQueue pqueue
        DTYPE_t ujpca, ujpca_new  # u_j + c_a
        DTYPE_t ui, fi, fa, beta
        size_t min_edge_idx
        size_t tail_vert_index
        size_t edge_idx
        ElementState edge_state
        DTYPE_t edge_val

    # setup #
    # ===== #

    # initialize c_a and f_a
    for i in range(<size_t>edge_count):

        c_a[i] = csc_trav_time[i]

        # replace infinite values with very large finite values
        if (freq[i] > INFFREQ):
            f_a[i] = INFFREQ
        # replace zeros with very small values
        elif (freq[i] < MINFREQ):
            f_a[i] = MINFREQ
        else:
            f_a[i] = freq[i]

    # first pass #
    #------------#

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and UNLABELED state
        pq.init_pqueue(&pqueue, <size_t>edge_count, <size_t>edge_count)

        # only the incoming edges of the target vertex are inserted into the 
        # priority queue
        for edge_idx in range(<size_t>csc_indptr[<size_t>target_vert_idx], 
            <size_t>csc_indptr[<size_t>(target_vert_idx + 1)]):
            pq.insert(&pqueue, edge_idx, c_a[edge_idx])

        # first pass
        while pqueue.size > 0:

            min_edge_idx = pq.extract_min(&pqueue)
            ujpca = pqueue.Elements[min_edge_idx].key
            tail_vert_index = <size_t>tail_indices[min_edge_idx]
            ui = u_i[tail_vert_index]

            if (ui >= ujpca):
            
                fi = f_i[tail_vert_index]
                fa = f_a[min_edge_idx]

                # compute the beta coefficient
                if (ui < DTYPE_INF) | (fi > 0.0):

                    beta = fi * ui

                else:

                    beta = 1.0

                # update u_i
                ui_new = (beta + fa * ujpca) / (fi + fa)
                u_i[tail_vert_index] = ui_new

                # loop on incoming edges
                for edge_idx in range(<size_t>csc_indptr[tail_vert_index], 
                    <size_t>csc_indptr[tail_vert_index + 1]):

                    edge_state = pqueue.Elements[edge_idx].state

                    if (edge_state != SCANNED):

                        ujpca_new = ui_new + c_a[edge_idx]
                        if (edge_state == UNLABELED):
                            pq.insert(&pqueue, edge_idx, ujpca_new)
                        elif (pqueue.Elements[edge_idx].key > ujpca_new):
                            decrease_val(heap, edge_idx, ujpca_new)

                # update f_i
                f_i[tail_vert_index] = fi + fa

                # add the edge to hyperpath
                h_a[min_edge_idx] = 1





