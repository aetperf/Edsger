

cimport numpy as cnp

from edsger.pq_bin_dec_0b as pq_bd0


cpdef cnp.ndarray compute_sssp_pq_bd0(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using the pq_bin_dec_0b priority queue.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        pq_bd0.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        pq_bd0.init_pqueue(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        pq_bd0.insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = pq_bd0.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        pq_bd0.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        pq_bd0.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = pq_bd0.copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    pq_bd0.free_pqueue(&pqueue)  

    return path_lengths