

cimport numpy as cnp

from edsger.commons cimport (
    DTYPE_INF, NOT_IN_HEAP, SCANNED, DTYPE_t, ElementState)
cimport edsger.pq_bin_dec_0b as pq_bd0  # priority queue based on a binary heap

cpdef cnp.ndarray compute_sssp_pq_bd0(
    cnp.uint32_t[::1] csr_indptr,
    cnp.uint32_t[::1] csr_indices,
    DTYPE_t[::1] csr_data,
    int source_vert_idx,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using the pq_bin_dec_0b priority queue.

        Does not return predecessors.

    input
    =====
    * cnp.uint32_t[::1] csr_indices : indices in the CSR format
    * cnp.uint32_t[::1] csr_indptr : pointers in the CSR format
    * DTYPE_t[::1] csr_data :  data (edge weights) in the CSR format
    * int source_vert_idx : source vertex index
    * int vertex_count : vertex count

    output
    ======
    * cnp.ndarray : shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, head_vert_val
        pq_bd0.PriorityQueue pqueue
        ElementState vert_state
        size_t source = <size_t>source_vert_idx

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        pq_bd0.init_pqueue(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        pq_bd0.insert(&pqueue, source, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = pq_bd0.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], 
                <size_t>csr_indptr[tail_vert_idx + 1]):

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


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #

from edsger.commons cimport DTYPE
import numpy as np

cdef generate_braess_network_csr():
    """ Generate a Braess-like network in CSR format.
    """

    csr_indptr = np.array([0, 2, 4, 5, 5], dtype=np.uint32)
    csr_indices = np.array([1, 2, 2, 3, 3], dtype=np.uint32)
    csr_data = np.array([1., 2., 0., 2., 1.], dtype=DTYPE)

    return csr_indptr, csr_indices, csr_data


cpdef compute_sssp_pq_bd0_01():
    """ A single edge from vertex 0 to vertex 1, with weight 1.0.
    """
    csr_indptr = np.array([0, 1, 1], dtype=np.uint32)
    csr_indices = np.array([1], dtype=np.uint32)
    csr_data = np.array([1.], dtype=DTYPE)

    # from vertex 0
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 0, 2)
    path_lengths_ref = np.array([0., 1.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 1, 2)
    path_lengths_ref = np.array([DTYPE_INF, 0.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)


cpdef compute_sssp_pq_bd0_02():
    """ Small network with 5 edges and 4 vertices.
    """

    csr_indptr, csr_indices, csr_data = generate_braess_network_csr()

    # from vertex 0
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 0, 4)
    path_lengths_ref = np.array([0., 1., 1., 2.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 1, 4)
    path_lengths_ref = np.array([DTYPE_INF, 0., 0., 1.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 2
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 2, 4)
    path_lengths_ref = np.array([DTYPE_INF, DTYPE_INF, 0., 1.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 3
    path_lengths = compute_sssp_pq_bd0(csr_indptr, csr_indices, csr_data, 3, 4)
    path_lengths_ref = np.array([DTYPE_INF, DTYPE_INF, DTYPE_INF, 0.], dtype=DTYPE)
    assert np.allclose(path_lengths_ref, path_lengths)