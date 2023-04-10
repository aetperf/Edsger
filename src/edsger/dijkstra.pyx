""" 
An implementation of Dijkstra's algorithm.
    
author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

cimport numpy as cnp

from edsger.commons cimport (
    DTYPE_INF, UNLABELED, SCANNED, DTYPE_t, ElementState)
cimport edsger.pq_4ary_dec_0b as pq  # priority queue


cpdef cnp.ndarray compute_sssp(
    cnp.uint32_t[::1] csr_indptr,
    cnp.uint32_t[::1] csr_indices,
    DTYPE_t[::1] csr_data,
    int source_vert_idx,
    int vertex_count,
    int heap_length):
    """ 
    Compute single-source shortest path (from one vertex to all vertices)
    using the pq_4ary_dec_0b priority queue.

    Does not return predecessors.

    input
    =====
    * cnp.uint32_t[::1] csr_indices : indices in the CSR format
    * cnp.uint32_t[::1] csr_indptr : pointers in the CSR format
    * DTYPE_t[::1] csr_data :  data (edge weights) in the CSR format
    * int source_vert_idx : source vertex index
    * int vertex_count : vertex count
    * int heap_length : heap length

    output
    ======
    * cnp.ndarray : shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t source = <size_t>source_vert_idx

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and UNLABELED state
        pq.init_pqueue(&pqueue, <size_t>heap_length, <size_t>vertex_count)

        # the key is set to zero for the source vertex,
        # which is inserted into the heap
        pq.insert(&pqueue, source, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = pq.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], 
                <size_t>csr_indptr[tail_vert_idx + 1]):

                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        pq.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = pq.copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    pq.free_pqueue(&pqueue)  

    return path_lengths


cpdef cnp.ndarray compute_sssp_w_path(
    cnp.uint32_t[::1] csr_indptr,
    cnp.uint32_t[::1] csr_indices,
    DTYPE_t[::1] csr_data,
    cnp.uint32_t[::1] predecessor,
    int source_vert_idx,
    int vertex_count,
    int heap_length):
    """ 
    Compute single-source shortest path (from one vertex to all vertices)
    using the pq_4ary_dec_0b priority queue.

    Compute predecessors.

    input
    =====
    * cnp.uint32_t[::1] csr_indices : indices in the CSR format
    * cnp.uint32_t[::1] csr_indptr : pointers in the CSR format
    * DTYPE_t[::1] csr_data :  data (edge weights) in the CSR format
    * cnp.uint32_t[::1] predecessor :  array of indices, one for each vertex of 
        the graph. Each vertex' entry contains the index of its predecessor in 
        a path from the source, through the graph. 
    * int source_vert_idx : source vertex index
    * int vertex_count : vertex count
    * int heap_length : heap length

    output
    ======
    * cnp.ndarray : shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t source = <size_t>source_vert_idx

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and UNLABELED state
        pq.init_pqueue(&pqueue, <size_t>heap_length, <size_t>vertex_count)

        # the key is set to zero for the source vertex,
        # which is inserted into the heap
        pq.insert(&pqueue, source, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = pq.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], 
                <size_t>csr_indptr[tail_vert_idx + 1]):

                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, head_vert_idx, head_vert_val)
                        predecessor[head_vert_idx] = tail_vert_idx
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        pq.decrease_key(&pqueue, head_vert_idx, head_vert_val)
                        predecessor[head_vert_idx] = tail_vert_idx

    # copy the results into a numpy array
    path_lengths = pq.copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    pq.free_pqueue(&pqueue)  

    return path_lengths


cpdef cnp.ndarray compute_stsp(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_indices,
    DTYPE_t[::1] csc_data,
    int target_vert_idx,
    int vertex_count,
    int heap_length):
    """ 
    Compute single-target shortest path (from all vertices to one vertex)
    using the pq_4ary_dec_0b priority queue.

    Does not return successors.

    input
    =====
    * cnp.uint32_t[::1] csc_indices : indices in the CSC format
    * cnp.uint32_t[::1] csc_indptr : pointers in the CSC format
    * DTYPE_t[::1] csc_data :  data (edge weights) in the CSC format
    * int target_vert_idx : source vertex index
    * int vertex_count : vertex count
    * int heap_length : heap length

    output
    ======
    * cnp.ndarray : shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t target = <size_t>target_vert_idx

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and UNLABELED state
        pq.init_pqueue(&pqueue, <size_t>heap_length, <size_t>vertex_count)

        # the key is set to zero for the target vertex,
        # which is inserted into the heap
        pq.insert(&pqueue, target, 0.0)

        # main loop
        while pqueue.size > 0:
            head_vert_idx = pq.extract_min(&pqueue)
            head_vert_val = pqueue.Elements[head_vert_idx].key

            # loop on incoming edges
            for idx in range(<size_t>csc_indptr[head_vert_idx], 
                <size_t>csc_indptr[head_vert_idx + 1]):

                tail_vert_idx = <size_t>csc_indices[idx]
                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    tail_vert_val = head_vert_val + csc_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, tail_vert_idx, tail_vert_val)
                    elif pqueue.Elements[tail_vert_idx].key > tail_vert_val:
                        pq.decrease_key(&pqueue, tail_vert_idx, tail_vert_val)

    # copy the results into a numpy array
    path_lengths = pq.copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    pq.free_pqueue(&pqueue)  

    return path_lengths


cpdef cnp.ndarray compute_stsp_w_path(
    cnp.uint32_t[::1] csc_indptr,
    cnp.uint32_t[::1] csc_indices,
    DTYPE_t[::1] csc_data,
    cnp.uint32_t[::1] successor,
    int target_vert_idx,
    int vertex_count,
    int heap_length):
    """ 
    Compute single-target shortest path (from all vertices to one vertex)
    using the pq_4ary_dec_0b priority queue.

    Compute successors.

    input
    =====
    * cnp.uint32_t[::1] csc_indices : indices in the CSC format
    * cnp.uint32_t[::1] csc_indptr : pointers in the CSC format
    * DTYPE_t[::1] csc_data :  data (edge weights) in the CSC format
    * cnp.uint32_t[::1] successor :  array of indices, one for each vertex of 
        the graph. Each vertex' entry contains the index of its successor in 
        a path to the target, through the graph. 
    * int target_vert_idx : source vertex index
    * int vertex_count : vertex count
    * int heap_length : heap length

    output
    ======
    * cnp.ndarray : shortest path length for each vertex
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t target = <size_t>target_vert_idx

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and UNLABELED state
        pq.init_pqueue(&pqueue, <size_t>heap_length, <size_t>vertex_count)

        # the key is set to zero for the target vertex,
        # which is inserted into the heap
        pq.insert(&pqueue, target, 0.0)

        # main loop
        while pqueue.size > 0:
            head_vert_idx = pq.extract_min(&pqueue)
            head_vert_val = pqueue.Elements[head_vert_idx].key

            # loop on incoming edges
            for idx in range(<size_t>csc_indptr[head_vert_idx], 
                <size_t>csc_indptr[head_vert_idx + 1]):

                tail_vert_idx = <size_t>csc_indices[idx]
                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    tail_vert_val = head_vert_val + csc_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, tail_vert_idx, tail_vert_val)
                        successor[tail_vert_idx] = head_vert_idx
                    elif pqueue.Elements[tail_vert_idx].key > tail_vert_val:
                        pq.decrease_key(&pqueue, tail_vert_idx, tail_vert_val)
                        successor[tail_vert_idx] = head_vert_idx

    # copy the results into a numpy array
    path_lengths = pq.copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    pq.free_pqueue(&pqueue)  

    return path_lengths


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #

from edsger.commons import DTYPE_PY
import numpy as np


cdef generate_single_edge_network_csr():
    """  
    Generate a single edge network in CSR format.

    This network has 1 edge and 2 vertices.
    """

    csr_indptr = np.array([0, 1, 1], dtype=np.uint32)
    csr_indices = np.array([1], dtype=np.uint32)
    csr_data = np.array([1.], dtype=DTYPE_PY)

    return csr_indptr, csr_indices, csr_data


cdef generate_single_edge_network_csc():
    """  
    Generate a single edge network in CSC format.

    This network has 1 edge and 2 vertices.
    """

    csc_indptr = np.array([0, 0, 1], dtype=np.uint32)
    csc_indices = np.array([0], dtype=np.uint32)
    csc_data = np.array([1.], dtype=DTYPE_PY)

    return csc_indptr, csc_indices, csc_data


cdef generate_braess_network_csr():
    """ 
    Generate a Braess-like network in CSR format.

    This network hs 5 edges and 4 vertices.
    """

    csr_indptr = np.array([0, 2, 4, 5, 5], dtype=np.uint32)
    csr_indices = np.array([1, 2, 2, 3, 3], dtype=np.uint32)
    csr_data = np.array([1., 2., 0., 2., 1.], dtype=DTYPE_PY)

    return csr_indptr, csr_indices, csr_data


cdef generate_braess_network_csc():
    """ 
    Generate a Braess-like network in CSC format.

    This network hs 5 edges and 4 vertices.
    """

    csc_indptr = np.array([0, 0, 1, 3, 5], dtype=np.uint32)
    csc_indices = np.array([0, 0, 1, 1, 2], dtype=np.uint32)
    csc_data = np.array([1., 2., 0., 2., 1.], dtype=DTYPE_PY)

    return csc_indptr, csc_indices, csc_data


cpdef compute_sssp_01():
    """ 
    Compute SSSP with the compute_sssp_pq_bd0 routine on a single edge 
    network.
    """

    csr_indptr, csr_indices, csr_data = generate_single_edge_network_csr()

    # from vertex 0
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 0, 2, 2)
    path_lengths_ref = np.array([0., 1.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 1, 2, 2)
    path_lengths_ref = np.array([DTYPE_INF, 0.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)


cpdef compute_stsp_01():
    """ 
    Compute TSSP with the compute_stsp_pq_bd0 routine on a single edge 
    network.
    """

    csc_indptr, csc_indices, csc_data = generate_single_edge_network_csc()

    # from vertex 0
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 0, 2, 2)
    path_lengths_ref = np.array([0., DTYPE_INF], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 1, 2, 2)
    path_lengths_ref = np.array([1., 0.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)


cpdef compute_sssp_02():
    """ 
    Compute SSSP with the compute_sssp_pq_bd0 routine on Braess-like 
    network.
    """

    csr_indptr, csr_indices, csr_data = generate_braess_network_csr()

    # from vertex 0
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 0, 4, 4)
    path_lengths_ref = np.array([0., 1., 1., 2.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 1, 4, 4)
    path_lengths_ref = np.array([DTYPE_INF, 0., 0., 1.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 2
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 2, 4, 4)
    path_lengths_ref = np.array([DTYPE_INF, DTYPE_INF, 0., 1.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 3
    path_lengths = compute_sssp(csr_indptr, csr_indices, csr_data, 3, 4, 4)
    path_lengths_ref = np.array([DTYPE_INF, DTYPE_INF, DTYPE_INF, 0.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)


cpdef compute_stsp_02():
    """ 
    Compute STSP with the compute_stsp_pq_bd0 routine on Braess-like 
    network.
    """

    csc_indptr, csc_indices, csc_data = generate_braess_network_csc()

    # from vertex 0
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 0, 4, 4)
    path_lengths_ref = np.array([0., DTYPE_INF, DTYPE_INF, DTYPE_INF], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 1
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 1, 4, 4)
    path_lengths_ref = np.array([1., 0., DTYPE_INF, DTYPE_INF], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 2
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 2, 4, 4)
    path_lengths_ref = np.array([1., 0., 0., DTYPE_INF], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)

    # from vertex 3
    path_lengths = compute_stsp(csc_indptr, csc_indices, csc_data, 3, 4, 4)
    path_lengths_ref = np.array([2., 1.0, 1., 0.], dtype=DTYPE_PY)
    assert np.allclose(path_lengths_ref, path_lengths)