"""
An implementation of Dijkstra's algorithm.

cpdef functions:

- compute_sssp
    Compute single-source shortest path (from one vertex to all vertices). Does
    not return predecessors.
- compute_sssp_w_path
    Compute single-source shortest path (from one vertex to all vertices).
    Compute predecessors.
- compute_sssp_early_termination
    Compute single-source shortest path with early termination when target nodes
    are reached. Does not return predecessors.
- compute_sssp_w_path_early_termination
    Compute single-source shortest path with early termination when target nodes
    are reached. Compute predecessors.
- compute_stsp
    Compute single-target shortest path (from all vertices to one vertex). Does
    not return successors.
- compute_stsp_w_path
    Compute single-target shortest path (from all vertices to one vertex).
    Compute successors.
- compute_stsp_early_termination
    Compute single-target shortest path with early termination when target nodes
    are reached. Does not return successors.
- compute_stsp_w_path_early_termination
    Compute single-target shortest path with early termination when target nodes
    are reached. Compute successors.
"""

cimport numpy as cnp

from edsger.commons cimport (
    DTYPE_INF, UNLABELED, SCANNED, DTYPE_t, ElementState)
cimport edsger.pq_4ary_dec_0b as pq  # priority queue

# memory prefetching support (x86/x64 only)
cdef extern from "prefetch_compat.h":
    void prefetch_hint(char*, int) nogil
    int PREFETCH_T0


cpdef cnp.ndarray compute_sssp(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        int source_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Compute single-source shortest path (from one vertex to all vertices). Does
    not return predecessors.

    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data DTYPE_t[::1]
        data (edge weights) in the CSR format
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
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

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csr_indptr[tail_vert_idx + 1]:
                    prefetch_hint(<char*>&csr_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csr_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[head_vert_idx], PREFETCH_T0)

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


cpdef cnp.ndarray compute_sssp_early_termination(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        cnp.uint32_t[::1] termination_nodes,
        int source_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Compute single-source shortest path with early termination when target
    nodes are reached.
    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data DTYPE_t[::1]
        data (edge weights) in the CSR format
    termination_nodes : cnp.uint32_t[::1]
        target node indices for early termination
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each termination node
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, i
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t source = <size_t>source_vert_idx
        size_t target_count = termination_nodes.shape[0]
        size_t visited_targets = 0
        size_t iteration_count = 0
        size_t check_frequency = 16

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

            # check for early termination every check_frequency iterations
            iteration_count += 1
            if iteration_count % check_frequency == 0:
                visited_targets = 0
                for i in range(target_count):
                    if pqueue.Elements[termination_nodes[i]].state == SCANNED:
                        visited_targets += 1
                if visited_targets == target_count:
                    break

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx],
                             <size_t>csr_indptr[tail_vert_idx + 1]):

                head_vert_idx = <size_t>csr_indices[idx]

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csr_indptr[tail_vert_idx + 1]:
                    prefetch_hint(<char*>&csr_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csr_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[head_vert_idx], PREFETCH_T0)

                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        pq.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy only the termination nodes' results into a numpy array
    cdef cnp.ndarray path_lengths = np.empty(target_count, dtype=DTYPE_PY)
    for i in range(target_count):
        path_lengths[i] = pqueue.Elements[termination_nodes[i]].key

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
    Compute single-source shortest path (from one vertex to all vertices).
    Compute predecessors.

    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format
    predecessor : cnp.uint32_t[::1]
        array of indices, one for each vertex of the graph. Each vertex'
        entry contains the index of its predecessor in a path from the
        source, through the graph.
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
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

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csr_indptr[tail_vert_idx + 1]:
                    prefetch_hint(<char*>&csr_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csr_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[head_vert_idx], PREFETCH_T0)

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


cpdef cnp.ndarray compute_sssp_w_path_early_termination(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        cnp.uint32_t[::1] predecessor,
        cnp.uint32_t[::1] termination_nodes,
        int source_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Compute single-source shortest path with path tracking and early termination.
    Parameters
    ----------
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format
    predecessor : cnp.uint32_t[::1]
        array of indices, one for each vertex of the graph. Each vertex'
        entry contains the index of its predecessor in a path from the
        source, through the graph.
    termination_nodes : cnp.uint32_t[::1]
        target node indices for early termination
    source_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each termination node
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, i
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t source = <size_t>source_vert_idx
        size_t target_count = termination_nodes.shape[0]
        size_t visited_targets = 0
        size_t iteration_count = 0
        size_t check_frequency = 16

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

            # check for early termination every check_frequency iterations
            iteration_count += 1
            if iteration_count % check_frequency == 0:
                visited_targets = 0
                for i in range(target_count):
                    if pqueue.Elements[termination_nodes[i]].state == SCANNED:
                        visited_targets += 1
                if visited_targets == target_count:
                    break

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx],
                             <size_t>csr_indptr[tail_vert_idx + 1]):

                head_vert_idx = <size_t>csr_indices[idx]

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csr_indptr[tail_vert_idx + 1]:
                    prefetch_hint(<char*>&csr_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csr_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[head_vert_idx], PREFETCH_T0)

                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, head_vert_idx, head_vert_val)
                        predecessor[head_vert_idx] = tail_vert_idx
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        pq.decrease_key(&pqueue, head_vert_idx, head_vert_val)
                        predecessor[head_vert_idx] = tail_vert_idx

    # copy only the termination nodes' results into a numpy array
    cdef cnp.ndarray path_lengths = np.empty(target_count, dtype=DTYPE_PY)
    for i in range(target_count):
        path_lengths[i] = pqueue.Elements[termination_nodes[i]].key

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
    Compute single-target shortest path (from all vertices to one vertex). Does
    not return successors.

    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        indices in the CSC format
    csc_indptr : cnp.uint32_t[::1]
        pointers in the CSC format
    csc_data : DTYPE_t[::1]
        data (edge weights) in the CSC format
    target_vert_idx : int
        source vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
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

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csc_indptr[head_vert_idx + 1]:
                    prefetch_hint(<char*>&csc_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csc_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[tail_vert_idx], PREFETCH_T0)

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
    Compute single-target shortest path (from all vertices to one vertex).
    Compute successors.

    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        Indices in the CSC format.
    csc_indices : cnp.uint32_t[::1]
        Pointers in the CSC format.
    csc_data : DTYPE_t[::1]
        Data (edge weights) in the CSC format.
    target_vert_idx : int
        Target vertex index.
    vertex_count : int
        Vertex count.
    heap_length : int
        heap_length.

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each vertex
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

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csc_indptr[head_vert_idx + 1]:
                    prefetch_hint(<char*>&csc_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csc_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[tail_vert_idx], PREFETCH_T0)

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


cpdef cnp.ndarray compute_stsp_early_termination(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        cnp.uint32_t[::1] termination_nodes,
        int target_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Compute single-target shortest path with early termination when target
    nodes are reached.
    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        indices in the CSC format
    csc_indptr : cnp.uint32_t[::1]
        pointers in the CSC format
    csc_data : DTYPE_t[::1]
        data (edge weights) in the CSC format
    termination_nodes : cnp.uint32_t[::1]
        target node indices for early termination
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count
    heap_length : int
        heap length

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each termination node
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, i
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t target = <size_t>target_vert_idx
        size_t target_count = termination_nodes.shape[0]
        size_t visited_targets = 0
        size_t iteration_count = 0
        size_t check_frequency = 16

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

            # check for early termination every check_frequency iterations
            iteration_count += 1
            if iteration_count % check_frequency == 0:
                visited_targets = 0
                for i in range(target_count):
                    if pqueue.Elements[termination_nodes[i]].state == SCANNED:
                        visited_targets += 1
                if visited_targets == target_count:
                    break

            # loop on incoming edges
            for idx in range(<size_t>csc_indptr[head_vert_idx],
                             <size_t>csc_indptr[head_vert_idx + 1]):

                tail_vert_idx = <size_t>csc_indices[idx]

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csc_indptr[head_vert_idx + 1]:
                    prefetch_hint(<char*>&csc_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csc_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[tail_vert_idx], PREFETCH_T0)

                    tail_vert_val = head_vert_val + csc_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, tail_vert_idx, tail_vert_val)
                    elif pqueue.Elements[tail_vert_idx].key > tail_vert_val:
                        pq.decrease_key(&pqueue, tail_vert_idx, tail_vert_val)

    # copy only the termination nodes' results into a numpy array
    cdef cnp.ndarray path_lengths = np.empty(target_count, dtype=DTYPE_PY)
    for i in range(target_count):
        path_lengths[i] = pqueue.Elements[termination_nodes[i]].key

    # cleanup
    pq.free_pqueue(&pqueue)

    return path_lengths


cpdef cnp.ndarray compute_stsp_w_path_early_termination(
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        cnp.uint32_t[::1] successor,
        cnp.uint32_t[::1] termination_nodes,
        int target_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Compute single-target shortest path with path tracking and early termination.
    Parameters
    ----------
    csc_indices : cnp.uint32_t[::1]
        Indices in the CSC format.
    csc_indptr : cnp.uint32_t[::1]
        Pointers in the CSC format.
    csc_data : DTYPE_t[::1]
        Data (edge weights) in the CSC format.
    successor : cnp.uint32_t[::1]
        Array of successor indices for path reconstruction.
    termination_nodes : cnp.uint32_t[::1]
        target node indices for early termination
    target_vert_idx : int
        Target vertex index.
    vertex_count : int
        Vertex count.
    heap_length : int
        heap_length.

    Returns
    -------
    path_lengths : cnp.ndarray
        shortest path length for each termination node
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx, i
        DTYPE_t tail_vert_val, head_vert_val
        pq.PriorityQueue pqueue
        ElementState vert_state
        size_t target = <size_t>target_vert_idx
        size_t target_count = termination_nodes.shape[0]
        size_t visited_targets = 0
        size_t iteration_count = 0
        size_t check_frequency = 16

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

            # check for early termination every check_frequency iterations
            iteration_count += 1
            if iteration_count % check_frequency == 0:
                visited_targets = 0
                for i in range(target_count):
                    if pqueue.Elements[termination_nodes[i]].state == SCANNED:
                        visited_targets += 1
                if visited_targets == target_count:
                    break

            # loop on incoming edges
            for idx in range(<size_t>csc_indptr[head_vert_idx],
                             <size_t>csc_indptr[head_vert_idx + 1]):

                tail_vert_idx = <size_t>csc_indices[idx]

                # prefetch next iteration data to improve cache performance
                if idx + 1 < <size_t>csc_indptr[head_vert_idx + 1]:
                    prefetch_hint(<char*>&csc_indices[idx + 1], PREFETCH_T0)
                    prefetch_hint(<char*>&csc_data[idx + 1], PREFETCH_T0)

                vert_state = pqueue.Elements[tail_vert_idx].state
                if vert_state != SCANNED:
                    # prefetch priority queue element data for the vertex
                    prefetch_hint(<char*>&pqueue.Elements[tail_vert_idx], PREFETCH_T0)

                    tail_vert_val = head_vert_val + csc_data[idx]
                    if vert_state == UNLABELED:
                        pq.insert(&pqueue, tail_vert_idx, tail_vert_val)
                        successor[tail_vert_idx] = head_vert_idx
                    elif pqueue.Elements[tail_vert_idx].key > tail_vert_val:
                        pq.decrease_key(&pqueue, tail_vert_idx, tail_vert_val)
                        successor[tail_vert_idx] = head_vert_idx

    # copy only the termination nodes' results into a numpy array
    cdef cnp.ndarray path_lengths = np.empty(target_count, dtype=DTYPE_PY)
    for i in range(target_count):
        path_lengths[i] = pqueue.Elements[termination_nodes[i]].key

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


cpdef compute_sssp_early_termination_01():
    """
    Test SSSP early termination on Braess-like network.
    """

    csr_indptr, csr_indices, csr_data = generate_braess_network_csr()

    # from vertex 0, stop when vertices 1 and 3 are reached
    target_nodes = np.array([1, 3], dtype=np.uint32)
    path_lengths = compute_sssp_early_termination(
        csr_indptr, csr_indices, csr_data, target_nodes, 0, 4, 4
    )

    # should return path lengths only for termination nodes [1, 3]
    path_lengths_ref = np.array([1., 2.], dtype=DTYPE_PY)  # lengths to nodes 1 and 3
    assert np.allclose(path_lengths_ref, path_lengths)

    # test with path tracking
    predecessor = np.arange(0, 4, dtype=np.uint32)
    path_lengths_w_path = compute_sssp_w_path_early_termination(
        csr_indptr, csr_indices, csr_data, predecessor, target_nodes, 0, 4, 4
    )
    assert np.allclose(path_lengths_ref, path_lengths_w_path)


cpdef compute_stsp_early_termination_01():
    """
    Test STSP early termination on Braess-like network.
    """

    csc_indptr, csc_indices, csc_data = generate_braess_network_csc()

    # to vertex 3, stop when vertices 0 and 2 are reached
    target_nodes = np.array([0, 2], dtype=np.uint32)
    path_lengths = compute_stsp_early_termination(
        csc_indptr, csc_indices, csc_data, target_nodes, 3, 4, 4
    )

    # should return path lengths only for termination nodes [0, 2]
    path_lengths_ref = np.array([2., 1.], dtype=DTYPE_PY)  # lengths to nodes 0 and 2
    assert np.allclose(path_lengths_ref, path_lengths)

    # test with path tracking
    successor = np.arange(0, 4, dtype=np.uint32)
    path_lengths_w_path = compute_stsp_w_path_early_termination(
        csc_indptr, csc_indices, csc_data, successor, target_nodes, 3, 4, 4
    )
    assert np.allclose(path_lengths_ref, path_lengths_w_path)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
