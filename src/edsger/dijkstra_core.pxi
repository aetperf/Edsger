"""
Dijkstra's algorithm: code shared by dijkstra.pyx and dijkstra_nodec.pyx.

This file is not a module. It is included by the two modules, after they have
defined the compile-time constant LAZY:

- LAZY = 0 : priority queue with decrease-key (dijkstra.pyx)
- LAZY = 1 : no decrease-key, a new entry is inserted into the heap each time
  the path length of a vertex is improved, and stale entries are skipped when
  they are extracted (dijkstra_nodec.pyx)

LAZY is a C constant: the branches that depend on it are removed by the C
compiler, each module only contains the code of its own variant.

cdef functions:

- _dijkstra
    Dijkstra's algorithm, from one origin vertex.
- _run
    Check the arguments, allocate the output and call _dijkstra.

cpdef functions: see the docstring of the including module.
"""

cimport numpy as cnp
import numpy as np

from edsger.commons import DTYPE_PY
from edsger.commons cimport DTYPE_INF, DTYPE_t, check_graph_arguments
from edsger.pq_4ary_inline cimport (
    HeapNode, HEAP_OFFSET, HEAP_SENTINELS, heap_alloc, heap_grow, heap_free,
    sift_down, sift_up, prefetch_t0, thp_alloc, thp_free)


cdef int _dijkstra(
        const cnp.uint32_t* indptr,
        const cnp.uint32_t* indices,
        const DTYPE_t* data,
        DTYPE_t* dist,
        cnp.uint32_t* path,
        const cnp.uint32_t* termination_nodes,
        size_t termination_count,
        size_t origin,
        size_t vertex_count,
        size_t heap_length) noexcept nogil:
    """
    Dijkstra's algorithm, from one origin vertex, on a graph given in a
    compressed sparse format. All the public functions call this one: SSSP on
    the CSR arrays, STSP on the CSC arrays.

    The tentative path lengths are stored in `dist`, which is also the
    output: DTYPE_INF means that the vertex has not been reached. Since the
    edge weights are non-negative, the path length of a scanned vertex cannot
    be improved, so that no vertex state is needed: an edge is relaxed if and
    only if it improves the path length of its head vertex.

    With decrease-key (LAZY = 0), the head vertex is then inserted into the
    heap if its path length was DTYPE_INF, or has its key decreased
    otherwise.

    Without decrease-key (LAZY = 1), a new entry is inserted into the heap
    in both cases. The keys inserted for a given vertex are strictly
    decreasing, so that an extracted entry is stale, and skipped, if and only
    if its key is larger than the path length of its vertex. With
    non-negative edge weights, each vertex is scanned once and each edge is
    relaxed at most once. The number of insertions is bounded accordingly, so
    that the algorithm stops, with an error code, on a graph with negative
    weights, on which it could otherwise loop forever.

    input
    =====
    * const cnp.uint32_t* indptr : pointers in the CSR/CSC format
    * const cnp.uint32_t* indices : indices in the CSR/CSC format
    * const DTYPE_t* data : edge weights in the CSR/CSC format
    * DTYPE_t* dist : path length of each vertex (output)
    * cnp.uint32_t* path : predecessor/successor of each vertex (output), or
      NULL if the paths are not tracked
    * const cnp.uint32_t* termination_nodes : vertices for early termination,
      or NULL
    * size_t termination_count : number of termination nodes, 0 for no early
      termination
    * size_t origin : origin vertex index
    * size_t vertex_count : vertex count
    * size_t heap_length : initial capacity of the heap, which grows if
      needed

    output
    ======
    * int : 0 on success, -1 if a memory allocation failed, -2 if more
      entries were inserted than is possible with non-negative edge weights
      (LAZY = 1 only)
    """

    cdef:
        size_t size, i, idx, edge_end, tail_vert_idx, head_vert_idx
        size_t next_vert_idx, next_idx, visited_count
        size_t iteration_count = 0
        size_t check_frequency = 16
        size_t capacity = heap_length if heap_length > 0 else 1
        size_t pos_bytes = vertex_count * sizeof(cnp.uint32_t)
        size_t insert_count = 1  # the origin vertex
        size_t insert_count_max = <size_t>indptr[vertex_count] + 1
        DTYPE_t tail_vert_val, head_vert_val, head_vert_val_old
        DTYPE_t inf = DTYPE_INF
        cnp.uint32_t* pos = NULL
        HeapNode* heap_base
        HeapNode* heap_base_new
        HeapNode* heap
        HeapNode last
        int status = 0

    # heap index of each vertex, for the decrease-key operation: it does not
    # need to be initialized
    if not LAZY:
        pos = <cnp.uint32_t*> thp_alloc(pos_bytes)
        if pos == NULL:
            return -1
    heap_base = heap_alloc(capacity)
    if heap_base == NULL:
        if not LAZY:
            thp_free(pos, pos_bytes)
        return -1
    heap = heap_base + HEAP_OFFSET

    # all vertices have an INFINITY path length
    for i in range(vertex_count):
        dist[i] = inf

    # the origin vertex is inserted into the heap with a zero key
    dist[origin] = 0.0
    heap[0].key = 0.0
    heap[0].vert = <cnp.uint32_t>origin
    if not LAZY:
        pos[origin] = 0
    size = 1
    for i in range(HEAP_SENTINELS):
        heap[size + i].key = inf

    # main loop
    while size > 0:

        # extract the entry with min key
        tail_vert_idx = heap[0].vert
        tail_vert_val = heap[0].key
        size -= 1
        last = heap[size]
        heap[size].key = inf
        if size > 0:
            sift_down(heap, pos, size, last.key, last.vert)

            # The new root is most likely the next vertex to be scanned, and
            # its children the ones after that. Their adjacency data is
            # requested now, so that it is in the cache when it is needed, as
            # well as the path length read by the stale entry test.
            next_vert_idx = heap[0].vert
            if LAZY:
                prefetch_t0(<void*>&dist[next_vert_idx])
            next_idx = indptr[next_vert_idx]
            if next_idx < indptr[next_vert_idx + 1]:
                prefetch_t0(<void*>&data[next_idx])
                prefetch_t0(<void*>&dist[indices[next_idx]])
            if size > 4:
                for i in range(1, 5):
                    next_idx = indptr[heap[i].vert]
                    prefetch_t0(<void*>&indices[next_idx])
                    prefetch_t0(<void*>&data[next_idx])

        # skip stale entries: the path length of the vertex has been improved
        # since this entry was inserted
        if LAZY and tail_vert_val > dist[tail_vert_idx]:
            continue

        # Check for early termination every check_frequency iterations. The
        # path length of a vertex is final as soon as it is not larger than
        # the key that has just been extracted.
        if termination_count > 0:
            iteration_count += 1
            if iteration_count % check_frequency == 0:
                visited_count = 0
                for i in range(termination_count):
                    if dist[termination_nodes[i]] <= tail_vert_val:
                        visited_count += 1
                if visited_count == termination_count:
                    break

        # loop on outgoing edges
        edge_end = indptr[tail_vert_idx + 1]
        for idx in range(indptr[tail_vert_idx], edge_end):

            head_vert_idx = indices[idx]
            head_vert_val = tail_vert_val + data[idx]
            head_vert_val_old = dist[head_vert_idx]

            if head_vert_val < head_vert_val_old:
                dist[head_vert_idx] = head_vert_val
                if LAZY or head_vert_val_old == inf:
                    # insert a new entry: first visit of the vertex, or any
                    # improvement of its path length without decrease-key
                    prefetch_t0(<void*>&indptr[head_vert_idx])
                    if LAZY:
                        insert_count += 1
                        if insert_count > insert_count_max:
                            # negative edge weights: leave the main loop
                            status = -2
                            size = 0
                            break
                    if size == capacity:
                        heap_base_new = heap_grow(heap_base, capacity, 2 * capacity)
                        if heap_base_new == NULL:
                            # leave the main loop
                            status = -1
                            size = 0
                            break
                        heap_base = heap_base_new
                        heap = heap_base + HEAP_OFFSET
                        capacity = 2 * capacity
                    i = size
                    size += 1
                    heap[size + HEAP_SENTINELS - 1].key = inf
                else:
                    # already in the heap: decrease key
                    i = pos[head_vert_idx]
                sift_up(heap, pos, i, head_vert_val, <cnp.uint32_t>head_vert_idx)
                if path != NULL:
                    path[head_vert_idx] = <cnp.uint32_t>tail_vert_idx

    # cleanup
    if not LAZY:
        thp_free(pos, pos_bytes)
    heap_free(heap_base, capacity)

    return status


cdef cnp.ndarray _run(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        DTYPE_t[::1] data,
        cnp.uint32_t[::1] path,
        cnp.uint32_t[::1] termination_nodes,
        int origin_vert_idx,
        int vertex_count,
        int heap_length):
    """
    Allocate the path length array, run Dijkstra's algorithm and return the
    path lengths: for each vertex, or for each termination node if any.

    `path` and `termination_nodes` may be None.

    The arguments are checked for consistency, in constant time. The content
    of the graph arrays is assumed to be valid: non-decreasing pointers,
    indices smaller than the vertex count, non-negative weights. The graphs
    built by edsger.path.Dijkstra are. For arrays coming from somewhere else,
    edsger.commons.validate_graph checks them, in linear time: it is meant to
    be called once per graph, not before each run.
    """

    cdef:
        cnp.ndarray path_lengths
        DTYPE_t[::1] dist
        cnp.uint32_t* path_ptr = NULL
        cnp.uint32_t* termination_ptr = NULL
        size_t termination_count = 0
        int status

    # bounds checking is disabled in this module
    check_graph_arguments(
        indptr, indices, data.shape[0], origin_vert_idx, vertex_count)
    if heap_length < 0:
        raise ValueError(f"the heap length should not be negative, got {heap_length}")
    if path is not None:
        if path.shape[0] < vertex_count:
            raise ValueError(
                f"the path array has {path.shape[0]} elements for {vertex_count} "
                "vertices"
            )
        path_ptr = &path[0]
    if termination_nodes is not None and termination_nodes.shape[0] > 0:
        if np.max(termination_nodes) >= vertex_count:
            raise IndexError(
                f"termination node index is out of bounds for {vertex_count} "
                "vertices"
            )
        termination_ptr = &termination_nodes[0]
        termination_count = termination_nodes.shape[0]

    # the output array is allocated once the arguments have been checked
    path_lengths = np.empty(vertex_count, dtype=DTYPE_PY)
    dist = path_lengths

    with nogil:
        status = _dijkstra(
            &indptr[0],
            &indices[0],
            &data[0],
            &dist[0],
            path_ptr,
            termination_ptr,
            termination_count,
            <size_t>origin_vert_idx,
            <size_t>vertex_count,
            <size_t>heap_length)

    if status == -2:
        raise ValueError(
            "negative edge weights are not supported by Dijkstra's algorithm"
        )
    if status != 0:
        raise MemoryError("unable to allocate the priority queue")

    if termination_nodes is not None:
        # only the termination nodes' results
        return path_lengths[np.asarray(termination_nodes)]

    return path_lengths


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

    return _run(
        csr_indptr, csr_indices, csr_data, None, None,
        source_vert_idx, vertex_count, heap_length)


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

    return _run(
        csr_indptr, csr_indices, csr_data, None, termination_nodes,
        source_vert_idx, vertex_count, heap_length)


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

    return _run(
        csr_indptr, csr_indices, csr_data, predecessor, None,
        source_vert_idx, vertex_count, heap_length)


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

    return _run(
        csr_indptr, csr_indices, csr_data, predecessor, termination_nodes,
        source_vert_idx, vertex_count, heap_length)


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

    return _run(
        csc_indptr, csc_indices, csc_data, None, None,
        target_vert_idx, vertex_count, heap_length)


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

    return _run(
        csc_indptr, csc_indices, csc_data, successor, None,
        target_vert_idx, vertex_count, heap_length)


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

    return _run(
        csc_indptr, csc_indices, csc_data, None, termination_nodes,
        target_vert_idx, vertex_count, heap_length)


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

    return _run(
        csc_indptr, csc_indices, csc_data, successor, termination_nodes,
        target_vert_idx, vertex_count, heap_length)


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #


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
    Compute SSSP with the compute_sssp routine on a single edge
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
    Compute TSSP with the compute_stsp routine on a single edge
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
    Compute SSSP with the compute_sssp routine on Braess-like
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
    Compute STSP with the compute_stsp routine on Braess-like
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
