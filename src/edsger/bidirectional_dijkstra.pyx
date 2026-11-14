# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Bidirectional Dijkstra's algorithm, for the shortest path between two
vertices.

Two searches are run alternately: a forward search from the source on the
outgoing edges (CSR format), and a backward search from the target on the
incoming edges (CSC format). At each step, the search whose smallest key is
the smallest scans one vertex. Each time an edge (u, v) is relaxed by one
search and v has a finite path length in the other one, the length of the
path through v is a candidate for the shortest path length mu. The searches
stop as soon as the sum of their smallest keys is not smaller than mu, or as
soon as one of them has no vertex left to scan: no path can be shorter than
mu.

Each search uses the inline 4-ary heap of `pq_4ary_inline`, without
decrease-key: a vertex is inserted again each time its path length is
improved, and the stale entries are skipped. They are removed from the top of
the heaps before the stopping test, so that the test uses the exact smallest
path lengths of the vertices left to scan.

The per-vertex arrays live in a `Workspace`, which is reused from one run to
the next: the path lengths are kept at infinity between the runs, and only the
vertices reached by a run are reset at its end. The cost of a run does not
depend on the vertex count, only on the part of the graph that is explored.

cdef classes:

- Workspace
    Per-vertex arrays reused by successive runs.

cpdef functions:

- compute_bidirectional
    Shortest path length between two vertices, and the vertex where the two
    half paths meet.
"""

cimport numpy as cnp
import numpy as np

from edsger.commons import DTYPE_PY
from edsger.commons cimport DTYPE_INF, DTYPE_t, check_graph_arguments
from edsger.pq_4ary_inline cimport (
    HeapNode, HEAP_OFFSET, HEAP_SENTINELS, heap_alloc, heap_grow, heap_free,
    sift_down, sift_up, prefetch_t0)


cdef struct Search:
    # graph in CSR (forward search) or CSC (backward search) format
    const cnp.uint32_t* indptr
    const cnp.uint32_t* indices
    const DTYPE_t* data
    # tentative path lengths: DTYPE_INF on entry and on exit
    DTYPE_t* dist
    # vertices reached by the search
    cnp.uint32_t* touched
    size_t touched_count
    # predecessor (forward) or successor (backward) of each vertex, or NULL
    cnp.uint32_t* path
    # heap
    HeapNode* heap_base
    HeapNode* heap
    size_t size
    size_t capacity


cdef inline int _search_init(
        Search* s, size_t origin, size_t heap_length) noexcept nogil:
    """Allocate the heap and insert the origin vertex. Returns -1 if the
    allocation failed."""
    cdef size_t i
    s.capacity = heap_length if heap_length > 0 else 1
    s.size = 0
    s.touched_count = 0
    s.heap_base = heap_alloc(s.capacity)
    if s.heap_base == NULL:
        return -1
    s.heap = s.heap_base + HEAP_OFFSET
    s.dist[origin] = 0.0
    s.touched[0] = <cnp.uint32_t>origin
    s.touched_count = 1
    if s.path != NULL:
        s.path[origin] = <cnp.uint32_t>origin
    s.heap[0].key = 0.0
    s.heap[0].vert = <cnp.uint32_t>origin
    s.size = 1
    for i in range(HEAP_SENTINELS):
        s.heap[s.size + i].key = DTYPE_INF
    return 0


cdef inline void _search_free(Search* s) noexcept nogil:
    """Reset the reached vertices to an infinite path length and free the
    heap."""
    cdef size_t i
    for i in range(s.touched_count):
        s.dist[s.touched[i]] = DTYPE_INF
    s.touched_count = 0
    if s.heap_base != NULL:
        heap_free(s.heap_base, s.capacity)
        s.heap_base = NULL


cdef inline void _pop(Search* s) noexcept nogil:
    """Remove the root of the heap."""
    cdef HeapNode last
    s.size -= 1
    last = s.heap[s.size]
    s.heap[s.size].key = DTYPE_INF
    if s.size > 0:
        sift_down(s.heap, NULL, s.size, last.key, last.vert)


cdef inline void _skip_stale(Search* s) noexcept nogil:
    """Remove the stale entries from the top of the heap: their key is larger
    than the path length of their vertex, which has been improved since they
    were inserted."""
    while s.size > 0 and s.heap[0].key > s.dist[s.heap[0].vert]:
        _pop(s)


cdef inline int _scan(
        Search* s,
        const DTYPE_t* dist_other,
        DTYPE_t* mu,
        size_t* meet) noexcept nogil:
    """
    Scan the vertex at the root of the heap, which is not stale, and update
    the shortest path length mu and the meeting vertex.

    output
    ======
    * int : 0 on success, -1 if a memory allocation failed, -2 if a negative
      edge weight was found
    """
    cdef:
        size_t tail_vert_idx = s.heap[0].vert
        DTYPE_t tail_vert_val = s.heap[0].key
        size_t idx, edge_end, head_vert_idx, next_vert_idx, next_idx, i
        DTYPE_t head_vert_val, candidate
        HeapNode* heap_base_new

    _pop(s)
    if s.size > 0:
        # adjacency of the next vertex to be scanned, most likely
        next_vert_idx = s.heap[0].vert
        next_idx = s.indptr[next_vert_idx]
        if next_idx < s.indptr[next_vert_idx + 1]:
            prefetch_t0(<void*>&s.data[next_idx])
            prefetch_t0(<void*>&s.dist[s.indices[next_idx]])

    edge_end = s.indptr[tail_vert_idx + 1]
    for idx in range(s.indptr[tail_vert_idx], edge_end):

        head_vert_idx = s.indices[idx]
        head_vert_val = tail_vert_val + s.data[idx]

        if head_vert_val < s.dist[head_vert_idx]:
            if head_vert_val < tail_vert_val:
                return -2
            if s.dist[head_vert_idx] == DTYPE_INF:
                # first visit of the vertex
                s.touched[s.touched_count] = <cnp.uint32_t>head_vert_idx
                s.touched_count += 1
                prefetch_t0(<void*>&s.indptr[head_vert_idx])
            s.dist[head_vert_idx] = head_vert_val
            if s.path != NULL:
                s.path[head_vert_idx] = <cnp.uint32_t>tail_vert_idx

            # insert a new entry
            if s.size == s.capacity:
                heap_base_new = heap_grow(s.heap_base, s.capacity, 2 * s.capacity)
                if heap_base_new == NULL:
                    return -1
                s.heap_base = heap_base_new
                s.heap = heap_base_new + HEAP_OFFSET
                s.capacity = 2 * s.capacity
            i = s.size
            s.size += 1
            s.heap[s.size + HEAP_SENTINELS - 1].key = DTYPE_INF
            sift_up(s.heap, NULL, i, head_vert_val, <cnp.uint32_t>head_vert_idx)

            # path through the head vertex, reached by the other search
            if dist_other[head_vert_idx] != DTYPE_INF:
                candidate = head_vert_val + dist_other[head_vert_idx]
                if candidate < mu[0]:
                    mu[0] = candidate
                    meet[0] = head_vert_idx

    return 0


cdef int _bidirectional(
        Search* fwd,
        Search* bwd,
        size_t source,
        size_t target,
        size_t heap_length,
        DTYPE_t* length,
        size_t* meet) noexcept nogil:
    """
    Bidirectional Dijkstra's algorithm from a source to a target vertex.

    input
    =====
    * Search* fwd : forward search, graph and arrays set
    * Search* bwd : backward search, graph and arrays set
    * size_t source : source vertex index
    * size_t target : target vertex index
    * size_t heap_length : initial capacity of each heap
    * DTYPE_t* length : shortest path length (output), DTYPE_INF if the target
      cannot be reached
    * size_t* meet : vertex of the shortest path where the two searches meet
      (output), unchanged if the target cannot be reached

    output
    ======
    * int : 0 on success, -1 if a memory allocation failed, -2 if a negative
      edge weight was found on the way
    """
    cdef:
        DTYPE_t mu = DTYPE_INF
        int status = 0

    length[0] = DTYPE_INF
    fwd.heap_base = NULL
    bwd.heap_base = NULL
    fwd.touched_count = 0
    bwd.touched_count = 0

    if source == target:
        length[0] = 0.0
        meet[0] = source
        if fwd.path != NULL:
            fwd.path[source] = <cnp.uint32_t>source
        if bwd.path != NULL:
            bwd.path[target] = <cnp.uint32_t>target
        return 0

    if _search_init(fwd, source, heap_length) != 0 or \
            _search_init(bwd, target, heap_length) != 0:
        _search_free(fwd)
        _search_free(bwd)
        return -1

    while True:
        _skip_stale(fwd)
        _skip_stale(bwd)
        # an empty heap has an infinite smallest key: stop
        if fwd.size == 0 or bwd.size == 0:
            break
        if fwd.heap[0].key + bwd.heap[0].key >= mu:
            break
        if fwd.heap[0].key <= bwd.heap[0].key:
            status = _scan(fwd, bwd.dist, &mu, meet)
        else:
            status = _scan(bwd, fwd.dist, &mu, meet)
        if status != 0:
            break

    if status == 0:
        length[0] = mu

    _search_free(fwd)
    _search_free(bwd)

    return status


cdef class Workspace:
    """
    Per-vertex arrays of the bidirectional Dijkstra's algorithm, reused by
    successive runs on the same graph.

    The path lengths are infinite between two runs: a run only resets the
    vertices it has reached. A workspace must not be used by two runs at the
    same time.

    Parameters
    ----------
    vertex_count : int
        Vertex count of the graph.
    """
    cdef:
        readonly int vertex_count
        DTYPE_t[::1] dist_fwd
        DTYPE_t[::1] dist_bwd
        cnp.uint32_t[::1] touched_fwd
        cnp.uint32_t[::1] touched_bwd
        cnp.uint32_t[::1] predecessor
        cnp.uint32_t[::1] successor

    def __init__(self, int vertex_count):
        if vertex_count < 1:
            raise ValueError(f"the vertex count should be positive, got {vertex_count}")
        self.vertex_count = vertex_count
        self.dist_fwd = np.full(vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        self.dist_bwd = np.full(vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        self.touched_fwd = np.empty(vertex_count, dtype=np.uint32)
        self.touched_bwd = np.empty(vertex_count, dtype=np.uint32)
        # allocated when needed
        self.predecessor = None
        self.successor = None

    def get_path_links(self):
        """Predecessor (forward search) and successor (backward search) arrays
        of the last run with path tracking, or None. Only the vertices of the
        shortest path are meaningful: from the meeting vertex, the
        predecessors lead to the source and the successors to the target."""
        if self.predecessor is None:
            return None
        return np.asarray(self.predecessor), np.asarray(self.successor)


cpdef tuple compute_bidirectional(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        cnp.uint32_t[::1] csc_indptr,
        cnp.uint32_t[::1] csc_indices,
        DTYPE_t[::1] csc_data,
        int source_vert_idx,
        int target_vert_idx,
        int vertex_count,
        bint path_tracking=False,
        Workspace workspace=None,
        int heap_length=1024):
    """
    Compute the shortest path length from a source to a target vertex with
    the bidirectional Dijkstra's algorithm.

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format, non-negative
    csc_indptr : cnp.uint32_t[::1]
        pointers in the CSC format, same graph
    csc_indices : cnp.uint32_t[::1]
        indices in the CSC format
    csc_data : DTYPE_t[::1]
        data (edge weights) in the CSC format
    source_vert_idx : int
        source vertex index
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count
    path_tracking : bool, optional
        whether to store the predecessors and the successors in the workspace
    workspace : Workspace, optional
        per-vertex arrays, reused from one run to the next. A temporary one is
        allocated if None.
    heap_length : int, optional
        initial capacity of each heap, which grows if needed

    Returns
    -------
    (length, meet) : (float, int)
        shortest path length, DTYPE_INF if the target cannot be reached, and
        the vertex of the shortest path where the two searches meet, -1 if the
        target cannot be reached
    """
    cdef:
        Search fwd, bwd
        DTYPE_t length = DTYPE_INF
        size_t meet = <size_t>vertex_count
        int status

    # bounds checking is disabled in this module
    check_graph_arguments(
        csr_indptr, csr_indices, csr_data.shape[0], source_vert_idx, vertex_count)
    check_graph_arguments(
        csc_indptr, csc_indices, csc_data.shape[0], target_vert_idx, vertex_count)
    if heap_length < 0:
        raise ValueError(f"the heap length should not be negative, got {heap_length}")
    if workspace is None:
        workspace = Workspace(vertex_count)
    elif workspace.vertex_count != vertex_count:
        raise ValueError(
            f"the workspace is for {workspace.vertex_count} vertices, the graph has "
            f"{vertex_count}"
        )

    fwd.indptr = &csr_indptr[0]
    fwd.indices = &csr_indices[0]
    fwd.data = &csr_data[0]
    fwd.dist = &workspace.dist_fwd[0]
    fwd.touched = &workspace.touched_fwd[0]
    fwd.path = NULL
    bwd.indptr = &csc_indptr[0]
    bwd.indices = &csc_indices[0]
    bwd.data = &csc_data[0]
    bwd.dist = &workspace.dist_bwd[0]
    bwd.touched = &workspace.touched_bwd[0]
    bwd.path = NULL
    if path_tracking:
        if workspace.predecessor is None:
            workspace.predecessor = np.empty(vertex_count, dtype=np.uint32)
            workspace.successor = np.empty(vertex_count, dtype=np.uint32)
        fwd.path = &workspace.predecessor[0]
        bwd.path = &workspace.successor[0]

    with nogil:
        status = _bidirectional(
            &fwd,
            &bwd,
            <size_t>source_vert_idx,
            <size_t>target_vert_idx,
            <size_t>heap_length,
            &length,
            &meet)

    if status == -2:
        raise ValueError(
            "negative edge weights are not supported by Dijkstra's algorithm"
        )
    if status != 0:
        raise MemoryError("unable to allocate the priority queue")

    if meet >= <size_t>vertex_count:
        return length, -1
    return length, <int>meet


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
