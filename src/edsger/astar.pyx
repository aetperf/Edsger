# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
A* algorithm, for the shortest path between two vertices.

The search is Dijkstra's algorithm on the reduced keys g(v) + h(v), where g(v)
is the tentative path length from the source and h(v) a lower bound of the
path length from v to the target (the heuristic). It stops as soon as the
target is extracted from the priority queue.

The priority queue is the inline 4-ary heap of `pq_4ary_inline`, without
decrease-key: a vertex is inserted again each time its path length is
improved, and the stale entries are skipped. A vertex whose path length is
improved after it has been scanned is scanned again, which is what makes the
result optimal with an admissible but inconsistent heuristic. With a
consistent heuristic, each vertex is scanned at most once.

The heuristic is either given as an array, or computed on the fly from vertex
coordinates (Euclidean or great-circle distance times a factor), only for the
vertices reached by the search.

The per-vertex arrays live in a `Workspace`, which is reused from one run to
the next: the path lengths are kept at infinity between the runs, and only the
vertices reached by a run are reset at its end. The cost of a run does not
depend on the vertex count, only on the part of the graph that is explored.

cdef classes:

- Workspace
    Per-vertex arrays reused by successive runs.

cpdef functions:

- compute_astar
    Shortest path length between two vertices, heuristic given as an array
    (or zero).
- compute_astar_coordinates
    Shortest path length between two vertices, heuristic computed from vertex
    coordinates.
"""

cimport numpy as cnp
import numpy as np
from libc.math cimport asin, cos, sin, sqrt

from edsger.commons import DTYPE_PY
from edsger.commons cimport DTYPE_INF, DTYPE_t, check_graph_arguments
from edsger.pq_4ary_inline cimport (
    HeapNode, HEAP_OFFSET, HEAP_SENTINELS, heap_alloc, heap_grow, heap_free,
    sift_down, sift_up, prefetch_t0)


# mean Earth radius in meters: the great-circle distances are in meters
EARTH_RADIUS = 6371008.8
cdef DTYPE_t EARTH_RADIUS_C = 6371008.8

cdef enum HeuristicKind:
    H_ARRAY = 0
    H_ZERO = 1
    H_EUCLIDEAN = 2
    H_HAVERSINE = 3

# names of the coordinate metrics, in the order of HeuristicKind
METRICS = ("euclidean", "haversine")


cdef struct Heuristic:
    int kind
    # H_ARRAY: value for each vertex
    const DTYPE_t* values
    # H_EUCLIDEAN, H_HAVERSINE: vertex coordinates (radians for haversine:
    # x is the longitude, y the latitude) and those of the target
    const DTYPE_t* x
    const DTYPE_t* y
    DTYPE_t x_t
    DTYPE_t y_t
    DTYPE_t cos_y_t
    # the distance is multiplied by this factor
    DTYPE_t factor


cdef inline DTYPE_t _coordinate_heuristic(
        const Heuristic* h, size_t vert) noexcept nogil:
    """Heuristic of a vertex computed from the coordinates (H_EUCLIDEAN,
    H_HAVERSINE), or zero (H_ZERO)."""
    cdef DTYPE_t dx, dy, a
    if h.kind == H_EUCLIDEAN:
        dx = h.x[vert] - h.x_t
        dy = h.y[vert] - h.y_t
        return h.factor * sqrt(dx * dx + dy * dy)
    if h.kind == H_HAVERSINE:
        dx = sin(0.5 * (h.x[vert] - h.x_t))
        dy = sin(0.5 * (h.y[vert] - h.y_t))
        a = dy * dy + cos(h.y[vert]) * h.cos_y_t * dx * dx
        if a > 1.0:
            a = 1.0
        return h.factor * 2.0 * EARTH_RADIUS_C * asin(sqrt(a))
    return 0.0


cdef class Workspace:
    """
    Per-vertex arrays of the A* algorithm, reused by successive runs on the
    same graph.

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
        DTYPE_t[::1] dist
        cnp.uint32_t[::1] touched
        DTYPE_t[::1] h_cache
        cnp.uint32_t[::1] predecessor

    def __init__(self, int vertex_count):
        if vertex_count < 1:
            raise ValueError(f"the vertex count should be positive, got {vertex_count}")
        self.vertex_count = vertex_count
        self.dist = np.full(vertex_count, DTYPE_INF, dtype=DTYPE_PY)
        self.touched = np.empty(vertex_count, dtype=np.uint32)
        # allocated when needed
        self.h_cache = None
        self.predecessor = None

    def get_predecessor(self):
        """Predecessor array of the last run with path tracking, or None. Only
        the vertices of the shortest path to the target are meaningful."""
        if self.predecessor is None:
            return None
        return np.asarray(self.predecessor)


cdef int _astar(
        const cnp.uint32_t* indptr,
        const cnp.uint32_t* indices,
        const DTYPE_t* data,
        const Heuristic* heur,
        DTYPE_t* dist,
        DTYPE_t* h_cache,
        cnp.uint32_t* touched,
        cnp.uint32_t* pred,
        size_t source,
        size_t target,
        size_t heap_length,
        DTYPE_t* length) noexcept nogil:
    """
    A* algorithm from a source to a target vertex, on a graph in CSR format.

    input
    =====
    * const cnp.uint32_t* indptr : pointers in the CSR format
    * const cnp.uint32_t* indices : indices in the CSR format
    * const DTYPE_t* data : edge weights in the CSR format
    * const Heuristic* heur : heuristic
    * DTYPE_t* dist : tentative path lengths, DTYPE_INF for each vertex on
      entry, and on exit
    * DTYPE_t* h_cache : heuristic of the reached vertices, for the
      heuristics computed on the fly (not H_ARRAY), uninitialized
    * cnp.uint32_t* touched : vertices reached by the search, uninitialized
    * cnp.uint32_t* pred : predecessor of each vertex (output), or NULL.
      Only the vertices reached by the search are written.
    * size_t source : source vertex index
    * size_t target : target vertex index
    * size_t heap_length : initial capacity of the heap, which grows if
      needed
    * DTYPE_t* length : path length from the source to the target (output),
      DTYPE_INF if the target cannot be reached

    output
    ======
    * int : 0 on success, -1 if a memory allocation failed, -2 if a negative
      edge weight was found on the way
    """
    cdef:
        const DTYPE_t* hval
        size_t size, i, idx, edge_end, tail_vert_idx, head_vert_idx
        size_t next_vert_idx, next_idx
        size_t touched_count = 0
        size_t capacity = heap_length if heap_length > 0 else 1
        DTYPE_t tail_vert_val, tail_vert_key, head_vert_val
        DTYPE_t inf = DTYPE_INF
        HeapNode* heap_base
        HeapNode* heap_base_new
        HeapNode* heap
        HeapNode last
        int status = 0

    length[0] = inf
    if heur.kind == H_ARRAY:
        hval = heur.values
    else:
        hval = h_cache

    heap_base = heap_alloc(capacity)
    if heap_base == NULL:
        return -1
    heap = heap_base + HEAP_OFFSET

    # the source vertex is inserted into the heap with key h(source)
    dist[source] = 0.0
    touched[0] = <cnp.uint32_t>source
    touched_count = 1
    if heur.kind != H_ARRAY:
        h_cache[source] = _coordinate_heuristic(heur, source)
    if pred != NULL:
        pred[source] = <cnp.uint32_t>source
    heap[0].key = hval[source]
    heap[0].vert = <cnp.uint32_t>source
    size = 1
    for i in range(HEAP_SENTINELS):
        heap[size + i].key = inf

    # main loop
    while size > 0:

        # extract the entry with min key
        tail_vert_idx = heap[0].vert
        tail_vert_key = heap[0].key
        size -= 1
        last = heap[size]
        heap[size].key = inf
        if size > 0:
            sift_down(heap, NULL, size, last.key, last.vert)

            # adjacency of the next vertex to be scanned, most likely
            next_vert_idx = heap[0].vert
            prefetch_t0(<void*>&dist[next_vert_idx])
            next_idx = indptr[next_vert_idx]
            if next_idx < indptr[next_vert_idx + 1]:
                prefetch_t0(<void*>&data[next_idx])
                prefetch_t0(<void*>&dist[indices[next_idx]])

        # skip stale entries: the key of a vertex is g + h, recomputed here
        # with the same operation as when the entry was inserted
        tail_vert_val = dist[tail_vert_idx]
        if tail_vert_key > tail_vert_val + hval[tail_vert_idx]:
            continue

        # the target is extracted: its path length is final
        if tail_vert_idx == target:
            length[0] = tail_vert_val
            break

        # loop on outgoing edges
        edge_end = indptr[tail_vert_idx + 1]
        for idx in range(indptr[tail_vert_idx], edge_end):

            head_vert_idx = indices[idx]
            head_vert_val = tail_vert_val + data[idx]

            if head_vert_val < dist[head_vert_idx]:
                if head_vert_val < tail_vert_val:
                    # negative edge weight: leave the main loop, the search
                    # could otherwise loop forever on a negative cycle
                    status = -2
                    size = 0
                    break
                if dist[head_vert_idx] == inf:
                    # first visit of the vertex
                    touched[touched_count] = <cnp.uint32_t>head_vert_idx
                    touched_count += 1
                    if heur.kind != H_ARRAY:
                        h_cache[head_vert_idx] = _coordinate_heuristic(
                            heur, head_vert_idx)
                    prefetch_t0(<void*>&indptr[head_vert_idx])
                dist[head_vert_idx] = head_vert_val
                if pred != NULL:
                    pred[head_vert_idx] = <cnp.uint32_t>tail_vert_idx

                # insert a new entry
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
                sift_up(
                    heap, NULL, i, head_vert_val + hval[head_vert_idx],
                    <cnp.uint32_t>head_vert_idx)

    # cleanup: the reached vertices are reset to an infinite path length
    for i in range(touched_count):
        dist[touched[i]] = inf
    heap_free(heap_base, capacity)

    return status


cdef DTYPE_t _run(
        cnp.uint32_t[::1] indptr,
        cnp.uint32_t[::1] indices,
        DTYPE_t[::1] data,
        Heuristic* heur,
        bint path_tracking,
        int source_vert_idx,
        int target_vert_idx,
        int vertex_count,
        Workspace workspace,
        int heap_length) except? -1.0:
    """
    Check the arguments, run the A* algorithm and return the path length.
    `heur` is filled by the caller, except for the coordinates of the target.
    """
    cdef:
        DTYPE_t length = DTYPE_INF
        cnp.uint32_t* pred_ptr = NULL
        DTYPE_t* h_cache_ptr = NULL
        int status

    # bounds checking is disabled in this module
    check_graph_arguments(
        indptr, indices, data.shape[0], source_vert_idx, vertex_count)
    if target_vert_idx < 0 or target_vert_idx >= vertex_count:
        raise IndexError(
            f"vertex index {target_vert_idx} is out of bounds for {vertex_count} "
            "vertices"
        )
    if heap_length < 0:
        raise ValueError(f"the heap length should not be negative, got {heap_length}")
    if workspace is None:
        workspace = Workspace(vertex_count)
    elif workspace.vertex_count != vertex_count:
        raise ValueError(
            f"the workspace is for {workspace.vertex_count} vertices, the graph has "
            f"{vertex_count}"
        )
    if heur.kind != H_ARRAY:
        if workspace.h_cache is None:
            workspace.h_cache = np.empty(vertex_count, dtype=DTYPE_PY)
        h_cache_ptr = &workspace.h_cache[0]
    if heur.kind == H_EUCLIDEAN or heur.kind == H_HAVERSINE:
        heur.x_t = heur.x[target_vert_idx]
        heur.y_t = heur.y[target_vert_idx]
        heur.cos_y_t = cos(heur.y_t)
    if path_tracking:
        if workspace.predecessor is None:
            workspace.predecessor = np.empty(vertex_count, dtype=np.uint32)
        pred_ptr = &workspace.predecessor[0]

    with nogil:
        status = _astar(
            &indptr[0],
            &indices[0],
            &data[0],
            heur,
            &workspace.dist[0],
            h_cache_ptr,
            &workspace.touched[0],
            pred_ptr,
            <size_t>source_vert_idx,
            <size_t>target_vert_idx,
            <size_t>heap_length,
            &length)

    if status == -2:
        raise ValueError("negative edge weights are not supported by the A* algorithm")
    if status != 0:
        raise MemoryError("unable to allocate the priority queue")

    return length


cpdef DTYPE_t compute_astar(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        int source_vert_idx,
        int target_vert_idx,
        int vertex_count,
        const DTYPE_t[::1] heuristic=None,
        bint path_tracking=False,
        Workspace workspace=None,
        int heap_length=1024) except? -1.0:
    """
    Compute the shortest path length from a source to a target vertex with
    the A* algorithm, the heuristic being given for each vertex.

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format, non-negative
    source_vert_idx : int
        source vertex index
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count
    heuristic : DTYPE_t[::1], optional
        lower bound of the path length from each vertex to the target: finite
        and non-negative. None for a zero heuristic (Dijkstra's algorithm
        stopped at the target).
    path_tracking : bool, optional
        whether to store the predecessors in the workspace
    workspace : Workspace, optional
        per-vertex arrays, reused from one run to the next. A temporary one is
        allocated if None.
    heap_length : int, optional
        initial capacity of the heap, which grows if needed

    Returns
    -------
    length : float
        shortest path length, DTYPE_INF if the target cannot be reached
    """
    cdef Heuristic heur

    heur.kind = H_ZERO
    heur.values = NULL
    heur.factor = 0.0
    if heuristic is not None:
        if heuristic.shape[0] != vertex_count:
            raise ValueError(
                f"the heuristic array has {heuristic.shape[0]} elements for "
                f"{vertex_count} vertices"
            )
        heur.kind = H_ARRAY
        heur.values = &heuristic[0]

    return _run(
        csr_indptr, csr_indices, csr_data, &heur, path_tracking,
        source_vert_idx, target_vert_idx, vertex_count, workspace, heap_length)


cpdef DTYPE_t compute_astar_coordinates(
        cnp.uint32_t[::1] csr_indptr,
        cnp.uint32_t[::1] csr_indices,
        DTYPE_t[::1] csr_data,
        int source_vert_idx,
        int target_vert_idx,
        int vertex_count,
        const DTYPE_t[::1] x,
        const DTYPE_t[::1] y,
        str metric,
        factor,
        bint path_tracking=False,
        Workspace workspace=None,
        int heap_length=1024) except? -1.0:
    """
    Compute the shortest path length from a source to a target vertex with
    the A* algorithm, the heuristic being computed from the vertex
    coordinates: `factor` times the distance between the vertex and the
    target.

    Parameters
    ----------
    csr_indptr : cnp.uint32_t[::1]
        pointers in the CSR format
    csr_indices : cnp.uint32_t[::1]
        indices in the CSR format
    csr_data : DTYPE_t[::1]
        data (edge weights) in the CSR format, non-negative
    source_vert_idx : int
        source vertex index
    target_vert_idx : int
        target vertex index
    vertex_count : int
        vertex count
    x : DTYPE_t[::1]
        first coordinate of each vertex: longitude in radians for
        'haversine'
    y : DTYPE_t[::1]
        second coordinate of each vertex: latitude in radians for
        'haversine'
    metric : str
        'euclidean' or 'haversine' (great-circle distance, in meters)
    factor : float
        the distance is multiplied by this factor, non-negative
    path_tracking : bool, optional
        whether to store the predecessors in the workspace
    workspace : Workspace, optional
        per-vertex arrays, reused from one run to the next. A temporary one is
        allocated if None.
    heap_length : int, optional
        initial capacity of the heap, which grows if needed

    Returns
    -------
    length : float
        shortest path length, DTYPE_INF if the target cannot be reached
    """
    cdef Heuristic heur

    if x is None or y is None:
        raise ValueError("the coordinates should not be None")
    if x.shape[0] != vertex_count or y.shape[0] != vertex_count:
        raise ValueError(
            f"the coordinate arrays have {x.shape[0]} and {y.shape[0]} elements for "
            f"{vertex_count} vertices"
        )
    if metric not in METRICS:
        raise ValueError(f"metric should be one of {METRICS}, got {metric!r}")
    # checked with NumPy, away from the fast-math flags of this module
    if not (np.isfinite(factor) and factor >= 0.0):
        raise ValueError(f"the factor should be finite and non-negative, got {factor}")

    heur.kind = H_EUCLIDEAN if metric == "euclidean" else H_HAVERSINE
    heur.values = NULL
    heur.x = &x[0]
    heur.y = &y[0]
    heur.factor = <DTYPE_t>factor

    return _run(
        csr_indptr, csr_indices, csr_data, &heur, path_tracking,
        source_vert_idx, target_vert_idx, vertex_count, workspace, heap_length)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
