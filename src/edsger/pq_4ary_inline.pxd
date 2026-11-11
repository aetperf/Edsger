"""
4-ary min-heap, inlined into the calling module.

This is the heap used by Dijkstra's algorithm, with (`dijkstra.pyx`) or
without (`dijkstra_nodec.pyx`) the decrease-key operation. It differs from
`pq_4ary_dec_0b` and `pq_4ary_nodec_0b` in its memory layout, not in its
nature: it is still an implicit 4-ary min-heap, zero-based.

Layout
------

- The heap is an array of `HeapNode` (key, vertex index). The keys live in
  the heap itself, so that comparing the children of a node only reads the
  heap array, which is small and stays in the CPU cache, and never the
  per-vertex arrays.
- With decrease-key, `pos[v]` is the heap index of vertex `v`. It is only
  meaningful while `v` is in the heap, and does not need to be initialized.
- The heap pointer is offset by `HEAP_OFFSET` nodes from the start of the
  allocation. When the allocation is cache-line aligned, which is the case of
  the large ones, the 4 children of any node share a single 64-byte cache
  line.
- Sentinels: the `HEAP_SENTINELS` nodes that follow the last node of the heap
  hold a key that is not smaller than any real key. A group of children can
  then always be read as 4 keys, without testing how many of them exist. The
  caller maintains this invariant when the heap size changes:

  - after `size -= 1` : `heap[size].key = sentinel_key`
  - after `size += 1` : `heap[size + HEAP_SENTINELS - 1].key = sentinel_key`

- An allocation of capacity `capacity` holds `capacity + HEAP_PAD` nodes. The
  caller grows the heap with `heap_grow` when `size == capacity`.

Why the child selection is branch-free
--------------------------------------

Which of the 4 children holds the smallest key is close to random, so a
compare-and-branch selection is mispredicted most of the time. The selection
below is written with arithmetic only (two comparisons in parallel, then a
final one), which the compilers turn into flag/min instructions. On the USA
road network this accounts for most of the speed-up over the previous heaps.

cdef inline functions:

- heap_alloc
    Allocate a heap.
- heap_grow
    Move a heap into a larger allocation.
- heap_free
    Free a heap.
- sift_down
    Fill the hole left at the root with a given element (with decrease-key).
- sift_up
    Place an element at a given heap index, or above (with decrease-key).
- sift_down_nodec
    Fill the hole left at the root with a given element (no decrease-key).
- sift_up_nodec
    Place an element at a given heap index, or above (no decrease-key).
"""

cimport numpy as cnp
from libc.string cimport memcpy

from edsger.commons cimport DTYPE_t

# large pages / THP memory allocation support
cdef extern from "memory_alloc.h":
    void* thp_alloc(size_t size) nogil
    void thp_free(void* ptr, size_t size) nogil

# memory prefetching support
cdef extern from "prefetch_compat.h":
    void prefetch_t0(void*) nogil


cdef struct HeapNode:
    DTYPE_t key
    cnp.uint32_t vert

cdef enum:
    HEAP_OFFSET = 3
    HEAP_SENTINELS = 3
    HEAP_PAD = 8


cdef inline HeapNode* heap_alloc(size_t capacity) noexcept nogil:
    """
    Allocate a heap.

    input
    =====
    * size_t capacity : maximum number of nodes in the heap

    output
    ======
    * HeapNode* : start of the allocation, or NULL. The heap array starts
      HEAP_OFFSET nodes further.
    """
    return <HeapNode*> thp_alloc((capacity + HEAP_PAD) * sizeof(HeapNode))


cdef inline void heap_free(HeapNode* heap_base, size_t capacity) noexcept nogil:
    """
    Free a heap.

    input
    =====
    * HeapNode* heap_base : start of the allocation
    * size_t capacity : capacity given at allocation time
    """
    thp_free(heap_base, (capacity + HEAP_PAD) * sizeof(HeapNode))


cdef inline HeapNode* heap_grow(
        HeapNode* heap_base,
        size_t capacity,
        size_t capacity_new) noexcept nogil:
    """
    Move a heap into a larger allocation, sentinels included.

    input
    =====
    * HeapNode* heap_base : start of the current allocation
    * size_t capacity : current capacity
    * size_t capacity_new : new capacity

    output
    ======
    * HeapNode* : start of the new allocation, or NULL if the allocation
      failed, in which case the current one is left untouched.

    assumption
    ==========
    * capacity_new > capacity
    """
    cdef HeapNode* heap_base_new = heap_alloc(capacity_new)

    if heap_base_new != NULL:
        memcpy(heap_base_new, heap_base, (capacity + HEAP_PAD) * sizeof(HeapNode))
        heap_free(heap_base, capacity)

    return heap_base_new


cdef inline void sift_down(
        HeapNode* heap,
        cnp.uint32_t* pos,
        size_t size,
        DTYPE_t key,
        cnp.uint32_t vert) noexcept nogil:
    """
    Fill the hole left at the root with the element (key, vert) and restore
    the heap order, keeping track of the heap index of each vertex.

    input
    =====
    * HeapNode* heap : heap array
    * cnp.uint32_t* pos : heap index of each vertex
    * size_t size : number of nodes in the heap, hole included
    * DTYPE_t key : key of the element to place
    * cnp.uint32_t vert : vertex index of the element to place

    assumptions
    ===========
    * size > 0
    * heap[size], ..., heap[size + HEAP_SENTINELS - 1] hold sentinel keys
    """
    cdef:
        size_t i = 0  # hole index
        size_t c = 1  # index of the first child of the hole
        size_t m, a, b
        HeapNode* p
        DTYPE_t k0, k1, k2, k3, ka, kb, km

    while c < size:
        p = &heap[c]

        # get the 16 grandchildren on their way (4 cache lines)
        prefetch_t0(&heap[4 * c + 1])
        prefetch_t0(&heap[4 * c + 5])
        prefetch_t0(&heap[4 * c + 9])
        prefetch_t0(&heap[4 * c + 13])

        # branch-free selection of the smallest child
        k0 = p[0].key
        k1 = p[1].key
        k2 = p[2].key
        k3 = p[3].key
        a = <size_t>(k1 < k0)
        ka = k1 if k1 < k0 else k0
        b = 2 + <size_t>(k3 < k2)
        kb = k3 if k3 < k2 else k2
        km = kb if kb < ka else ka
        m = c + (a ^ ((a ^ b) & (-<size_t>(kb < ka))))

        if km < key:
            # move the smallest child up into the hole
            heap[i] = heap[m]
            pos[heap[i].vert] = <cnp.uint32_t>i
            i = m
            c = 4 * m + 1
        else:
            break

    heap[i].key = key
    heap[i].vert = vert
    pos[vert] = <cnp.uint32_t>i


cdef inline void sift_up(
        HeapNode* heap,
        cnp.uint32_t* pos,
        size_t node_idx,
        DTYPE_t key,
        cnp.uint32_t vert) noexcept nogil:
    """
    Place the element (key, vert) at the heap index node_idx, or above, and
    restore the heap order, keeping track of the heap index of each vertex.
    This is both the insert operation (node_idx is the first free index) and
    the decrease-key operation (node_idx is the current index of the vertex).

    input
    =====
    * HeapNode* heap : heap array
    * cnp.uint32_t* pos : heap index of each vertex
    * size_t node_idx : heap index of the hole
    * DTYPE_t key : key of the element to place
    * cnp.uint32_t vert : vertex index of the element to place

    assumption
    ==========
    * key is not larger than the keys found below node_idx
    """
    cdef:
        size_t i = node_idx, j

    while i > 0:
        j = (i - 1) >> 2
        if heap[j].key > key:
            # move the parent down into the hole
            heap[i] = heap[j]
            pos[heap[i].vert] = <cnp.uint32_t>i
            i = j
        else:
            break

    heap[i].key = key
    heap[i].vert = vert
    pos[vert] = <cnp.uint32_t>i


cdef inline void sift_down_nodec(
        HeapNode* heap,
        size_t size,
        DTYPE_t key,
        cnp.uint32_t vert) noexcept nogil:
    """
    Fill the hole left at the root with the element (key, vert) and restore
    the heap order. Same as sift_down, without the heap index of the
    vertices.

    input
    =====
    * HeapNode* heap : heap array
    * size_t size : number of nodes in the heap, hole included
    * DTYPE_t key : key of the element to place
    * cnp.uint32_t vert : vertex index of the element to place

    assumptions
    ===========
    * size > 0
    * heap[size], ..., heap[size + HEAP_SENTINELS - 1] hold sentinel keys
    """
    cdef:
        size_t i = 0  # hole index
        size_t c = 1  # index of the first child of the hole
        size_t m, a, b
        HeapNode* p
        DTYPE_t k0, k1, k2, k3, ka, kb, km

    while c < size:
        p = &heap[c]

        # get the 16 grandchildren on their way (4 cache lines)
        prefetch_t0(&heap[4 * c + 1])
        prefetch_t0(&heap[4 * c + 5])
        prefetch_t0(&heap[4 * c + 9])
        prefetch_t0(&heap[4 * c + 13])

        # branch-free selection of the smallest child
        k0 = p[0].key
        k1 = p[1].key
        k2 = p[2].key
        k3 = p[3].key
        a = <size_t>(k1 < k0)
        ka = k1 if k1 < k0 else k0
        b = 2 + <size_t>(k3 < k2)
        kb = k3 if k3 < k2 else k2
        km = kb if kb < ka else ka
        m = c + (a ^ ((a ^ b) & (-<size_t>(kb < ka))))

        if km < key:
            # move the smallest child up into the hole
            heap[i] = heap[m]
            i = m
            c = 4 * m + 1
        else:
            break

    heap[i].key = key
    heap[i].vert = vert


cdef inline void sift_up_nodec(
        HeapNode* heap,
        size_t node_idx,
        DTYPE_t key,
        cnp.uint32_t vert) noexcept nogil:
    """
    Place the element (key, vert) at the heap index node_idx, or above, and
    restore the heap order. Same as sift_up, without the heap index of the
    vertices.

    input
    =====
    * HeapNode* heap : heap array
    * size_t node_idx : heap index of the hole
    * DTYPE_t key : key of the element to place
    * cnp.uint32_t vert : vertex index of the element to place
    """
    cdef:
        size_t i = node_idx, j

    while i > 0:
        j = (i - 1) >> 2
        if heap[j].key > key:
            # move the parent down into the hole
            heap[i] = heap[j]
            i = j
        else:
            break

    heap[i].key = key
    heap[i].vert = vert


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
