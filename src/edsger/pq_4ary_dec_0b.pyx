"""
Priority queue based on a minimum 4-ary heap.

- dec: priority queue with decrease-key operation.
- 0b: indices are zero-based

cdef functions:

- init_pqueue
    Initialize the priority queue.
- _initialize_element
    Initialize a single element.
- free_pqueue
    Free the priority queue.
- insert
    Insert an element into the heap and reorder the heap.
- decrease_key
    Decrease the key of a element in the heap, given its element index.
- peek
    Find heap min key.
- is_empty
    Check if the heap is empty.
- extract_min
    Extract element with min keay from the heap,
    and return its element index.
- copy_keys_to_numpy
    Copy the keys into a numpy array.
- _exchange_nodes
    Exchange two nodes in the heap.
- _min_heapify
    Re-order sub-tree under a given node (given its node index)
    until it satisfies the heap property.
- _decrease_key_from_node_index
    Decrease the key of an element in the heap, given its tree index.

"""

cimport numpy as cnp
from libc.stdlib cimport free, malloc

from edsger.commons import DTYPE_PY
from edsger.commons cimport (
    DTYPE_INF, LABELED, UNLABELED, SCANNED, DTYPE_t
)


cdef void init_pqueue(
        PriorityQueue* pqueue,
        size_t heap_length,
        size_t element_count) noexcept nogil:
    """
    Initialize the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t length : length (maximum size) of the heap
    * size_t element_count : element count

    assumptions
    ===========
    * heap_length <= element_count
    """
    cdef size_t i

    pqueue.length = heap_length
    pqueue.size = 0
    pqueue.A = <size_t*> malloc(heap_length * sizeof(size_t))
    pqueue.Elements = <Element*> malloc(element_count * sizeof(Element))

    for i in range(heap_length):
        pqueue.A[i] = heap_length
        _initialize_element(pqueue, i)
    for i in range(heap_length, element_count):
        _initialize_element(pqueue, i)


cdef void _initialize_element(
        PriorityQueue* pqueue,
        size_t element_idx) noexcept nogil:
    """
    Initialize a single element.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    """
    pqueue.Elements[element_idx].key = DTYPE_INF
    pqueue.Elements[element_idx].state = UNLABELED
    pqueue.Elements[element_idx].node_idx = pqueue.length


cdef void free_pqueue(
        PriorityQueue* pqueue) noexcept nogil:
    """
    Free the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    """
    free(pqueue.A)
    free(pqueue.Elements)


cdef void insert(
        PriorityQueue* pqueue,
        size_t element_idx,
        DTYPE_t key) noexcept nogil:
    """
    Insert an element into the heap and reorder the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element pqueue.Elements[element_idx] is not in the heap
    * its new key is smaller than DTYPE_INF
    """
    cdef size_t node_idx = pqueue.size

    pqueue.size += 1
    pqueue.Elements[element_idx].state = LABELED
    pqueue.Elements[element_idx].node_idx = node_idx
    pqueue.A[node_idx] = element_idx
    _decrease_key_from_node_index(pqueue, node_idx, key)


cdef void decrease_key(
        PriorityQueue* pqueue,
        size_t element_idx,
        DTYPE_t key_new) noexcept nogil:
    """
    Decrease the key of a element in the heap, given its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key_new : new value of the element key

    assumption
    ==========
    * pqueue.Elements[idx] is in the heap
    """
    _decrease_key_from_node_index(
        pqueue,
        pqueue.Elements[element_idx].node_idx,
        key_new)


cdef DTYPE_t peek(PriorityQueue* pqueue) noexcept nogil:
    """
    Find heap min key.

    input
    =====
    * PriorityQueue* pqueue : priority queue

    output
    ======
    * DTYPE_t : key value of the min element

    assumption
    ==========
    * pqueue.size > 0
    * heap is heapified
    """
    return pqueue.Elements[pqueue.A[0]].key


cdef bint is_empty(PriorityQueue* pqueue) noexcept nogil:
    """
    Check if the heap is empty.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    """
    cdef bint isempty = 0

    if pqueue.size == 0:
        isempty = 1

    return isempty


cdef size_t extract_min(PriorityQueue* pqueue) noexcept nogil:
    """
    Extract element with min keay from the heap,
    and return its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue

    output
    ======
    * size_t : element index with min key

    assumption
    ==========
    * pqueue.size > 0
    """
    cdef:
        size_t element_idx = pqueue.A[0]  # min element index
        size_t node_idx = pqueue.size - 1  # last leaf node index

    # exchange the root node with the last leaf node
    _exchange_nodes(pqueue, 0, node_idx)

    # remove this element from the heap
    pqueue.Elements[element_idx].state = SCANNED
    pqueue.Elements[element_idx].node_idx = pqueue.length
    pqueue.A[node_idx] = pqueue.length
    pqueue.size -= 1

    # reorder the tree Elements from the root node
    _min_heapify(pqueue, 0)

    return element_idx


cdef cnp.ndarray copy_keys_to_numpy(
    PriorityQueue* pqueue,
    size_t vertex_count
):
    """
    Copy the keys into a numpy array.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * int vertex_count : vertex count
    * int num_threads : number of threads for the parallel job

    output
    ======
    * cnp.ndarray : NumPy array with all the keys
    """

    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE_PY)

    cdef:
        size_t i  # loop counter
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    return path_lengths


cdef void _exchange_nodes(
        PriorityQueue* pqueue,
        size_t node_i,
        size_t node_j) noexcept nogil:
    """
    Exchange two nodes in the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_i: first node index
    * size_t node_j: second node index
    """
    cdef:
        size_t element_i = pqueue.A[node_i]
        size_t element_j = pqueue.A[node_j]

    # exchange element indices in the heap array
    pqueue.A[node_i] = element_j
    pqueue.A[node_j] = element_i

    # exchange node indices in the element array
    pqueue.Elements[element_j].node_idx = node_i
    pqueue.Elements[element_i].node_idx = node_j


cdef void _min_heapify(
        PriorityQueue* pqueue,
        size_t node_idx) noexcept nogil:
    """
    Re-order sub-tree under a given node (given its node index)
    until it satisfies the heap property.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    """
    cdef:
        size_t c1, c2, c3, c4, i = node_idx, s
        DTYPE_t val_tmp, val_min

    while True:

        c1 = 4 * i + 1
        c2 = c1 + 1
        c3 = c2 + 1
        c4 = c3 + 1

        s = i
        val_min = pqueue.Elements[pqueue.A[s]].key
        if (c4 < pqueue.size):
            val_tmp = pqueue.Elements[pqueue.A[c4]].key
            if val_tmp < val_min:
                s = c4
                val_min = val_tmp
            val_tmp = pqueue.Elements[pqueue.A[c3]].key
            if val_tmp < val_min:
                s = c3
                val_min = val_tmp
            val_tmp = pqueue.Elements[pqueue.A[c2]].key
            if val_tmp < val_min:
                s = c2
                val_min = val_tmp
            val_tmp = pqueue.Elements[pqueue.A[c1]].key
            if val_tmp < val_min:
                s = c1
        else:
            if (c3 < pqueue.size):
                val_tmp = pqueue.Elements[pqueue.A[c3]].key
                if val_tmp < val_min:
                    s = c3
                    val_min = val_tmp
                val_tmp = pqueue.Elements[pqueue.A[c2]].key
                if val_tmp < val_min:
                    s = c2
                    val_min = val_tmp
                val_tmp = pqueue.Elements[pqueue.A[c1]].key
                if val_tmp < val_min:
                    s = c1
            else:
                if (c2 < pqueue.size):
                    val_tmp = pqueue.Elements[pqueue.A[c2]].key
                    if val_tmp < val_min:
                        s = c2
                        val_min = val_tmp
                    val_tmp = pqueue.Elements[pqueue.A[c1]].key
                    if val_tmp < val_min:
                        s = c1
                else:
                    if (c1 < pqueue.size):
                        val_tmp = pqueue.Elements[pqueue.A[c1]].key
                        if val_tmp < val_min:
                            s = c1

        if s != i:
            _exchange_nodes(pqueue, i, s)
            i = s
        else:
            break


cdef void _decrease_key_from_node_index(
        PriorityQueue* pqueue,
        size_t node_idx,
        DTYPE_t key_new) noexcept nogil:
    """
    Decrease the key of an element in the heap, given its tree index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    * DTYPE_t key_new : new key value

    assumptions
    ===========
    * pqueue.Elements[pqueue.A[node_idx]] is in the heap (node_idx < pqueue.size)
    * key_new < pqueue.Elements[pqueue.A[node_idx]].key
    """
    cdef:
        size_t i = node_idx, j
        DTYPE_t key_j

    pqueue.Elements[pqueue.A[i]].key = key_new
    while i > 0:
        j = (i - 1) // 4
        key_j = pqueue.Elements[pqueue.A[j]].key
        if key_j > key_new:
            _exchange_nodes(pqueue, i, j)
            i = j
        else:
            break


# ============================================================================ #
# tests                                                                        #
# ============================================================================ #

import numpy as np


cpdef init_01(int length=4):
    """
    Initialize an empty priority queue.
    """

    cdef:
        PriorityQueue pqueue
        size_t length_val = <size_t>length

    init_pqueue(&pqueue, length_val, length_val)

    assert pqueue.length == length_val
    assert pqueue.size == 0
    for i in range(length_val):
        assert pqueue.A[i] == pqueue.length
        assert pqueue.Elements[i].key == DTYPE_INF
        assert pqueue.Elements[i].state == UNLABELED
        assert pqueue.Elements[i].node_idx == pqueue.length

    free_pqueue(&pqueue)


cpdef insert_01():
    """
    Insert an element into an empty priority queue of length 1.
    """

    cdef:
        PriorityQueue pqueue
        DTYPE_t key

    init_pqueue(&pqueue, 1, 1)
    assert pqueue.length == 1
    key = 1.0
    insert(&pqueue, 0, key)
    assert pqueue.size == 1
    assert pqueue.A[0] == 0
    assert pqueue.Elements[0].key == key
    assert pqueue.Elements[0].state == LABELED
    assert pqueue.Elements[0].node_idx == 0

    free_pqueue(&pqueue)


cpdef insert_02():
    """
    Insert 4 elements into an empty priority queue of length 4.
    """

    cdef:
        PriorityQueue pqueue
        DTYPE_t key

    init_pqueue(&pqueue, 4, 4)

    elem_idx = 1
    key = 3.0
    insert(&pqueue, elem_idx, key)
    A_ref = [1, 4, 4, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == LABELED
    assert pqueue.Elements[1].node_idx == 0
    assert pqueue.size == 1

    elem_idx = 0
    key = 2.0
    insert(&pqueue, elem_idx, key)
    A_ref = [0, 1, 4, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == LABELED
    assert pqueue.Elements[0].node_idx == 0
    assert pqueue.Elements[1].node_idx == 1
    assert pqueue.size == 2

    elem_idx = 3
    key = 4.0
    insert(&pqueue, elem_idx, key)
    A_ref = [0, 1, 3, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == LABELED
    assert pqueue.Elements[0].node_idx == 0
    assert pqueue.Elements[1].node_idx == 1
    assert pqueue.Elements[3].node_idx == 2
    assert pqueue.size == 3

    elem_idx = 2
    key = 1.0
    insert(&pqueue, elem_idx, key)
    A_ref = [2, 1, 3, 0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[2].key == key
    assert pqueue.Elements[2].state == LABELED
    assert pqueue.Elements[0].node_idx == 3
    assert pqueue.Elements[1].node_idx == 1
    assert pqueue.Elements[2].node_idx == 0
    assert pqueue.Elements[3].node_idx == 2
    assert pqueue.size == 4

    free_pqueue(&pqueue)


cpdef insert_03(int length=4):
    """
    Insert elements with equal key values.
    """

    cdef:
        PriorityQueue pqueue
        size_t i, length_val = <size_t>length
        DTYPE_t key = 1.0

    init_pqueue(&pqueue, length_val, length_val)
    for i in range(length_val):
        insert(&pqueue, i, key)
    for i in range(length_val):
        assert pqueue.A[i] == i

    free_pqueue(&pqueue)


cpdef peek_01():
    """
    Successively insert elements into a priority queue and
    check the min element of the priority without extracting it.
    """

    cdef PriorityQueue pqueue

    init_pqueue(&pqueue, 4, 4)

    insert(&pqueue, 0, 9.0)
    assert peek(&pqueue) == 9.0
    insert(&pqueue, 1, 9.0)
    assert peek(&pqueue) == 9.0
    insert(&pqueue, 2, 5.0)
    assert peek(&pqueue) == 5.0
    insert(&pqueue, 3, 3.0)
    assert peek(&pqueue) == 3.0

    free_pqueue(&pqueue)


cpdef extract_min_01():
    """
    Insert 4 elements into a priority queue and
    extract them all.
    """

    cdef PriorityQueue pqueue

    init_pqueue(&pqueue, 4, 4)
    insert(&pqueue, 1, 3.0)
    insert(&pqueue, 0, 2.0)
    insert(&pqueue, 3, 4.0)
    insert(&pqueue, 2, 1.0)
    idx = extract_min(&pqueue)
    assert idx == 2
    assert pqueue.size == 3
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 0
    assert pqueue.size == 2
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 1
    assert pqueue.size == 1
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 3
    assert pqueue.size == 0
    assert pqueue.Elements[idx].state == SCANNED

    free_pqueue(&pqueue)


cpdef is_empty_01():
    """
    Insert an element and extract it.
    """

    cdef PriorityQueue pqueue

    init_pqueue(&pqueue, 2, 2)

    assert is_empty(&pqueue) == 1
    insert(&pqueue, 1, 3.0)
    assert is_empty(&pqueue) == 0
    _ = extract_min(&pqueue)
    assert is_empty(&pqueue) == 1

    free_pqueue(&pqueue)


cpdef decrease_key_01():
    """
    Insert elements into a priority queue and decrease the largest
    key value to become the smallest.
    """

    cdef PriorityQueue pqueue

    init_pqueue(&pqueue, 4, 4)

    insert(&pqueue, 1, 3.0)
    insert(&pqueue, 0, 2.0)
    insert(&pqueue, 3, 4.0)
    insert(&pqueue, 2, 1.0)

    assert pqueue.size == 4
    A_ref = [2, 1, 3, 0]
    n_ref = [3, 1, 0, 2]
    key_ref = [2.0, 3.0, 1.0, 4.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == LABELED
        assert pqueue.Elements[i].key == key_ref[i]

    decrease_key(&pqueue, 3, 0.0)

    assert pqueue.size == 4
    A_ref = [3, 1, 2, 0]
    n_ref = [3, 1, 2, 0]
    key_ref = [2.0, 3.0, 1.0, 0.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == LABELED
        assert pqueue.Elements[i].key == key_ref[i]

    decrease_key(&pqueue, 1, -1.0)

    assert pqueue.size == 4
    A_ref = [1, 3, 2, 0]
    n_ref = [3, 0, 2, 1]
    key_ref = [2.0, -1.0, 1.0, 0.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == LABELED
        assert pqueue.Elements[i].key == key_ref[i]

    free_pqueue(&pqueue)


cdef void heapsort(DTYPE_t[::1] values_in, DTYPE_t[::1] values_out) noexcept nogil:
    """
    Heap sort by inerting all the values into the priority queue,
    and extracting them.
    """

    cdef:
        size_t i, length_val = <size_t>values_in.shape[0]
        PriorityQueue pqueue

    init_pqueue(&pqueue, length_val, length_val)
    for i in range(length_val):
        insert(&pqueue, i, values_in[i])
    for i in range(length_val):
        values_out[i] = pqueue.Elements[extract_min(&pqueue)].key
    free_pqueue(&pqueue)


cpdef sort_01(int n, random_seed=124):
    """
    Create an array of random number, sort it with the heapsort function
    and with the numpy default sort function, compare the results.
    """

    rng = np.random.default_rng(random_seed)
    values_in = rng.random(size=n)
    values_out = np.empty_like(values_in, dtype=DTYPE_PY)
    heapsort(values_in, values_out)
    values_in_sorted = np.sort(values_in)
    np.testing.assert_array_equal(values_in_sorted, values_out)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
