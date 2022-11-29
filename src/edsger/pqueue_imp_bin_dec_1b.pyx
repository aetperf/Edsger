""" Priority queue based on a minimum binary heap.

    - Priority queue with decrease-key operation.
    - Binary heap is implicit, implemented with a static array.
    - Indices are 0-based.
    
author : François Pacull
copyright : Architecture & Performance
license : MIT
"""

# cython: boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc

# data type for the key value
ctypedef cnp.float64_t DTYPE_t
cdef DTYPE_t DTYPE_INF = <DTYPE_t>np.finfo(dtype=np.float64).max

cdef enum ElementState:
   SCANNED     = 1     # popped from the heap
   NOT_IN_HEAP = 2     # never been in the heap
   IN_HEAP     = 3     # in the heap

cdef struct Element:
    ElementState state # element state wrt the heap
    ssize_t node_idx   # index of the corresponding node in the tree
    DTYPE_t key        # key value

cdef struct PriorityQueue:
    ssize_t  length    # maximum heap size
    ssize_t  size      # number of elements in the heap
    ssize_t* A         # array storing the binary tree
    Element* Elements  # array storing the elements

cdef void init_pqueue(
    PriorityQueue* pqueue,
    ssize_t length) nogil:
    """Initialize the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t length : length (maximum size) of the heap
    """
    cdef ssize_t i

    pqueue.length = length
    pqueue.size = 0
    pqueue.A = <ssize_t*> malloc(length * sizeof(ssize_t))
    pqueue.Elements = <Element*> malloc(length * sizeof(Element))

    for i in range(length):
        pqueue.A[i] = length
        _initialize_element(pqueue, i)

cdef inline void _initialize_element(
    PriorityQueue* pqueue,
    ssize_t element_idx) nogil:
    """Initialize a single element.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t element_idx : index of the element in the element array
    """
    pqueue.Elements[element_idx].key = DTYPE_INF
    pqueue.Elements[element_idx].state = NOT_IN_HEAP
    pqueue.Elements[element_idx].node_idx = pqueue.length

cdef void free_pqueue(
    PriorityQueue* pqueue) nogil:
    """Free the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    """
    free(pqueue.A)
    free(pqueue.Elements)

cdef void insert(
    PriorityQueue* pqueue,
    ssize_t element_idx,
    DTYPE_t key) nogil:
    """Insert an element into the priority queue and reorder the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element pqueue.Elements[element_idx] is not in the heap
    * its new key is smaller than DTYPE_INF
    """
    cdef ssize_t node_idx = pqueue.size

    pqueue.size += 1
    pqueue.Elements[element_idx].state = IN_HEAP
    pqueue.Elements[element_idx].node_idx = node_idx
    pqueue.A[node_idx] = element_idx
    _decrease_key_from_node_index(pqueue, node_idx, key)

cdef void decrease_key(
    PriorityQueue* pqueue,
    ssize_t element_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of a element in the priority queue, 
    given its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t element_idx : index of the element in the element array
    * DTYPE_t key_new : new value of the element key 

    assumption
    ==========
    * pqueue.Elements[idx] is in the heap
    """
    _decrease_key_from_node_index(
        pqueue, 
        pqueue.Elements[element_idx].node_idx, 
        key_new)

cdef ssize_t extract_min(PriorityQueue* pqueue) nogil:
    """Extract element with min key from the priority queue, 
    and return its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue

    output
    ======
    * ssize_t : element index with min key

    assumption
    ==========
    * pqueue.size > 0
    """
    cdef: 
        ssize_t element_idx = pqueue.A[0]  # min element index
        ssize_t node_idx = pqueue.size - 1  # last leaf node index

    # exchange the root node with the last leaf node
    _exchange_nodes(pqueue, 0, node_idx)

    # remove this element from the heap
    pqueue.Elements[element_idx].state = SCANNED
    pqueue.Elements[element_idx].node_idx = pqueue.length
    pqueue.A[node_idx] = pqueue.length
    pqueue.size -= 1

    # reorder the tree elements from the root node
    _min_heapify(pqueue, 0)

    return element_idx

cdef inline void _exchange_nodes(
    PriorityQueue* pqueue, 
    ssize_t node_i,
    ssize_t node_j) nogil:
    """Exchange two nodes in the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t node_i: first node index
    * ssize_t node_j: second node index
    """
    cdef: 
        ssize_t element_i = pqueue.A[node_i]
        ssize_t element_j = pqueue.A[node_j]
    
    # exchange element indices in the heap array
    pqueue.A[node_i] = element_j
    pqueue.A[node_j] = element_i

    # exchange node indices in the element array
    pqueue.Elements[element_j].node_idx = node_i
    pqueue.Elements[element_i].node_idx = node_j

    
cdef inline void _min_heapify(
    PriorityQueue* pqueue,
    ssize_t node_idx) nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t node_idx : node index
    """
    cdef: 
        ssize_t l, r, i = node_idx, s

    while True:

        l =  2 * i + 1  
        r = l + 1
        
        if (
            (l < pqueue.size) and 
            (pqueue.Elements[pqueue.A[l]].key < pqueue.Elements[pqueue.A[i]].key)
        ):
            s = l
        else:
            s = i

        if (
            (r < pqueue.size) and 
            (pqueue.Elements[pqueue.A[r]].key < pqueue.Elements[pqueue.A[s]].key)
        ):
            s = r

        if s != i:
            _exchange_nodes(pqueue, i, s)
            i = s
        else:
            break
    
cdef inline void _decrease_key_from_node_index(
    PriorityQueue* pqueue,
    ssize_t node_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of an element in the priority queue, given its tree index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * ssize_t node_idx : node index
    * DTYPE_t key_new : new key value

    assumptions
    ===========
    * pqueue.elements[pqueue.A[node_idx]] is in the heap (node_idx < pqueue.size)
    * key_new < pqueue.elements[pqueue.A[node_idx]].key
    """
    cdef:
        ssize_t i = node_idx, j
        DTYPE_t key_j

    pqueue.Elements[pqueue.A[i]].key = key_new
    while i > 0: 
        j = (i - 1) // 2  
        key_j = pqueue.Elements[pqueue.A[j]].key
        if key_j > key_new:
            _exchange_nodes(pqueue, i, j)
            i = j
        else:
            break
