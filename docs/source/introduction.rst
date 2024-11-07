Edsger
======

Edsger is a Python package for finding the shortest paths between nodes in directed graphs with positive edge weights. It provides an implementation of Dijkstra's algorithm, as well as functions for tracking the shortest path(s) from the source/target vertex to/from all other vertices in the graph.

The 4-ary array-based heap used in Edsger is a variant of the traditional binary heap, where each node has up to four children instead of two. This allows for a more compact representation of the heap, as well as faster insertion and extraction of elements due to the reduced number of pointer chasing operations. The heap is implemented in Cython.

