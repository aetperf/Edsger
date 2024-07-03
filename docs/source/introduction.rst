Introduction
============

Edsger is a Python package for finding the shortest paths between nodes in directed graphs with positive edge weights. It provides an implementation of Dijkstra's algorithm, as well as functions for tracking the shortest path(s) from the source/target vertex to/from all other vertices in the graph.

The main class in the package is `Dijkstra`, which takes a DataFrame containing the edges of the graph and allows the user to specify the names of the columns that contain the IDs of the edge starting and ending nodes, as well as the (positive) weights of the edges. The user can also choose the orientation of Dijkstra's algorithm (single source shortest paths or single target shortest path) and whether to check if the edges DataFrame is well-formed and/or permute the IDs of the nodes to have contiguous node indices (without large gaps).

The `Dijkstra` class has a `run` method that computes the shortest path length and returns the results as a 1D Numpy array or a Pandas Series object indexed by vertex indices. The user can also choose whether to track the shortest path(s) with predecessors/successors, whether to return path length(s) as infinity (np.inf) when no path exists or a very large number, and the heap length as a fraction of the number of vertices.

The 4-ary array-based heap used in Edsger is a variant of the traditional binary heap, where each node has up to four children instead of two. This allows for a more compact representation of the heap, as well as faster insertion and extraction of elements due to the reduced number of pointer chasing operations. The heap is implemented in Cython.