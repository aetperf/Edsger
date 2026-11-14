# API reference

The four algorithms are classes of the `edsger.path` module. This page is generated from the docstrings.

## Dijkstra

```{eval-rst}
.. autoclass:: edsger.path.Dijkstra

   .. automethod:: run
   .. automethod:: run_many
   .. automethod:: get_path
   .. automethod:: get_vertices
   .. autoproperty:: n_vertices
   .. autoproperty:: n_edges
   .. autoproperty:: edges
   .. autoproperty:: orientation
   .. autoproperty:: permute
   .. autoproperty:: lazy
   .. autoproperty:: path_links
```

## BellmanFord

```{eval-rst}
.. autoclass:: edsger.path.BellmanFord

   .. automethod:: run
   .. automethod:: get_path
   .. automethod:: get_vertices
   .. automethod:: has_negative_cycle
   .. autoproperty:: n_vertices
   .. autoproperty:: n_edges
   .. autoproperty:: edges
   .. autoproperty:: orientation
   .. autoproperty:: permute
   .. autoproperty:: path_links
```

## BFS

```{eval-rst}
.. autoclass:: edsger.path.BFS

   .. automethod:: run
   .. automethod:: get_path
   .. automethod:: get_vertices
   .. autoproperty:: UNREACHABLE
   .. autoproperty:: n_vertices
   .. autoproperty:: n_edges
   .. autoproperty:: edges
   .. autoproperty:: orientation
   .. autoproperty:: permute
   .. autoproperty:: path_links
```

## HyperpathGenerating

```{eval-rst}
.. autoclass:: edsger.path.HyperpathGenerating

   .. automethod:: run
```

## Graph validation

The classes above build the graph arrays themselves. The Cython functions of the library can also be called directly on arrays in CSR or CSC format, in which case the arrays are read without bounds checking. `validate_graph` checks such arrays once, before any number of runs.

```{eval-rst}
.. autofunction:: edsger.commons.validate_graph
```

## Test networks

```{eval-rst}
.. autofunction:: edsger.networks.create_sf_network

.. autoclass:: edsger.networks.SiouxFalls

   .. autoproperty:: edges
```
