"""
Minimum spanning tree (forest) of a graph seen as undirected.

classes:

- MinimumSpanningTree
    Kruskal's algorithm, with a union-find structure.
"""

from typing import Any

import numpy as np
import pandas as pd

from edsger.graph_importer import standardize_graph_dataframe
from edsger.kruskal import kruskal  # pylint: disable=no-name-in-module
from edsger.path import OUTPUT_TYPES, _Graph, _import_polars

# numpy output of MinimumSpanningTree.run: (tails, heads, weights)
ForestArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


class MinimumSpanningTree(_Graph):
    """Minimum spanning tree of a graph seen as undirected, with Kruskal's
    algorithm.

    The edge (u, v) joins the vertices u and v whatever its direction. If the
    graph is not connected, the result is a minimum spanning forest: one
    minimum spanning tree per connected component.

    Parameters
    ----------
    edges : pandas.DataFrame or polars.DataFrame
        The edges of the graph, one row per edge. pandas dataframes can use NumPy
        or Arrow data types.
    tail : str, optional (default='tail')
        Name of the column with the tail vertex of each edge. The vertex
        identifiers are integers, non-negative unless `permute` is True.
    head : str, optional (default='head')
        Name of the column with the head vertex of each edge.
    weight : str or None, optional (default='weight')
        Name of the column with the edge weights. The weights must be finite,
        they can be negative. With None, all the edges have a weight of 1: the
        result is a spanning forest with no particular property.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and non-finite values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Negative identifiers are only accepted with this
        option. The results use the original identifiers in both cases.
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    Parallel and antiparallel edges: the graph is undirected, so the edges
    (u, v) and (v, u) join the same pair of vertices. Of all the edges joining
    a pair of vertices, only one with the minimum weight can be in the forest.
    It is returned with its tail and head as given. Self-loops are ignored.

    Ties: among edges of equal weight, the one whose (tail, head) comes first
    in the order of the internal vertex indices is considered first, so the
    result is deterministic. Several minimum spanning forests may exist; they
    all have the same total weight.

    The vertices are the tails and heads of the edges: with `permute=False`,
    the integers between 0 and the largest identifier that are not used by
    any edge are not vertices, and do not count as trees.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.spanning_tree import MinimumSpanningTree
    >>> edges = pd.DataFrame({
    ...     "tail": [0, 1, 0, 2],
    ...     "head": [1, 2, 2, 3],
    ...     "weight": [1.0, 2.0, 4.0, 3.0],
    ... })
    >>> mst = MinimumSpanningTree(edges)
    >>> mst.run()
       tail  head  weight
    0     0     1     1.0
    1     1     2     2.0
    2     2     3     3.0
    >>> mst.total_weight, mst.n_trees
    (6.0, 1)
    """

    _algorithm_name = "minimum spanning tree algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: str | None = "weight",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        edges = standardize_graph_dataframe(edges, tail, head, weight)
        if (
            weight is not None
            and len(edges) > 0
            and not np.isfinite(np.asarray(edges[weight], dtype=np.float64)).all()
        ):
            raise ValueError(f"edges['{weight}'] should be finite")
        self._weight_name = weight
        self._build_graph(edges, tail, head, weight, "out", permute, verbose)
        self._total_weight: float | None = None
        self._n_trees: int | None = None

    @property
    def total_weight(self) -> float | None:
        """Total weight of the forest computed by the last run, None before."""
        return self._total_weight

    @property
    def n_trees(self) -> int | None:
        """Number of trees of the forest computed by the last run, that is the
        number of connected components of the undirected graph, isolated
        vertices excluded since they are not vertices. None before the first
        run."""
        return self._n_trees

    def run(self, output: str = "pandas") -> pd.DataFrame | ForestArrays | Any:
        """Compute a minimum spanning forest.

        Parameters
        ----------
        output : {'pandas', 'polars', 'numpy'}, optional (default='pandas')
            Type of the result.

        Returns
        -------
        pandas.DataFrame, polars.DataFrame or tuple of numpy.ndarray
            The edges of the forest by increasing weight, with the original
            vertex identifiers: a dataframe with the tail, head and weight
            columns (named as in the input, no weight column if `weight` is
            None), or with output='numpy' the tuple of arrays
            (tails, heads, weights).

        Notes
        -----
        The total weight and the number of trees are then available as the
        `total_weight` and `n_trees` attributes.
        """
        if output not in OUTPUT_TYPES:
            raise ValueError(
                f"argument 'output' should be one of {OUTPUT_TYPES}, got {output!r}"
            )
        tails = np.asarray(self._edges[self._tail_name])
        heads = np.asarray(self._edges[self._head_name])
        if self._weight_name is None:
            weights = np.ones(len(tails), dtype=np.float64)
            order = np.arange(len(tails), dtype=np.int64)
        else:
            weights = np.asarray(self._edges[self._weight_name], dtype=np.float64)
            order = np.argsort(weights, kind="stable").astype(np.int64, copy=False)

        if self._n_vertices > 0:
            selected = kruskal(
                tails.astype(np.uint32),
                heads.astype(np.uint32),
                order,
                self._n_vertices,
            )
        else:
            selected = np.empty(0, dtype=np.int64)

        forest_weights = weights[selected]
        self._total_weight = float(forest_weights.sum())
        self._n_trees = self._vertex_count(tails, heads) - len(selected)

        forest_tails = self._labels_of(tails[selected])
        forest_heads = self._labels_of(heads[selected])
        if output == "numpy":
            return forest_tails, forest_heads, forest_weights
        data = {self._tail_name: forest_tails, self._head_name: forest_heads}
        if self._weight_name is not None:
            data[self._weight_name] = forest_weights
        if output == "pandas":
            return pd.DataFrame(data)
        return _import_polars().DataFrame(data)

    def _vertex_count(self, tails: np.ndarray, heads: np.ndarray) -> int:
        """Number of vertices that are the tail or the head of an edge."""
        if self._labels is not None:
            return len(self._labels)
        is_vertex = np.zeros(self._n_vertices, dtype=bool)
        is_vertex[tails] = True
        is_vertex[heads] = True
        return int(np.count_nonzero(is_vertex))
