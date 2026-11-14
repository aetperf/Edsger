"""
Connected components of a directed graph.
"""

from typing import Optional

import numpy as np
import pandas as pd

from edsger.graph_importer import standardize_graph_dataframe
from edsger.path import VertexResult, _resolve_output, _Graph
from edsger.components_core import (  # pylint: disable=no-name-in-module
    strong_components_csr,
    weak_components_csr,
)

CONNECTION_TYPES = ("weak", "strong")


class ConnectedComponents(_Graph):
    """Weakly or strongly connected components of a directed graph.

    The graph is built once from a dataframe of edges. The `run` method then
    gives the component of each vertex, for either kind of connection.

    Two vertices are in the same weakly connected component if they are
    connected when the edge directions are ignored, and in the same strongly
    connected component if there is a path from each one to the other.

    Parameters
    ----------
    edges : pandas.DataFrame or polars.DataFrame
        The edges of the graph, one row per edge. pandas dataframes can use NumPy
        or Arrow data types.
        A weight column, if there is one, is ignored.
    tail : str, optional (default='tail')
        Name of the column with the tail (start) vertex of each edge. The vertex
        identifiers are integers, non-negative unless `permute` is True.
    head : str, optional (default='head')
        Name of the column with the head (end) vertex of each edge.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types and missing values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Without it, the internal arrays have one element per
        integer between 0 and the largest vertex identifier. The results are the
        same in both cases.
        Negative identifiers are only accepted with this option, and the results
        must then be labelled by vertex (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    The vertices of the graph are the tails and heads of the edges. The
    components are numbered from 0 in increasing order of their smallest vertex:
    the smallest vertex is in component 0, the smallest vertex that is not in
    component 0 is in component 1, and so on. The labels do not depend on the
    order of the edges or on `permute`.

    The weakly connected components are computed with a union-find structure,
    the strongly connected components with Pearce's variant of Tarjan's
    algorithm, iterative: both run in linear time, and are not limited by the
    depth of the call stack.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.components import ConnectedComponents
    >>> edges = pd.DataFrame({"tail": [0, 1, 2, 3], "head": [1, 0, 1, 4]})
    >>> cc = ConnectedComponents(edges)
    >>> cc.run(connection="weak")
    array([0, 0, 0, 1, 1], dtype=int32)
    >>> cc.run(connection="strong")
    array([0, 0, 1, 2, 3], dtype=int32)
    >>> cc.n_components
    4
    """

    _algorithm_name = "connected components algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        if check_edges:
            self._check_edges(edges, tail, head)
        # Convert to standardized NumPy-backed pandas DataFrame, without weights
        edges = standardize_graph_dataframe(edges, tail, head)
        self._build_graph(edges, tail, head, None, "out", permute, verbose)
        self._n_components: Optional[int] = None

    @property
    def n_components(self) -> Optional[int]:
        """Number of components found by the last run, or None before the first
        run."""
        return self._n_components

    def run(
        self,
        connection: str = "weak",
        return_series: bool = False,
        output: Optional[str] = None,
    ) -> VertexResult:
        """Compute the connected component of each vertex.

        Parameters
        ----------
        connection : {'weak', 'strong'}, optional (default='weak')
            Kind of connection: 'weak' ignores the edge directions, 'strong'
            requires a path in each direction.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'component'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'component'.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The component label of each vertex, as int32, numbered from 0 in
            increasing order of the smallest vertex of each component. The array
            has one element per integer between 0 and the largest vertex
            identifier: the integers that are not vertices get -1. The Series and
            the DataFrame have one row per vertex, with the same -1 values when
            `permute` is False. The number of components is then given by
            `n_components`.

        Raises
        ------
        ValueError
            If `connection` is not 'weak' or 'strong', or if the graph has
            negative vertex identifiers and the output is 'numpy'.
        """
        if connection not in CONNECTION_TYPES:
            raise ValueError(
                f"argument 'connection' should be one of {CONNECTION_TYPES}, "
                f"got {connection!r}"
            )
        output = _resolve_output(output, return_series)
        self._check_output(output)

        if self._n_vertices == 0:
            labels = np.empty(0, dtype=np.int32)
            n_components = 0
        else:
            compute = (
                weak_components_csr if connection == "weak" else strong_components_csr
            )
            labels_u32, n_components = compute(
                self._indptr, self._indices, self._n_vertices
            )
            # the labels are smaller than the vertex count, which is an int32
            labels = labels_u32.view(np.int32)
            if self._labels is None:
                labels, n_components = self._drop_non_vertices(labels, n_components)

        self._n_components = n_components
        return self._vertex_result(labels, "component", -1, output)

    def _drop_non_vertices(self, labels: np.ndarray, n_components: int):
        """Without `permute`, the integers between 0 and the largest vertex
        identifier that are not vertices are isolated in the internal graph, with
        a component each. They get the label -1, and the components are
        renumbered without them."""
        is_vertex = np.zeros(self._n_vertices, dtype=bool)
        is_vertex[self._indices] = True
        is_vertex[np.diff(self._indptr) > 0] = True
        if is_vertex.all():
            return labels, n_components
        kept = np.zeros(n_components, dtype=bool)
        kept[labels[is_vertex]] = True
        new_labels = (np.cumsum(kept) - 1).astype(np.int32)
        labels = np.where(is_vertex, new_labels[labels], np.int32(-1))
        return labels.astype(np.int32, copy=False), int(kept.sum())


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
