"""
Centrality measures of the vertices of a directed graph.

classes:

- Centrality
    PageRank, HITS hubs and authorities, Katz centrality, by power iteration.
- ConvergenceError
    Raised when the power iterations do not converge.
"""

import math
import numbers
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from edsger.graph_importer import standardize_graph_dataframe
from edsger.path import VertexResult, _resolve_output, _VertexGraph
from edsger.power_iteration import (  # pylint: disable=no-name-in-module
    hits_csr_csc,
    katz_csc,
    pagerank_csc,
)


class ConvergenceError(RuntimeError):
    """The power iterations did not converge within `max_iter` iterations."""


def _check_iteration_arguments(tol: float, max_iter: int) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, numbers.Integral):
        raise TypeError(f"max_iter should be an integer, got {max_iter!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter should be at least 1, got {max_iter}")
    if not isinstance(tol, numbers.Real) or not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol should be a positive number, got {tol!r}")


def _check_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} should be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} should be finite, got {value!r}")
    return float(value)


class Centrality(_VertexGraph):
    """Centrality measures of the vertices of a directed graph: PageRank, HITS
    hubs and authorities, and Katz centrality.

    The graph is built once from a dataframe of edges, then each measure is
    computed by power iteration, with the methods `pagerank`, `hits` and
    `katz`. The results follow the conventions of NetworkX (see the Notes).

    Parameters
    ----------
    edges : pandas.DataFrame or polars.DataFrame
        The edges of the graph, one row per edge. pandas dataframes can use NumPy
        or Arrow data types.
    tail : str, optional (default='tail')
        Name of the column with the tail (start) vertex of each edge. The vertex
        identifiers are integers, non-negative unless `permute` is True.
    head : str, optional (default='head')
        Name of the column with the head (end) vertex of each edge.
    weight : str or None, optional (default=None)
        Name of the column with the edge weights, which must be non-negative
        and finite. With None, the graph is unweighted: all the edges have a
        weight of 1.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types and missing values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Negative identifiers are only accepted with this
        option, and the results must then be labelled by vertex
        (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    When several edges have the same tail and head vertices, only one is kept,
    with the minimum weight (as for the other algorithms of edsger, whereas a
    NetworkX MultiDiGraph would add up their weights).

    The vertices are the tails and heads of the edges. With `permute=False`,
    the integers between 0 and the largest identifier that are not used by any
    edge are not vertices: they do not take part in the computations, and get
    a value of 0 in the results.

    Each method raises `ConvergenceError` if the power iterations do not
    converge within `max_iter` iterations, or diverge.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.centrality import Centrality
    >>> edges = pd.DataFrame({"tail": [0, 1, 2, 2], "head": [1, 2, 0, 1]})
    >>> c = Centrality(edges)
    >>> c.pagerank().round(3)
    array([0.215, 0.397, 0.388])
    """

    _algorithm_name = "centrality algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: Optional[str] = None,
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        if check_edges:
            self._check_edges(edges, tail, head, weight, nonnegative=True)
        edges = standardize_graph_dataframe(edges, tail, head, weight)
        if weight is not None and len(edges) > 0:
            weights = np.asarray(edges[weight], dtype=np.float64)
            # rejects the NaN weights too
            if not weights.min() >= 0.0:
                raise ValueError(f"edges['{weight}'] should be nonnegative")
            if not np.isfinite(weights.max()):
                raise ValueError(f"edges['{weight}'] should be finite")
        # CSC format: the iterations pull the values of the predecessors
        self._build_graph(edges, tail, head, weight, "in", permute, verbose)
        self._weighted = weight is not None

        n = self._n_vertices
        # predecessor (tail) of each edge, in CSC order
        preds = self._indices
        if self._labels is not None:
            self._is_vertex = np.ones(n, dtype=bool)
        else:
            self._is_vertex = np.diff(self._indptr) > 0
            self._is_vertex[preds] = True
        self._n_actual = int(np.count_nonzero(self._is_vertex))
        self._data = (
            self._edge_weights
            if self._edge_weights is not None
            else np.empty(0, dtype=np.float64)
        )
        out_weight = np.bincount(
            preds, weights=self._edge_weights if self._weighted else None, minlength=n
        ).astype(np.float64)
        self._inv_out_weight = np.zeros(n, dtype=np.float64)
        positive = out_weight > 0.0
        self._inv_out_weight[positive] = 1.0 / out_weight[positive]
        # CSR format, for HITS only: built on first use
        self._csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------ helpers

    def _uniform(self) -> np.ndarray:
        """Uniform distribution over the vertices."""
        return self._is_vertex / float(self._n_actual)

    def _vertex_vector(self, values: Any, name: str) -> np.ndarray:
        """Dense vector over the internal vertices from a mapping vertex ->
        value (dict or pandas Series). Missing vertices get 0."""
        if isinstance(values, pd.Series):
            keys = values.index.to_numpy()
            vals = values.to_numpy()
        elif isinstance(values, Mapping):
            keys = np.array(list(values.keys()))
            vals = np.array(list(values.values()))
        else:
            raise TypeError(
                f"{name} should be a dict or a pandas Series mapping vertices to "
                f"values, got {type(values).__name__}"
            )
        vector = np.zeros(self._n_vertices, dtype=np.float64)
        if len(keys) == 0:
            return vector
        if not np.issubdtype(keys.dtype, np.integer):
            raise TypeError(f"the vertices of {name} should be integers")
        if not np.issubdtype(vals.dtype, np.number) or np.issubdtype(
            vals.dtype, np.complexfloating
        ):
            raise TypeError(f"the values of {name} should be real numbers")
        vals = vals.astype(np.float64)
        if not np.isfinite(vals).all() or (vals < 0.0).any():
            raise ValueError(f"the values of {name} should be nonnegative and finite")
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"the vertices of {name} should be unique")
        internal = self._internal_vertices(keys, f"{name} vertex", f"{name} vertices")
        vector[internal] = vals
        return vector

    def _get_csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The graph in CSR format, from the CSC format."""
        if self._csr is None:
            n = self._n_vertices
            heads = np.repeat(
                np.arange(n, dtype=np.uint32), np.diff(self._indptr).astype(np.int64)
            )
            order = np.argsort(self._indices, kind="stable")
            indptr = np.zeros(n + 1, dtype=np.uint32)
            np.cumsum(np.bincount(self._indices, minlength=n), out=indptr[1:])
            data = self._data[order] if self._weighted else self._data
            self._csr = (indptr, heads[order], data)
        return self._csr

    def _result(
        self, values: np.ndarray, name: str, output: Optional[str], return_series: bool
    ) -> VertexResult:
        output = _resolve_output(output, return_series)
        self._check_output(output)
        return self._vertex_result(values, name, 0.0, output)

    def _check_result(self, values: np.ndarray, n_iter: int, message: str) -> None:
        if n_iter < 0 or not np.isfinite(values).all():
            raise ConvergenceError(message)

    # ------------------------------------------------------------ measures

    def pagerank(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        damping: float = 0.85,
        personalization: Optional[Mapping] = None,
        tol: float = 1.0e-6,
        max_iter: int = 100,
        output: Optional[str] = None,
        return_series: bool = False,
    ) -> VertexResult:
        """PageRank of the vertices.

        The PageRank is the stationary distribution of a random walk that
        follows an outgoing edge with probability `damping`, chosen in
        proportion to the edge weights, and otherwise jumps to a vertex drawn
        from the personalization distribution (uniform by default). From a
        dangling vertex, without outgoing edge (or with zero out-weight), the
        walk always jumps to a vertex drawn from the personalization
        distribution.

        This is `networkx.pagerank` with `alpha=damping` and `dangling=None`.

        Parameters
        ----------
        damping : float, optional (default=0.85)
            Damping factor, in [0, 1] (`alpha` in NetworkX).
        personalization : dict or pandas.Series, optional (default=None)
            Teleportation weights of the vertices, as a mapping from vertex to a
            non-negative weight, normalized to sum 1. The vertices that are
            not given get 0. Uniform over the vertices if None.
        tol : float, optional (default=1e-6)
            The iterations stop when the L1 norm of the change of the PageRank
            vector is smaller than `n * tol`, n being the number of vertices.
        max_iter : int, optional (default=100)
            Maximum number of iterations.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'pagerank'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'pagerank'.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The PageRank of each vertex, summing to 1. The array is indexed by
            vertex identifier.

        Raises
        ------
        ConvergenceError
            If the iterations do not converge within `max_iter` iterations.
        """
        damping = _check_real(damping, "damping")
        if not 0.0 <= damping <= 1.0:
            raise ValueError(f"damping should be in [0, 1], got {damping}")
        _check_iteration_arguments(tol, max_iter)
        if self._n_vertices == 0:
            return self._result(np.empty(0), "pagerank", output, return_series)

        if personalization is None:
            teleport = self._uniform()
        else:
            teleport = self._vertex_vector(personalization, "personalization")
            total = teleport.sum()
            if not total > 0.0:
                raise ValueError("the personalization values should not all be 0")
            teleport /= total

        values, n_iter = pagerank_csc(
            self._indptr,
            self._indices,
            self._data,
            self._weighted,
            self._inv_out_weight,
            teleport,
            teleport,
            self._uniform(),
            damping,
            self._n_actual * float(tol),
            int(max_iter),
        )
        self._check_result(
            values,
            n_iter,
            f"PageRank did not converge within {max_iter} iterations (tol={tol}): "
            "increase max_iter or tol",
        )
        return self._result(values, "pagerank", output, return_series)

    def hits(
        self,
        tol: float = 1.0e-8,
        max_iter: int = 100,
        output: Optional[str] = None,
        return_series: bool = False,
    ) -> Tuple[VertexResult, VertexResult]:
        """HITS hub and authority scores of the vertices.

        The authority of a vertex is the weighted sum of the hub scores of its
        predecessors, the hub score of a vertex is the weighted sum of the
        authorities of its successors: these are the principal left and right
        singular vectors of the adjacency matrix. They are computed by power
        iteration, starting from uniform hub scores.

        This is `networkx.hits` with `normalized=True`.

        Parameters
        ----------
        tol : float, optional (default=1e-8)
            The iterations stop when the L1 norm of the change of the hub
            vector, scaled so that its largest element is 1, is smaller than
            `tol`.
        max_iter : int, optional (default=100)
            Maximum number of iterations.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the results. 'pandas' returns Series named 'hub' and
            'authority' indexed by vertex ('vertex_idx'), 'polars' DataFrames
            with the columns 'vertex_idx' and 'hub' or 'authority'.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.

        Returns
        -------
        tuple
            The hub scores and the authority scores, each summing to 1. The
            arrays are indexed by vertex identifier.

        Raises
        ------
        ConvergenceError
            If the iterations do not converge within `max_iter` iterations.

        Notes
        -----
        If the largest singular value of the adjacency matrix is not simple,
        for example with several connected components having the same one,
        the result depends on the starting vector.
        """
        _check_iteration_arguments(tol, max_iter)
        if self._n_vertices == 0:
            empty = np.empty(0)
            return (
                self._result(empty, "hub", output, return_series),
                self._result(empty, "authority", output, return_series),
            )
        if self._weighted and not self._data.max() > 0.0:
            raise ValueError("HITS is not defined when all the edge weights are 0")

        csr_indptr, csr_indices, csr_data = self._get_csr()
        hubs, authorities, n_iter = hits_csr_csc(
            csr_indptr,
            csr_indices,
            csr_data,
            self._indptr,
            self._indices,
            self._data,
            self._weighted,
            self._uniform(),
            float(tol),
            int(max_iter),
        )
        self._check_result(
            hubs,
            n_iter,
            f"HITS did not converge within {max_iter} iterations (tol={tol}): "
            "increase max_iter or tol",
        )
        hubs /= hubs.sum()
        authorities /= authorities.sum()
        return (
            self._result(hubs, "hub", output, return_series),
            self._result(authorities, "authority", output, return_series),
        )

    def katz(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        tol: float = 1.0e-6,
        max_iter: int = 1000,
        normalized: bool = True,
        output: Optional[str] = None,
        return_series: bool = False,
    ) -> VertexResult:
        """Katz centrality of the vertices.

        The Katz centrality x is the solution of x = alpha A^T x + beta, where
        A is the (weighted) adjacency matrix: the centrality of a vertex is
        beta plus alpha times the weighted sum of the centralities of its
        predecessors. It is computed by power iteration starting from 0, which
        converges if alpha is smaller than the inverse of the spectral radius
        of A. Checking this is left to the user: a too large alpha is detected
        as non-convergence.

        This is `networkx.katz_centrality` with a scalar beta.

        Parameters
        ----------
        alpha : float, optional (default=0.1)
            Attenuation factor.
        beta : float, optional (default=1.0)
            Constant term, the same for all the vertices.
        tol : float, optional (default=1e-6)
            The iterations stop when the L1 norm of the change of the
            centrality vector is smaller than `n * tol`, n being the number of
            vertices.
        max_iter : int, optional (default=1000)
            Maximum number of iterations.
        normalized : bool, optional (default=True)
            Whether to divide the result by its Euclidean norm, as NetworkX.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'katz' indexed
            by vertex ('vertex_idx'), 'polars' a DataFrame with the columns
            'vertex_idx' and 'katz'.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The Katz centrality of each vertex. The array is indexed by vertex
            identifier.

        Raises
        ------
        ConvergenceError
            If the iterations do not converge within `max_iter` iterations, for
            example because alpha is too large.
        """
        alpha = _check_real(alpha, "alpha")
        beta = _check_real(beta, "beta")
        _check_iteration_arguments(tol, max_iter)
        if self._n_vertices == 0:
            return self._result(np.empty(0), "katz", output, return_series)

        values, n_iter = katz_csc(
            self._indptr,
            self._indices,
            self._data,
            self._weighted,
            beta * self._is_vertex.astype(np.float64),
            alpha,
            self._n_actual * float(tol),
            int(max_iter),
        )
        self._check_result(
            values,
            n_iter,
            f"Katz centrality did not converge within {max_iter} iterations "
            f"(tol={tol}): alpha={alpha} may be larger than the inverse of the "
            "spectral radius of the adjacency matrix, otherwise increase max_iter "
            "or tol",
        )
        if normalized:
            norm = float(np.sqrt(np.dot(values, values)))
            if norm > 0.0:
                values /= norm
        return self._result(values, "katz", output, return_series)
