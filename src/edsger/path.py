"""
Path-related methods.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import numbers
import os
import threading
from typing import TYPE_CHECKING, Optional, Union, List, Any, cast
import warnings

import numpy as np
import pandas as pd

from edsger.graph_importer import standardize_graph_dataframe
from edsger.commons import (
    A_VERY_SMALL_TIME_INTERVAL_PY,
    DTYPE_INF_PY,
    DTYPE_PY,
    INF_FREQ_PY,
    MIN_FREQ_PY,
)
from edsger.bellman_ford import (
    compute_bf_sssp,
    compute_bf_sssp_w_path,
    compute_bf_stsp,
    compute_bf_stsp_w_path,
    detect_negative_cycle,
    detect_negative_cycle_csc,
)
from edsger import dijkstra as dijkstra_dec
from edsger import dijkstra_nodec as dijkstra_lazy
from edsger.path_tracking import compute_path
from edsger.spiess_florian import compute_SF_in
from edsger.star import (
    convert_graph_to_csc_float64,
    convert_graph_to_csc_uint32,
    convert_graph_to_csr_float64,
    convert_graph_to_csr_uint32,
)
from edsger.bfs import bfs_csr, bfs_csc  # pylint: disable=no-name-in-module
from edsger.dfs import dfs_csr, dfs_csc  # pylint: disable=no-name-in-module
from edsger import astar as astar_core  # pylint: disable=no-name-in-module
from edsger import (
    bidirectional_dijkstra as bidir_core,
)  # pylint: disable=no-name-in-module

if TYPE_CHECKING:
    import polars

# Result of the methods returning one value per vertex
VertexResult = Union[np.ndarray, pd.Series, "polars.DataFrame"]

OUTPUT_TYPES = ("numpy", "pandas", "polars")


def _resolve_output(output: Optional[str], return_series: bool) -> str:
    """Return the output type of a method, from its `output` and `return_series`
    arguments: `return_series=True` is the same as `output='pandas'`."""
    if output is None:
        return "pandas" if return_series else "numpy"
    if output not in OUTPUT_TYPES:
        raise ValueError(
            f"argument 'output' should be one of {OUTPUT_TYPES}, got {output!r}"
        )
    if return_series and output != "pandas":
        raise ValueError(
            f"return_series=True means output='pandas', it conflicts with "
            f"output={output!r}"
        )
    return output


def _import_polars():
    """Import Polars, which is an optional dependency."""
    try:
        import polars as pl  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "output='polars' requires Polars: pip install polars"
        ) from exc
    return pl


def _labelled_result(index, values: np.ndarray, name: str, output: str) -> Any:
    """Return values labelled by vertex: a pandas Series indexed by vertex, or a
    Polars DataFrame with a 'vertex_idx' column."""
    if output == "pandas":
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        index.name = "vertex_idx"
        return pd.Series(values, index=index, name=name)
    pl = _import_polars()
    return pl.DataFrame({"vertex_idx": np.asarray(index), name: values})


def _edges_as_pandas(edges, columns):
    """Return the edges as a pandas DataFrame, for the checks of the input data.

    A Polars DataFrame is converted, for the columns to check that it has.
    The other types are rejected.
    """
    if isinstance(edges, pd.DataFrame):
        return edges
    if type(edges).__module__.startswith("polars") and hasattr(edges, "to_pandas"):
        return edges.select(
            [col for col in columns if col in edges.columns]
        ).to_pandas()
    raise TypeError("edges should be a pandas DataFrame or a Polars DataFrame")


def _check_orientation(orientation):
    """Checks the orientation argument."""
    if orientation not in ["in", "out"]:
        raise ValueError("orientation should be either 'in' or 'out'")


class _Graph:
    """Common part of the classes working on a graph built from a dataframe of
    edges.

    It builds the graph in compressed format (CSR for the 'out' orientation, CSC
    for 'in'), renumbers the vertices if asked to, and translates the vertices
    and the results between the original labels and the internal indices.

    With `permute=True`, `_labels` is the sorted array of the original vertex
    labels: the internal index of a vertex is the position of its label in this
    array. Otherwise `_labels` is None and the internal indices are the labels.
    """

    _algorithm_name = "algorithm"

    # set by _build_graph
    _edges: pd.DataFrame
    _tail_name: str
    _head_name: str
    _n_edges: int
    _n_vertices: int
    _n_vertices_init: int
    _orientation: str
    _permute: bool
    _labels: Optional[np.ndarray]
    _negative_labels: bool
    _indptr: np.ndarray
    _indices: np.ndarray
    _edge_weights: Optional[np.ndarray]
    _path_links: Optional[np.ndarray]
    _path_links_permuted: Optional[np.ndarray]
    _verbose: bool

    # ------------------------------------------------------------ graph building

    def _build_graph(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges,
        tail: str,
        head: str,
        weight: Optional[str],
        orientation: str,
        permute: bool,
        verbose: bool,
    ) -> None:
        """Build the graph from standardized edges (NumPy-backed pandas DataFrame).

        `weight` is None for an unweighted graph.
        """
        self._edges = edges
        self._tail_name = tail
        self._head_name = head
        self._verbose = verbose
        self._remove_parallel_edges(tail, head, weight)

        # reindex the vertices
        self._permute = permute
        self._labels = None
        # negative vertex labels are only possible with permute=True
        self._negative_labels = False
        if self._n_edges == 0:
            self._n_vertices = 0
            self._n_vertices_init = 0
        elif permute:
            self._renumber_vertices(tail, head, weight)
            assert self._labels is not None
            self._negative_labels = bool(self._labels[0] < 0)
            self._n_vertices = len(self._labels)
            self._n_vertices_init = int(self._labels[-1]) + 1
        else:
            self._n_vertices = int(
                max(
                    np.asarray(self._edges[tail]).max(),
                    np.asarray(self._edges[head]).max(),
                )
                + 1
            )
            self._n_vertices_init = self._n_vertices

        # convert to CSR ('out') or CSC ('in')
        _check_orientation(orientation)
        self._orientation = orientation
        convert = (
            convert_graph_to_csr_float64
            if orientation == "out"
            else convert_graph_to_csc_float64
        )
        if weight is None:
            # the conversion needs a data column, unused for an unweighted graph
            weight_col = "_dummy_weight"
            edges_tmp = self._edges.assign(**{weight_col: 1.0})
        else:
            weight_col = weight
            edges_tmp = self._edges
        indptr, indices, data = convert(
            edges_tmp, tail, head, weight_col, self._n_vertices
        )
        self._indptr = indptr.astype(np.uint32)
        self._indices = indices.astype(np.uint32)
        self._edge_weights = None if weight is None else data.astype(DTYPE_PY)

        self._path_links = None
        self._path_links_permuted = None

    def _remove_parallel_edges(self, tail, head, weight):
        """Keep one edge per pair of vertices: the one with the smallest weight, or
        any of them for an unweighted graph."""
        original_count = len(self._edges)
        grouped = self._edges.groupby([tail, head], as_index=False)
        if weight is None:
            self._edges = cast(pd.DataFrame, grouped.first())
        else:
            self._edges = cast(pd.DataFrame, grouped[weight].min())
        self._n_edges = len(self._edges)

        removed = original_count - self._n_edges
        if removed > 0 and self._verbose:
            kept = (
                "All the edges count for one."
                if weight is None
                else "For each pair of vertices, kept the edge with minimum weight."
            )
            print(f"Automatically removed {removed} parallel edge(s). {kept}")

    def _renumber_vertices(self, tail, head, weight):
        """Renumber the vertices with contiguous indices starting from 0, in the
        order of their labels."""
        tails = np.asarray(self._edges[tail])
        heads = np.asarray(self._edges[head])
        labels, inverse = np.unique(np.concatenate((tails, heads)), return_inverse=True)
        inverse = inverse.reshape(-1)
        data = {tail: inverse[: len(tails)], head: inverse[len(tails) :]}
        if weight is not None:
            data[weight] = np.asarray(self._edges[weight])
        self._edges = pd.DataFrame(data)
        self._labels = labels

    def _check_edges(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, edges, tail, head, weight=None, nonnegative=False
    ):
        """Checks if the edges DataFrame is well-formed. If not, raises an
        appropriate error."""
        columns = [tail, head] + ([] if weight is None else [weight])
        edges = _edges_as_pandas(edges, columns)

        for role, col in (("tail", tail), ("head", head), ("weight", weight)):
            if col is not None and col not in edges:
                raise KeyError(
                    f"edge {role} column '{col}' not found in graph edges dataframe"
                )

        if edges[columns].isnull().to_numpy().any():
            raise ValueError(
                f"edges[[{', '.join(columns)}]] should not have any missing value"
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"edges['{col}'] should be of integer type")

        if weight is not None:
            if not pd.api.types.is_numeric_dtype(edges[weight].dtype):
                raise TypeError(f"edges['{weight}'] should be of numeric type")
            if nonnegative and edges[weight].min() < 0.0:
                raise ValueError(f"edges['{weight}'] should be nonnegative")
            if not np.isfinite(edges[weight]).all():
                raise ValueError(f"edges['{weight}'] should be finite")

    # ------------------------------------------------------------ properties

    @property
    def edges(self) -> Any:
        """The edges of the graph as they are stored, without the parallel edges,
        and with the internal vertex indices if `permute` is True.
        """
        return self._edges

    @property
    def n_edges(self) -> int:
        """Number of edges of the graph, without the parallel edges."""
        return self._n_edges

    @property
    def n_vertices(self) -> int:
        """Number of vertices. This is the largest vertex identifier plus one, or
        the number of distinct vertices if `permute` is True.
        """
        return self._n_vertices

    @property
    def permute(self) -> bool:
        """Whether the vertices are renumbered internally."""
        return self._permute

    # ------------------------------------------------------------ vertex labels

    def _check_output(self, output: str) -> None:
        """An array indexed by vertex cannot hold negative vertex labels."""
        if self._negative_labels and output == "numpy":
            raise ValueError(
                "the graph has negative vertex labels: the result cannot be an "
                "array indexed by vertex, use output='pandas' "
                "(return_series=True) or output='polars'"
            )

    def _internal_vertex(self, vertex_idx: int) -> int:
        """Return the internal index of a vertex given by its label."""
        if self._labels is not None:
            position = int(np.searchsorted(self._labels, vertex_idx))
            if position == len(self._labels) or self._labels[position] != vertex_idx:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            return position
        if vertex_idx < 0:
            raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be non-negative")
        if vertex_idx >= self._n_vertices:
            raise ValueError(f"vertex {vertex_idx} not found in graph")
        return int(vertex_idx)

    def _internal_vertices(
        self, vertices: np.ndarray, what: str, what_plural: str
    ) -> np.ndarray:
        """Return the internal indices of an array of vertex labels, as uint32."""
        if self._labels is not None:
            positions = np.searchsorted(self._labels, vertices)
            found = positions < len(self._labels)
            found[found] = self._labels[positions[found]] == vertices[found]
            if not found.all():
                raise ValueError(f"{what} {vertices[~found][0]} not found in graph")
            return positions.astype(np.uint32)
        if np.any(vertices >= self._n_vertices) or np.any(vertices < 0):
            raise ValueError(f"{what_plural} contain invalid vertex indices")
        return vertices.astype(np.uint32)

    def _vertex_result(
        self, values: np.ndarray, name: str, fill, output: str
    ) -> VertexResult:
        """Return a result given for each internal vertex, in the output type.

        A NumPy result is indexed by vertex label, from 0 to the largest label:
        the integers that are not vertices get the `fill` value.
        """
        if output == "numpy":
            if self._labels is None:
                return values
            dense = np.full(self._n_vertices_init, fill, dtype=values.dtype)
            dense[self._labels] = values
            return dense
        if self._labels is None:
            index = pd.RangeIndex(self._n_vertices)
        else:
            index = self._labels
        return _labelled_result(index, values, name, output)

    def _labels_of(self, internal: np.ndarray) -> np.ndarray:
        """Return the labels of internal vertex indices."""
        if self._labels is None:
            return internal
        return self._labels[internal]

    # ------------------------------------------------------------ public methods

    def get_vertices(self) -> Any:
        """Return the vertices of the graph.

        Returns
        -------
        numpy.ndarray
            The sorted identifiers of the vertices that are the tail or the head of
            at least one edge.
        """
        if self._labels is not None:
            return self._labels.copy()
        return np.union1d(
            np.asarray(self._edges[self._tail_name]),
            np.asarray(self._edges[self._head_name]),
        )


class _VertexGraph(_Graph):
    """Common part of the algorithms run from or to one vertex, with an
    orientation and an optional path tree: `Dijkstra`, `BellmanFord`, `BFS`,
    `DFS`."""

    @property
    def orientation(self) -> str:
        """Orientation of the algorithm, either 'out' or 'in'."""
        return self._orientation

    @property
    def path_links(self) -> Optional[np.ndarray]:
        """Path tree of the last run, or None if the paths were not tracked. This
        is the predecessor ('out') or the successor ('in') of each vertex. The
        vertex given to `run` and the vertices that cannot be reached are linked
        to themselves.
        """
        return self._path_links

    # ------------------------------------------------------------ path tree

    def _set_path_links(self, path_links: Optional[np.ndarray], output: str) -> None:
        """Store the path tree computed on the internal indices."""
        if path_links is None:
            self._path_links = None
            return
        if self._labels is None:
            self._path_links = path_links
            return
        # kept for get_path, which follows the path on the internal indices
        self._path_links_permuted = path_links
        linked_labels = self._labels[path_links]
        if output == "numpy":
            # indexed by vertex label
            self._path_links = np.arange(self._n_vertices_init, dtype=np.uint32)
            self._path_links[self._labels] = linked_labels
        else:
            # in the order of the vertex labels, like the Series or DataFrame
            self._path_links = linked_labels.astype(np.uint32)

    # ------------------------------------------------------------ public methods

    def get_path(self, vertex_idx: int) -> Optional[np.ndarray]:
        """Return the shortest path between a vertex and the vertex of the last run.

        The last call to `run` must have been made with `path_tracking=True`.

        Parameters
        ----------
        vertex_idx : int
            The other end of the path: the target vertex if the orientation is
            'out', the source vertex if it is 'in'.

        Returns
        -------
        numpy.ndarray or None
            The vertices of the path, starting with `vertex_idx` and ending with the
            vertex given to `run`. With the 'out' orientation, this is the path in
            reverse order, from the target back to the source. With the 'in'
            orientation, it is the path in travel order. If there is no path, the
            array only contains `vertex_idx`. None is returned, with a warning, if
            no path was tracked.
        """
        if self._path_links is None:
            warnings.warn(
                f"Current {type(self).__name__} instance has no path attribute: "
                "make sure path_tracking is set to True, and run the "
                f"{self._algorithm_name}",
                UserWarning,
            )
            return None
        if self._labels is not None:
            assert self._path_links_permuted is not None
            position = self._internal_vertex(vertex_idx)
            path_vertices = self._labels[
                compute_path(self._path_links_permuted, position)
            ]
            if self._labels[0] >= 0:
                path_vertices = path_vertices.astype(np.uint32)
            return path_vertices
        return compute_path(self._path_links, vertex_idx)


class Dijkstra(_VertexGraph):
    """Dijkstra's algorithm, for the shortest paths in a directed graph with
    non-negative edge weights.

    The graph is built once from a dataframe of edges. The `run` method can then
    be called any number of times, from or to different vertices.

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
    weight : str, optional (default='weight')
        Name of the column with the edge weights. Negative and NaN weights are
        rejected with a ValueError, whatever the value of `check_edges`.
    orientation : {'out', 'in'}, optional (default='out')
        'out' for the paths from one vertex to all the others (single source),
        'in' for the paths from all the vertices to one vertex (single target).
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and non-finite values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Without it, the internal arrays have one element per
        integer between 0 and the largest vertex identifier. The arguments and
        the results of the methods use the original identifiers in both cases.
        Negative identifiers are only accepted with this option, and the results
        must then be labelled by vertex (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.
    lazy : bool, optional (default=True)
        Whether to use the variant of the algorithm without decrease-key
        operation, in which a vertex can be inserted several times in the
        priority queue. Both variants return the same path lengths.

    Notes
    -----
    When several edges have the same tail and head vertices, only the one with
    the smallest weight is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import Dijkstra
    >>> edges = pd.DataFrame(
    ...     {"tail": [0, 0, 1], "head": [1, 2, 2], "weight": [1.0, 4.0, 2.0]}
    ... )
    >>> Dijkstra(edges).run(vertex_idx=0)
    array([0., 1., 3.])
    """

    _algorithm_name = "shortest path algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: str = "weight",
        orientation: str = "out",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
        lazy: bool = True,
    ) -> None:
        # Select the dijkstra module based on the lazy parameter
        self._dijkstra = dijkstra_lazy if lazy else dijkstra_dec
        self._lazy = lazy

        if check_edges:
            self._check_edges(edges, tail, head, weight, nonnegative=True)
        # Convert to standardized NumPy-backed pandas DataFrame
        edges = standardize_graph_dataframe(edges, tail, head, weight)
        # Dijkstra's algorithm is not valid with negative weights: they are
        # rejected whatever the value of check_edges, and so are NaN weights
        # (the min is NaN if any weight is)
        if len(edges) > 0 and not np.asarray(edges[weight]).min() >= 0.0:
            raise ValueError(f"edges['{weight}'] should be nonnegative")

        self._build_graph(edges, tail, head, weight, orientation, permute, verbose)

    @property
    def lazy(self) -> bool:
        """Whether the variant without decrease-key operation is used."""
        return self._lazy

    def _heap_length(self, heap_length_ratio: float) -> int:
        """Initial capacity of the priority queue."""
        if not isinstance(heap_length_ratio, float):
            raise TypeError(
                f"argument 'heap_length_ratio={heap_length_ratio}' must be of float type"
            )
        heap_length_ratio = min(heap_length_ratio, 1.0)
        if heap_length_ratio <= 0.0:
            raise ValueError(
                f"argument 'heap_length_ratio={heap_length_ratio}' must be strictly "
                "positive"
            )
        return int(np.rint(heap_length_ratio * self._n_vertices))

    def _termination_nodes(self, termination_nodes) -> Optional[np.ndarray]:
        """Check the termination nodes and return their internal indices."""
        if termination_nodes is None:
            return None
        try:
            nodes = np.array(termination_nodes, dtype=np.int64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise TypeError(
                "argument 'termination_nodes' must be array-like of integers"
            ) from exc
        if nodes.ndim != 1:
            raise ValueError("argument 'termination_nodes' must be 1-dimensional")
        if len(nodes) == 0:
            raise ValueError("argument 'termination_nodes' must not be empty")
        return self._internal_vertices(nodes, "termination node", "termination nodes")

    def _compute(
        self,
        vertex: int,
        path_links: Optional[np.ndarray],
        termination: Optional[np.ndarray],
        heap_length: int,
    ) -> np.ndarray:
        """Run the Cython routine matching the orientation and the options."""
        name = "compute_" + ("sssp" if self._orientation == "out" else "stsp")
        args: List[Any] = [self._indptr, self._indices, self._edge_weights]
        if path_links is not None:
            name += "_w_path"
            args.append(path_links)
        if termination is not None:
            name += "_early_termination"
            args.append(termination)
        args += [vertex, self._n_vertices, heap_length]
        return getattr(self._dijkstra, name)(*args)

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_inf: bool = True,
        return_series: bool = False,
        heap_length_ratio: float = 1.0,
        termination_nodes: Optional[List[int]] = None,
        output: Optional[str] = None,
    ) -> VertexResult:
        """Compute the shortest path lengths from or to a given vertex.

        Parameters
        ----------
        vertex_idx : int
            The source vertex if the orientation is 'out', the target vertex if it
            is 'in'.
        path_tracking : bool, optional (default=False)
            Whether to store the shortest path tree, so that the paths can be
            retrieved with `get_path`.
        return_inf : bool, optional (default=True)
            Whether to return `numpy.inf` as the path length of the vertices that
            cannot be reached. If False, the largest float64 value is returned
            instead.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.
        heap_length_ratio : float, optional (default=1.0)
            Initial capacity of the priority queue, as a fraction of the number of
            vertices, in (0, 1]. The queue grows when it is full, so this only
            changes the size of the first memory allocation.
        termination_nodes : array-like of int, optional (default=None)
            Vertices at which to stop. The algorithm ends as soon as the path
            lengths of all these vertices are known, and only their path lengths
            are returned, in the given order.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'path_length'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'path_length'.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The path lengths, as float64. Without `termination_nodes`, the array
            has one element per integer between 0 and the largest vertex
            identifier: element i is the length of the shortest path from the
            source to vertex i ('out'), or from vertex i to the target ('in').
            The Series and the DataFrame have one row per vertex. With
            `termination_nodes`, there is one element per termination node.

        Raises
        ------
        ValueError
            If `vertex_idx` or a termination node is not a vertex of the graph, or
            if the graph has negative vertex identifiers and the output is
            'numpy'.
        """
        output = _resolve_output(output, return_series)
        self._check_output(output)
        vertex_new = self._internal_vertex(vertex_idx)
        heap_length = self._heap_length(heap_length_ratio)
        termination = self._termination_nodes(termination_nodes)

        path_links = (
            np.arange(self._n_vertices, dtype=np.uint32) if path_tracking else None
        )
        path_lengths = self._compute(vertex_new, path_links, termination, heap_length)
        self._set_path_links(path_links, output)

        # deal with infinity, in place: the array has just been allocated by the
        # Dijkstra routine, and a copy of it is costly on large graphs
        if return_inf:
            path_lengths[path_lengths == DTYPE_INF_PY] = np.inf

        if termination is not None:
            # one path length per termination node, in the given order
            if output == "numpy":
                return path_lengths
            return _labelled_result(
                np.asarray(termination_nodes), path_lengths, "path_length", output
            )

        fill = np.inf if return_inf else DTYPE_INF_PY
        return self._vertex_result(path_lengths, "path_length", fill, output)

    def run_many(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        vertices: List[int],
        termination_nodes: Optional[List[int]] = None,
        return_inf: bool = True,
        output: Optional[str] = None,
        n_jobs: Optional[int] = None,
        heap_length_ratio: float = 1.0,
    ) -> Union[np.ndarray, pd.DataFrame, "polars.DataFrame"]:
        """Compute the shortest path lengths from or to several vertices, in parallel.

        Each vertex is a separate run of the algorithm. The runs are distributed
        over a pool of threads: the Cython routine does not hold the GIL.

        Parameters
        ----------
        vertices : array-like of int
            The source vertices if the orientation is 'out', the target vertices
            if it is 'in'.
        termination_nodes : array-like of int, optional (default=None)
            The other ends of the paths: if given, each run stops as soon as the
            path lengths of these vertices are known, and only these path lengths
            are returned. With the 'out' orientation, this gives an
            origin-destination matrix of the path lengths from `vertices` to
            `termination_nodes`.
        return_inf : bool, optional (default=True)
            Whether to return `numpy.inf` as the path length of the vertices that
            cannot be reached. If False, the largest float64 value is returned
            instead.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result, see below.
        n_jobs : int, optional (default=None)
            Number of threads. None uses all the CPUs (`os.cpu_count()`).
        heap_length_ratio : float, optional (default=1.0)
            Initial capacity of the priority queue, as in `run`.

        Returns
        -------
        numpy.ndarray, pandas.DataFrame or polars.DataFrame
            The path lengths, as float64. The NumPy array has one row per vertex
            of `vertices`, and one column per integer between 0 and the largest
            vertex identifier, or one column per termination node. The pandas
            DataFrame has the same rows, labelled by `vertices`, and one column
            per vertex of the graph or per termination node. The Polars DataFrame
            has one row per pair of vertices, with the columns 'origin',
            'destination' and 'path_length'. The pandas index and columns are
            named 'origin' and 'destination' as well. With the 'in' orientation,
            `vertices` are the destinations.

        Raises
        ------
        ValueError
            If a vertex or a termination node is not a vertex of the graph, or if
            the graph has negative vertex identifiers and the output is 'numpy'.

        Notes
        -----
        Without termination nodes, the result holds one path length per pair of
        vertices: 8 bytes times the number of runs times the number of vertices.
        """
        output = _resolve_output(output, False)
        self._check_output(output)
        try:
            vertex_labels = np.array(vertices, dtype=np.int64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise TypeError(
                "argument 'vertices' must be array-like of integers"
            ) from exc
        if vertex_labels.ndim != 1:
            raise ValueError("argument 'vertices' must be 1-dimensional")
        sources = self._internal_vertices(vertex_labels, "vertex", "vertices")
        heap_length = self._heap_length(heap_length_ratio)
        termination = self._termination_nodes(termination_nodes)
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
            raise TypeError(f"argument 'n_jobs' must be an integer, got {n_jobs!r}")
        if n_jobs < 1:
            raise ValueError(f"argument 'n_jobs' must be positive, got {n_jobs}")

        n_columns = self._n_vertices if termination is None else len(termination)
        path_lengths = np.empty((len(sources), n_columns), dtype=DTYPE_PY)

        def compute_row(i):
            path_lengths[i] = self._compute(
                int(sources[i]), None, termination, heap_length
            )

        n_jobs = min(int(n_jobs), len(sources))
        if n_jobs <= 1:
            for i in range(len(sources)):
                compute_row(i)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # list() re-raises the exceptions of the threads
                list(executor.map(compute_row, range(len(sources))))

        if return_inf:
            path_lengths[path_lengths == DTYPE_INF_PY] = np.inf

        # column labels
        if termination is not None:
            columns = np.asarray(termination_nodes)
        elif self._labels is not None:
            columns = self._labels
            if output == "numpy":
                # one column per integer between 0 and the largest label
                dense = np.full(
                    (len(sources), self._n_vertices_init),
                    np.inf if return_inf else DTYPE_INF_PY,
                )
                dense[:, self._labels] = path_lengths
                path_lengths = dense
        else:
            columns = np.arange(self._n_vertices)
        if output == "numpy":
            return path_lengths

        if self._orientation == "out":
            row_name, column_name = "origin", "destination"
        else:
            row_name, column_name = "destination", "origin"
        if output == "pandas":
            return pd.DataFrame(
                path_lengths,
                index=pd.Index(vertex_labels, name=row_name),
                columns=pd.Index(columns, name=column_name),
            )
        pl = _import_polars()
        long = pl.DataFrame(
            {
                row_name: np.repeat(vertex_labels, len(columns)),
                column_name: np.tile(columns, len(vertex_labels)),
                "path_length": path_lengths.reshape(-1),
            }
        )
        return long.select(["origin", "destination", "path_length"])


class BellmanFord(_VertexGraph):
    """Bellman-Ford algorithm, for the shortest paths in a directed graph whose
    edge weights can be negative.

    The graph is built once from a dataframe of edges. The `run` method can then
    be called any number of times, from or to different vertices. The shortest
    paths are not defined when the graph has a cycle of negative total weight:
    such a cycle is detected and reported.

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
    weight : str, optional (default='weight')
        Name of the column with the edge weights, which can be negative.
    orientation : {'out', 'in'}, optional (default='out')
        'out' for the paths from one vertex to all the others (single source),
        'in' for the paths from all the vertices to one vertex (single target).
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and non-finite values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Without it, the internal arrays have one element per
        integer between 0 and the largest vertex identifier. The arguments and
        the results of the methods use the original identifiers in both cases.
        Negative identifiers are only accepted with this option, and the results
        must then be labelled by vertex (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    When several edges have the same tail and head vertices, only the one with
    the smallest weight is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import BellmanFord
    >>> edges = pd.DataFrame(
    ...     {"tail": [0, 0, 1], "head": [1, 2, 2], "weight": [1.0, 4.0, -2.0]}
    ... )
    >>> BellmanFord(edges).run(vertex_idx=0)
    array([ 0.,  1., -1.])
    """

    _algorithm_name = "shortest path algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: str = "weight",
        orientation: str = "out",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        # Convert to standardized NumPy-backed pandas DataFrame
        edges = standardize_graph_dataframe(edges, tail, head, weight)
        self._build_graph(edges, tail, head, weight, orientation, permute, verbose)

        # the negative cycles are only searched for if there is a negative weight
        assert self._edge_weights is not None
        self._has_negative_weights = bool(np.any(self._edge_weights < 0))
        self._has_negative_cycle = False

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_inf: bool = True,
        return_series: bool = False,
        detect_negative_cycles: bool = True,
        output: Optional[str] = None,
    ) -> VertexResult:
        """Compute the shortest path lengths from or to a given vertex.

        Parameters
        ----------
        vertex_idx : int
            The source vertex if the orientation is 'out', the target vertex if it
            is 'in'.
        path_tracking : bool, optional (default=False)
            Whether to store the shortest path tree, so that the paths can be
            retrieved with `get_path`.
        return_inf : bool, optional (default=True)
            Whether to return `numpy.inf` as the path length of the vertices that
            cannot be reached. If False, the largest float64 value is returned
            instead.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.
        detect_negative_cycles : bool, optional (default=True)
            Whether to look for a negative cycle once the path lengths are computed,
            and raise a ValueError if there is one. The graph is only searched if
            it has at least one negative weight.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'path_length'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'path_length'.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The path lengths, as float64. The array has one element per integer
            between 0 and the largest vertex identifier: element i is the length of
            the shortest path from the source to vertex i ('out'), or from vertex i
            to the target ('in'). The Series and the DataFrame have one row per
            vertex.

        Raises
        ------
        ValueError
            If `vertex_idx` is not a vertex of the graph, if a negative cycle is
            detected, or if the graph has negative vertex identifiers and the
            output is 'numpy'.
        """
        output = _resolve_output(output, return_series)
        self._check_output(output)
        vertex_new = self._internal_vertex(vertex_idx)

        out = self._orientation == "out"
        weights = self._edge_weights
        assert weights is not None
        if path_tracking:
            path_links = np.arange(self._n_vertices, dtype=np.uint32)
            compute = compute_bf_sssp_w_path if out else compute_bf_stsp_w_path
            path_lengths = compute(
                self._indptr,
                self._indices,
                weights,
                path_links,
                vertex_new,
                self._n_vertices,
            )
        else:
            path_links = None
            compute = compute_bf_sssp if out else compute_bf_stsp
            path_lengths = compute(
                self._indptr,
                self._indices,
                weights,
                vertex_new,
                self._n_vertices,
            )
        self._set_path_links(path_links, output)

        # detect negative cycles if requested (only if negative weights exist)
        if detect_negative_cycles and self._has_negative_weights:
            detect = detect_negative_cycle if out else detect_negative_cycle_csc
            self._has_negative_cycle = detect(
                self._indptr,
                self._indices,
                weights,
                path_lengths,
                self._n_vertices,
            )
            if self._has_negative_cycle:
                raise ValueError("Negative cycle detected in the graph")

        # deal with infinity, in place: the array has just been allocated
        if return_inf:
            path_lengths[path_lengths == DTYPE_INF_PY] = np.inf

        fill = np.inf if return_inf else DTYPE_INF_PY
        return self._vertex_result(path_lengths, "path_length", fill, output)

    def has_negative_cycle(self):
        """Tell whether a negative cycle was found.

        Returns
        -------
        bool
            The result of the last run made with `detect_negative_cycles=True`.
        """
        return self._has_negative_cycle


class HyperpathGenerating:
    """Spiess and Florian's algorithm, for the optimal strategies (hyperpaths)
    in a frequency-based transit network.

    Each edge has a travel time and a frequency. At a vertex, a traveller
    boards the first vehicle to come among a set of attractive edges, so the
    demand is split between these edges in proportion to their frequencies, and
    the travel time includes an expected waiting time. The algorithm computes
    the strategy with the smallest expected travel time to a destination, and
    the resulting volume on each edge.

    Parameters
    ----------
    edges : pandas.DataFrame or polars.DataFrame
        The edges of the graph, one row per edge, without missing values.
    tail : str, optional (default='tail')
        Name of the column with the tail (start) vertex of each edge. The vertex
        identifiers are non-negative integers.
    head : str, optional (default='head')
        Name of the column with the head (end) vertex of each edge.
    trav_time : str, optional (default='trav_time')
        Name of the column with the travel time of each edge.
    freq : str, optional (default='freq')
        Name of the column with the frequency of each edge, in vehicles per
        unit of travel time. An infinite frequency means that there is no
        waiting time on the edge.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and negative values.
    orientation : {'in', 'out'}, optional (default='in')
        Only 'in' (many origins to one destination) is implemented.

    Attributes
    ----------
    edge_count : int
        Number of edges.
    vertex_count : int
        Number of vertices: the largest vertex identifier plus one.
    u_i_vec : numpy.ndarray or None
        After a run, the expected travel time from each vertex to the
        destination.

    Notes
    -----
    The travel times are bounded below by 1e-12 and the frequencies are kept
    in the range [1e-20, 1e20].

    References
    ----------
    Spiess, H. and Florian, M. (1989). Optimal strategies: a new assignment
    model for transit networks. Transportation Research Part B 23(2), 83-102.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import HyperpathGenerating
    >>> edges = pd.DataFrame(
    ...     {
    ...         "tail": [0, 0, 1],
    ...         "head": [2, 1, 2],
    ...         "trav_time": [25.0, 7.0, 10.0],
    ...         "freq": [1 / 12, 1 / 12, 1 / 6],
    ...     }
    ... )
    >>> hp = HyperpathGenerating(edges)
    >>> hp.run(origin=0, destination=2, volume=100.0)
    >>> hp.u_i_vec
    array([30., 16.,  0.])
    """

    def __init__(
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        trav_time: str = "trav_time",
        freq: str = "freq",
        check_edges: bool = False,
        orientation: str = "in",
    ) -> None:
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq)
        # Convert to standardized NumPy-backed pandas DataFrame
        self._edges = standardize_graph_dataframe(
            edges, tail, head, trav_time=trav_time, freq=freq
        )
        self.edge_count = len(self._edges)

        # remove inf values if any, and values close to zero
        self._edges[trav_time] = np.where(
            self._edges[trav_time] > DTYPE_INF_PY, DTYPE_INF_PY, self._edges[trav_time]
        )
        self._edges[trav_time] = np.where(
            self._edges[trav_time] < A_VERY_SMALL_TIME_INTERVAL_PY,
            A_VERY_SMALL_TIME_INTERVAL_PY,
            self._edges[trav_time],
        )
        self._edges[freq] = np.where(
            self._edges[freq] > INF_FREQ_PY, INF_FREQ_PY, self._edges[freq]
        )
        self._edges[freq] = np.where(
            self._edges[freq] < MIN_FREQ_PY, MIN_FREQ_PY, self._edges[freq]
        )

        # create an edge index column
        self._edges = self._edges.reset_index(drop=True)
        data_col = "edge_idx"
        self._edges[data_col] = self._edges.index

        # convert to CSR/CSC format
        self.vertex_count = self._edges[[tail, head]].max().max() + 1
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' or 'out'")
        self._orientation = orientation
        if self._orientation == "out":
            fs_indptr, _, fs_data = convert_graph_to_csr_uint32(
                self._edges, tail, head, data_col, self.vertex_count
            )
            self.__indptr = fs_indptr.astype(np.uint32)
            self._edge_idx = fs_data.astype(np.uint32)
        else:
            rs_indptr, _, rs_data = convert_graph_to_csc_uint32(
                self._edges, tail, head, data_col, self.vertex_count
            )
            self.__indptr = rs_indptr.astype(np.uint32)
            self._edge_idx = rs_data.astype(np.uint32)

        # edge attributes
        self._trav_time = np.asarray(self._edges[trav_time]).astype(DTYPE_PY)
        self._freq = np.asarray(self._edges[freq]).astype(DTYPE_PY)
        self._tail = np.asarray(self._edges[tail]).astype(np.uint32)
        self._head = np.asarray(self._edges[head]).astype(np.uint32)

        # node attribute
        self.u_i_vec = None

    def run(
        self,
        origin: Union[int, List[int]],
        destination: int,
        volume: Union[float, List[float]],
        return_inf: bool = False,
    ) -> None:
        """Compute the optimal strategy to a destination and assign the demand.

        The results are stored in the object: the expected travel times in
        `u_i_vec`, and the edge volumes in the 'volume' column of the `_edges`
        dataframe.

        Parameters
        ----------
        origin : int or list of int
            The origin vertex, or the origin vertices, of the demand.
        destination : int
            The destination vertex.
        volume : float or list of float
            The demand from each origin to the destination: non-negative numbers,
            as many as origins.
        return_inf : bool, optional (default=False)
            Not used.

        Raises
        ------
        NotImplementedError
            If the orientation is 'out'.
        TypeError
            If a vertex is not an integer or a volume is not a number.
        ValueError
            If a vertex is not in the graph, if a volume is negative or NaN, or if
            the origins and the volumes do not have the same length.
        """
        # column storing the resulting edge volumes
        self._edges["volume"] = 0.0
        self.u_i_vec = None

        # vertex least travel time
        u_i_vec = DTYPE_INF_PY * np.ones(self.vertex_count, dtype=DTYPE_PY)

        # input check
        if not isinstance(volume, list):
            volume = [volume]

        if self._orientation == "out":
            raise NotImplementedError(
                "one-to-many Spiess & Florian's algorithm not implemented yet"
            )

        # Only "in" orientation is supported currently
        if not isinstance(origin, list):
            origin = [origin]
        if len(origin) != len(volume):
            raise ValueError(
                f"origin and volume should have the same length, "
                f"got {len(origin)} and {len(volume)}"
            )
        for i, item in enumerate(origin):
            self._check_vertex_idx(item)
            self._check_volume(volume[i])
        self._check_vertex_idx(destination)
        demand_indices = np.array(origin, dtype=np.uint32)

        if not isinstance(return_inf, bool):
            raise TypeError("argument 'return_inf' should be a bool")

        demand_values = np.array(volume, dtype=DTYPE_PY)

        # The edge volumes are computed in their own array and then stored in the
        # edges dataframe: with the copy-on-write of pandas 3, the values of a column
        # are read-only and cannot be filled in place.
        edge_volumes = np.zeros(len(self._edges), dtype=DTYPE_PY)

        compute_SF_in(
            self.__indptr,
            self._edge_idx,
            self._trav_time,
            self._freq,
            self._tail,
            self._head,
            demand_indices,  # source vertex indices
            demand_values,
            edge_volumes,
            u_i_vec,
            self.vertex_count,
            destination,
        )
        self._edges["volume"] = edge_volumes
        self.u_i_vec = u_i_vec

    def _check_vertex_idx(self, idx):
        if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
            raise TypeError(
                f"vertex index should be an integer, got {type(idx).__name__}"
            )
        if idx < 0 or idx >= self.vertex_count:
            raise ValueError(f"vertex {idx} not found in graph")

    def _check_volume(self, v):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise TypeError(f"volume should be a number, got {type(v).__name__}")
        if not float(v) >= 0.0:
            raise ValueError(f"volume should be non-negative, got {v}")

    def _check_edges(self, edges, tail, head, trav_time, freq):
        edges = _edges_as_pandas(edges, [tail, head, trav_time, freq])

        for col in [tail, head, trav_time, freq]:
            if col not in edges:
                raise KeyError(
                    f"edge column '{col}' not found in graph edges dataframe"
                )

        if edges[[tail, head, trav_time, freq]].isnull().to_numpy().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{tail}, {head}, {trav_time}, {freq}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of integer type")

        for col in [trav_time, freq]:
            if not pd.api.types.is_numeric_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of numeric type")

            if edges[col].min() < 0.0:
                raise ValueError(f"column '{col}' should be nonnegative")


class BFS(_VertexGraph):
    """Breadth-first search, for the paths with the fewest edges in a directed
    graph.

    The edge weights are not used: all the edges count for one. The graph is
    built once from a dataframe of edges. The `run` method can then be called
    any number of times, from or to different vertices.

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
    orientation : {'out', 'in'}, optional (default='out')
        'out' for the paths from one vertex to all the others (single source),
        'in' for the paths from all the vertices to one vertex (single target).
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types and missing values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Without it, the internal arrays have one element per
        integer between 0 and the largest vertex identifier. The arguments and
        the results of the methods use the original identifiers in both cases.
        Negative identifiers are only accepted with this option, and the results
        must then be labelled by vertex (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.
    sentinel : int, optional (default=-9999)
        Value returned for the vertices without predecessor: the start vertex
        and the vertices that cannot be reached. It must be a negative int32,
        different from all the vertex identifiers.

    Notes
    -----
    When several edges have the same tail and head vertices, only one is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import BFS
    >>> edges = pd.DataFrame({"tail": [0, 0, 1], "head": [1, 2, 3]})
    >>> BFS(edges).run(vertex_idx=0)
    array([-9999,     0,     0,     1], dtype=int32)
    """

    _algorithm_name = "BFS algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        orientation: str = "out",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
        sentinel: int = -9999,
    ) -> None:
        # Validate sentinel value (runtime check for callers without type checking)
        if not isinstance(
            sentinel, int
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"sentinel must be an integer, got {type(sentinel).__name__}"
            )
        if sentinel >= 0:
            raise ValueError(f"sentinel must be negative, got {sentinel}")
        if sentinel < np.iinfo(np.int32).min or sentinel > np.iinfo(np.int32).max:
            raise ValueError(
                f"sentinel must fit in int32 range [{np.iinfo(np.int32).min}, "
                f"{np.iinfo(np.int32).max}], got {sentinel}"
            )
        self._sentinel = sentinel

        if check_edges:
            self._check_edges(edges, tail, head)
        # Convert to standardized NumPy-backed pandas DataFrame, without weights
        edges = standardize_graph_dataframe(edges, tail, head)
        self._build_graph(edges, tail, head, None, orientation, permute, verbose)

        if self._labels is not None and np.any(self._labels == self._sentinel):
            raise ValueError(
                f"sentinel {self._sentinel} is also a vertex label: "
                "use another sentinel value"
            )

    @property
    def UNREACHABLE(self) -> int:  # pylint: disable=invalid-name
        """The sentinel value, returned for the vertices without predecessor."""
        return self._sentinel

    def run(
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_series: bool = False,
        output: Optional[str] = None,
    ) -> VertexResult:
        """Compute the breadth-first search tree from or to a given vertex.

        Parameters
        ----------
        vertex_idx : int
            The source vertex if the orientation is 'out', the target vertex if it
            is 'in'.
        path_tracking : bool, optional (default=False)
            Whether to store the search tree, so that the paths can be retrieved
            with `get_path`.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'predecessor'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'predecessor'.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The predecessor ('out') or the successor ('in') of each vertex on a
            path with the fewest edges, as int32. The array has one element per
            integer between 0 and the largest vertex identifier. The start vertex
            and the vertices that cannot be reached have the sentinel value. The
            Series and the DataFrame have one row per vertex.

        Raises
        ------
        ValueError
            If `vertex_idx` is not a vertex of the graph, or if the graph has
            negative vertex identifiers and the output is 'numpy'.
        """
        output = _resolve_output(output, return_series)
        self._check_output(output)
        vertex_new = self._internal_vertex(vertex_idx)

        bfs = bfs_csr if self._orientation == "out" else bfs_csc
        predecessors = bfs(
            self._indptr,
            self._indices,
            vertex_new,
            self._n_vertices,
            self._sentinel,
        )
        reachable = predecessors != self._sentinel

        if path_tracking:
            # the vertices without predecessor are linked to themselves
            path_links = np.arange(self._n_vertices, dtype=np.uint32)
            path_links[reachable] = predecessors[reachable].astype(np.uint32)
        else:
            path_links = None
        self._set_path_links(path_links, output)

        if self._labels is not None:
            # predecessors with their original labels
            predecessors[reachable] = self._labels[predecessors[reachable]].astype(
                np.int32
            )
        return self._vertex_result(predecessors, "predecessor", self._sentinel, output)


def _check_sentinel(sentinel) -> None:
    """Checks the sentinel value of the algorithms returning predecessors."""
    if not isinstance(sentinel, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"sentinel must be an integer, got {type(sentinel).__name__}")
    if sentinel >= 0:
        raise ValueError(f"sentinel must be negative, got {sentinel}")
    if sentinel < np.iinfo(np.int32).min or sentinel > np.iinfo(np.int32).max:
        raise ValueError(
            f"sentinel must fit in int32 range [{np.iinfo(np.int32).min}, "
            f"{np.iinfo(np.int32).max}], got {sentinel}"
        )


class DFS(_VertexGraph):
    """Depth-first search in a directed graph.

    The search is iterative, so it is not limited by the depth of the call
    stack. The edge weights are not used. The graph is built once from a
    dataframe of edges. The `run` method can then be called any number of times,
    from or to different vertices.

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
    orientation : {'out', 'in'}, optional (default='out')
        'out' to follow the edges from one vertex (the search tree gives a path
        from this vertex to each reachable vertex), 'in' to follow them backward
        to one vertex (a path from each vertex that can reach it).
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types and missing values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. Without it, the internal arrays have one element per
        integer between 0 and the largest vertex identifier. The arguments and
        the results of the methods use the original identifiers in both cases.
        Negative identifiers are only accepted with this option, and the results
        must then be labelled by vertex (`output='pandas'` or `'polars'`).
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.
    sentinel : int, optional (default=-9999)
        Value returned for the vertices without predecessor: the start vertex
        and the vertices that cannot be reached. It must be a negative int32,
        different from all the vertex identifiers.

    Notes
    -----
    When several edges have the same tail and head vertices, only one is kept.

    The visiting order is the one of the recursive depth-first search: from the
    current vertex, the search goes to its first neighbour that is not
    discovered yet, and backtracks when there is none. The neighbours of a
    vertex are taken in increasing order of their identifiers: the heads of its
    outgoing edges with the 'out' orientation, the tails of its incoming edges
    with 'in'. This is also the order of `scipy.sparse.csgraph.depth_first_order`
    on a matrix with sorted indices.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import DFS
    >>> edges = pd.DataFrame({"tail": [0, 0, 1], "head": [1, 2, 3]})
    >>> dfs = DFS(edges)
    >>> dfs.run(vertex_idx=0)
    array([-9999,     0,     0,     1], dtype=int32)
    >>> dfs.get_order()
    array([0, 1, 3, 2], dtype=uint32)
    """

    _algorithm_name = "DFS algorithm"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        orientation: str = "out",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
        sentinel: int = -9999,
    ) -> None:
        _check_sentinel(sentinel)
        self._sentinel = sentinel
        self._order: Optional[np.ndarray] = None

        if check_edges:
            self._check_edges(edges, tail, head)
        # Convert to standardized NumPy-backed pandas DataFrame, without weights
        edges = standardize_graph_dataframe(edges, tail, head)
        self._build_graph(edges, tail, head, None, orientation, permute, verbose)

        if self._labels is not None and np.any(self._labels == self._sentinel):
            raise ValueError(
                f"sentinel {self._sentinel} is also a vertex label: "
                "use another sentinel value"
            )

    @property
    def UNREACHABLE(self) -> int:  # pylint: disable=invalid-name
        """The sentinel value, returned for the vertices without predecessor."""
        return self._sentinel

    def run(
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_series: bool = False,
        output: Optional[str] = None,
    ) -> VertexResult:
        """Compute the depth-first search tree from or to a given vertex.

        The discovery order of the vertices is kept, see `get_order`.

        Parameters
        ----------
        vertex_idx : int
            The start vertex: the source vertex if the orientation is 'out', the
            target vertex if it is 'in'.
        path_tracking : bool, optional (default=False)
            Whether to store the search tree, so that the paths can be retrieved
            with `get_path`.
        return_series : bool, optional (default=False)
            Same as `output='pandas'`.
        output : {'numpy', 'pandas', 'polars'}, optional (default='numpy')
            Type of the result. 'pandas' returns a Series named 'predecessor'
            indexed by vertex ('vertex_idx'), 'polars' a DataFrame with the
            columns 'vertex_idx' and 'predecessor'.

        Returns
        -------
        numpy.ndarray, pandas.Series or polars.DataFrame
            The predecessor ('out') or the successor ('in') of each vertex in the
            depth-first search tree, as int32. The array has one element per
            integer between 0 and the largest vertex identifier. The start vertex
            and the vertices that cannot be reached have the sentinel value. The
            Series and the DataFrame have one row per vertex.

        Raises
        ------
        ValueError
            If `vertex_idx` is not a vertex of the graph, or if the graph has
            negative vertex identifiers and the output is 'numpy'.
        """
        output = _resolve_output(output, return_series)
        self._check_output(output)
        vertex_new = self._internal_vertex(vertex_idx)

        dfs = dfs_csr if self._orientation == "out" else dfs_csc
        predecessors, order = dfs(
            self._indptr,
            self._indices,
            vertex_new,
            self._n_vertices,
            self._sentinel,
        )
        self._order = order
        reachable = predecessors != self._sentinel

        if path_tracking:
            # the vertices without predecessor are linked to themselves
            path_links = np.arange(self._n_vertices, dtype=np.uint32)
            path_links[reachable] = predecessors[reachable].astype(np.uint32)
        else:
            path_links = None
        self._set_path_links(path_links, output)

        if self._labels is not None:
            # predecessors with their original labels
            predecessors[reachable] = self._labels[predecessors[reachable]].astype(
                np.int32
            )
        return self._vertex_result(predecessors, "predecessor", self._sentinel, output)

    def get_path(self, vertex_idx: int) -> Optional[np.ndarray]:
        """Return the path between a vertex and the vertex of the last run in the
        depth-first search tree.

        The last call to `run` must have been made with `path_tracking=True`. This
        path is generally not a shortest one.

        Parameters
        ----------
        vertex_idx : int
            The other end of the path: the target vertex if the orientation is
            'out', the source vertex if it is 'in'.

        Returns
        -------
        numpy.ndarray or None
            The vertices of the path, starting with `vertex_idx` and ending with the
            vertex given to `run`: in reverse order with the 'out' orientation, in
            travel order with 'in'. If there is no path, the array only contains
            `vertex_idx`. None is returned, with a warning, if no path was tracked.
        """
        return super().get_path(vertex_idx)

    def get_order(self) -> Optional[np.ndarray]:
        """Return the vertices reached by the last run, in discovery order.

        Returns
        -------
        numpy.ndarray or None
            The vertices reached from ('out') or reaching ('in') the vertex given
            to `run`, in the order in which the search discovered them (preorder),
            starting with this vertex. None is returned, with a warning, if `run`
            was not called.
        """
        if self._order is None:
            warnings.warn(
                f"Current {type(self).__name__} instance has no discovery order: "
                f"run the {self._algorithm_name} first",
                UserWarning,
            )
            return None
        if self._labels is None:
            return self._order.copy()
        vertices = self._labels[self._order]
        if self._labels[0] >= 0:
            vertices = vertices.astype(np.uint32)
        return vertices


# Initial capacity of the priority queues of the single pair algorithms: they
# grow when needed
_PAIR_HEAP_LENGTH = 1024

# Relative margin taken on the factor of the coordinate heuristic, computed
# from the edges, so that the rounding errors cannot make it overestimate
_HEURISTIC_FACTOR_MARGIN = 1.0e-9

# The heuristic values are clipped to this value: an infinite value (the target
# cannot be reached from the vertex) or one close to the largest float64 would
# make the keys overflow. Lowering a heuristic keeps it admissible and
# consistent.
_HEURISTIC_MAX = 1.0e300


def _distances(x0, y0, x1, y1, metric):
    """Distances between points, with the same formulas as the A* routine: the
    Euclidean distance, or the great-circle distance in meters (coordinates in
    radians)."""
    if metric == "euclidean":
        return np.hypot(x1 - x0, y1 - y0)
    sin_dx = np.sin(0.5 * (x1 - x0))
    sin_dy = np.sin(0.5 * (y1 - y0))
    a = sin_dy * sin_dy + np.cos(y1) * np.cos(y0) * sin_dx * sin_dx
    return 2.0 * astar_core.EARTH_RADIUS * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


class _PairGraph(_Graph):
    """Common part of the shortest path algorithms between two vertices:
    `AStar` and `BidirectionalDijkstra`.

    The graph is built in CSR format (orientation 'out'). The per-vertex arrays
    of the Cython routine are kept in a workspace that is reused by the
    successive runs, so that a run costs time in proportion to the part of the
    graph it explores, not to the vertex count. A run that finds the workspace
    in use by another thread gets a temporary one.
    """

    _algorithm_name = "shortest path algorithm"
    _weight_name: str
    _path: Optional[np.ndarray]
    _path_tracked: bool
    _workspace: Any
    _workspace_lock: Any
    _layout: Optional[tuple]

    def _init_pair(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, edges, tail, head, weight, check_edges, permute, verbose
    ) -> None:
        """Check the edges and build the graph."""
        if check_edges:
            self._check_edges(edges, tail, head, weight, nonnegative=True)
        edges = standardize_graph_dataframe(edges, tail, head, weight)
        # negative and NaN weights are rejected whatever the value of
        # check_edges (the min is NaN if any weight is)
        if len(edges) > 0 and not np.asarray(edges[weight]).min() >= 0.0:
            raise ValueError(f"edges['{weight}'] should be nonnegative")
        self._weight_name = weight
        self._build_graph(edges, tail, head, weight, "out", permute, verbose)
        self._path = None
        self._path_tracked = False
        self._workspace = None
        self._workspace_lock = threading.Lock()
        self._layout = None

    def _new_workspace(self) -> Any:
        """Return a new workspace for the Cython routine."""
        raise NotImplementedError

    @contextmanager
    def _borrowed_workspace(self):
        """The workspace of the instance, or a temporary one if it is in use."""
        if self._workspace_lock.acquire(blocking=False):
            try:
                if self._workspace is None:
                    self._workspace = self._new_workspace()
                yield self._workspace
            finally:
                self._workspace_lock.release()
        else:
            yield self._new_workspace()

    def _pair_vertices(self, source: int, target: int):
        """Internal indices of the source and the target vertices."""
        return self._internal_vertex(source), self._internal_vertex(target)

    def _vertex_layout(self):
        """The vertex labels (`get_vertices`), and, if some internal indices are
        not vertices (`permute=False` and gaps between the labels), a boolean
        mask of these indices, None otherwise. Computed once."""
        if self._layout is None:
            vertices = self.get_vertices()
            non_vertices = None
            if self._labels is None and len(vertices) < self._n_vertices:
                non_vertices = np.ones(self._n_vertices, dtype=bool)
                non_vertices[vertices] = False
            self._layout = (vertices, non_vertices)
        return self._layout

    def _scatter(self, values: np.ndarray) -> np.ndarray:
        """Return values given in the order of `get_vertices`, for each internal
        index: the indices that are not vertices get a zero."""
        vertices, non_vertices = self._vertex_layout()
        if non_vertices is None:
            # the internal indices are the positions in get_vertices
            return np.ascontiguousarray(values)
        result = np.zeros((self._n_vertices,) + values.shape[1:], dtype=values.dtype)
        result[vertices] = values
        return result

    def _set_pair_path(self, path: Optional[np.ndarray], tracked: bool) -> None:
        """Store the path of the last run, given with the internal indices."""
        self._path_tracked = tracked
        if path is None:
            self._path = None
            return
        labels = self._labels_of(path)
        if self._labels is not None and not self._negative_labels:
            labels = labels.astype(np.uint32)
        self._path = labels

    @staticmethod
    def _pair_length(length: float, return_inf: bool) -> float:
        """Path length returned by `run`."""
        if return_inf and length == DTYPE_INF_PY:
            return float(np.inf)
        return float(length)

    def get_path(self) -> Optional[np.ndarray]:
        """Return the shortest path found by the last run.

        The last call to `run` must have been made with `path_tracking=True`.

        Returns
        -------
        numpy.ndarray or None
            The vertices of the path, in travel order: from the source to the
            target, both included. None if the target cannot be reached from the
            source, and None with a warning if no path was tracked.
        """
        if not self._path_tracked:
            warnings.warn(
                f"Current {type(self).__name__} instance has no path attribute: "
                "make sure path_tracking is set to True, and run the "
                f"{self._algorithm_name}",
                UserWarning,
            )
            return None
        return None if self._path is None else self._path.copy()


class AStar(_PairGraph):
    """A* algorithm, for the shortest path between two vertices of a directed
    graph with non-negative edge weights.

    A* is Dijkstra's algorithm guided by a heuristic: a lower bound of the path
    length from each vertex to the target. The vertices are scanned in the
    order of their path length from the source plus their heuristic, and the
    search stops as soon as the target is reached. The closer the heuristic is
    to the actual path lengths, the fewer vertices are scanned. The heuristic
    is given to `run`, or computed from vertex coordinates (see
    `set_coordinates`).

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
    weight : str, optional (default='weight')
        Name of the column with the edge weights. Negative and NaN weights are
        rejected with a ValueError, whatever the value of `check_edges`.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and non-finite values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. The arguments and the results of the methods use the
        original identifiers in both cases. Negative identifiers are only
        accepted with this option, and the heuristic must then be given as a
        pandas Series.
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    Let d(v) be the length of the shortest path from vertex v to the target,
    and h(v) the heuristic.

    - The heuristic is *admissible* if h(v) <= d(v) for every vertex v. The
      path length returned by `run` is then the shortest one.
    - It is *consistent* if h(u) <= w(u, v) + h(v) for every edge (u, v) of
      weight w(u, v), and h(target) = 0. A consistent heuristic is admissible,
      and each vertex is then scanned at most once.
    - An admissible but inconsistent heuristic still gives the shortest path,
      but some vertices may be scanned several times, which can be slow.
    - With a heuristic that is not admissible, the result is the length of a
      path from the source to the target, which may not be the shortest one.
      This cannot be detected, and no error is raised.

    The zero heuristic is consistent: A* is then Dijkstra's algorithm stopped
    at the target. When several edges have the same tail and head vertices,
    only the one with the smallest weight is kept.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from edsger.path import AStar
    >>> edges = pd.DataFrame(
    ...     {"tail": [0, 0, 1], "head": [1, 2, 2], "weight": [1.0, 4.0, 2.0]}
    ... )
    >>> astar = AStar(edges)
    >>> astar.run(source=0, target=2, heuristic=np.array([2.5, 2.0, 0.0]))
    3.0
    """

    _coordinates: Optional[tuple]

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: str = "weight",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        self._init_pair(edges, tail, head, weight, check_edges, permute, verbose)
        self._coordinates = None

    def _new_workspace(self) -> Any:
        return astar_core.Workspace(max(self._n_vertices, 1))

    def set_coordinates(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        coordinates: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        metric: str = "euclidean",
        factor: Optional[float] = None,
    ) -> float:
        """Set the vertex coordinates, from which the heuristic is computed when
        `run` is called without one.

        The heuristic of a vertex is then `factor` times its distance to the
        target. It is only computed for the vertices reached by the search, so
        that a run does not cost time in proportion to the vertex count.

        Parameters
        ----------
        coordinates : pandas.DataFrame
            The coordinates of the vertices, indexed by vertex identifier. Every
            vertex of the graph must have finite coordinates, the other rows are
            ignored.
        x : str, optional (default='x')
            Name of the column with the first coordinate: the longitude in
            degrees for the 'haversine' metric.
        y : str, optional (default='y')
            Name of the column with the second coordinate: the latitude in
            degrees for the 'haversine' metric.
        metric : {'euclidean', 'haversine'}, optional (default='euclidean')
            The distance between the vertices: Euclidean distance, or
            great-circle distance in meters on a spherical Earth.
        factor : float, optional (default=None)
            The distance is multiplied by this factor, to be compared with the
            edge weights: for example the inverse of the largest speed if the
            weights are travel times. If None, the largest factor such that
            factor * distance(u, v) <= w(u, v) for every edge (u, v) is computed
            from the edges, minus a small margin for the rounding errors. This
            makes the heuristic consistent (the distance satisfies the triangle
            inequality), so the paths found are the shortest ones. A factor
            given here is used as is: if it is too large, the paths may not be
            the shortest ones.

        Returns
        -------
        float
            The factor used.

        Raises
        ------
        ValueError
            If a vertex has no coordinates or non-finite ones, if a latitude is
            not in [-90, 90] with the 'haversine' metric, or if the factor is not
            finite and non-negative.
        """
        if not isinstance(coordinates, pd.DataFrame):
            raise TypeError("coordinates should be a pandas DataFrame")
        if metric not in astar_core.METRICS:
            raise ValueError(
                f"metric should be one of {astar_core.METRICS}, got {metric!r}"
            )
        for col in (x, y):
            if col not in coordinates:
                raise KeyError(f"coordinate column '{col}' not found in coordinates")
        if not coordinates.index.is_unique:
            raise ValueError("the index of the coordinates should be unique")

        vertices, _ = self._vertex_layout()
        xy = coordinates[[x, y]].reindex(vertices).to_numpy(dtype=DTYPE_PY)
        if not np.isfinite(xy).all():
            raise ValueError("every vertex of the graph should have finite coordinates")
        if metric == "haversine":
            if np.abs(xy[:, 1]).max(initial=0.0) > 90.0:
                raise ValueError("the latitudes should be in [-90, 90] degrees")
            xy = np.radians(xy)
        xy = self._scatter(xy)
        x_int = np.ascontiguousarray(xy[:, 0])
        y_int = np.ascontiguousarray(xy[:, 1])

        if factor is None:
            factor = 0.0
            tails = np.asarray(self._edges[self._tail_name])
            heads = np.asarray(self._edges[self._head_name])
            weights = np.asarray(self._edges[self._weight_name], dtype=DTYPE_PY)
            ratio_min = np.inf
            # by chunks of edges, to bound the memory
            chunk = 1 << 22
            for start in range(0, len(tails), chunk):
                t = tails[start : start + chunk]
                h = heads[start : start + chunk]
                dist = _distances(x_int[t], y_int[t], x_int[h], y_int[h], metric)
                positive = dist > 0.0
                if positive.any():
                    w = weights[start : start + chunk]
                    ratio_min = min(ratio_min, (w[positive] / dist[positive]).min())
            if np.isfinite(ratio_min):
                factor = float(ratio_min) * (1.0 - _HEURISTIC_FACTOR_MARGIN)
        else:
            if not (np.isfinite(factor) and factor >= 0.0):
                raise ValueError(
                    f"the factor should be finite and non-negative, got {factor}"
                )
            factor = float(factor)

        self._coordinates = (x_int, y_int, metric, factor)
        return factor

    def _internal_heuristic(self, heuristic) -> np.ndarray:
        """Check the heuristic and return it for the internal vertex indices."""
        if isinstance(heuristic, pd.Series):
            if not heuristic.index.is_unique:
                raise ValueError("the index of the heuristic should be unique")
            vertices, _ = self._vertex_layout()
            values = heuristic.reindex(vertices).to_numpy(dtype=DTYPE_PY)
            if np.isnan(values).any():
                raise ValueError(
                    "the heuristic should have a value, not NaN, for every vertex"
                )
            values = self._scatter(values)
        else:
            values = np.asarray(heuristic)
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError("the heuristic should be a numeric array")
            if values.ndim != 1 or len(values) != self._n_vertices_init:
                raise ValueError(
                    "the heuristic array should have one element per integer "
                    "between 0 and the largest vertex identifier: "
                    f"{self._n_vertices_init}, got shape {values.shape}"
                )
            if self._negative_labels:
                raise ValueError(
                    "the graph has negative vertex labels: the heuristic should be "
                    "a pandas Series indexed by vertex"
                )
            values = values.astype(DTYPE_PY, copy=False)
            if self._labels is not None:
                values = values[self._labels]
            else:
                _, non_vertices = self._vertex_layout()
                if non_vertices is not None:
                    # the integers that are not vertices are ignored
                    values = np.where(non_vertices, 0.0, values)
            values = np.ascontiguousarray(values)
        # NaN values make the min NaN: compared as Python floats
        value_min = float(values.min())
        if not value_min >= 0.0:
            if value_min != value_min:
                raise ValueError("the heuristic should not contain NaN values")
            raise ValueError("the heuristic should be non-negative")
        if float(values.max()) > _HEURISTIC_MAX:
            values = np.minimum(values, _HEURISTIC_MAX)
        return values

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        source: int,
        target: int,
        heuristic: Union[np.ndarray, pd.Series, None] = None,
        path_tracking: bool = False,
        return_inf: bool = True,
    ) -> float:
        """Compute the shortest path length from a source to a target vertex.

        Parameters
        ----------
        source : int
            The source vertex.
        target : int
            The target vertex.
        heuristic : numpy.ndarray or pandas.Series, optional (default=None)
            A lower bound of the path length from each vertex to the target,
            non-negative (see the Notes of the class), infinite if the target
            cannot be reached from the vertex. A NumPy array
            has one element per integer between 0 and the largest vertex
            identifier, like the NumPy results of `Dijkstra.run`: the elements
            that are not vertices are ignored. A pandas Series is indexed by
            vertex identifier and must have a value for every vertex; it is
            required if the graph has negative vertex identifiers. If None, the
            heuristic is computed from the coordinates given to
            `set_coordinates`, or is zero if there are none. The heuristic
            depends on the target: it must be given again for another target.
        path_tracking : bool, optional (default=False)
            Whether to store the shortest path, so that it can be retrieved with
            `get_path`.
        return_inf : bool, optional (default=True)
            Whether to return `numpy.inf` if the target cannot be reached. If
            False, the largest float64 value is returned instead.

        Returns
        -------
        float
            The length of the shortest path from the source to the target.

        Raises
        ------
        ValueError
            If the source or the target is not a vertex of the graph, or if the
            heuristic does not have the right length, has missing, NaN or
            negative values.
        """
        source_new, target_new = self._pair_vertices(source, target)
        values = None if heuristic is None else self._internal_heuristic(heuristic)
        assert self._edge_weights is not None
        graph = (self._indptr, self._indices, self._edge_weights)
        path = None
        with self._borrowed_workspace() as workspace:
            if values is not None or self._coordinates is None:
                length = astar_core.compute_astar(
                    *graph,
                    source_new,
                    target_new,
                    self._n_vertices,
                    values,
                    path_tracking,
                    workspace,
                    _PAIR_HEAP_LENGTH,
                )
            else:
                x, y, metric, factor = self._coordinates
                length = astar_core.compute_astar_coordinates(
                    *graph,
                    source_new,
                    target_new,
                    self._n_vertices,
                    x,
                    y,
                    metric,
                    factor,
                    path_tracking,
                    workspace,
                    _PAIR_HEAP_LENGTH,
                )
            if path_tracking and length != DTYPE_INF_PY:
                # from the target back to the source
                path = compute_path(workspace.get_predecessor(), target_new)[::-1]
        self._set_pair_path(path, path_tracking)
        return self._pair_length(length, return_inf)


class BidirectionalDijkstra(_PairGraph):
    """Bidirectional Dijkstra's algorithm, for the shortest path between two
    vertices of a directed graph with non-negative edge weights.

    Two searches are run alternately: from the source on the outgoing edges,
    and from the target on the incoming edges. They stop as soon as no path
    through the vertices that are left to scan can be shorter than the best
    path found where the two searches meet.

    The graph is stored in both CSR and CSC formats, which takes about twice
    the memory of a `Dijkstra` instance.

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
    weight : str, optional (default='weight')
        Name of the column with the edge weights. Negative and NaN weights are
        rejected with a ValueError, whatever the value of `check_edges`.
    check_edges : bool, optional (default=False)
        Whether to check the edges before building the graph: column names,
        data types, missing and non-finite values.
    permute : bool, optional (default=False)
        Whether to renumber the vertices internally, with contiguous indices
        starting from 0. The arguments and the results of the methods use the
        original identifiers in both cases. Negative identifiers are only
        accepted with this option.
    verbose : bool, optional (default=False)
        Whether to print the number of parallel edges that were removed.

    Notes
    -----
    The path length is the sum of the lengths of the two half paths, from the
    source to the meeting vertex and from the meeting vertex to the target. It
    is not summed in the same order as by `Dijkstra.run`, so the two may differ
    in the last bits with non-integer weights. When several edges have the
    same tail and head vertices, only the one with the smallest weight is kept.

    Examples
    --------
    >>> import pandas as pd
    >>> from edsger.path import BidirectionalDijkstra
    >>> edges = pd.DataFrame(
    ...     {"tail": [0, 0, 1], "head": [1, 2, 2], "weight": [1.0, 4.0, 2.0]}
    ... )
    >>> bd = BidirectionalDijkstra(edges)
    >>> bd.run(source=0, target=2, path_tracking=True)
    3.0
    >>> bd.get_path()
    array([0, 1, 2], dtype=uint32)
    """

    _csc_indptr: np.ndarray
    _csc_indices: np.ndarray
    _csc_weights: np.ndarray

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        edges: pd.DataFrame,
        tail: str = "tail",
        head: str = "head",
        weight: str = "weight",
        check_edges: bool = False,
        permute: bool = False,
        verbose: bool = False,
    ) -> None:
        self._init_pair(edges, tail, head, weight, check_edges, permute, verbose)
        # incoming edges, for the backward search
        indptr, indices, data = convert_graph_to_csc_float64(
            self._edges, tail, head, weight, self._n_vertices
        )
        self._csc_indptr = indptr.astype(np.uint32)
        self._csc_indices = indices.astype(np.uint32)
        self._csc_weights = data.astype(DTYPE_PY)

    def _new_workspace(self) -> Any:
        return bidir_core.Workspace(max(self._n_vertices, 1))

    def run(
        self,
        source: int,
        target: int,
        path_tracking: bool = False,
        return_inf: bool = True,
    ) -> float:
        """Compute the shortest path length from a source to a target vertex.

        Parameters
        ----------
        source : int
            The source vertex.
        target : int
            The target vertex.
        path_tracking : bool, optional (default=False)
            Whether to store the shortest path, so that it can be retrieved with
            `get_path`.
        return_inf : bool, optional (default=True)
            Whether to return `numpy.inf` if the target cannot be reached. If
            False, the largest float64 value is returned instead.

        Returns
        -------
        float
            The length of the shortest path from the source to the target.

        Raises
        ------
        ValueError
            If the source or the target is not a vertex of the graph.
        """
        source_new, target_new = self._pair_vertices(source, target)
        assert self._edge_weights is not None
        path = None
        with self._borrowed_workspace() as workspace:
            length, meet = bidir_core.compute_bidirectional(
                self._indptr,
                self._indices,
                self._edge_weights,
                self._csc_indptr,
                self._csc_indices,
                self._csc_weights,
                source_new,
                target_new,
                self._n_vertices,
                path_tracking,
                workspace,
                _PAIR_HEAP_LENGTH,
            )
            if path_tracking and meet >= 0:
                predecessor, successor = workspace.get_path_links()
                # source -> meeting vertex, then meeting vertex -> target
                forward = compute_path(predecessor, meet)[::-1]
                backward = compute_path(successor, meet)
                path = np.concatenate((forward, backward[1:]))
        self._set_pair_path(path, path_tracking)
        return self._pair_length(length, return_inf)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
