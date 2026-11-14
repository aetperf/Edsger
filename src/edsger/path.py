"""
Path-related methods.
"""

import numbers
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


class _VertexGraph:
    """Common part of the algorithms run from or to one vertex of a graph built
    from a dataframe of edges: `Dijkstra`, `BellmanFord` and `BFS`.

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
    def orientation(self) -> str:
        """Orientation of the algorithm, either 'out' or 'in'."""
        return self._orientation

    @property
    def permute(self) -> bool:
        """Whether the vertices are renumbered internally."""
        return self._permute

    @property
    def path_links(self) -> Optional[np.ndarray]:
        """Path tree of the last run, or None if the paths were not tracked. This
        is the predecessor ('out') or the successor ('in') of each vertex. The
        vertex given to `run` and the vertices that cannot be reached are linked
        to themselves.
        """
        return self._path_links

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

    def _internal_vertices(self, vertices: np.ndarray, what: str) -> np.ndarray:
        """Return the internal indices of an array of vertex labels, as uint32."""
        if self._labels is not None:
            positions = np.searchsorted(self._labels, vertices)
            found = positions < len(self._labels)
            found[found] = self._labels[positions[found]] == vertices[found]
            if not found.all():
                raise ValueError(f"{what} {vertices[~found][0]} not found in graph")
            return positions.astype(np.uint32)
        if np.any(vertices >= self._n_vertices) or np.any(vertices < 0):
            raise ValueError(f"{what}s contain invalid vertex indices")
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
        return self._internal_vertices(nodes, "termination node")

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
        if not v >= 0.0:
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


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
