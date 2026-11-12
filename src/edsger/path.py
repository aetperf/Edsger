"""
Path-related methods.
"""

import numbers
from typing import Optional, Union, List, Any, cast
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


def _compute_path_with_permutation(path_links, permutation, vertex_idx):
    """
    Compute a path on a graph whose vertices have been re-indexed.

    The path is followed on the contiguous vertex indices and returned with the original
    vertex labels: neither the time nor the memory depends on the values of the labels,
    which may be negative.

    Parameters
    ----------
    path_links : numpy.ndarray
        Predecessors or successors, for the re-indexed vertices.
    permutation : pandas.DataFrame
        Permutation of the vertices, sorted by new vertex index.
    vertex_idx : int
        Original label of the source or target vertex.

    Returns
    -------
    path_vertices : numpy.ndarray
        Original labels of the vertices of the path, of np.uint32 type unless some labels
        are negative.
    """
    # the new indices are the ranks of the sorted labels
    labels = permutation["vert_idx_old"].to_numpy()
    position = int(np.searchsorted(labels, vertex_idx))
    if position == len(labels) or labels[position] != vertex_idx:
        raise ValueError(f"vertex {vertex_idx} not found in graph")
    path_vertices = labels[compute_path(path_links, position)]
    if labels[0] >= 0:
        path_vertices = path_vertices.astype(np.uint32)
    return path_vertices


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


class Dijkstra:
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
        must then be returned as Series (`return_series=True`).
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

    def __init__(
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

        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        # Convert to standardized NumPy-backed pandas DataFrame
        self._edges = standardize_graph_dataframe(edges, tail, head, weight)
        self._tail_name = tail
        self._head_name = head
        self._n_edges = len(self._edges)
        # Dijkstra's algorithm is not valid with negative weights: they are
        # rejected whatever the value of check_edges, and so are NaN weights
        # (the min is NaN if any weight is)
        if self._n_edges > 0 and not np.asarray(self._edges[weight]).min() >= 0.0:
            raise ValueError(f"edges['{weight}'] should be nonnegative")
        self._verbose = verbose

        # preprocess edges to handle parallel edges
        self._preprocess_edges(tail, head, weight)

        # reindex the vertices
        self._permute = permute
        # negative vertex labels are only possible with permute=True
        self._negative_labels = False
        if len(self._edges) == 0:
            # Handle empty graphs
            self._permutation = None
            self._n_vertices = 0
            self.__n_vertices_init = 0
        elif self._permute:
            self._negative_labels = bool(
                self._edges[[tail, head]].min(axis=0).min() < 0
            )
            self.__n_vertices_init = self._edges[[tail, head]].max(axis=0).max() + 1
            self._permutation = self._permute_graph(tail, head)
            self._n_vertices = len(self._permutation)
        else:
            self._permutation = None
            self._n_vertices = self._edges[[tail, head]].max(axis=0).max() + 1
            self.__n_vertices_init = self._n_vertices

        # convert to CSR/CSC:
        # __indices: numpy.ndarray
        #     1D array containing the indices of the indices of the forward or reverse star of
        #     the graph in compressed format.
        # __indptr: numpy.ndarray
        #     1D array containing the indices of the pointer of the forward or reverse star of
        #     the graph in compressed format.
        # __edge_weights: numpy.ndarray
        #     1D array containing the weights of the edges in the graph.

        self._check_orientation(orientation)
        self._orientation = orientation
        if self._orientation == "out":
            fs_indptr, fs_indices, fs_data = convert_graph_to_csr_float64(
                self._edges, tail, head, weight, self._n_vertices
            )
            self.__indices = fs_indices.astype(np.uint32)
            self.__indptr = fs_indptr.astype(np.uint32)
            self.__edge_weights = fs_data.astype(DTYPE_PY)
        else:
            rs_indptr, rs_indices, rs_data = convert_graph_to_csc_float64(
                self._edges, tail, head, weight, self._n_vertices
            )
            self.__indices = rs_indices.astype(np.uint32)
            self.__indptr = rs_indptr.astype(np.uint32)
            self.__edge_weights = rs_data.astype(DTYPE_PY)

        self._path_links = None
        self._path_links_permuted = None

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
    def lazy(self) -> bool:
        """Whether the variant without decrease-key operation is used."""
        return self._lazy

    @property
    def path_links(self) -> Optional[np.ndarray]:
        """Path tree of the last run, or None if the paths were not tracked. This
        is the predecessor ('out') or the successor ('in') of each vertex. The
        vertex given to `run` and the vertices that cannot be reached are linked
        to themselves.
        """
        return self._path_links

    def _preprocess_edges(self, tail, head, weight):
        """
        Preprocess edges to handle parallel edges by keeping only the minimum weight edge
        between any pair of vertices.

        Parameters
        ----------
        tail : str
            The column name for tail vertices
        head : str
            The column name for head vertices
        weight : str
            The column name for edge weights
        """
        original_count = len(self._edges)
        self._edges = self._edges.groupby([tail, head], as_index=False)[weight].min()
        final_count = len(self._edges)

        if original_count > final_count:
            parallel_edges_removed = original_count - final_count
            if self._verbose:
                print(
                    f"Automatically removed {parallel_edges_removed} parallel edge(s). "
                    f"For each pair of vertices, kept the edge with minimum weight."
                )

        self._n_edges = len(self._edges)

    def _check_edges(self, edges, tail, head, weight):
        """Checks if the edges DataFrame is well-formed. If not, raises an appropriate error."""
        edges = _edges_as_pandas(edges, [tail, head, weight])

        if tail not in edges:
            raise KeyError(
                f"edge tail column '{tail}' not found in graph edges dataframe"
            )

        if head not in edges:
            raise KeyError(
                f"edge head column '{head}' not found in graph edges dataframe"
            )

        if weight not in edges:
            raise KeyError(
                f"edge weight column '{weight}' not found in graph edges dataframe"
            )

        if edges[[tail, head, weight]].isnull().to_numpy().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{tail}, {head}, {weight}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"edges['{col}'] should be of integer type")

        if not pd.api.types.is_numeric_dtype(edges[weight].dtype):
            raise TypeError(f"edges['{weight}'] should be of numeric type")

        if edges[weight].min() < 0.0:
            raise ValueError(f"edges['{weight}'] should be nonnegative")

        if not np.isfinite(edges[weight]).all():
            raise ValueError(f"edges['{weight}'] should be finite")

    def _permute_graph(self, tail, head):
        """Permute the IDs of the nodes to start from 0 and be contiguous.
        Returns a DataFrame with the permuted IDs."""

        permutation = pd.DataFrame(
            data={
                "vert_idx": np.union1d(
                    np.asarray(self._edges[tail]), np.asarray(self._edges[head])
                )
            }
        )
        permutation["vert_idx_new"] = permutation.index
        permutation.index.name = "index"

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=tail,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([tail, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": tail})

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([head, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": head})

        permutation = permutation.rename(columns={"vert_idx": "vert_idx_old"})
        permutation = permutation.reset_index(drop=True)
        permutation = permutation.sort_values(by="vert_idx_new")

        permutation.index.name = "index"
        self._edges.index.name = "index"

        return permutation

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' or 'out'")

    def run(
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_inf: bool = True,
        return_series: bool = False,
        heap_length_ratio: float = 1.0,
        termination_nodes: Optional[List[int]] = None,
    ) -> Union[np.ndarray, pd.Series]:
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
            Whether to return a pandas Series indexed by vertex instead of a NumPy
            array.
        heap_length_ratio : float, optional (default=1.0)
            Initial capacity of the priority queue, as a fraction of the number of
            vertices, in (0, 1]. The queue grows when it is full, so this only
            changes the size of the first memory allocation.
        termination_nodes : array-like of int, optional (default=None)
            Vertices at which to stop. The algorithm ends as soon as the path
            lengths of all these vertices are known, and only their path lengths
            are returned, in the given order.

        Returns
        -------
        numpy.ndarray or pandas.Series
            The path lengths, as float64. Without `termination_nodes`, the array
            has one element per integer between 0 and the largest vertex
            identifier: element i is the length of the shortest path from the
            source to vertex i ('out'), or from vertex i to the target ('in').
            With `termination_nodes`, it has one element per termination node. If
            `return_series` is True, the result is a Series named 'path_length'
            with the vertex identifiers as index.

        Raises
        ------
        ValueError
            If `vertex_idx` or a termination node is not a vertex of the graph, or
            if the graph has negative vertex identifiers and `return_series` is
            False.
        """
        # validate the input arguments - type checking handled by static typing
        if self._negative_labels and not return_series:
            raise ValueError(
                "the graph has negative vertex labels: the result cannot be an "
                "array indexed by vertex, use return_series=True"
            )
        if self._permute and self._permutation is not None:
            if vertex_idx not in self._permutation.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._permutation.loc[
                self._permutation.vert_idx_old == vertex_idx, "vert_idx_new"
            ].iloc[0]
        else:
            if vertex_idx < 0:
                raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be positive")
            if vertex_idx >= self._n_vertices:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = vertex_idx
        # Type checking is now handled by static typing
        if not isinstance(heap_length_ratio, float):
            raise TypeError(
                f"argument 'heap_length_ratio=f{heap_length_ratio}' must be of float type"
            )

        heap_length_ratio = np.amin([heap_length_ratio, 1.0])
        if heap_length_ratio <= 0.0:
            raise ValueError(
                f"argument 'heap_length_ratio={heap_length_ratio}' must be strictly positive "
            )
        heap_length = int(np.rint(heap_length_ratio * self._n_vertices))

        # validate and process termination_nodes
        termination_nodes_array = None
        if termination_nodes is not None:
            try:
                termination_nodes_array = np.array(termination_nodes, dtype=np.int64)
            except (ValueError, TypeError, OverflowError) as exc:
                raise TypeError(
                    "argument 'termination_nodes' must be array-like of integers"
                ) from exc

            if termination_nodes_array.ndim != 1:
                raise ValueError("argument 'termination_nodes' must be 1-dimensional")

            if len(termination_nodes_array) == 0:
                raise ValueError("argument 'termination_nodes' must not be empty")

            # handle vertex permutation if needed
            if self._permute and self._permutation is not None:
                termination_nodes_permuted = []
                for termination_node in termination_nodes_array:
                    if termination_node not in self._permutation.vert_idx_old.values:
                        raise ValueError(
                            f"termination node {termination_node} not found in graph"
                        )
                    termination_node_new = self._permutation.loc[
                        self._permutation.vert_idx_old == termination_node,
                        "vert_idx_new",
                    ].iloc[0]
                    termination_nodes_permuted.append(termination_node_new)
                termination_nodes_array = np.array(
                    termination_nodes_permuted, dtype=np.uint32
                )
            else:
                # validate that termination nodes exist
                if np.any(termination_nodes_array >= self._n_vertices) or np.any(
                    termination_nodes_array < 0
                ):
                    raise ValueError(
                        "termination_nodes contains invalid vertex indices"
                    )
                termination_nodes_array = termination_nodes_array.astype(np.uint32)

        # compute path length
        if not path_tracking:
            self._path_links = None
            if termination_nodes_array is not None:
                # use early termination algorithms
                if self._orientation == "in":
                    path_length_values = self._dijkstra.compute_stsp_early_termination(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        termination_nodes_array,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = self._dijkstra.compute_sssp_early_termination(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        termination_nodes_array,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
            else:
                # use standard algorithms
                if self._orientation == "in":
                    path_length_values = self._dijkstra.compute_stsp(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = self._dijkstra.compute_sssp(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
        else:
            self._path_links = np.arange(0, self._n_vertices, dtype=np.uint32)
            if termination_nodes_array is not None:
                # use early termination algorithms with path tracking
                if self._orientation == "in":
                    path_length_values = (
                        self._dijkstra.compute_stsp_w_path_early_termination(
                            self.__indptr,
                            self.__indices,
                            self.__edge_weights,
                            self._path_links,
                            termination_nodes_array,
                            vertex_new,
                            self._n_vertices,
                            heap_length,
                        )
                    )
                else:
                    path_length_values = (
                        self._dijkstra.compute_sssp_w_path_early_termination(
                            self.__indptr,
                            self.__indices,
                            self.__edge_weights,
                            self._path_links,
                            termination_nodes_array,
                            vertex_new,
                            self._n_vertices,
                            heap_length,
                        )
                    )
            else:
                # use standard algorithms with path tracking
                if self._orientation == "in":
                    path_length_values = self._dijkstra.compute_stsp_w_path(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = self._dijkstra.compute_sssp_w_path(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )

            if self._permute and self._permutation is not None:
                # permute back the path vertex indices
                self._path_links_permuted = self._path_links
                path_df = pd.DataFrame(
                    data={
                        "vertex_idx": np.arange(self._n_vertices),
                        "associated_idx": self._path_links,
                    }
                )
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="vertex_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["vertex_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "vertex_idx"})
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["associated_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "associated_idx"})

                if return_series:
                    path_df = path_df.set_index("vertex_idx")
                    self._path_links = path_df.associated_idx.astype(
                        np.uint32
                    ).to_numpy()
                else:
                    self._path_links = np.arange(
                        self.__n_vertices_init, dtype=np.uint32
                    )
                    self._path_links[path_df.vertex_idx.values] = (
                        path_df.associated_idx.values
                    )

        # deal with infinity, in place: the array has just been allocated by the
        # Dijkstra routine, and a copy of it is costly on large graphs
        if return_inf:
            path_length_values[path_length_values == DTYPE_INF_PY] = np.inf

        # reorder path lengths
        if return_series:
            if (
                self._permute
                and termination_nodes_array is None
                and self._permutation is not None
            ):
                self._permutation["path_length"] = path_length_values
                path_lengths_df = cast(
                    pd.DataFrame,
                    self._permutation[["vert_idx_old", "path_length"]].copy(),
                )
                path_lengths_df = path_lengths_df.sort_values(by="vert_idx_old")
                path_lengths_df = path_lengths_df.set_index("vert_idx_old", drop=True)
                path_lengths_df.index.name = "vertex_idx"
                path_lengths_series = path_lengths_df.path_length
            else:
                path_lengths_series = pd.Series(path_length_values)
                if termination_nodes is not None:
                    # early termination: one path length per termination node,
                    # in the given order
                    path_lengths_series = path_lengths_series.set_axis(
                        np.asarray(termination_nodes)
                    )
                path_lengths_series.index.name = "vertex_idx"
                path_lengths_series.name = "path_length"

            return path_lengths_series

        # For early termination, return results directly (already in correct order)
        if termination_nodes_array is not None:
            return path_length_values

        if self._permute and self._permutation is not None:
            self._permutation["path_length"] = path_length_values
            if return_inf:
                path_length_values = np.inf * np.ones(self.__n_vertices_init)
            else:
                path_length_values = DTYPE_INF_PY * np.ones(self.__n_vertices_init)
            assert self._permutation is not None  # guaranteed by condition above
            path_length_values[self._permutation.vert_idx_old.values] = (
                self._permutation.path_length.values
            )

        return path_length_values

    def get_vertices(self) -> Any:
        """Return the vertices of the graph.

        Returns
        -------
        numpy.ndarray
            The sorted identifiers of the vertices that are the tail or the head of
            at least one edge.
        """
        if self._permute and self._permutation is not None:
            return np.asarray(self._permutation.vert_idx_old)
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
                "Current Dijkstra instance has not path attribute : \
                make sure path_tracking is set to True, and run the \
                shortest path algorithm",
                UserWarning,
            )
            return None
        if self._permute and self._permutation is not None:
            return _compute_path_with_permutation(
                self._path_links_permuted, self._permutation, vertex_idx
            )
        return compute_path(self._path_links, vertex_idx)


class BellmanFord:
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
        must then be returned as Series (`return_series=True`).
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

    def __init__(
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
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        # Convert to standardized NumPy-backed pandas DataFrame
        self._edges = standardize_graph_dataframe(edges, tail, head, weight)
        self._tail_name = tail
        self._head_name = head
        self._n_edges = len(self._edges)
        self._verbose = verbose

        # preprocess edges to handle parallel edges
        self._preprocess_edges(tail, head, weight)

        # reindex the vertices
        self._permute = permute
        # negative vertex labels are only possible with permute=True
        self._negative_labels = False
        if len(self._edges) == 0:
            # Handle empty graphs
            self._permutation = None
            self._n_vertices = 0
            self.__n_vertices_init = 0
        elif self._permute:
            self._negative_labels = bool(
                self._edges[[tail, head]].min(axis=0).min() < 0
            )
            self.__n_vertices_init = self._edges[[tail, head]].max(axis=0).max() + 1
            self._permutation = self._permute_graph(tail, head)
            self._n_vertices = len(self._permutation)
        else:
            self._permutation = None
            self._n_vertices = self._edges[[tail, head]].max(axis=0).max() + 1
            self.__n_vertices_init = self._n_vertices

        # convert to CSR/CSC:
        self._check_orientation(orientation)
        self._orientation = orientation
        if self._orientation == "out":
            fs_indptr, fs_indices, fs_data = convert_graph_to_csr_float64(
                self._edges, tail, head, weight, self._n_vertices
            )
            self.__indices = fs_indices.astype(np.uint32)
            self.__indptr = fs_indptr.astype(np.uint32)
            self.__edge_weights = fs_data.astype(DTYPE_PY)
        else:
            rs_indptr, rs_indices, rs_data = convert_graph_to_csc_float64(
                self._edges, tail, head, weight, self._n_vertices
            )
            self.__indices = rs_indices.astype(np.uint32)
            self.__indptr = rs_indptr.astype(np.uint32)
            self.__edge_weights = rs_data.astype(DTYPE_PY)

        # Check if graph has any negative weights (for optimization)
        self._has_negative_weights = np.any(self.__edge_weights < 0)

        self._path_links = None
        self._path_links_permuted = None
        self._has_negative_cycle = False

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

    def _preprocess_edges(self, tail, head, weight):
        """
        Preprocess edges to handle parallel edges by keeping only the minimum weight edge
        between any pair of vertices.

        Parameters
        ----------
        tail : str
            The column name for tail vertices
        head : str
            The column name for head vertices
        weight : str
            The column name for edge weights
        """
        original_count = len(self._edges)
        self._edges = self._edges.groupby([tail, head], as_index=False)[weight].min()
        final_count = len(self._edges)

        if original_count > final_count:
            parallel_edges_removed = original_count - final_count
            if self._verbose:
                print(
                    f"Automatically removed {parallel_edges_removed} parallel edge(s). "
                    f"For each pair of vertices, kept the edge with minimum weight."
                )

        self._n_edges = len(self._edges)

    def _check_edges(self, edges, tail, head, weight):
        """Checks if the edges DataFrame is well-formed. If not, raises an appropriate error."""
        edges = _edges_as_pandas(edges, [tail, head, weight])

        if tail not in edges:
            raise KeyError(
                f"edge tail column '{tail}' not found in graph edges dataframe"
            )

        if head not in edges:
            raise KeyError(
                f"edge head column '{head}' not found in graph edges dataframe"
            )

        if weight not in edges:
            raise KeyError(
                f"edge weight column '{weight}' not found in graph edges dataframe"
            )

        if edges[[tail, head, weight]].isnull().to_numpy().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{tail}, {head}, {weight}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"edges['{col}'] should be of integer type")

        if not pd.api.types.is_numeric_dtype(edges[weight].dtype):
            raise TypeError(f"edges['{weight}'] should be of numeric type")

        # Note: Unlike Dijkstra, we allow negative weights for Bellman-Ford
        if not np.isfinite(edges[weight]).all():
            raise ValueError(f"edges['{weight}'] should be finite")

    def _permute_graph(self, tail, head):
        """Permute the IDs of the nodes to start from 0 and be contiguous.
        Returns a DataFrame with the permuted IDs."""

        permutation = pd.DataFrame(
            data={
                "vert_idx": np.union1d(
                    np.asarray(self._edges[tail]), np.asarray(self._edges[head])
                )
            }
        )
        permutation["vert_idx_new"] = permutation.index
        permutation.index.name = "index"

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=tail,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([tail, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": tail})

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([head, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": head})

        permutation = permutation.rename(columns={"vert_idx": "vert_idx_old"})
        permutation = permutation.reset_index(drop=True)
        permutation = permutation.sort_values(by="vert_idx_new")

        permutation.index.name = "index"
        self._edges.index.name = "index"

        return permutation

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' or 'out'")

    def run(
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_inf: bool = True,
        return_series: bool = False,
        detect_negative_cycles: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
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
            Whether to return a pandas Series indexed by vertex instead of a NumPy
            array.
        detect_negative_cycles : bool, optional (default=True)
            Whether to look for a negative cycle once the path lengths are computed,
            and raise a ValueError if there is one. The graph is only searched if
            it has at least one negative weight.

        Returns
        -------
        numpy.ndarray or pandas.Series
            The path lengths, as float64. The array has one element per integer
            between 0 and the largest vertex identifier: element i is the length of
            the shortest path from the source to vertex i ('out'), or from vertex i
            to the target ('in'). If `return_series` is True, the result is a
            Series named 'path_length' with the vertex identifiers as index.

        Raises
        ------
        ValueError
            If `vertex_idx` is not a vertex of the graph, if a negative cycle is
            detected, or if the graph has negative vertex identifiers and
            `return_series` is False.
        """
        # validate the input arguments - type checking handled by static typing
        if self._negative_labels and not return_series:
            raise ValueError(
                "the graph has negative vertex labels: the result cannot be an "
                "array indexed by vertex, use return_series=True"
            )
        if self._permute and self._permutation is not None:
            if vertex_idx not in self._permutation.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._permutation.loc[
                self._permutation.vert_idx_old == vertex_idx, "vert_idx_new"
            ].iloc[0]
        else:
            if vertex_idx < 0:
                raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be positive")
            if vertex_idx >= self._n_vertices:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = vertex_idx
        # Type checking is now handled by static typing

        # compute path length
        if not path_tracking:
            self._path_links = None
            if self._orientation == "in":
                path_length_values = compute_bf_stsp(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    vertex_new,
                    self._n_vertices,
                )
            else:
                path_length_values = compute_bf_sssp(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    vertex_new,
                    self._n_vertices,
                )
        else:
            self._path_links = np.arange(0, self._n_vertices, dtype=np.uint32)
            if self._orientation == "in":
                path_length_values = compute_bf_stsp_w_path(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    self._path_links,
                    vertex_new,
                    self._n_vertices,
                )
            else:
                path_length_values = compute_bf_sssp_w_path(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    self._path_links,
                    vertex_new,
                    self._n_vertices,
                )

            if self._permute and self._permutation is not None:
                # permute back the path vertex indices
                self._path_links_permuted = self._path_links
                path_df = pd.DataFrame(
                    data={
                        "vertex_idx": np.arange(self._n_vertices),
                        "associated_idx": self._path_links,
                    }
                )
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="vertex_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["vertex_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "vertex_idx"})
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["associated_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "associated_idx"})

                if return_series:
                    path_df = path_df.set_index("vertex_idx")
                    self._path_links = path_df.associated_idx.astype(
                        np.uint32
                    ).to_numpy()
                else:
                    self._path_links = np.arange(
                        self.__n_vertices_init, dtype=np.uint32
                    )
                    self._path_links[path_df.vertex_idx.values] = (
                        path_df.associated_idx.values
                    )

        # detect negative cycles if requested (only if negative weights exist)
        if detect_negative_cycles and self._has_negative_weights:
            if self._orientation == "out":
                # CSR format - can use detect_negative_cycle directly
                self._has_negative_cycle = detect_negative_cycle(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    path_length_values,
                    self._n_vertices,
                )
            else:
                # CSC format - use CSC-specific negative cycle detection
                # Much more efficient than converting CSC→CSR
                self._has_negative_cycle = detect_negative_cycle_csc(
                    self.__indptr,
                    self.__indices,
                    self.__edge_weights,
                    path_length_values,
                    self._n_vertices,
                )

            if self._has_negative_cycle:
                raise ValueError("Negative cycle detected in the graph")

        # deal with infinity
        if return_inf:
            path_length_values = np.where(
                path_length_values == DTYPE_INF_PY, np.inf, path_length_values
            )

        # reorder path lengths
        if return_series:
            if self._permute and self._permutation is not None:
                path_df = pd.DataFrame(
                    data={"path_length": path_length_values[: self._n_vertices]}
                )
                path_df["vert_idx_new"] = path_df.index
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="vert_idx_new",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["vert_idx_new"], axis=1)
                path_df = path_df.set_index("vert_idx_old")
                path_lengths_series = path_df.path_length.astype(DTYPE_PY)
            else:
                path_lengths_series = pd.Series(
                    data=path_length_values[: self._n_vertices], dtype=DTYPE_PY
                )
                path_lengths_series.index = np.arange(self._n_vertices)
            path_lengths_series.index.name = None
            return path_lengths_series

        # No else needed - de-indent the code
        if self._permute and self._permutation is not None:
            path_df = pd.DataFrame(
                data={"path_length": path_length_values[: self._n_vertices]}
            )
            path_df["vert_idx_new"] = path_df.index
            path_df = pd.merge(
                path_df,
                self._permutation,
                left_on="vert_idx_new",
                right_on="vert_idx_new",
                how="left",
            )
            path_df = path_df.drop(["vert_idx_new"], axis=1)
            path_length_values = np.full(self.__n_vertices_init, DTYPE_INF_PY)
            path_length_values[path_df.vert_idx_old.values] = path_df.path_length.values
            if return_inf:
                path_length_values = np.where(
                    path_length_values == DTYPE_INF_PY, np.inf, path_length_values
                )
        return path_length_values

    def get_vertices(self) -> Any:
        """Return the vertices of the graph.

        Returns
        -------
        numpy.ndarray
            The sorted identifiers of the vertices that are the tail or the head of
            at least one edge.
        """
        if self._permute and self._permutation is not None:
            return np.asarray(self._permutation.vert_idx_old)
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
                "Current BellmanFord instance has not path attribute : \
                make sure path_tracking is set to True, and run the \
                shortest path algorithm",
                UserWarning,
            )
            return None
        if self._permute and self._permutation is not None:
            return _compute_path_with_permutation(
                self._path_links_permuted, self._permutation, vertex_idx
            )
        return compute_path(self._path_links, vertex_idx)

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


class BFS:
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
        must then be returned as Series (`return_series=True`).
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

    def __init__(
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

        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head)
        # Convert to standardized NumPy-backed pandas DataFrame
        # Note: BFS doesn't need weights, but standardize_graph_dataframe handles that
        self._edges = standardize_graph_dataframe(edges, tail, head)
        self._tail_name = tail
        self._head_name = head
        self._n_edges = len(self._edges)
        self._verbose = verbose

        # preprocess edges to handle parallel edges
        self._preprocess_edges(tail, head)

        # reindex the vertices
        self._permute = permute
        # negative vertex labels are only possible with permute=True
        self._negative_labels = False
        if len(self._edges) == 0:
            # Handle empty graphs
            self._permutation = None
            self._n_vertices = 0
            self.__n_vertices_init = 0
        elif self._permute:
            self._negative_labels = bool(
                self._edges[[tail, head]].min(axis=0).min() < 0
            )
            self.__n_vertices_init = self._edges[[tail, head]].max(axis=0).max() + 1
            self._permutation = self._permute_graph(tail, head)
            if (
                self._negative_labels
                and (self._permutation.vert_idx_old == self._sentinel).any()
            ):
                raise ValueError(
                    f"sentinel {self._sentinel} is also a vertex label: "
                    "use another sentinel value"
                )
            self._n_vertices = len(self._permutation)
        else:
            self._permutation = None
            self._n_vertices = self._edges[[tail, head]].max(axis=0).max() + 1
            self.__n_vertices_init = self._n_vertices

        # convert to CSR/CSC
        self._check_orientation(orientation)
        self._orientation = orientation
        if self._orientation == "out":
            # Use dummy weight column for conversion (BFS doesn't use weights)
            self._edges["_bfs_dummy_weight"] = 1.0
            fs_indptr, fs_indices, _ = convert_graph_to_csr_float64(
                self._edges, tail, head, "_bfs_dummy_weight", self._n_vertices
            )
            self._edges = self._edges.drop("_bfs_dummy_weight", axis=1)
            self.__indices = fs_indices.astype(np.uint32)
            self.__indptr = fs_indptr.astype(np.uint32)
        else:
            self._edges["_bfs_dummy_weight"] = 1.0
            rs_indptr, rs_indices, _ = convert_graph_to_csc_float64(
                self._edges, tail, head, "_bfs_dummy_weight", self._n_vertices
            )
            self._edges = self._edges.drop("_bfs_dummy_weight", axis=1)
            self.__indices = rs_indices.astype(np.uint32)
            self.__indptr = rs_indptr.astype(np.uint32)

        self._path_links = None
        self._path_links_permuted = None

    @property
    def UNREACHABLE(self) -> int:
        """The sentinel value, returned for the vertices without predecessor."""
        return self._sentinel

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

    def _preprocess_edges(self, tail, head):
        """
        Preprocess edges to handle parallel edges by keeping only one edge
        between any pair of vertices (BFS doesn't use weights).

        Parameters
        ----------
        tail : str
            The column name for tail vertices
        head : str
            The column name for head vertices
        """
        original_count = len(self._edges)
        self._edges = self._edges.groupby([tail, head], as_index=False).first()
        final_count = len(self._edges)

        if original_count > final_count:
            parallel_edges_removed = original_count - final_count
            if self._verbose:
                print(
                    f"Automatically removed {parallel_edges_removed} parallel edge(s). "
                    f"BFS treats all edges equally."
                )

        self._n_edges = len(self._edges)

    def _check_edges(self, edges, tail, head):
        """Checks if the edges DataFrame is well-formed. If not, raises an appropriate error."""
        edges = _edges_as_pandas(edges, [tail, head])

        if tail not in edges:
            raise KeyError(
                f"edge tail column '{tail}' not found in graph edges dataframe"
            )

        if head not in edges:
            raise KeyError(
                f"edge head column '{head}' not found in graph edges dataframe"
            )

        if edges[[tail, head]].isnull().to_numpy().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{tail}, {head}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"edges['{col}'] should be of integer type")

    def _permute_graph(self, tail, head):
        """Permute the IDs of the nodes to start from 0 and be contiguous.
        Returns a DataFrame with the permuted IDs."""

        permutation = pd.DataFrame(
            data={
                "vert_idx": np.union1d(
                    np.asarray(self._edges[tail]), np.asarray(self._edges[head])
                )
            }
        )
        permutation["vert_idx_new"] = permutation.index
        permutation.index.name = "index"

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=tail,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([tail, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": tail})

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges = self._edges.drop([head, "vert_idx"], axis=1)
        self._edges = self._edges.rename(columns={"vert_idx_new": head})

        permutation = permutation.rename(columns={"vert_idx": "vert_idx_old"})
        permutation = permutation.reset_index(drop=True)
        permutation = permutation.sort_values(by="vert_idx_new")

        permutation.index.name = "index"
        self._edges.index.name = "index"

        return permutation

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' or 'out'")

    def run(
        self,
        vertex_idx: int,
        path_tracking: bool = False,
        return_series: bool = False,
    ) -> Union[np.ndarray, pd.Series]:
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
            Whether to return a pandas Series indexed by vertex instead of a NumPy
            array.

        Returns
        -------
        numpy.ndarray or pandas.Series
            The predecessor ('out') or the successor ('in') of each vertex on a
            path with the fewest edges, as int32. The array has one element per
            integer between 0 and the largest vertex identifier. The start vertex
            and the vertices that cannot be reached have the sentinel value. If
            `return_series` is True, the result is a Series named 'predecessor'
            with the vertex identifiers as index.

        Raises
        ------
        ValueError
            If `vertex_idx` is not a vertex of the graph, or if the graph has
            negative vertex identifiers and `return_series` is False.
        """
        # validate the input arguments
        if self._negative_labels and not return_series:
            raise ValueError(
                "the graph has negative vertex labels: the result cannot be an "
                "array indexed by vertex, use return_series=True"
            )
        if self._permute and self._permutation is not None:
            if vertex_idx not in self._permutation.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._permutation.loc[
                self._permutation.vert_idx_old == vertex_idx, "vert_idx_new"
            ].iloc[0]
        else:
            if vertex_idx < 0:
                raise ValueError(
                    f"argument 'vertex_idx={vertex_idx}' must be non-negative"
                )
            if vertex_idx >= self._n_vertices:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = vertex_idx

        # compute BFS predecessors
        if self._orientation == "out":
            predecessors = bfs_csr(
                self.__indptr,
                self.__indices,
                vertex_new,
                self._n_vertices,
                self._sentinel,
            )
        else:
            predecessors = bfs_csc(
                self.__indptr,
                self.__indices,
                vertex_new,
                self._n_vertices,
                self._sentinel,
            )

        # store path links if tracking is enabled
        if path_tracking:
            # Convert predecessors to path_links format (uint32)
            # Replace sentinel value with vertex's own index (like Dijkstra does)
            self._path_links = np.arange(self._n_vertices, dtype=np.uint32)
            reachable_mask = predecessors != self._sentinel
            self._path_links[reachable_mask] = predecessors[reachable_mask].astype(
                np.uint32
            )

            if self._permute and self._permutation is not None:
                # permute back the path vertex indices (same approach as Dijkstra)
                self._path_links_permuted = self._path_links
                path_df = pd.DataFrame(
                    data={
                        "vertex_idx": np.arange(self._n_vertices),
                        "associated_idx": self._path_links,
                    }
                )
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="vertex_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["vertex_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "vertex_idx"})
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df = path_df.drop(["associated_idx", "vert_idx_new"], axis=1)
                path_df = path_df.rename(columns={"vert_idx_old": "associated_idx"})

                if return_series:
                    path_df = path_df.set_index("vertex_idx")
                    self._path_links = path_df.associated_idx.astype(
                        np.uint32
                    ).to_numpy()
                else:
                    self._path_links = np.arange(
                        self.__n_vertices_init, dtype=np.uint32
                    )
                    self._path_links[path_df.vertex_idx.values] = (
                        path_df.associated_idx.values
                    )
        else:
            self._path_links = None

        # reorder predecessors for permuted graphs
        if return_series:
            if self._permute and self._permutation is not None:
                pred_df = pd.DataFrame(data={"predecessor": predecessors})
                pred_df["vert_idx_new"] = pred_df.index
                pred_df = pd.merge(
                    pred_df,
                    self._permutation,
                    left_on="vert_idx_new",
                    right_on="vert_idx_new",
                    how="left",
                )

                # Map predecessor values back to original IDs
                valid_mask = pred_df["predecessor"] != self._sentinel
                if valid_mask.any():
                    pred_df_valid = pred_df[valid_mask].copy()
                    pred_df_valid = pd.merge(
                        pred_df_valid,
                        self._permutation,
                        left_on="predecessor",
                        right_on="vert_idx_new",
                        how="left",
                        suffixes=("", "_pred"),
                    )
                    pred_df.loc[valid_mask, "predecessor"] = pred_df_valid[
                        "vert_idx_old_pred"
                    ].values.astype(np.int32)

                pred_df = pred_df.set_index("vert_idx_old")
                predecessors_series = pred_df.predecessor.astype(np.int32)
                predecessors_series.index.name = "vertex_idx"
                predecessors_series.name = "predecessor"
            else:
                predecessors_series = pd.Series(predecessors, dtype=np.int32)
                predecessors_series.index.name = "vertex_idx"
                predecessors_series.name = "predecessor"

            return predecessors_series

        # For array output with permutation
        if self._permute and self._permutation is not None:
            pred_df = pd.DataFrame(data={"predecessor": predecessors})
            pred_df["vert_idx_new"] = pred_df.index
            pred_df = pd.merge(
                pred_df,
                self._permutation,
                left_on="vert_idx_new",
                right_on="vert_idx_new",
                how="left",
            )

            # Map predecessor values back to original IDs
            valid_mask = pred_df["predecessor"] != self._sentinel
            if valid_mask.any():
                pred_df_valid = pred_df[valid_mask].copy()
                pred_df_valid = pd.merge(
                    pred_df_valid,
                    self._permutation,
                    left_on="predecessor",
                    right_on="vert_idx_new",
                    how="left",
                    suffixes=("", "_pred"),
                )
                pred_df.loc[valid_mask, "predecessor"] = pred_df_valid[
                    "vert_idx_old_pred"
                ].values.astype(np.int32)

            predecessors_array = np.full(
                self.__n_vertices_init, self._sentinel, dtype=np.int32
            )
            predecessors_array[pred_df.vert_idx_old.values] = (
                pred_df.predecessor.values.astype(np.int32)
            )
            return predecessors_array

        return predecessors

    def get_vertices(self) -> Any:
        """Return the vertices of the graph.

        Returns
        -------
        numpy.ndarray
            The sorted identifiers of the vertices that are the tail or the head of
            at least one edge.
        """
        if self._permute and self._permutation is not None:
            return np.asarray(self._permutation.vert_idx_old)
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
                "Current BFS instance has no path attribute: "
                "make sure path_tracking is set to True, and run the "
                "BFS algorithm",
                UserWarning,
            )
            return None
        if self._permute and self._permutation is not None:
            return _compute_path_with_permutation(
                self._path_links_permuted, self._permutation, vertex_idx
            )
        return compute_path(self._path_links, vertex_idx)


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
