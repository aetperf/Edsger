"""
Path-related methods.
"""

import warnings

import numpy as np
import pandas as pd

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
from edsger.dijkstra import (
    compute_sssp,
    compute_sssp_w_path,
    compute_sssp_early_termination,
    compute_sssp_w_path_early_termination,
    compute_stsp,
    compute_stsp_w_path,
    compute_stsp_early_termination,
    compute_stsp_w_path_early_termination,
)
from edsger.path_tracking import compute_path
from edsger.spiess_florian import compute_SF_in
from edsger.star import (
    convert_graph_to_csc_float64,
    convert_graph_to_csc_uint32,
    convert_graph_to_csr_float64,
    convert_graph_to_csr_uint32,
)


class Dijkstra:
    """
    Dijkstra's algorithm for finding the shortest paths between nodes in directed graphs with
    positive edge weights.

    Note: If parallel edges exist between the same pair of vertices, only the edge with the minimum
    weight will be kept automatically during initialization.

    Parameters:
    -----------
    edges: pandas.DataFrame
        DataFrame containing the edges of the graph. It should have three columns: 'tail', 'head',
        and 'weight'. The 'tail' column should contain the IDs of the starting nodes, the 'head'
        column should contain the IDs of the ending nodes, and the 'weight' column should contain
        the (positive) weights of the edges.
    tail: str, optional (default='tail')
        The name of the column in the DataFrame that contains the IDs of the edge starting nodes.
    head: str, optional (default='head')
        The name of the column in the DataFrame that contains the IDs of the edge ending nodes.
    weight: str, optional (default='weight')
        The name of the column in the DataFrame that contains the (positive) weights of the edges.
    orientation: str, optional (default='out')
        The orientation of Dijkstra's algorithm. It can be either 'out' for single source shortest
        paths or 'in' for single target shortest path.
    check_edges: bool, optional (default=False)
        Whether to check if the edges DataFrame is well-formed. If set to True, the edges DataFrame
        will be checked for missing values and invalid data types.
    permute: bool, optional (default=False)
        Whether to permute the IDs of the nodes. If set to True, the node IDs will be reindexed to
        start from 0 and be contiguous.
    verbose: bool, optional (default=False)
        Whether to print messages about parallel edge removal.
    """

    def __init__(
        self,
        edges,
        tail="tail",
        head="head",
        weight="weight",
        orientation="out",
        check_edges=False,
        permute=False,
        verbose=False,
    ):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        self._edges = edges[[tail, head, weight]].copy(deep=True)
        self._n_edges = len(self._edges)
        self._verbose = verbose

        # preprocess edges to handle parallel edges
        self._preprocess_edges(tail, head, weight)

        # reindex the vertices
        self._permute = permute
        if self._permute:
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

    @property
    def edges(self):
        """
        Getter for the graph edge dataframe.

        Returns
        -------
        edges: pandas.DataFrame
            DataFrame containing the edges of the graph.
        """
        return self._edges

    @property
    def n_edges(self):
        """
        Getter for the number of graph edges.

        Returns
        -------
        n_edges: int
            The number of edges in the graph.
        """
        return self._n_edges

    @property
    def n_vertices(self):
        """
        Getter for the number of graph vertices.

        Returns
        -------
        n_vertices: int
            The number of nodes in the graph (after permutation, if _permute is True).
        """
        return self._n_vertices

    @property
    def orientation(self):
        """
        Getter of Dijkstra's algorithm orientation ("in" or "out").

        Returns
        -------
        orientation : str
            The orientation of Dijkstra's algorithm.
        """
        return self._orientation

    @property
    def permute(self):
        """
        Getter for the graph permutation/reindexing option.

        Returns
        -------
        permute : bool
            Whether to permute the IDs of the nodes.
        """
        return self._permute

    @property
    def path_links(self):
        """
        Getter for the graph permutation/reindexing option.

        Returns
        -------
        path_links: numpy.ndarray
            predecessors or successors node index if the path tracking is activated.
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
        if not isinstance(edges, pd.core.frame.DataFrame):
            raise TypeError("edges should be a pandas DataFrame")

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

        if edges[[tail, head, weight]].isna().any().any():
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
                    self._edges[tail].values, self._edges[head].values
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
        self._edges.drop([tail, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": tail}, inplace=True)

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([head, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": head}, inplace=True)

        permutation.rename(columns={"vert_idx": "vert_idx_old"}, inplace=True)
        permutation.reset_index(drop=True, inplace=True)
        permutation.sort_values(by="vert_idx_new", inplace=True)

        permutation.index.name = "index"
        self._edges.index.name = "index"

        return permutation

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' on 'out'")

    def run(
        self,
        vertex_idx,
        path_tracking=False,
        return_inf=True,
        return_series=False,
        heap_length_ratio=1.0,
        termination_nodes=None,
    ):
        """
        Runs shortest path algorithm between a given vertex and all other vertices in the graph.

        Parameters
        ----------
        vertex_idx : int
            The index of the source/target vertex.
        path_tracking : bool, optional (default=False)
            Whether to track the shortest path(s) from the source vertex to all other vertices in
            the graph.
        return_inf : bool, optional (default=True)
            Whether to return path length(s) as infinity (np.inf) when no path exists.
        return_series : bool, optional (default=False)
            Whether to return a Pandas Series object indexed by vertex indices with path length(s)
            as values.
        heap_length_ratio : float, optional (default=1.0)
            The heap length as a fraction of the number of vertices. Must be in the range (0, 1].
        termination_nodes : array-like, optional (default=None)
            List or array of vertex indices for early termination. For SSSP (orientation='out'),
            these are target nodes to reach. For STSP (orientation='in'), these are source nodes
            to find paths from. When provided, the algorithm stops once all specified nodes have
            been processed, potentially improving performance. If None, runs to completion.

        Returns
        -------
        path_length_values or path_lengths_series : array_like or Pandas Series
            If `return_series=False`, a 1D Numpy array of shape (n_vertices,) with the shortest
            path length from the source vertex to each vertex in the graph (`orientation="out"`), or
            from each vertex to the target vertex (`orientation="in"`). If `return_series=True`, a
            Pandas Series object with the same data and the vertex indices as index.

        """
        # validate the input arguments
        if not isinstance(vertex_idx, int):
            try:
                vertex_idx = int(vertex_idx)
            except ValueError as exc:
                raise TypeError(
                    f"argument 'vertex_idx={vertex_idx}' must be an integer"
                ) from exc
        if vertex_idx < 0:
            raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be positive")
        if self._permute:
            if vertex_idx not in self._permutation.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._permutation.loc[
                self._permutation.vert_idx_old == vertex_idx, "vert_idx_new"
            ].iloc[0]
        else:
            if vertex_idx >= self._n_vertices:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = vertex_idx
        if not isinstance(path_tracking, bool):
            raise TypeError(
                f"argument 'path_tracking=f{path_tracking}' must be of bool type"
            )
        if not isinstance(return_inf, bool):
            raise TypeError(f"argument 'return_inf=f{return_inf}' must be of bool type")
        if not isinstance(return_series, bool):
            raise TypeError(
                f"argument 'return_series=f{return_series}' must be of bool type"
            )
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
                termination_nodes_array = np.array(termination_nodes, dtype=np.uint32)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    "argument 'termination_nodes' must be array-like of integers"
                ) from exc

            if termination_nodes_array.ndim != 1:
                raise ValueError("argument 'termination_nodes' must be 1-dimensional")

            if len(termination_nodes_array) == 0:
                raise ValueError("argument 'termination_nodes' must not be empty")

            # handle vertex permutation if needed
            if self._permute:
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

        # compute path length
        if not path_tracking:
            self._path_links = None
            if termination_nodes_array is not None:
                # use early termination algorithms
                if self._orientation == "in":
                    path_length_values = compute_stsp_early_termination(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        termination_nodes_array,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = compute_sssp_early_termination(
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
                    path_length_values = compute_stsp(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = compute_sssp(
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
                    path_length_values = compute_stsp_w_path_early_termination(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        termination_nodes_array,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = compute_sssp_w_path_early_termination(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        termination_nodes_array,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
            else:
                # use standard algorithms with path tracking
                if self._orientation == "in":
                    path_length_values = compute_stsp_w_path(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )
                else:
                    path_length_values = compute_sssp_w_path(
                        self.__indptr,
                        self.__indices,
                        self.__edge_weights,
                        self._path_links,
                        vertex_new,
                        self._n_vertices,
                        heap_length,
                    )

            if self._permute:
                # permute back the path vertex indices
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
                path_df.drop(["vertex_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "vertex_idx"}, inplace=True)
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df.drop(["associated_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "associated_idx"}, inplace=True)

                if return_series:
                    path_df.set_index("vertex_idx", inplace=True)
                    self._path_links = path_df.associated_idx.astype(np.uint32)
                else:
                    self._path_links = np.arange(
                        self.__n_vertices_init, dtype=np.uint32
                    )
                    self._path_links[path_df.vertex_idx.values] = (
                        path_df.associated_idx.values
                    )

        # deal with infinity
        if return_inf:
            path_length_values = np.where(
                path_length_values == DTYPE_INF_PY, np.inf, path_length_values
            )

        # reorder path lengths
        if return_series:
            if self._permute and termination_nodes_array is None:
                self._permutation["path_length"] = path_length_values
                path_lengths_df = self._permutation[
                    ["vert_idx_old", "path_length"]
                ].sort_values(by="vert_idx_old")
                path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
                path_lengths_df.index.name = "vertex_idx"
                path_lengths_series = path_lengths_df.path_length
            else:
                path_lengths_series = pd.Series(path_length_values)
                path_lengths_series.index.name = "vertex_idx"
                path_lengths_series.name = "path_length"
                if self._permute and termination_nodes_array is not None:
                    # For early termination with permutation, use original termination node indices
                    path_lengths_series.index = termination_nodes

            return path_lengths_series

        # For early termination, return results directly (already in correct order)
        if termination_nodes_array is not None:
            return path_length_values

        if self._permute:
            self._permutation["path_length"] = path_length_values
            if return_inf:
                path_length_values = np.inf * np.ones(self.__n_vertices_init)
            else:
                path_length_values = DTYPE_INF_PY * np.ones(self.__n_vertices_init)
            path_length_values[self._permutation.vert_idx_old.values] = (
                self._permutation.path_length.values
            )

        return path_length_values

    def get_vertices(self):
        """
        Get the unique vertices from the graph.

        If the graph has been permuted, this method returns the vertices based on the original
        indexing. Otherwise, it returns the union of tail and head vertices from the edges.

        Returns
        -------
        vertices : ndarray
            A 1-D array containing the unique vertices.
        """
        if self._permute:
            return self._permutation.vert_idx_old.values
        return np.union1d(self._edges["tail"], self._edges["head"])

    def get_path(self, vertex_idx):
        """Compute path from predecessors or successors.

        Parameters:
        -----------

        vertex_idx : int
            source or target vertex index.

        Returns
        -------

        path_vertices : numpy.ndarray
            Array of np.uint32 type storing the path from or to the given vertex index. If we are
            dealing with the sssp algorithm, the input vertex is the target vertex and the path to
            the source is given backward from the target to the source using the predecessors. If
            we are dealing with the stsp algorithm, the input vertex is the source vertex and the
            path to the target is given backward from the target to the source using the
            successors.

        """
        if self._path_links is None:
            warnings.warn(
                "Current Dijkstra instance has not path attribute : \
                make sure path_tracking is set to True, and run the \
                shortest path algorithm",
                UserWarning,
            )
            return None
        if isinstance(self._path_links, pd.Series):
            path_vertices = compute_path(self._path_links.values, vertex_idx)
        else:
            path_vertices = compute_path(self._path_links, vertex_idx)
        return path_vertices


class BellmanFord:
    """
    Bellman-Ford algorithm for finding the shortest paths between nodes in directed graphs.
    Supports negative edge weights and detects negative cycles.

    Note: If parallel edges exist between the same pair of vertices, only the edge with the minimum
    weight will be kept automatically during initialization.

    Parameters:
    -----------
    edges: pandas.DataFrame
        DataFrame containing the edges of the graph. It should have three columns: 'tail', 'head',
        and 'weight'. The 'tail' column should contain the IDs of the starting nodes, the 'head'
        column should contain the IDs of the ending nodes, and the 'weight' column should contain
        the weights of the edges (can be negative).
    tail: str, optional (default='tail')
        The name of the column in the DataFrame that contains the IDs of the edge starting nodes.
    head: str, optional (default='head')
        The name of the column in the DataFrame that contains the IDs of the edge ending nodes.
    weight: str, optional (default='weight')
        The name of the column in the DataFrame that contains the weights of the edges.
    orientation: str, optional (default='out')
        The orientation of Bellman-Ford's algorithm. It can be either 'out' for single source
        shortest paths or 'in' for single target shortest path.
    check_edges: bool, optional (default=False)
        Whether to check if the edges DataFrame is well-formed. If set to True, the edges
        DataFrame will be checked for missing values and invalid data types. Note: negative
        weights are allowed.
    permute: bool, optional (default=False)
        Whether to permute the IDs of the nodes. If set to True, the node IDs will be reindexed to
        start from 0 and be contiguous.
    verbose: bool, optional (default=False)
        Whether to print messages about parallel edge removal.
    """

    def __init__(
        self,
        edges,
        tail="tail",
        head="head",
        weight="weight",
        orientation="out",
        check_edges=False,
        permute=False,
        verbose=False,
    ):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        self._edges = edges[[tail, head, weight]].copy(deep=True)
        self._n_edges = len(self._edges)
        self._verbose = verbose

        # preprocess edges to handle parallel edges
        self._preprocess_edges(tail, head, weight)

        # reindex the vertices
        self._permute = permute
        if self._permute:
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
        self._has_negative_cycle = False

    @property
    def edges(self):
        """
        Getter for the graph edge dataframe.

        Returns
        -------
        edges: pandas.DataFrame
            DataFrame containing the edges of the graph.
        """
        return self._edges

    @property
    def n_edges(self):
        """
        Getter for the number of graph edges.

        Returns
        -------
        n_edges: int
            The number of edges in the graph.
        """
        return self._n_edges

    @property
    def n_vertices(self):
        """
        Getter for the number of graph vertices.

        Returns
        -------
        n_vertices: int
            The number of nodes in the graph (after permutation, if _permute is True).
        """
        return self._n_vertices

    @property
    def orientation(self):
        """
        Getter of Bellman-Ford's algorithm orientation ("in" or "out").

        Returns
        -------
        orientation : str
            The orientation of Bellman-Ford's algorithm.
        """
        return self._orientation

    @property
    def permute(self):
        """
        Getter for the graph permutation/reindexing option.

        Returns
        -------
        permute : bool
            Whether to permute the IDs of the nodes.
        """
        return self._permute

    @property
    def path_links(self):
        """
        Getter for the path links (predecessors or successors).

        Returns
        -------
        path_links: numpy.ndarray
            predecessors or successors node index if the path tracking is activated.
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
        if not isinstance(edges, pd.core.frame.DataFrame):
            raise TypeError("edges should be a pandas DataFrame")

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

        if edges[[tail, head, weight]].isna().any().any():
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
                    self._edges[tail].values, self._edges[head].values
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
        self._edges.drop([tail, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": tail}, inplace=True)

        self._edges = pd.merge(
            self._edges,
            permutation[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([head, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": head}, inplace=True)

        permutation.rename(columns={"vert_idx": "vert_idx_old"}, inplace=True)
        permutation.reset_index(drop=True, inplace=True)
        permutation.sort_values(by="vert_idx_new", inplace=True)

        permutation.index.name = "index"
        self._edges.index.name = "index"

        return permutation

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError("orientation should be either 'in' on 'out'")

    def run(
        self,
        vertex_idx,
        path_tracking=False,
        return_inf=True,
        return_series=False,
        detect_negative_cycles=True,
    ):
        """
        Runs Bellman-Ford shortest path algorithm between a given vertex and all other vertices
        in the graph.

        Parameters
        ----------
        vertex_idx : int
            The index of the source/target vertex.
        path_tracking : bool, optional (default=False)
            Whether to track the shortest path(s) from the source vertex to all other vertices in
            the graph.
        return_inf : bool, optional (default=True)
            Whether to return path length(s) as infinity (np.inf) when no path exists.
        return_series : bool, optional (default=False)
            Whether to return a Pandas Series object indexed by vertex indices with path length(s)
            as values.
        detect_negative_cycles : bool, optional (default=True)
            Whether to detect negative cycles in the graph. If True and a negative cycle is
            detected,
            raises a ValueError.

        Returns
        -------
        path_length_values or path_lengths_series : array_like or Pandas Series
            If `return_series=False`, a 1D Numpy array of shape (n_vertices,) with the shortest
            path length from the source vertex to each vertex in the graph (`orientation="out"`), or
            from each vertex to the target vertex (`orientation="in"`). If `return_series=True`, a
            Pandas Series object with the same data and the vertex indices as index.

        Raises
        ------
        ValueError
            If detect_negative_cycles is True and a negative cycle is detected in the graph.
        """
        # validate the input arguments
        if not isinstance(vertex_idx, int):
            try:
                vertex_idx = int(vertex_idx)
            except ValueError as exc:
                raise TypeError(
                    f"argument 'vertex_idx={vertex_idx}' must be an integer"
                ) from exc
        if vertex_idx < 0:
            raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be positive")
        if self._permute:
            if vertex_idx not in self._permutation.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._permutation.loc[
                self._permutation.vert_idx_old == vertex_idx, "vert_idx_new"
            ].iloc[0]
        else:
            if vertex_idx >= self._n_vertices:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = vertex_idx
        if not isinstance(path_tracking, bool):
            raise TypeError(
                f"argument 'path_tracking=f{path_tracking}' must be of bool type"
            )
        if not isinstance(return_inf, bool):
            raise TypeError(f"argument 'return_inf=f{return_inf}' must be of bool type")
        if not isinstance(return_series, bool):
            raise TypeError(
                f"argument 'return_series=f{return_series}' must be of bool type"
            )
        if not isinstance(detect_negative_cycles, bool):
            raise TypeError(
                f"argument 'detect_negative_cycles={detect_negative_cycles}' must be of bool type"
            )

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

            if self._permute:
                # permute back the path vertex indices
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
                path_df.drop(["vertex_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "vertex_idx"}, inplace=True)
                path_df = pd.merge(
                    path_df,
                    self._permutation,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df.drop(["associated_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "associated_idx"}, inplace=True)

                if return_series:
                    path_df.set_index("vertex_idx", inplace=True)
                    self._path_links = path_df.associated_idx.astype(np.uint32)
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
            if self._permute:
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
                path_df.drop(["vert_idx_new"], axis=1, inplace=True)
                path_df.set_index("vert_idx_old", inplace=True)
                path_lengths_series = path_df.path_length.astype(DTYPE_PY)
            else:
                path_lengths_series = pd.Series(
                    data=path_length_values[: self._n_vertices], dtype=DTYPE_PY
                )
                path_lengths_series.index = np.arange(self._n_vertices)
            path_lengths_series.index.name = None
            return path_lengths_series

        # No else needed - de-indent the code
        if self._permute:
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
            path_df.drop(["vert_idx_new"], axis=1, inplace=True)
            path_length_values = np.full(self.__n_vertices_init, DTYPE_INF_PY)
            path_length_values[path_df.vert_idx_old.values] = path_df.path_length.values
            if return_inf:
                path_length_values = np.where(
                    path_length_values == DTYPE_INF_PY, np.inf, path_length_values
                )
        return path_length_values

    def get_path(self, vertex_idx):
        """Compute path from predecessors or successors.

        Parameters:
        -----------

        vertex_idx : int
            source or target vertex index.

        Returns
        -------

        path_vertices : numpy.ndarray
            Array of np.uint32 type storing the path from or to the given vertex index. If we are
            dealing with the sssp algorithm, the input vertex is the target vertex and the path to
            the source is given backward from the target to the source using the predecessors. If
            we are dealing with the stsp algorithm, the input vertex is the source vertex and the
            path to the target is given backward from the target to the source using the
            successors.

        """
        if self._path_links is None:
            warnings.warn(
                "Current BellmanFord instance has not path attribute : \
                make sure path_tracking is set to True, and run the \
                shortest path algorithm",
                UserWarning,
            )
            return None
        if isinstance(self._path_links, pd.Series):
            path_vertices = compute_path(self._path_links.values, vertex_idx)
        else:
            path_vertices = compute_path(self._path_links, vertex_idx)
        return path_vertices

    def has_negative_cycle(self):
        """
        Check if the last run detected a negative cycle.

        Returns
        -------
        has_negative_cycle : bool
            True if a negative cycle was detected in the last run, False otherwise.
        """
        return self._has_negative_cycle


class HyperpathGenerating:
    """
    A class for constructing and managing hyperpath-based routing and analysis in transportation
    or graph-based systems.

    Parameters
    ----------
    edges : pandas.DataFrame
        A DataFrame containing graph edge information with columns specified by `tail`, `head`,
        `trav_time`, and `freq`. Must not contain missing values.
    tail : str, optional
        Name of the column in `edges` representing the tail nodes (source nodes), by default "tail".
    head : str, optional
        Name of the column in `edges` representing the head nodes (target nodes), by default "head".
    trav_time : str, optional
        Name of the column in `edges` representing travel times for edges, by default "trav_time".
    freq : str, optional
        Name of the column in `edges` representing frequencies of edges, by default "freq".
    check_edges : bool, optional
        Whether to validate the structure and data types of `edges`, by default False.
    orientation : {"in", "out"}, optional
        Determines the orientation of the graph structure for traversal.
        - "in": Graph traversal is from destination to origin.
        - "out": Graph traversal is from origin to destination.
        By default "in".

    Attributes
    ----------
    edge_count : int
        The number of edges in the graph.
    vertex_count : int
        The total number of vertices in the graph.
    u_i_vec : numpy.ndarray
        An array storing the least travel time for each vertex after running the algorithm.
    _edges : pandas.DataFrame
        Internal DataFrame containing the edges with additional metadata.
    _trav_time : numpy.ndarray
        Array of travel times for edges.
    _freq : numpy.ndarray
        Array of frequencies for edges.
    _tail : numpy.ndarray
        Array of tail nodes (source nodes) for edges.
    _head : numpy.ndarray
        Array of head nodes (target nodes) for edges.
    __indptr : numpy.ndarray
        Array for compressed row (or column) pointers in the CSR/CSC representation.
    _edge_idx : numpy.ndarray
        Array of edge indices in the CSR/CSC representation.

    Methods
    -------
    run(origin, destination, volume, return_inf=False)
        Computes the hyperpath and updates edge volumes based on the input demand and configuration.
    _check_vertex_idx(idx)
        Validates a vertex index to ensure it is within the graph's bounds.
    _check_volume(v)
        Validates a volume value to ensure it is a non-negative float.
    _check_edges(edges, tail, head, trav_time, freq)
        Validates the structure and data types of the input edges DataFrame.
    """

    def __init__(
        self,
        edges,
        tail="tail",
        head="head",
        trav_time="trav_time",
        freq="freq",
        check_edges=False,
        orientation="in",
    ):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq)
        self._edges = edges[[tail, head, trav_time, freq]].copy(deep=True)
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
        assert orientation in ["out", "in"]
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
        self._trav_time = self._edges[trav_time].values.astype(DTYPE_PY)
        self._freq = self._edges[freq].values.astype(DTYPE_PY)
        self._tail = self._edges[tail].values.astype(np.uint32)
        self._head = self._edges[head].values.astype(np.uint32)

        # node attribute
        self.u_i_vec = None

    def run(self, origin, destination, volume, return_inf=False):
        """
        Computes the hyperpath and updates edge volumes based on the input demand and configuration.

        Parameters
        ----------
        origin : int or list of int
            The starting vertex or vertices of the demand. If `self._orientation` is "in",
            this can be a list of origins corresponding to the demand volumes.
        destination : int or list of int
            The target vertex or vertices of the demand. If `self._orientation` is "out",
            this can be a list of destinations corresponding to the demand volumes.
        volume : float or list of float
            The demand volume associated with each origin or destination. Must be non-negative.
            If a single float is provided, it is applied to a single origin-destination pair.
        return_inf : bool, optional
            If True, returns additional information from the computation (not yet implemented).
            Default is False.

        Raises
        ------
        NotImplementedError
            If `self._orientation` is "out", as the one-to-many algorithm is not yet implemented.
        AssertionError
            If the lengths of `origin` or `destination` and `volume` do not match.
            If any vertex index or volume is invalid.
        TypeError
            If `volume` is not a float or list of floats.
        ValueError
            If any volume value is negative.

        Notes
        -----
        - The method modifies the `self._edges` DataFrame by adding a "volume" column representing
        edge volumes based on the computed hyperpath.
        - The `self.u_i_vec` array is updated to store the least travel time for each vertex.
        - Only "in" orientation is currently supported.
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
        assert len(origin) == len(volume)
        for i, item in enumerate(origin):
            self._check_vertex_idx(item)
            self._check_volume(volume[i])
        self._check_vertex_idx(destination)
        demand_indices = np.array(origin, dtype=np.uint32)

        assert isinstance(return_inf, bool)

        demand_values = np.array(volume, dtype=DTYPE_PY)

        compute_SF_in(
            self.__indptr,
            self._edge_idx,
            self._trav_time,
            self._freq,
            self._tail,
            self._head,
            demand_indices,  # source vertex indices
            demand_values,
            self._edges["volume"].values,
            u_i_vec,
            self.vertex_count,
            destination,
        )
        self.u_i_vec = u_i_vec

    def _check_vertex_idx(self, idx):
        assert isinstance(idx, int)
        assert idx >= 0
        assert idx < self.vertex_count

    def _check_volume(self, v):
        assert isinstance(v, float)
        assert v >= 0.0

    def _check_edges(self, edges, tail, head, trav_time, freq):
        if not isinstance(edges, pd.core.frame.DataFrame):
            raise TypeError("edges should be a pandas DataFrame")

        for col in [tail, head, trav_time, freq]:
            if col not in edges:
                raise KeyError(
                    f"edge column '{col}' not found in graph edges dataframe"
                )

        if edges[[tail, head, trav_time, freq]].isna().any().any():
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


# author : Francois Pacull
# copyright : Architecture & Performance
# email: francois.pacull@architecture-performance.fr
# license : MIT
