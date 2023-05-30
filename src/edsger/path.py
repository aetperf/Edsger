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
from edsger.dijkstra import (
    compute_sssp,
    compute_sssp_w_path,
    compute_stsp,
    compute_stsp_w_path,
)
from edsger.spiess_florian import compute_SF_in
from edsger.star import (
    convert_graph_to_csc_float64,
    convert_graph_to_csc_uint32,
    convert_graph_to_csr_float64,
    convert_graph_to_csr_uint32,
)
from edsger.path_tracking import compute_path


class Dijkstra:
    """
    Dijkstra's algorithm for finding the shortest paths between nodes in directed graphs with
    positive edge weights.

    Parameters:
    -----------
    edges: pandas.DataFrame
        DataFrame containing the edges of the graph. It should have three columns: 'tail', 'head',
        and 'weight'. The 'tail' column should contain the IDs of the starting nodes, the 'head'
        column should contain the IDs of the ending nodes, and the 'weight' column should contain
        the (positive) weights of the edges.

    tail: str, optional (default='tail')
        The name of the column in the DataFrame that contains the IDs of the starting nodes.

    head: str, optional (default='head')
        The name of the column in the DataFrame that contains the IDs of the ending nodes.

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

    Attributes:
    -----------
    _edges: pandas.DataFrame
        DataFrame containing the edges of the graph.

    _n_edges: int
        The number of edges in the graph.

    _permute: bool
        Whether to permute the IDs of the nodes.

    _vertices: pandas.DataFrame or None
        DataFrame containing the old and new IDs of the nodes if the IDs have been permuted.

    _n_vertices: int
        The number of nodes in the graph (after permutation, if _permute is True).

    __n_vertices_init: int
        The number of nodes in the original graph (not permuted).

    _orientation: str
        The orientation of Dijkstra's algorithm.

    __indices: numpy.ndarray
        1D array containing the indices of the indices of the forward or reverse star of the graph
        in compressed format.

    __indptr: numpy.ndarray
        1D array containing the indices of the pointer of the forward or reverse star of the graph
        in compressed format.

    __edge_weights: numpy.ndarray
        1D array containing the weights of the edges in the graph.

    _path_links: numpy.ndarray
        predecessors or successors node index if the path tracking is activated.

    Methods:
    --------
    _check_edges(edges, tail, head, weight)
        Checks if the edges DataFrame is well-formed. If not, raises an appropriate error.

    _permute_graph(tail, head)
        Permute the IDs of the nodes to start from 0 and be contiguous. Returns a DataFrame with
        the permuted IDs.

    _check_orientation(orientation):
        Checks the orientation attribute.

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
    ):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, weight)
        self._edges = edges[[tail, head, weight]].copy(deep=True)
        self._n_edges = len(self._edges)

        # reindex the vertices
        self._permute = permute
        if self._permute:
            self.__n_vertices_init = self._edges[[tail, head]].max(axis=0).max() + 1
            self._vertices = self._permute_graph(tail, head)
            self._n_vertices = len(self._vertices)
        else:
            self._vertices = None
            self._n_vertices = self._edges[[tail, head]].max(axis=0).max() + 1
            self.__n_vertices_init = self._n_vertices

        # convert to CSR/CSC
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
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def n_edges(self):
        return self._n_edges

    @property
    def n_vertices(self):
        return self._n_vertices

    @property
    def orientation(self):
        return self._orientation

    @property
    def permute(self):
        return self._permute

    @property
    def permute(self):
        return self._path_links

    def _check_edges(self, edges, tail, head, weight):
        """Checks if the edges DataFrame is well-formed. If not, raises an appropriate error."""
        if type(edges) != pd.core.frame.DataFrame:
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

        vertices = pd.DataFrame(
            data={
                "vert_idx": np.union1d(
                    self._edges[tail].values, self._edges[head].values
                )
            }
        )
        vertices["vert_idx_new"] = vertices.index
        vertices.index.name = "index"

        self._edges = pd.merge(
            self._edges,
            vertices[["vert_idx", "vert_idx_new"]],
            left_on=tail,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([tail, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": tail}, inplace=True)

        self._edges = pd.merge(
            self._edges,
            vertices[["vert_idx", "vert_idx_new"]],
            left_on=head,
            right_on="vert_idx",
            how="left",
        )
        self._edges.drop([head, "vert_idx"], axis=1, inplace=True)
        self._edges.rename(columns={"vert_idx_new": head}, inplace=True)

        vertices.rename(columns={"vert_idx": "vert_idx_old"}, inplace=True)
        vertices.reset_index(drop=True, inplace=True)
        vertices.sort_values(by="vert_idx_new", inplace=True)

        vertices.index.name = "index"
        self._edges.index.name = "index"

        return vertices

    def _check_orientation(self, orientation):
        """Checks the orientation attribute."""
        if orientation not in ["in", "out"]:
            raise ValueError(f"orientation should be either 'in' on 'out'")

    def run(
        self,
        vertex_idx,
        path_tracking=False,
        return_inf=True,
        return_Series=False,
        heap_length_ratio=1.0,
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
        return_Series : bool, optional (default=False)
            Whether to return a Pandas Series object indexed by vertex indices with path length(s)
            as values.
        heap_length_ratio : float, optional (default=1.0)
            The heap length as a fraction of the number of vertices. Must be in the range (0, 1].

        Returns
        -------
        path_length_values : array_like or Pandas Series
            If `return_Series=False`, a 1D Numpy array of shape (n_vertices,) with the shortest
            path length from the source vertex to each vertex in the graph (`orientation="out"`), or
            from each vertex to the target vertex (`orientation="in"`). If `return_Series=True`, a
            Pandas Series object with the same data
            and the vertex indices as index.

        """
        # validate the input arguments
        if not isinstance(vertex_idx, int):
            raise TypeError(f"argument 'vertex_idx=f{vertex_idx}' must be of int type")
        if vertex_idx < 0:
            raise ValueError(f"argument 'vertex_idx={vertex_idx}' must be positive")
        if self._permute:
            if vertex_idx not in self._vertices.vert_idx_old.values:
                raise ValueError(f"vertex {vertex_idx} not found in graph")
            vertex_new = self._vertices.loc[
                self._vertices.vert_idx_old == vertex_idx, "vert_idx_new"
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
        if not isinstance(return_Series, bool):
            raise TypeError(
                f"argument 'return_Series=f{return_Series}' must be of bool type"
            )
        if not isinstance(heap_length_ratio, float):
            raise TypeError(
                f"argument 'heap_length_ratio=f{heap_length_ratio}' must be of float type"
            )
        if heap_length_ratio > 1.0:
            heap_length_ratio = 1.0
        if heap_length_ratio <= 0.0:
            raise ValueError(
                f"argument 'heap_length_ratio={heap_length_ratio}' must be strictly positive "
            )
        heap_length = int(np.rint(heap_length_ratio * self._n_vertices))

        # compute path length
        if not path_tracking:
            self._path_links = None
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
                    self._vertices,
                    left_on="vertex_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df.drop(["vertex_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "vertex_idx"}, inplace=True)
                path_df = pd.merge(
                    path_df,
                    self._vertices,
                    left_on="associated_idx",
                    right_on="vert_idx_new",
                    how="left",
                )
                path_df.drop(["associated_idx", "vert_idx_new"], axis=1, inplace=True)
                path_df.rename(columns={"vert_idx_old": "associated_idx"}, inplace=True)

                if return_Series:
                    path_df.set_index("vertex_idx", inplace=True)
                    self._path_links = path_df.associated_idx.astype(np.uint32)
                else:
                    self._path_links = np.arange(
                        self.__n_vertices_init, dtype=np.uint32
                    )
                    self._path_links[
                        path_df.vertex_idx.values
                    ] = path_df.associated_idx.values

        # deal with infinity
        if return_inf:
            path_length_values = np.where(
                path_length_values == DTYPE_INF_PY, np.inf, path_length_values
            )

        # reorder path lengths
        if return_Series:
            if self._permute:
                self._vertices["path_length"] = path_length_values
                path_lengths_df = self._vertices[
                    ["vert_idx_old", "path_length"]
                ].sort_values(by="vert_idx_old")
                path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
                path_lengths_df.index.name = "vertex_idx"
                path_lengths_series = path_lengths_df.path_length
            else:
                path_lengths_series = pd.Series(path_length_values)
                path_lengths_series.index.name = "vertex_idx"
                path_lengths_series.name = "path_length"

            return path_lengths_series

        else:
            if self._permute:
                self._vertices["path_length"] = path_length_values
                if return_inf:
                    path_length_values = np.inf * np.ones(self.__n_vertices_init)
                else:
                    path_length_values = DTYPE_INF_PY * np.ones(self.__n_vertices_init)
                path_length_values[
                    self._vertices.vert_idx_old.values
                ] = self._vertices.path_length.values

            return path_length_values

    def get_path(self, vertex_idx):
        if self._path_links is None:
            warnings.warn(
                "Current Dijkstra instance has not path attribute : \
                make sure path_tracking is set to True, and run the \
                shortest path algorithm",
                UserWarning,
            )
        else:
            if isinstance(self._path_links, pd.Series):
                path_vertices = compute_path(self._path_links.values, vertex_idx)
            else:
                path_vertices = compute_path(self._path_links, vertex_idx)
            return path_vertices


class HyperpathGenerating:
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

    def run(self, origin, destination, volume, return_inf=False):
        # column storing the resulting edge volumes
        self._edges["volume"] = 0.0
        self.u_i_vec = None

        # vertex least travel time
        u_i_vec = DTYPE_INF_PY * np.ones(self.vertex_count, dtype=DTYPE_PY)

        # input check
        if type(volume) is not list:
            volume = [volume]
        if self._orientation == "out":
            self._check_vertex_idx(origin)
            if type(destination) is not list:
                destination = [destination]
            assert len(destination) == len(volume)
            for i, item in enumerate(destination):
                self._check_vertex_idx(item)
                self._check_volume(volume[i])
            demand_indices = np.array(destination, dtype=np.uint32)
        elif self._orientation == "in":
            if type(origin) is not list:
                origin = [origin]
            assert len(origin) == len(volume)
            for i, item in enumerate(origin):
                self._check_vertex_idx(item)
                self._check_volume(volume[i])
            self._check_vertex_idx(destination)
            demand_indices = np.array(origin, dtype=np.uint32)
        assert isinstance(return_inf, bool)

        demand_values = np.array(volume, dtype=DTYPE_PY)

        if self._orientation == "out":
            raise NotImplementedError(
                "one-to-many Spiess & Florian's algorithm not implemented yet"
            )
        elif self._orientation == "in":
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
        if type(edges) != pd.core.frame.DataFrame:
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
