""" 
Path-related methods.
"""

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
    compute_stsp,
    compute_sssp_w_path,
    compute_stsp_w_path,
)
from edsger.spiess_florian import compute_SF_in
from edsger.star import (
    convert_graph_to_csc_float64,
    convert_graph_to_csc_uint32,
    convert_graph_to_csr_float64,
    convert_graph_to_csr_uint32,
)


class Dijkstra:
    """
    Dijkstra's algorithm for directed graphs with positive edge weights.
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
        self.n_edges = len(self._edges)

        # reindex the vertices
        self._permute = permute
        if self._permute:
            self._n_vertices_old = self._edges[[tail, head]].max(axis=0).max() + 1
            self._vertices = self._permute_graph(tail, head)
            self.n_vertices = len(self._vertices)
        else:
            self._vertices = None
            self.n_vertices = self._edges[[tail, head]].max(axis=0).max() + 1
            self._n_vertices_old = self.n_vertices

        # convert to CSR/CSC
        self._check_orientation(orientation)
        self._orientation = orientation
        if self._orientation == "out":
            fs_indptr, fs_indices, fs_data = convert_graph_to_csr_float64(
                self._edges, tail, head, weight, self.n_vertices
            )
            self._indices = fs_indices.astype(np.uint32)
            self._indptr = fs_indptr.astype(np.uint32)
            self._edge_weights = fs_data.astype(DTYPE_PY)
        else:
            rs_indptr, rs_indices, rs_data = convert_graph_to_csc_float64(
                self._edges, tail, head, weight, self.n_vertices
            )
            self._indices = rs_indices.astype(np.uint32)
            self._indptr = rs_indptr.astype(np.uint32)
            self._edge_weights = rs_data.astype(DTYPE_PY)

    def _check_edges(self, edges, tail, head, weight):
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
        """Create a vertex table and reindex the vertices."""

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
            ]
        else:
            if vertex_idx >= self.n_vertices:
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
        heap_length = int(np.rint(heap_length_ratio * self.n_vertices))

        # compute path length
        if not path_tracking:
            self.path = None
            if self._orientation == "in":
                path_length_values = compute_stsp(
                    self._indptr,
                    self._indices,
                    self._edge_weights,
                    vertex_new,
                    self.n_vertices,
                    heap_length,
                )
            else:
                path_length_values = compute_sssp(
                    self._indptr,
                    self._indices,
                    self._edge_weights,
                    vertex_new,
                    self.n_vertices,
                    heap_length,
                )
        else:
            self.path = np.arange(0, self.n_vertices, dtype=np.uint32)
            if self._orientation == "in":
                path_length_values = compute_stsp_w_path(
                    self._indptr,
                    self._indices,
                    self._edge_weights,
                    self.path,
                    vertex_new,
                    self.n_vertices,
                    heap_length,
                )
            else:
                path_length_values = compute_sssp_w_path(
                    self._indptr,
                    self._indices,
                    self._edge_weights,
                    self.path,
                    vertex_new,
                    self.n_vertices,
                    heap_length,
                )

            if self._permute:
                # permute back the path vertex indices
                path_df = pd.DataFrame(
                    data={
                        "vertex_idx": np.arange(self.n_vertices),
                        "associated_idx": self.path,
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
                    self.path = path_df.associated_idx
                else:
                    self.path = np.arange(self._n_vertices_old)
                    self.path[path_df.vertex_idx.values] = path_df.associated_idx.values

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
                path_lengths_df = self._vertices[
                    ["vert_idx_old", "path_length"]
                ].sort_values(by="vert_idx_old")
                path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
                path_lengths_df.index.name = "vertex_idx"
                path_lengths_series = path_lengths_df.path_length
                path_length_values = path_lengths_series.values

            return path_length_values


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
            self._indptr = fs_indptr.astype(np.uint32)
            self._edge_idx = fs_data.astype(np.uint32)
        else:
            rs_indptr, _, rs_data = convert_graph_to_csc_uint32(
                self._edges, tail, head, data_col, self.vertex_count
            )
            self._indptr = rs_indptr.astype(np.uint32)
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
                self._indptr,
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
