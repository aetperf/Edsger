""" Path-related methods.
"""

import numpy as np
import pandas as pd

from edsger.commons import DTYPE_INF_PY
from edsger.networks import create_Spiess_network
from edsger.star import convert_graph_to_csr_uint32, convert_graph_to_csc_uint32


class HyperpathGenerating:
    def __init__(
        self,
        edges_df,
        tail="tail",
        head="head",
        trav_time="trav_time",
        freq="freq",
        check_edges=False,
        algo="SF",
        orientation="many-to-one",
    ):

        # load the edges
        if check_edges:
            self._check_edges(edges_df, tail, head, trav_time, freq)
        self._edges = edges_df[[tail, head, trav_time, freq]].copy(deep=True)
        self.edge_count = len(self._edges)

        # remove inf values if any
        for col in [trav_time, freq]:
            self._edges[col] = np.where(
                np.isinf(self._edges[col]), DTYPE_INF_PY, self._edges[col]
            )

        # create an edge index column
        self._edges = self._edges.reset_index(drop=True)
        data_col = "edge_idx"
        self._edges[data_col] = self._edges.index

        # convert to CSR/CSC format
        self.vertex_count = self._edges[[tail, head]].max().max() + 1
        assert orientation in ["one-to-many", "many-to-one"]
        self._orientation = orientation
        if self._orientation == "one-to-many":
            fs_indptr, fs_indices, fs_data = convert_graph_to_csr_uint32(
                self._edges, tail, head, data_col, self.vertex_count, self.edge_count
            )
            self._indices = fs_indices.astype(np.uint32)
            self._indptr = fs_indptr.astype(np.uint32)
            self._edge_idx = fs_data.astype(np.uint32)
            raise NotImplementedError(
                "one-to-many Spiess & Florian's algorithm not implemented yet"
            )
        else:
            rs_indptr, rs_indices, rs_data = convert_graph_to_csc_uint32(
                self._edges, tail, head, data_col, self.vertex_count, self.edge_count
            )
            self._indices = rs_indices.astype(np.uint32)
            self._indptr = rs_indptr.astype(np.uint32)
            self._edge_idx = rs_data.astype(np.uint32)

    def _check_edges(self, edges_df, tail, head, trav_time, freq):

        if type(edges_df) != pd.core.frame.DataFrame:
            raise TypeError("edges_df should be a pandas DataFrame")

        for col in [tail, head, trav_time, freq]:
            if col not in edges_df:
                raise KeyError(
                    f"edge column '{col}' not found in graph edges dataframe"
                )

        if edges_df[[tail, head, trav_time, freq]].isna().any().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges_df[[{tail}, {head}, {trav_time}, {freq}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges_df[col].dtype):
                raise TypeError(f"column '{col}' should be of integer type")

        for col in [trav_time, freq]:
            if not pd.api.types.is_numeric_dtype(edges_df[col].dtype):
                raise TypeError(f"column '{col}' should be of numeric type")

            if edges_df[col].min() < 0.0:
                raise ValueError(f"column '{col}' should be nonnegative")


if __name__ == "__main__":

    edges_df = create_Spiess_network()
    hp = HyperpathGenerating(edges_df, check_edges=True)
    print(hp._edges)
    print("done")


# class ShortestPath:
#     def __init__(
#         self,
#         edges_df,
#         source="source",
#         target="target",
#         weight="weight",
#         orientation="one-to-all",
#         check_edges=True,
#         permute=False,
#         heap_type="bin",
#     ):
#         self.time = {}
#         self._return_Series = True

#         t = Timer()
#         t.start()
#         # load the edges
#         if check_edges:
#             self._check_edges(edges_df, source, target, weight)
#         self._edges = edges_df  # not a copy (be careful not to modify it)
#         self.n_edges = len(self._edges)
#         t.stop()
#         self.time["load the edges"] = t.interval

#         self._heap_type = heap_type

#         # reindex the vertices
#         t = Timer()
#         t.start()
#         self._permute = permute
#         if self._permute:
#             self._vertices = self._permute_graph(source, target)
#             self.n_vertices = len(self._vertices)
#         else:
#             self.n_vertices = self._edges[[source, target]].max().max() + 1
#         t.stop()
#         self.time["reindex the vertices"] = t.interval

#         # convert to CSR/CSC
#         t = Timer()
#         t.start()
#         self._check_orientation(orientation)
#         self._orientation = orientation
#         if self._orientation == "one-to-all":
#             fs_indptr, fs_indices, fs_data = convert_graph_to_csr(
#                 self._edges, source, target, weight, self.n_vertices, self.n_edges
#             )
#             self._indices = fs_indices.astype(np.uint32)
#             self._indptr = fs_indptr.astype(np.uint32)
#             self._edge_weights = fs_data.astype(DTYPE_PY)
#         else:
#             rs_indptr, rs_indices, rs_data = convert_graph_to_csc(
#                 self._edges, source, target, weight, self.n_vertices, self.n_edges
#             )
#             self._indices = rs_indices.astype(np.uint32)
#             self._indptr = rs_indptr.astype(np.uint32)
#             self._edge_weights = rs_data.astype(DTYPE_PY)
#             raise NotImplementedError("one-to_all shortest path not implemented yet")
#         t.stop()
#         self.time["convert to CSR/CSC"] = t.interval

#     def _check_edges(self, edges_df, source, target, weight):

#         if type(edges_df) != pd.core.frame.DataFrame:
#             raise TypeError("edges_df should be a pandas DataFrame")

#         if source not in edges_df:
#             raise KeyError(
#                 f"edge source column '{source}'  not found in graph edges dataframe"
#             )

#         if target not in edges_df:
#             raise KeyError(
#                 f"edge target column '{target}' not found in graph edges dataframe"
#             )

#         if weight not in edges_df:
#             raise KeyError(
#                 f"edge weight column '{weight}' not found in graph edges dataframe"
#             )

#         if edges_df[[source, target, weight]].isna().any().any():
#             raise ValueError(
#                 " ".join(
#                     [
#                         f"edges_df[[{source}, {target}, {weight}]] ",
#                         "should not have any missing value",
#                     ]
#                 )
#             )

#         for col in [source, target]:
#             if not pd.api.types.is_integer_dtype(edges_df[col].dtype):
#                 raise TypeError(f"edges_df['{col}'] should be of integer type")

#         if not pd.api.types.is_numeric_dtype(edges_df[weight].dtype):
#             raise TypeError(f"edges_df['{weight}'] should be of numeric type")

#         if edges_df[weight].min() < 0.0:
#             raise ValueError(f"edges_df['{weight}'] should be nonnegative")

#         if not np.isfinite(edges_df[weight]).all():
#             raise ValueError(f"edges_df['{weight}'] should be finite")


#     def _permute_graph(self, source, target):
#         """Create a vertex table and reindex the vertices."""

#         vertices = pd.DataFrame(
#             data={
#                 "vert_idx": np.union1d(
#                     self._edges[source].values, self._edges[target].values
#                 )
#             }
#         )
#         vertices["vert_idx_new"] = vertices.index
#         vertices.index.name = "index"

#         self._edges = pd.merge(
#             self._edges,
#             vertices[["vert_idx", "vert_idx_new"]],
#             left_on=source,
#             right_on="vert_idx",
#             how="left",
#         )
#         self._edges.drop([source, "vert_idx"], axis=1, inplace=True)
#         self._edges.rename(columns={"vert_idx_new": source}, inplace=True)

#         self._edges = pd.merge(
#             self._edges,
#             vertices[["vert_idx", "vert_idx_new"]],
#             left_on=target,
#             right_on="vert_idx",
#             how="left",
#         )
#         self._edges.drop([target, "vert_idx"], axis=1, inplace=True)
#         self._edges.rename(columns={"vert_idx_new": target}, inplace=True)

#         vertices.rename(columns={"vert_idx": "vert_idx_old"}, inplace=True)
#         vertices.reset_index(drop=True, inplace=True)
#         vertices.sort_values(by="vert_idx_new", inplace=True)

#         vertices.index.name = "index"
#         self._edges.index.name = "index"

#         return vertices

#     def _check_orientation(self, orientation):
#         if orientation not in ["one-to-all", "all-to-one"]:
#             raise ValueError(
#                 f"orientation should be either 'one-to-all' or 'all-to-one'"
#             )

#     def run(self, vertex_idx, return_inf=False, return_Series=True, heap_length_ratio=1.0):

#         self._return_Series = return_Series

#         # check the source/target vertex
#         t = Timer()
#         t.start()
#         if self._permute:
#             if vertex_idx not in self._vertices.vert_idx_old.values:
#                 raise ValueError(f"vertex {vertex_idx} not found in graph")
#             vertex_new = self._vertices.loc[
#                 self._vertices.vert_idx_old == vertex_idx, "vert_idx_new"
#             ]
#         else:
#             if vertex_idx >= self.n_vertices:
#                 raise ValueError(f"vertex {vertex_idx} not found in graph")
#             vertex_new = vertex_idx
#         t.stop()
#         self.time["check the source/target vertex"] = t.interval

#         # compute path length
#         t = Timer()
#         t.start()
#         if self._orientation == "one-to-all":
#             if self._heap_type == "bin":
#                 path_length_values = path_length_from_bin(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#             elif self._heap_type == "fib":
#                 path_length_values = path_length_from_fib(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#             elif self._heap_type == "3ary":
#                 path_length_values = path_length_from_3ary(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#             elif self._heap_type == "4ary":
#                 path_length_values = path_length_from_4ary(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#             elif self._heap_type == "bin_basic":
#                 path_length_values = path_length_from_bin_basic(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#             elif self._heap_type == "bin_length":
#                 assert heap_length_ratio <= 1.0
#                 assert heap_length_ratio > 0.0
#                 heap_length = int(np.rint(heap_length_ratio * self.n_vertices))
#                 path_length_values = path_length_from_bhl(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                     heap_length
#                 )
#             else:  # bin_basic_insert_all
#                 path_length_values = path_length_from_bin_basic_insert_all(
#                     self._indices,
#                     self._indptr,
#                     self._edge_weights,
#                     vertex_new,
#                     self.n_vertices,
#                 )
#         t.stop()
#         self.time["compute path length"] = t.interval

#         # deal with infinity
#         if return_inf:
#             path_length_values = np.where(
#                 path_length_values == DTYPE_INF_PY, np.inf, path_length_values
#             )

#         # reorder results
#         if self._return_Series:

#             t = Timer()
#             t.start()

#             if self._permute:
#                 self._vertices["path_length"] = path_length_values
#                 path_lengths_df = self._vertices[
#                     ["vert_idx_old", "path_length"]
#                 ].sort_values(by="vert_idx_old")
#                 path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
#                 path_lengths_df.index.name = "vertex_idx"
#                 path_lengths_series = path_lengths_df.path_length
#             else:
#                 path_lengths_series = pd.Series(path_length_values)
#                 path_lengths_series.index.name = "vertex_idx"
#                 path_lengths_series.name = "path_length"

#             t.stop()
#             self.time["reorder results"] = t.interval

#             return path_lengths_series

#         else:

#             t = Timer()
#             t.start()

#             if self._permute:
#                 self._vertices["path_length"] = path_length_values
#                 path_lengths_df = self._vertices[
#                     ["vert_idx_old", "path_length"]
#                 ].sort_values(by="vert_idx_old")
#                 path_lengths_df.set_index("vert_idx_old", drop=True, inplace=True)
#                 path_lengths_df.index.name = "vertex_idx"
#                 path_lengths_series = path_lengths_df.path_length
#                 path_length_values = path_lengths_series.values

#             t.stop()
#             self.time["reorder results"] = t.interval

#             return path_length_values

#     def get_timings(self):
#         return pd.DataFrame.from_dict(self.time, orient="index", columns=["et_s"])
