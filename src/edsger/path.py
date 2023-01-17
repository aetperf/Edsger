""" Path-related methods.
"""

import numpy as np
import pandas as pd

from edsger.commons import DTYPE_INF_PY


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
    ):

        # load the edges
        if check_edges:
            self._check_edges(edges_df, tail, head, trav_time, freq)
        self._edges = edges_df.copy(deep=True)

        # remove inf values in any
        for col in [trav_time, freq]:
            self._edges[col] = np.where(
                np.isinf(self._edges[col]), DTYPE_INF_PY, self._edges[col]
            )

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


def create_Spiess_network():

    a_very_small_time_interval = 1.0e-06
    dwell_time = 5.0
    boarding_time = 0.5 * dwell_time + a_very_small_time_interval
    alighting_time = 0.5 * dwell_time + a_very_small_time_interval

    line1_freq = 1.0 / (60.0 * 12.0)
    line2_freq = 1.0 / (60.0 * 12.0)
    line3_freq = 1.0 / (60.0 * 30.0)
    line4_freq = 1.0 / (60.0 * 6.0)

    # stop A
    # 0 stop vertex
    # 1 boarding vertex
    # 2 boarding vertex

    # stop X
    # 3 stop vertex
    # 4 boarding vertex
    # 5 alighting vertex
    # 6 boarding vertex

    # stop Y
    # 7  stop vertex
    # 8  boarding vertex
    # 9  alighting vertex
    # 10 boarding vertex
    # 11 alighting vertex

    # stop B
    # 12 stop vertex
    # 13 alighting vertex
    # 14 alighting vertex
    # 15 alighting vertex

    tail = []
    head = []
    trav_time = []
    freq = []

    # edge 0
    # stop A : to line 1
    # boarding edge
    tail.append(0)
    head.append(2)
    freq.append(line1_freq)
    trav_time.append(boarding_time)

    # edge 1
    # stop A : to line 2
    # boarding edge
    tail.append(0)
    head.append(1)
    freq.append(line2_freq)
    trav_time.append(boarding_time)

    # edge 2
    # line 1 : first segment
    # on-board edge
    tail.append(2)
    head.append(15)
    freq.append(np.inf)
    trav_time.append(25.0 * 60.0)

    # edge 3
    # line 2 : first segment
    # on-board edge
    tail.append(1)
    head.append(5)
    freq.append(np.inf)
    trav_time.append(7.0 * 60.0)

    # edge 4
    # stop X : from line 2
    # alighting edge
    tail.append(5)
    head.append(3)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 5
    # stop X : in line 2
    # dwell edge
    tail.append(5)
    head.append(6)
    freq.append(np.inf)
    trav_time.append(dwell_time)

    # edge 6
    # stop X : from line 2 to line 3
    # transfer edge
    tail.append(5)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(dwell_time)

    # edge 7
    # stop X : to line 2
    # boarding edge
    tail.append(3)
    head.append(6)
    freq.append(line2_freq)
    trav_time.append(boarding_time)

    # edge 8
    # stop X : to line 3
    # boarding edge
    tail.append(3)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(boarding_time)

    # edge 9
    # line 2 : second segment
    # on-board edge
    tail.append(6)
    head.append(11)
    freq.append(np.inf)
    trav_time.append(6.0 * 60.0)

    # edge 10
    # line 3 : first segment
    # on-board edge
    tail.append(4)
    head.append(9)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)

    # edge 11
    # stop Y : from line 3
    # alighting edge
    tail.append(9)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 12
    # stop Y : from line 2
    # alighting edge
    tail.append(11)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 13
    # stop Y : from line 2 to line 3
    # transfer edge
    tail.append(11)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(dwell_time)

    # edge 14
    # stop Y : from line 2 to line 4
    # transfer edge
    tail.append(11)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)

    # edge 15
    # stop Y : from line 3 to line 4
    # transfer edge
    tail.append(9)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)

    # edge 16
    # stop Y : in line 3
    # dwell edge
    tail.append(9)
    head.append(10)
    freq.append(np.inf)
    trav_time.append(dwell_time)

    # edge 17
    # stop Y : to line 3
    # boarding edge
    tail.append(7)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(boarding_time)

    # edge 18
    # stop Y : to line 4
    # boarding edge
    tail.append(7)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(boarding_time)

    # edge 19
    # line 3 : second segment
    # on-board edge
    tail.append(10)
    head.append(14)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)

    # edge 20
    # line 4 : first segment
    # on-board edge
    tail.append(8)
    head.append(13)
    freq.append(np.inf)
    trav_time.append(10.0 * 60.0)

    # edge 21
    # stop Y : from line 1
    # alighting edge
    tail.append(15)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 22
    # stop Y : from line 3
    # alighting edge
    tail.append(14)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    # edge 23
    # stop Y : from line 4
    # alighting edge
    tail.append(13)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)

    edges_df = pd.DataFrame(
        data={"tail": tail, "head": head, "trav_time": trav_time, "freq": freq}
    )
    return edges_df


if __name__ == "__main__":

    edges_df = create_Spiess_network()
    hp = HyperpathGenerating(edges_df)
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
