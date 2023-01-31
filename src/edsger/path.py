""" 
Path-related methods.
"""

import numpy as np
import pandas as pd

from edsger.commons import DTYPE_PY, DTYPE_INF_PY, MIN_FREQ_PY, INF_FREQ_PY
from edsger.spiess_florian import compute_SF_in
from edsger.star import convert_graph_to_csr_uint32, convert_graph_to_csc_uint32


class HyperpathGenerating:
    def __init__(
        self,
        edges,
        tail="tail",
        head="head",
        trav_time="trav_time",
        freq="freq",
        check_edges=False,
        algo="SF",
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
            fs_indptr, fs_indices, fs_data = convert_graph_to_csr_uint32(
                self._edges, tail, head, data_col, self.vertex_count
            )
            self._indptr = fs_indptr.astype(np.uint32)
            self._indices = fs_indices.astype(np.uint32)
            self._edge_idx = fs_data.astype(np.uint32)
        else:
            rs_indptr, rs_indices, rs_data = convert_graph_to_csc_uint32(
                self._edges, tail, head, data_col, self.vertex_count
            )
            self._indptr = rs_indptr.astype(np.uint32)
            self._indices = rs_indices.astype(np.uint32)
            self._edge_idx = rs_data.astype(np.uint32)

        # edge attributes
        self._trav_time = self._edges[trav_time].values.astype(DTYPE_PY)
        self._freq = self._edges[freq].values.astype(DTYPE_PY)
        self._tail = self._edges[tail].values.astype(np.uint32)
        self._head = self._edges[head].values.astype(np.uint32)

    def run(self, origin, destination, volume, return_inf=False):

        # column storing the resulting edge volumes
        self._edges["volume"] = 0.0

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
            # compute_SF_out(
            #     self._indptr,
            #     self._indices,
            #     self._edge_idx,
            #     self._trav_time,
            #     self._freq,
            #     self._tail
            #     self._head,
            #     demand_indices,  # destination vertex indices
            #     demand_values,
            #     self._edges["volumes"].values,
            #     u_i_vec,
            #     self.vertex_count,
            #     origin,
            # )
            raise NotImplementedError(
                "one-to-many Spiess & Florian's algorithm not implemented yet"
            )
        elif self._orientation == "in":

            compute_SF_in(
                self._indptr,
                self._indices,
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


# if __name__ == "__main__":

#     tail = [0, 0]
#     head = [1, 1]
#     trav_time = [1.0, 1.0]
#     freq = [2.0, 1.0]
#     vol = [0.666666666, 0.333333333]
#     edges = pd.DataFrame(
#         data={
#             "tail": tail,
#             "head": head,
#             "trav_time": trav_time,
#             "freq": freq,
#             "volume": vol,
#         }
#     )
#     hp = HyperpathGenerating(edges, check_edges=True)
#     print(hp._indptr)
#     print(hp._indices)
#     print(hp._edge_idx)
