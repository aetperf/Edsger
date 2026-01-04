"""Type stubs for edsger.star Cython module."""

from typing import Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd

def convert_graph_to_csr_uint32(
    edges: pd.DataFrame,
    tail: str,
    head: str,
    data: str,
    vertex_count: int,
) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def convert_graph_to_csc_uint32(
    edges: pd.DataFrame,
    tail: str,
    head: str,
    data: str,
    vertex_count: int,
) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def convert_graph_to_csr_float64(
    edges: pd.DataFrame,
    tail: str,
    head: str,
    data: str,
    vertex_count: int,
) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float64]]: ...
def convert_graph_to_csc_float64(
    edges: pd.DataFrame,
    tail: str,
    head: str,
    data: str,
    vertex_count: int,
) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float64]]: ...
