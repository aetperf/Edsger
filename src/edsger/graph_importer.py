"""
Graph importer module for converting various DataFrame formats to NumPy-backed pandas DataFrames.

This module provides a unified interface for importing graph data from different DataFrame libraries
(pandas with NumPy backend, pandas with Arrow backend, Polars, etc.) and converting them to a
standardized NumPy-backed pandas DataFrame format that is optimal for the graph algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import warnings

import numpy as np
import pandas as pd


class GraphImporter(ABC):
    """
    Abstract base class for importing graph data from various DataFrame libraries.

    All importers convert their input format to a NumPy-backed pandas DataFrame
    with contiguous memory layout for optimal performance in Cython algorithms.
    """

    def __init__(self, edges_df):
        """
        Initialize the importer with a DataFrame.

        Parameters
        ----------
        edges_df : DataFrame-like
            The edges DataFrame in the specific library format.
        """
        self.edges_df = edges_df

    @staticmethod
    def from_dataframe(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        edges,
        tail: str = "tail",
        head: str = "head",
        weight: Optional[str] = None,
        trav_time: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> "GraphImporter":
        """
        Factory method to create the appropriate importer based on DataFrame type.

        Parameters
        ----------
        edges : DataFrame-like
            The edges DataFrame to import.
        tail : str
            Column name for tail vertices.
        head : str
            Column name for head vertices.
        weight : str, optional
            Column name for edge weights (for shortest path algorithms).
        trav_time : str, optional
            Column name for travel time (for hyperpath algorithms).
        freq : str, optional
            Column name for frequency (for hyperpath algorithms).

        Returns
        -------
        GraphImporter
            An instance of the appropriate importer subclass.
        """
        # Note: tail, head, weight, trav_time, freq are part of API but not used here
        # They're used by calling code after factory creates the importer
        # pylint: disable=unused-argument
        try:
            # Check for Polars DataFrame
            if hasattr(edges, "__class__") and edges.__class__.__module__.startswith(
                "polars"
            ):
                return PolarsImporter(edges)
        except (AttributeError, TypeError):
            # If __class__ or __module__ access fails, continue to other checks
            pass

        # Check for pandas DataFrame
        if isinstance(edges, pd.DataFrame):
            try:
                # Check if any column has Arrow backend
                has_arrow = any(
                    hasattr(dtype, "pyarrow_dtype") for dtype in edges.dtypes
                )

                if has_arrow:
                    return PandasArrowImporter(edges)
                return PandasNumpyImporter(edges)
            except (AttributeError, TypeError):
                # If dtype checking fails, assume NumPy backend
                return PandasNumpyImporter(edges)

        # Unknown type - try to convert to pandas
        warnings.warn(
            f"Unknown DataFrame type {type(edges)}. Attempting to convert to pandas.",
            UserWarning,
        )
        return PandasNumpyImporter(pd.DataFrame(edges))

    @abstractmethod
    def to_numpy_edges(self, columns: List[str]) -> pd.DataFrame:
        """
        Convert the DataFrame to a NumPy-backed pandas DataFrame.

        Parameters
        ----------
        columns : List[str]
            List of column names to extract.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with NumPy backend and contiguous memory.
        """

    def _ensure_contiguous(self, array: np.ndarray) -> np.ndarray:
        """
        Ensure the array is C-contiguous.

        Parameters
        ----------
        array : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            C-contiguous array.
        """
        if not array.flags["C_CONTIGUOUS"]:
            return np.ascontiguousarray(array)
        return array


class PandasNumpyImporter(GraphImporter):
    """
    Importer for pandas DataFrames with NumPy backend.

    This is the most efficient case as it requires minimal conversion.
    """

    def to_numpy_edges(self, columns: List[str]) -> pd.DataFrame:
        """
        Extract columns and ensure they are NumPy-backed.

        For NumPy-backed pandas, this is mostly a pass-through operation
        with validation to ensure contiguous memory.
        """
        # Extract only the needed columns
        result_df = self.edges_df[columns].copy(deep=True)

        # Ensure all columns are contiguous NumPy arrays
        for col in columns:
            if not isinstance(result_df[col].values, np.ndarray):
                # Convert to NumPy if somehow not already
                result_df[col] = result_df[col].to_numpy()

        return result_df


class PandasArrowImporter(GraphImporter):
    """
    Importer for pandas DataFrames with Arrow backend.

    Converts Arrow-backed columns to NumPy arrays with proper data types.
    """

    def to_numpy_edges(self, columns: List[str]) -> pd.DataFrame:
        """
        Convert Arrow-backed columns to NumPy arrays.

        Uses to_numpy() method which ensures contiguous memory layout.
        """
        result_data = {}

        for col in columns:
            series = self.edges_df[col]

            # Determine target dtype based on column values
            if col in columns[:2]:  # Assume first two are vertex indices (tail, head)
                # Try to use uint32 for vertex indices if possible
                max_val = series.max()
                if max_val < np.iinfo(np.uint32).max:
                    target_dtype = np.uint32
                else:
                    target_dtype = np.uint64
            else:
                # Use float64 for weights/times/frequencies
                target_dtype = np.float64

            # Convert to NumPy with specified dtype
            if hasattr(series, "to_numpy"):
                # Use to_numpy() for Arrow-backed series
                result_data[col] = series.to_numpy(dtype=target_dtype, copy=True)
            else:
                # Fallback for older pandas versions
                result_data[col] = series.values.astype(target_dtype)

            # Ensure contiguous
            result_data[col] = self._ensure_contiguous(result_data[col])

        return pd.DataFrame(result_data)


class PolarsImporter(GraphImporter):
    """
    Importer for Polars DataFrames.

    Converts Polars DataFrames to NumPy-backed pandas DataFrames.
    """

    def to_numpy_edges(
        self, columns: List[str]
    ) -> pd.DataFrame:  # pylint: disable=too-many-branches
        """
        Convert Polars DataFrame to NumPy-backed pandas DataFrame.

        Uses Polars' to_pandas() method or to_numpy() depending on what's available.
        """
        import importlib.util

        if importlib.util.find_spec("polars") is None:
            raise ImportError(
                "Polars is required to import Polars DataFrames. "
                "Install it with: pip install polars"
            )

        # Select only needed columns
        selected_df = self.edges_df.select(columns)

        # Method 1: Direct to_pandas() conversion (simplest)
        if hasattr(selected_df, "to_pandas"):
            result_df = selected_df.to_pandas()

            # Handle empty DataFrames
            if len(result_df) == 0:
                return result_df

            # Ensure proper dtypes
            for i, col in enumerate(columns):
                if i < 2:  # Vertex indices
                    # Check if column contains numeric data
                    if np.issubdtype(result_df[col].dtype, np.integer):
                        # Try to use uint32 for efficiency
                        max_val = result_df[col].max()
                        if not pd.isna(max_val) and max_val < np.iinfo(np.uint32).max:
                            result_df[col] = result_df[col].astype(np.uint32)
                    # If not numeric (e.g., strings), leave as is
                else:
                    # Weights/times/frequencies
                    result_df[col] = result_df[col].astype(np.float64)

            return result_df

        # Method 2: Column-by-column conversion
        result_data = {}

        # Handle empty DataFrames
        if len(selected_df) == 0:
            return selected_df.to_pandas()

        for i, col in enumerate(columns):
            series = selected_df[col]

            # Determine target dtype
            if i < 2:  # Vertex indices
                # Check if the series contains numeric data
                if hasattr(series, "dtype") and series.dtype.is_integer():
                    max_val = series.max()
                    if max_val is not None and max_val < np.iinfo(np.uint32).max:
                        target_dtype = np.uint32
                    else:
                        target_dtype = np.uint64
                else:
                    # Non-numeric columns, convert to pandas as is
                    result_data[col] = series.to_pandas()
                    continue
            else:
                target_dtype = np.float64

            # Convert to NumPy
            if hasattr(series, "to_numpy"):
                np_array = series.to_numpy().astype(target_dtype)
            else:
                # Fallback for older Polars versions
                np_array = series.to_list()
                np_array = np.array(np_array, dtype=target_dtype)

            # Ensure contiguous
            result_data[col] = self._ensure_contiguous(np_array)

        return pd.DataFrame(result_data)


def standardize_graph_dataframe(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    edges,
    tail: str = "tail",
    head: str = "head",
    weight: Optional[str] = None,
    trav_time: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to standardize any DataFrame format to NumPy-backed pandas.

    Parameters
    ----------
    edges : DataFrame-like
        Input edges DataFrame in any supported format.
    tail : str
        Column name for tail vertices.
    head : str
        Column name for head vertices.
    weight : str, optional
        Column name for edge weights.
    trav_time : str, optional
        Column name for travel time.
    freq : str, optional
        Column name for frequency.

    Returns
    -------
    pd.DataFrame
        NumPy-backed pandas DataFrame with only the specified columns.
    """
    # Determine which columns to extract
    columns = [tail, head]
    if weight is not None:
        columns.append(weight)
    if trav_time is not None:
        columns.append(trav_time)
    if freq is not None:
        columns.append(freq)

    # Create appropriate importer and convert
    importer = GraphImporter.from_dataframe(edges, tail, head, weight, trav_time, freq)
    return importer.to_numpy_edges(columns)
