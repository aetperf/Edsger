"""Tests for path_tracking.pyx.

py.test tests/test_path_tracking.py
"""

import numpy as np
import pytest
from edsger.path_tracking import compute_path


def test_01(n=10):
    path_links = np.arange(n, dtype=np.uint32)
    path_links[1:] -= 1

    path_vertices = compute_path(path_links, n - 1)
    path_vertices_ref = np.array(list(range(n))[::-1], dtype=np.uint32)

    np.testing.assert_array_equal(path_vertices, path_vertices_ref)


@pytest.mark.parametrize("vertex_idx", [-1, 10, 1000])
def test_out_of_bounds_vertex(vertex_idx, n=10):
    """Bounds checking is disabled in path_tracking.pyx: the vertex index is
    checked explicitly."""
    path_links = np.arange(n, dtype=np.uint32)
    path_links[1:] -= 1

    with pytest.raises(IndexError):
        compute_path(path_links, vertex_idx)
