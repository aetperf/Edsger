"""Tests for path_tracking.pyx.

py.test tests/test_path_tracking.py

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np
from edsger.path_tracking import compute_path


def test_01(n=10):
    path_links = np.arange(n, dtype=np.uint32)
    path_links[1:] -= 1

    path_vertices = compute_path(path_links, n - 1)
    path_vertices_ref = np.array(list(range(n))[::-1], dtype=np.uint32)

    np.testing.assert_array_equal(path_vertices, path_vertices_ref)
