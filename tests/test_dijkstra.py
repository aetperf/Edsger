"""Tests for dijkstra.pyx.

py.test tests/test_dijkstra.py

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

from edsger.dijkstra import (
    compute_sssp_01,
    compute_sssp_02,
    compute_stsp_01,
    compute_stsp_02,
)


def test_compute_sssp_01():
    compute_sssp_01()


def test_compute_stsp_01():
    compute_stsp_01()


def test_compute_sssp_02():
    compute_sssp_02()


def test_compute_stsp_02():
    compute_stsp_02()
