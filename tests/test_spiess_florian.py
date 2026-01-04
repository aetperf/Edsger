"""Tests for spiess_florian.pyx.

py.test tests/test_spiess_florian.py
"""

from edsger.spiess_florian import (  # type: ignore
    compute_SF_in_01,
    compute_SF_in_02,
)


def test_compute_SF_in_01():
    compute_SF_in_01()


def test_compute_SF_in_02():
    compute_SF_in_02()
