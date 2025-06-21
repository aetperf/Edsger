"""Tests for pq_bin_dec_0b.pyx.

py.test tests/test_pq_4ary_dec_0b.py

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

from edsger.pq_4ary_dec_0b import *


def test_init_01():
    init_01()


def test_insert_01():
    insert_01()


def test_insert_02():
    insert_02()


def test_insert_03():
    insert_03()


def test_peek_01():
    peek_01()


def test_extract_min_01():
    extract_min_01()


def test_is_empty_01():
    is_empty_01()


def test_decrease_key_01():
    decrease_key_01()


def test_sort_01():
    sort_01(1000)
