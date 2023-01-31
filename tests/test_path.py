""" Tests for path.py.

author : Francois Pacull
copyright : Architecture & Performance
email: francois.pacull@architecture-performance.fr
license : MIT
"""

import numpy as np

from edsger.networks import create_Spiess_network
from edsger.path import HyperpathGenerating


def test_SF_in_01():

    edges = create_Spiess_network(dwell_time=1.0e-8, a_very_small_time_interval=1.0e-8)
    hp = HyperpathGenerating(edges, check_edges=False)
    hp.run(origin=0, destination=12, volume=1.0)

    assert np.allclose(edges["volume"].values, hp._edges["volume"].values)

    u_i_vec_ref = np.array(
        [
            1.66500000e03,
            1.47000000e03,
            1.50000000e03,
            1.14428572e03,
            4.80000000e02,
            1.05000000e03,
            1.05000000e03,
            6.90000000e02,
            6.00000000e02,
            2.40000000e02,
            2.40000000e02,
            6.90000000e02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    )
    assert np.allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-06, atol=1e-06)
