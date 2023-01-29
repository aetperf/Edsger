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

    edges = create_Spiess_network()
    hp = HyperpathGenerating(edges, check_edges=False)
    hp.run(origin=0, destination=12, volume=1.0)

    assert np.allclose(edges["volume"].values, hp._edges["volume"].values)
