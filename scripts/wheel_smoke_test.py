"""Smoke test of an installed Edsger wheel, run by cibuildwheel.

Imports every compiled module and runs each family of algorithms on a small
graph, to check that the wheel can be loaded and works, without the test
dependencies.
"""

import importlib
import importlib.machinery
import pathlib

import numpy as np
import pandas as pd

import edsger

package_dir = pathlib.Path(edsger.__file__).parent
# the extension modules built for this interpreter, e.g. with the suffix
# .cpython-314-x86_64-linux-gnu.so or .cp314-win_amd64.pyd
suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
compiled = sorted(
    path.name.removesuffix(suffix)
    for path in package_dir.iterdir()
    if path.name.endswith(suffix)
)
assert len(compiled) >= 15, compiled
for name in compiled:
    importlib.import_module(f"edsger.{name}")

edges = pd.DataFrame(
    {
        "tail": [0, 0, 1, 2, 2, 3],
        "head": [1, 2, 2, 3, 4, 4],
        "weight": [1.0, 4.0, 2.0, 1.5, 3.0, 1.0],
    }
)
expected = np.array([0.0, 1.0, 3.0, 4.5, 5.5])
np.testing.assert_array_equal(edsger.Dijkstra(edges).run(0), expected)
np.testing.assert_array_equal(edsger.BellmanFord(edges).run(0), expected)
assert edsger.BidirectionalDijkstra(edges).run(0, 4) == 5.5
astar = edsger.AStar(edges)
coordinates = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0] * 5})
astar.set_coordinates(coordinates, metric="haversine")
assert astar.run(0, 4) == 5.5
assert edsger.BFS(edges).run(0).tolist() == [-9999, 0, 0, 2, 2]
assert edsger.DFS(edges).run(0).tolist() == [-9999, 0, 1, 2, 3]
assert edsger.ConnectedComponents(edges).run("strong").tolist() == [0, 1, 2, 3, 4]
mst = edsger.MinimumSpanningTree(edges)
mst.run()
assert mst.total_weight == 5.5
assert abs(edsger.Centrality(edges).pagerank().sum() - 1.0) < 1e-9
assert edsger.Dijkstra(edges).run_many([0, 1], n_jobs=2).shape == (2, 5)
print(
    f"edsger {edsger.__version__}: {len(compiled)} compiled modules, smoke test passed"
)
