"""
Run Dijkstra's algorithm (SSSP) on DIMACS networks.

Example :
> python dijkstra_dimacs.py -n USA -r 4 -c True
"""

import os
import sys
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd
from edsger.path import Dijkstra
from loguru import logger
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra

logger.remove()
fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    + " <level>{message}</level>"
)
logger.add(sys.stderr, format=fmt)

parser = ArgumentParser(description="Command line interface to dijkstra_dimacs.py")
parser.add_argument(
    "-d",
    "--dir",
    dest="data_dir",
    help="Data folder with network sub-folders",
    metavar="TXT",
    type=str,
    required=False,
    default=os.getenv("DIMACS_DATA_DIR", "/home/francois/Data/DIMACS_road_networks/"),
)
parser.add_argument(
    "-n",
    "--network",
    dest="network_name",
    help="network name, must be 'NY', 'BAY', 'COL', 'FLA', 'NW', "
    + "'NE', 'CAL', 'LKS', 'E', 'W', 'CTR', 'USA'",
    metavar="TXT",
    type=str,
    required=True,
)
parser.add_argument(
    "-l",
    "--library",
    dest="library_name",
    help='library name, must be "E" (Edsger), "GT" (graph-tool), "NK" (NetworKit)',
    metavar="TXT",
    type=str,
    required=False,
    default="E",
)
parser.add_argument(
    "-f",
    "--from",
    dest="idx_from",
    help="source vertex index",
    metavar="INT",
    type=int,
    required=False,
    default=1000,
)
parser.add_argument(
    "-r",
    "--repeat",
    dest="repeat",
    help="repeat",
    metavar="INT",
    type=int,
    required=False,
    default=1,
)
parser.add_argument(
    "-c",
    "--check_result",
    dest="check_result",
    help="check the resulting path lengths against SciPy",
    action="store_true",
)
parser.add_argument(
    "-q",
    "--heap_length_ratio",
    dest="heap_length_ratio",
    help="heap length ratio (0, 1]",
    metavar="FLOAT",
    type=float,
    required=False,
    default=1.0,
)
args = parser.parse_args()
data_dir = args.data_dir
data_dir = os.path.abspath(data_dir)
reg = args.network_name
reg = reg.upper()
idx_from = args.idx_from
repeat = args.repeat
check_result = args.check_result
lib = args.library_name.upper()
heap_length_ratio = args.heap_length_ratio

# lib name check
assert lib in ["E", "GT", "NK"]

data_dir_found = os.path.exists(data_dir)
if data_dir_found:
    logger.info(f"data dir : {data_dir}")
else:
    logger.critical(f"data dir '{data_dir}' not found")
    sys.exit()

# network name check
regions_usa = [
    "NY",
    "BAY",
    "COL",
    "FLA",
    "NW",
    "NE",
    "CAL",
    "LKS",
    "E",
    "W",
    "CTR",
    "USA",
]
if reg in regions_usa:
    logger.info(f"region : {reg}")
else:
    logger.critical(f"region '{reg}' invalid")
    sys.exit()

if isinstance(idx_from, int) and (idx_from >= 0):
    logger.info(f"idx_from : {idx_from}")
else:
    logger.critical(f"invalid value '{idx_from}' for idx_from")
    sys.exit()

if isinstance(repeat, int) and (repeat > 0):
    logger.info(f"repeat : {repeat}")
else:
    logger.critical(f"invalid value '{repeat}' for repeat")
    sys.exit()

if isinstance(check_result, bool):
    logger.info(f"check result : {check_result}")
else:
    logger.critical(f"invalid value '{check_result}' for check_result")
    sys.exit()

# locate the parquet file
network_file_path = os.path.join(data_dir, f"{reg}/USA-road-t.{reg}.gr.parquet")
network_file_found = os.path.exists(network_file_path)
if network_file_found:
    logger.info(f"network file path : {network_file_path}")
else:
    logger.critical(f"network file path '{network_file_path}' not found")
    sys.exit()

# load the network into a Pandas dataframe
edges = pd.read_parquet(network_file_path)
edges.rename(columns={"source": "tail", "target": "head", "tt": "weight"}, inplace=True)
edge_count = len(edges)
vertex_count = edges[["tail", "head"]].max().max() + 1
logger.info(f"{edge_count} edges and {vertex_count} vertices")

if lib == "E":
    # Edsger
    # ------

    logger.info("Edsger init")
    edges[["tail", "head"]] = edges[["tail", "head"]].astype(np.uint32)
    sp = Dijkstra(edges, orientation="out", check_edges=False)

    # SSSP

    results = []
    logger.info("Edsger run")
    for i in range(repeat):
        d = {}

        start = perf_counter()

        dist_matrix = sp.run(
            vertex_idx=idx_from, return_inf=True, heap_length_ratio=heap_length_ratio
        )

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"Edsger Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "edsger",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    df = pd.DataFrame.from_records(results)
    logger.info(f"Edsger min elapsed time : {df.elapsed_time.min():8.4f} s")

elif lib == "GT":
    # graph-tool
    # ----------

    logger.info("graph-tool init")

    import graph_tool as gt
    from graph_tool import topology

    # create the graph
    g = gt.Graph(directed=True)

    # create the vertices
    g.add_vertex(vertex_count)

    # create the edges
    g.add_edge_list(edges[["tail", "head"]].values)

    # edge property for the travel time
    eprop_t = g.new_edge_property("float")
    g.edge_properties["t"] = eprop_t  # internal property
    g.edge_properties["t"].a = edges["weight"].values

    # SSSP

    results = []
    logger.info("graph-tool Run")
    for i in range(repeat):
        d = {}

        start = perf_counter()

        dist = gt.topology.shortest_distance(
            g,
            source=g.vertex(idx_from),
            weights=g.ep.t,
            negative_weights=False,
            directed=True,
        )
        dist_matrix = dist.a

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"graph-tool Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "graph-tool",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    df = pd.DataFrame.from_records(results)
    logger.info(f"graph-tool min elapsed time : {df.elapsed_time.min():8.4f} s")

elif lib == "NK":
    # NetworKit
    # ---------

    import networkit as nk

    nk_file_format = nk.graphio.Format.NetworkitBinary
    networkit_file_path = os.path.join(
        data_dir, f"{reg}/USA-road-t.{reg}.gr.NetworkitBinary"
    )

    if os.path.exists(networkit_file_path):
        g = nk.graphio.readGraph(networkit_file_path, nk_file_format)

    else:
        g = nk.Graph(n=vertex_count, weighted=True, directed=True, edgesIndexed=False)

        for row in edges.itertuples():
            g.addEdge(row.tail, row.head, w=row.weight)

        nk.graphio.writeGraph(g, networkit_file_path, nk_file_format)

    nk_dijkstra = nk.distance.Dijkstra(
        g, idx_from, storePaths=False, storeNodesSortedByDistance=False
    )

    # SSSP

    results = []
    logger.info("NetworKit Run")
    for i in range(repeat):
        d = {}

        start = perf_counter()

        nk_dijkstra.run()
        dist_matrix = np.asarray(nk_dijkstra.getDistances(asarray=True))
        dist_matrix = np.where(dist_matrix >= 1.79769313e308, np.inf, dist_matrix)

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"NetworKit Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "NetworKit",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
        }
        results.append(d)

    df = pd.DataFrame.from_records(results)
    logger.info(f"NetworKit min elapsed time : {df.elapsed_time.min():8.4f} s")


if check_result:
    logger.info("result check")

    # SciPy
    logger.info("SciPy init")
    data = edges["weight"].values
    row = edges["tail"].values.astype(np.int32)
    col = edges["head"].values.astype(np.int32)
    graph_coo = coo_array((data, (row, col)), shape=(vertex_count, vertex_count))
    graph_csr = graph_coo.tocsr()

    logger.info("SciPy run")
    start = perf_counter()
    dist_matrix_ref = dijkstra(
        csgraph=graph_csr,
        directed=True,
        indices=idx_from,
        return_predecessors=False,
    )
    end = perf_counter()
    elapsed_time = end - start

    logger.info(f"SciPy Dijkstra - Elapsed time: {elapsed_time:8.4f} s")
    logger.info(f"isinf : {np.isinf(dist_matrix_ref).any()}")
    logger.info(
        f"allclose : {np.allclose(dist_matrix_ref, dist_matrix, equal_nan=True)}"
    )


logger.info("exit")
