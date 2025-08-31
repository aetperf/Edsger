"""
Run Bellman-Ford algorithm (SSSP) on DIMACS networks.

The Bellman-Ford algorithm can handle negative edge weights and detect negative cycles,
unlike Dijkstra's algorithm which requires non-negative weights.

Example :
> python bellman_ford_dimacs.py -n USA -r 4 -c True
"""

import os
import sys
import platform
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import coo_array
from scipy.sparse.csgraph import bellman_ford

from edsger.path import BellmanFord

logger.remove()
FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    + " <level>{message}</level>"
)
logger.add(sys.stderr, format=FMT)

# Determine default data directory based on OS
if platform.system() == "Windows":
    # Check common Windows paths
    if os.path.exists(
        r"C:\Users\fpacu\Documents\Workspace\Edsger\data\DIMACS_road_networks"
    ):
        DEFAULT_DATA_DIR = (
            r"C:\Users\fpacu\Documents\Workspace\Edsger\data\DIMACS_road_networks"
        )
    else:
        DEFAULT_DATA_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "DIMACS_road_networks",
        )
else:
    DEFAULT_DATA_DIR = "/home/francois/Data/DIMACS_road_networks/"

# Use environment variable if set, otherwise use OS-specific default
DEFAULT_DATA_DIR = os.getenv("DIMACS_DATA_DIR", DEFAULT_DATA_DIR)

parser = ArgumentParser(description="Command line interface to bellman_ford_dimacs.py")
parser.add_argument(
    "-d",
    "--dir",
    dest="data_dir",
    help="Data folder with network sub-folders",
    metavar="TXT",
    type=str,
    required=False,
    default=DEFAULT_DATA_DIR,
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
    help='library name, must be "E" (Edsger), "GT" (graph-tool)',
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
    "--detect-negative-cycles",
    dest="detect_negative_cycles",
    help="enable negative cycle detection (default: disabled)",
    action="store_true",
    default=False,
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
detect_negative_cycles = args.detect_negative_cycles

# lib name check
if lib not in ["E", "GT"]:
    logger.critical(
        f"library '{lib}' invalid for Bellman-Ford. Must be 'E' (Edsger) or 'GT' (graph-tool)"
    )
    sys.exit()

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

logger.info(f"detect negative cycles : {detect_negative_cycles}")

# locate the parquet file
network_file_path = os.path.join(
    data_dir, os.path.join(reg, f"USA-road-t.{reg}.gr.parquet")
)
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

# Initialize dist_matrix to None to avoid possibly-used-before-assignment
dist_matrix = None  # pylint: disable=invalid-name

if lib == "E":
    # Edsger
    # ------

    logger.info("Edsger init")
    edges[["tail", "head"]] = edges[["tail", "head"]].astype(np.uint32)
    sp = BellmanFord(edges, orientation="out", check_edges=False, verbose=False)

    # SSSP

    results = []
    logger.info("Edsger run")
    for i in range(repeat):
        d = {}

        start = perf_counter()

        try:
            dist_matrix = sp.run(
                vertex_idx=idx_from,
                return_inf=True,
                detect_negative_cycles=detect_negative_cycles,
            )
            negative_cycle_detected = False  # pylint: disable=invalid-name
        except ValueError as e:
            if "negative cycle" in str(e).lower():
                logger.warning(f"Negative cycle detected in trial {i+1}")
                negative_cycle_detected = True  # pylint: disable=invalid-name
                dist_matrix = None  # pylint: disable=invalid-name
            else:
                raise

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"Edsger Bellman-Ford {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "edsger",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
            "negative_cycle": negative_cycle_detected,
        }
        results.append(d)

    df = pd.DataFrame.from_records(results)
    logger.info(f"Edsger min elapsed time : {df.elapsed_time.min():8.4f} s")
    if df.negative_cycle.any():
        logger.info(f"Negative cycles detected in {df.negative_cycle.sum()} trials")

elif lib == "GT":
    # graph-tool
    # ----------

    logger.info("graph-tool init")

    import graph_tool as gt

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

        try:
            dist = gt.topology.shortest_distance(
                g,
                source=g.vertex(idx_from),
                weights=g.ep.t,
                negative_weights=True,  # Enable negative weights for Bellman-Ford
                directed=True,
            )
            dist_matrix = dist.a  # pylint: disable=invalid-name
            negative_cycle_detected = False  # pylint: disable=invalid-name
        except (RuntimeError, ValueError) as e:
            if "negative" in str(e).lower() and "cycle" in str(e).lower():
                logger.warning(f"Negative cycle detected in trial {i+1}")
                negative_cycle_detected = True  # pylint: disable=invalid-name
                dist_matrix = None  # pylint: disable=invalid-name
            else:
                raise

        end = perf_counter()
        elapsed_time = end - start
        logger.info(
            f"graph-tool Bellman-Ford {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s"
        )

        d = {
            "library": "graph-tool",
            "network": reg,
            "trial": i,
            "elapsed_time": elapsed_time,
            "negative_cycle": negative_cycle_detected,
        }
        results.append(d)

    df = pd.DataFrame.from_records(results)
    logger.info(f"graph-tool min elapsed time : {df.elapsed_time.min():8.4f} s")
    if df.negative_cycle.any():
        logger.info(f"Negative cycles detected in {df.negative_cycle.sum()} trials")


if check_result and dist_matrix is not None:
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

    try:
        dist_matrix_ref = bellman_ford(
            csgraph=graph_csr,
            directed=True,
            indices=idx_from,
            return_predecessors=False,
        )
        scipy_negative_cycle = False  # pylint: disable=invalid-name
    except (RuntimeError, ValueError) as e:
        if "negative cycle" in str(e).lower():
            logger.warning("SciPy detected negative cycle")
            scipy_negative_cycle = True  # pylint: disable=invalid-name
            dist_matrix_ref = None  # pylint: disable=invalid-name
        else:
            raise

    end = perf_counter()
    elapsed_time = end - start

    logger.info(f"SciPy Bellman-Ford - Elapsed time: {elapsed_time:8.4f} s")

    if dist_matrix_ref is not None and dist_matrix is not None:
        logger.info(f"isinf : {np.isinf(dist_matrix_ref).any()}")
        logger.info(
            f"allclose : {np.allclose(dist_matrix_ref, dist_matrix, equal_nan=True)}"
        )
    elif scipy_negative_cycle:
        logger.info("Both algorithms detected negative cycles (if applicable)")


logger.info("exit")
