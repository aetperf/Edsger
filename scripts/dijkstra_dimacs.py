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
from loguru import logger
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra

from edsger.path import Dijkstra


logger.remove()
fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    + " <level>{message}</level>"
)
logger.add(sys.stderr, format=fmt)

parser = ArgumentParser(description="Command line interface to perf_01.py")
parser.add_argument(
    "-d",
    "--dir",
    dest="data_dir",
    help="Data folder with network sub-folders",
    metavar="TXT",
    type=str,
    required=False,
    default="/home/francois/Data/DIMACS_road_networks/",
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
    help="check the resulting path lengths aginst SciPy",
    metavar="BOOL",
    type=bool,
    required=False,
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
    logger.critical(f"invalid value'{repeat}' for repeat")
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


# Edsger

logger.info("Edsger init")
edges[["tail", "head"]] = edges[["tail", "head"]].astype(np.uint32)
sp = Dijkstra(edges, orientation="out", check_edges=False)

results = []
logger.info("Edsger Run")
for i in range(repeat):
    d = {}

    start = perf_counter()

    path_lengths = sp.run(vertex_idx=idx_from, return_inf=True)
    dist_matrix = path_lengths.values

    end = perf_counter()
    elapsed_time = end - start
    logger.info(f"Edsger Dijkstra {i+1}/{repeat} - Elapsed time: {elapsed_time:8.4f} s")

    d = {
        "library": "edsger",
        "network": reg,
        "trial": i,
        "elapsed_time": elapsed_time,
    }
    results.append(d)

df = pd.DataFrame.from_records(results)
logger.info(f"Edsger min elapsed time : {df.elapsed_time.min():8.4f} s")

if check_result:
    logger.info(f"result check")

    # SciPy
    logger.info("SciPy init")
    data = edges["weight"].values
    row = edges["tail"].values
    col = edges["head"].values
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
