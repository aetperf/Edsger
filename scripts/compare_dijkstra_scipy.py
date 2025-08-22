#!/usr/bin/env python3
"""
Compare Dijkstra implementations between edsger and SciPy.

This script generates random graphs and verifies that both implementations
produce identical shortest path distances. Useful for validation and testing.
"""

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from edsger.path import Dijkstra
from edsger.utils import generate_random_network


def create_sparse_matrix(edges_df, n_vertices):
    """
    Convert edge DataFrame to SciPy sparse matrix format.

    Note: This function handles duplicate edges by keeping the minimum weight
    edge between each pair of vertices, which is consistent with shortest path algorithms.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        DataFrame with columns 'tail', 'head', 'weight'
    n_vertices : int
        Number of vertices in the graph

    Returns
    -------
    csr_matrix
        Sparse matrix in CSR format
    """
    # Handle duplicate edges by keeping the minimum weight
    clean_edges = edges_df.groupby(["tail", "head"])["weight"].min().reset_index()

    return csr_matrix(
        (clean_edges["weight"], (clean_edges["tail"], clean_edges["head"])),
        shape=(n_vertices, n_vertices),
    )


def compare_single_source(edges_df, source_vertex, n_vertices, tolerance=1e-10):
    """
    Compare Dijkstra results from a single source vertex.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge list DataFrame
    source_vertex : int
        Source vertex index
    n_vertices : int
        Total number of vertices
    tolerance : float
        Numerical tolerance for comparison

    Returns
    -------
    dict
        Comparison results including match status and statistics
    """

    # Run edsger Dijkstra
    edsger_dijkstra = Dijkstra(edges_df)
    start_time = time.time()
    edsger_distances = edsger_dijkstra.run(vertex_idx=source_vertex)
    edsger_time = time.time() - start_time

    # Run SciPy Dijkstra
    sparse_matrix = create_sparse_matrix(edges_df, n_vertices)
    start_time = time.time()
    scipy_distances = scipy_dijkstra(
        csgraph=sparse_matrix,
        directed=True,
        indices=source_vertex,
        return_predecessors=False,
    )
    scipy_time = time.time() - start_time

    # Compare results
    # Handle infinity values specially
    edsger_inf_mask = np.isinf(edsger_distances)
    scipy_inf_mask = np.isinf(scipy_distances)

    # Check if infinity positions match
    inf_match = np.array_equal(edsger_inf_mask, scipy_inf_mask)

    # Compare finite values
    if inf_match:
        finite_mask = ~edsger_inf_mask
        if np.any(finite_mask):
            # Only compare finite values
            edsger_finite = edsger_distances[finite_mask]
            scipy_finite = scipy_distances[finite_mask]
            differences = np.abs(edsger_finite - scipy_finite)
            max_diff = np.max(differences) if len(differences) > 0 else 0.0
            mean_diff = np.mean(differences) if len(differences) > 0 else 0.0
            all_match = np.all(differences <= tolerance)
        else:
            # All values are infinity
            max_diff = 0.0
            mean_diff = 0.0
            all_match = True
    else:
        # Infinity positions don't match - this is a real mismatch
        all_match = False
        # Calculate differences only for finite values to get meaningful error metrics
        both_finite_mask = (~edsger_inf_mask) & (~scipy_inf_mask)
        if np.any(both_finite_mask):
            edsger_finite = edsger_distances[both_finite_mask]
            scipy_finite = scipy_distances[both_finite_mask]
            differences = np.abs(edsger_finite - scipy_finite)
            max_diff = np.max(differences) if len(differences) > 0 else 0.0
            mean_diff = np.mean(differences) if len(differences) > 0 else 0.0
        else:
            max_diff = np.inf
            mean_diff = np.inf

    return {
        "match": all_match and inf_match,
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "infinity_match": inf_match,
        "n_reachable": int(np.sum(~edsger_inf_mask)),
        "n_unreachable": int(np.sum(edsger_inf_mask)),
        "edsger_time": float(edsger_time),
        "scipy_time": float(scipy_time),
        "edsger_distances": edsger_distances,
        "scipy_distances": scipy_distances,
    }


def compare_dijkstra_implementations(
    n_vertices=100,
    n_edges=500,
    seed=42,
    n_sources=10,
    tolerance=1e-10,
    verbose=True,
    weight_range=(0.1, 10.0),
):
    """
    Compare Dijkstra implementations on a random graph.

    Parameters
    ----------
    n_vertices : int
        Number of vertices in the graph
    n_edges : int
        Number of edges in the graph
    seed : int
        Random seed for reproducibility
    n_sources : int
        Number of source vertices to test
    tolerance : float
        Numerical tolerance for comparison
    verbose : bool
        Whether to print detailed output
    weight_range : tuple
        Range of edge weights (min, max)

    Returns
    -------
    dict
        Overall comparison results
    """
    if verbose:
        print("=" * 60)
        print("Dijkstra Implementation Comparison: edsger vs SciPy")
        print("=" * 60)
        print(f"Graph: {n_vertices} vertices, {n_edges} edges")
        print(f"Weight range: [{weight_range[0]:.2f}, {weight_range[1]:.2f}]")
        print(f"Testing from {n_sources} random source vertices...")
        print()

    # Generate random graph
    edges_df = generate_random_network(
        n_edges=n_edges,
        n_verts=n_vertices,
        seed=seed,
        sort=True,
        weight_range=weight_range,
    )

    # Select random source vertices
    np.random.seed(seed)
    source_vertices = np.random.choice(
        n_vertices, size=min(n_sources, n_vertices), replace=False
    )

    # Run comparisons
    results = []
    all_match = True
    total_edsger_time = 0.0
    total_scipy_time = 0.0

    for source in source_vertices:
        result = compare_single_source(edges_df, source, n_vertices, tolerance)
        results.append({"source": int(source), **result})

        total_edsger_time += result["edsger_time"]
        total_scipy_time += result["scipy_time"]

        if verbose:
            if result["match"]:
                print(
                    f"Source {source:3d}: ✓ All {n_vertices} distances match "
                    f"(max diff: {result['max_difference']:.2e}, "
                    f"reachable: {result['n_reachable']}/{n_vertices})"
                )
            else:
                print(
                    f"Source {source:3d}: ✗ MISMATCH DETECTED! "
                    f"(max diff: {result['max_difference']:.2e})"
                )
                all_match = False

        # Remove large arrays from stored results to save memory
        results[-1].pop("edsger_distances", None)
        results[-1].pop("scipy_distances", None)

    # Calculate statistics
    avg_edsger_time = total_edsger_time / len(source_vertices)
    avg_scipy_time = total_scipy_time / len(source_vertices)
    speedup = avg_scipy_time / avg_edsger_time if avg_edsger_time > 0 else 0

    if verbose:
        print("\n" + "-" * 40)
        print("Performance Comparison:")
        print(f"  Edsger average: {avg_edsger_time*1000:.3f} ms")
        print(f"  SciPy average:  {avg_scipy_time*1000:.3f} ms")
        print(f"  Speedup factor: {speedup:.2f}x")
        print()

        if all_match:
            print("✅ SUCCESS: All tests passed! Results match perfectly.")
        else:
            print("❌ FAILURE: Some tests failed. Results do not match.")
        print("=" * 60)

    return {
        "timestamp": datetime.now().isoformat(),
        "graph_info": {
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "seed": seed,
            "weight_range": weight_range,
        },
        "test_info": {"n_sources_tested": len(source_vertices), "tolerance": tolerance},
        "results": {"all_match": all_match, "individual_results": results},
        "performance": {
            "avg_edsger_time_ms": avg_edsger_time * 1000,
            "avg_scipy_time_ms": avg_scipy_time * 1000,
            "speedup_factor": speedup,
        },
    }


def main():
    """Main function to run the comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare Dijkstra implementations between edsger and SciPy"
    )
    parser.add_argument(
        "--vertices",
        "-v",
        type=int,
        default=100,
        help="Number of vertices in the graph (default: 100)",
    )
    parser.add_argument(
        "--edges",
        "-e",
        type=int,
        default=500,
        help="Number of edges in the graph (default: 500)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sources",
        "-n",
        type=int,
        default=10,
        help="Number of source vertices to test (default: 10)",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-10,
        help="Numerical tolerance for comparison (default: 1e-10)",
    )
    parser.add_argument(
        "--weight-min",
        type=float,
        default=0.1,
        help="Minimum edge weight (default: 0.1)",
    )
    parser.add_argument(
        "--weight-max",
        type=float,
        default=10.0,
        help="Maximum edge weight (default: 10.0)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    # Validate arguments
    if args.vertices <= 0:
        print("Error: Number of vertices must be positive", file=sys.stderr)
        sys.exit(1)
    if args.edges <= 0:
        print("Error: Number of edges must be positive", file=sys.stderr)
        sys.exit(1)
    if args.weight_min <= 0 or args.weight_max <= args.weight_min:
        print("Error: Invalid weight range", file=sys.stderr)
        sys.exit(1)

    # Run comparison
    results = compare_dijkstra_implementations(
        n_vertices=args.vertices,
        n_edges=args.edges,
        seed=args.seed,
        n_sources=args.sources,
        tolerance=args.tolerance,
        verbose=not args.quiet,
        weight_range=(args.weight_min, args.weight_max),
    )

    # Save results if requested
    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"\nResults saved to: {args.save_results}")

    # Exit with appropriate code
    sys.exit(0 if results["results"]["all_match"] else 1)


if __name__ == "__main__":
    main()
