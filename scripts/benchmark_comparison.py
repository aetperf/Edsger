#!/usr/bin/env python3
"""
Comprehensive benchmarking script for Dijkstra implementations.

Runs dijkstra_dimacs.py for Edsger, NetworKit, Graph-tool, and SciPy,
then creates a comparison bar plot with system and package version information.
"""

import os
import sys
import subprocess
import time
import platform
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json


def get_system_info():
    """Collect system and package version information."""
    info = {}

    # Python version
    info["python_version"] = platform.python_version()

    # Platform info
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()

    # Anaconda/Conda version
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["conda_version"] = result.stdout.strip()
        else:
            info["conda_version"] = "Not available"
    except FileNotFoundError:
        info["conda_version"] = "Not installed"

    # Package versions
    packages = ["edsger", "networkit", "graph-tool", "scipy", "numpy", "pandas"]
    for package in packages:
        try:
            if package == "edsger":
                import edsger

                info[f"{package}_version"] = getattr(edsger, "__version__", "dev")
            elif package == "networkit":
                import networkit as nk

                info[f"{package}_version"] = nk.__version__
            elif package == "graph-tool":
                import graph_tool as gt

                info[f"{package}_version"] = gt.__version__
            elif package == "scipy":
                import scipy

                info[f"{package}_version"] = scipy.__version__
            elif package == "numpy":
                import numpy as np

                info[f"{package}_version"] = np.__version__
            elif package == "pandas":
                import pandas as pd

                info[f"{package}_version"] = pd.__version__
        except ImportError:
            info[f"{package}_version"] = "Not installed"
        except Exception as e:
            info[f"{package}_version"] = f"Error: {str(e)}"

    return info


def run_benchmark(library, repeat=5):
    """Run dijkstra_dimacs.py for a specific library."""
    print(f"\n[BENCHMARK] Running {library} benchmark ({repeat} iterations)...")

    # Get the project root directory (parent of scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # For SciPy, we need to run with -c flag to get SciPy timing
    if library == "SciPy":
        cmd = [
            sys.executable,
            "dijkstra_dimacs.py",
            "-n",
            "USA",
            "-l",
            "E",  # Run Edsger but with -c to get SciPy comparison
            "-r",
            str(repeat),
            "-c",
        ]
    else:
        cmd = [
            sys.executable,
            "dijkstra_dimacs.py",
            "-n",
            "USA",
            "-l",
            library,
            "-r",
            str(repeat),
        ]

    # Set environment to include project root in Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    try:
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        print(f"[DEBUG] Working directory: {os.getcwd()}")
        print(f"[DEBUG] Project root: {project_root}")

        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env
        )
        end_time = time.time()

        print(f"[DEBUG] Return code: {result.returncode}")
        print(f"[DEBUG] Stdout length: {len(result.stdout)}")
        print(f"[DEBUG] Stderr length: {len(result.stderr)}")

        if result.returncode != 0:
            print(f"[ERROR] {library} benchmark failed!")
            print(f"[ERROR] stderr: {result.stderr}")
            print(f"[DEBUG] stdout: {result.stdout}")
            return None

        # Parse output to extract timing information
        # The output might be in stderr due to logging configuration
        output_text = result.stdout + result.stderr
        lines = output_text.strip().split("\n")
        times = []

        # Map library codes to full names used in output
        library_names = {
            "E": "Edsger",
            "NK": "NetworKit",
            "GT": "graph-tool",
            "SciPy": "SciPy",
        }

        expected_name = library_names.get(library, library)

        for line in lines:
            if f"{expected_name} Dijkstra" in line and "Elapsed time:" in line:
                # Extract time from line like "Edsger Dijkstra 1/5 - Elapsed time:   2.5410 s"
                parts = line.split("Elapsed time:")
                if len(parts) > 1:
                    time_str = parts[1].strip().split()[0]
                    try:
                        elapsed_time = float(time_str)
                        times.append(elapsed_time)
                    except ValueError:
                        continue

        if not times:
            print(f"[WARNING] Could not parse timing data for {library}")
            print(f"[DEBUG] Expected pattern: '{expected_name} Dijkstra'")
            print(f"[DEBUG] Sample output lines:")
            for i, line in enumerate(lines[-20:]):  # Show last 20 lines
                if "Dijkstra" in line or "elapsed" in line or "Elapsed" in line:
                    print(f"[DEBUG]   TIMING: {line}")
                elif i < 5:  # Show first few lines regardless
                    print(f"[DEBUG]   {line}")
            return None

        min_time = min(times)
        avg_time = np.mean(times)
        std_time = np.std(times)

        print(
            f"[BENCHMARK] {library} completed: min={min_time:.4f}s, avg={avg_time:.4f}s, std={std_time:.4f}s"
        )

        return {
            "library": library,
            "times": times,
            "min_time": min_time,
            "avg_time": avg_time,
            "std_time": std_time,
            "total_duration": end_time - start_time,
        }

    except subprocess.TimeoutExpired:
        print(f"[ERROR] {library} benchmark timed out!")
        return None
    except Exception as e:
        print(f"[ERROR] {library} benchmark failed with exception: {e}")
        return None


def create_comparison_plot(
    results, system_info, output_file="dijkstra_benchmark_comparison.png"
):
    """Create a clean comparison bar plot suitable for README."""

    # Prepare data
    libraries = [r["library"] for r in results if r is not None]
    min_times = [r["min_time"] for r in results if r is not None]
    avg_times = [r["avg_time"] for r in results if r is not None]
    std_times = [r["std_time"] for r in results if r is not None]

    if not libraries:
        print("[ERROR] No valid benchmark results to plot!")
        return

    # Create clean figure for README
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Use tol-colors for beautiful, distinctive colors
    try:
        from tol_colors import colorsets

        colors_palette = colorsets["bright"]
        colors = {
            "E": colors_palette[0],  # Blue
            "NK": colors_palette[1],  # Red
            "GT": colors_palette[2],  # Green
            "SciPy": colors_palette[3],  # Yellow
        }
    except (ImportError, KeyError):
        # Fallback to nice colors if tol-colors not available
        colors = {"E": "#4477AA", "NK": "#EE6677", "GT": "#228833", "SciPy": "#CCBB44"}

    # Library names with versions (clean up graph-tool version)
    def clean_version(version_str):
        """Clean up version string to remove commit hashes and other metadata."""
        if not version_str or version_str == "?":
            return version_str

        # Handle various graph-tool version formats
        # "2.97 (commit 4f33d8da, )" -> "2.97"
        # "2.97+git.abc123" -> "2.97"
        if "(" in version_str:
            version_str = version_str.split("(")[0].strip()
        if "+" in version_str:
            version_str = version_str.split("+")[0].strip()

        return version_str

    library_names = {
        "E": f"Edsger {system_info.get('edsger_version', '?')}",
        "NK": f"NetworKit {system_info.get('networkit_version', '?')}",
        "GT": f"Graph-tool {clean_version(system_info.get('graph-tool_version', '?'))}",
        "SciPy": f"SciPy {system_info.get('scipy_version', '?')}",
    }

    bar_colors = [colors.get(lib, "#808080") for lib in libraries]
    bar_labels = [library_names.get(lib, lib) for lib in libraries]

    # Create bars for best times
    x_pos = np.arange(len(libraries))
    bars = ax.bar(
        x_pos,
        min_times,
        color=bar_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
        width=0.6,
    )

    # Add error bars showing standard deviation around best time
    ax.errorbar(
        x_pos,
        min_times,
        yerr=std_times,
        fmt="none",
        ecolor="black",
        capsize=8,
        capthick=2,
        alpha=0.7,
    )

    # Styling
    ax.set_xlabel("Library", fontsize=14, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Dijkstra Algorithm Performance on USA Road Network\n(23.9M vertices, 57.7M edges)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # Add grid for better readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std_times[i] + 0.1,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Remove the "Fastest" annotation as requested

    # Improve layout
    ax.set_ylim(0, max(min_times) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add compact system info
    info_text = f"Python {system_info['python_version']} • {datetime.now().strftime('%Y-%m-%d')}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
    )

    plt.tight_layout()

    # Save to scripts folder (for README)
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"\n[PLOT] Saved README-ready plot to: {output_file}")

    # Also save to docs assets folder
    docs_output = f"../docs/source/assets/{os.path.basename(output_file)}"
    if os.path.exists("../docs/source/assets/"):
        plt.savefig(
            docs_output,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"[PLOT] Also saved to docs assets: {docs_output}")

    return output_file


def save_results_json(results, system_info, output_file="benchmark_results.json"):
    """Save detailed results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "benchmark_results": [r for r in results if r is not None],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[DATA] Saved detailed results to: {output_file}")


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("DIJKSTRA ALGORITHM PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Check if we're in the right directory
    if not os.path.exists("dijkstra_dimacs.py"):
        print(
            "[ERROR] dijkstra_dimacs.py not found. Run this script from the scripts/ directory."
        )
        sys.exit(1)

    # Collect system information
    print("\n[INFO] Collecting system and package information...")
    system_info = get_system_info()

    print(f"[INFO] Python: {system_info['python_version']}")
    print(f"[INFO] Platform: {system_info['platform']}")
    print(f"[INFO] Conda: {system_info['conda_version']}")

    # Define libraries to benchmark
    libraries = ["E", "NK", "GT", "SciPy"]  # Include SciPy directly
    repeat = 5

    print(f"\n[INFO] Running benchmarks with {repeat} iterations each...")

    # Run benchmarks
    results = []
    for lib in libraries:
        result = run_benchmark(lib, repeat)
        results.append(result)

        # Small delay between benchmarks to let system cool down
        if result:
            time.sleep(2)

    # Filter successful results
    successful_results = [r for r in results if r is not None]

    if not successful_results:
        print("[ERROR] No benchmarks completed successfully!")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for result in successful_results:
        print(
            f"{result['library']:>10}: {result['min_time']:7.4f}s (min), {result['avg_time']:7.4f}s (avg) ± {result['std_time']:6.4f}s"
        )

    # Create comparison plot
    print("\n[PLOT] Creating comparison visualization...")
    plot_file = create_comparison_plot(successful_results, system_info)

    # Save detailed results
    save_results_json(successful_results, system_info)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results plot: {plot_file}")
    print(f"Detailed data: benchmark_results.json")


if __name__ == "__main__":
    main()
