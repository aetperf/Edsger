#!/usr/bin/env python3
"""
Comprehensive benchmarking script for Dijkstra implementations with OS-specific output.

Runs dijkstra_dimacs.py for Edsger, NetworKit, Graph-tool, and SciPy,
then saves results to OS-specific JSON files for later comparison.
"""

import os
import sys
import subprocess
import time
import platform
import numpy as np
from datetime import datetime
import json
from argparse import ArgumentParser


def get_system_info():
    """Collect system and package version information."""
    info = {}

    # Python version
    info["python_version"] = platform.python_version()

    # Platform info
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    info["os_name"] = platform.system()  # Linux, Windows, Darwin

    # Anaconda/Conda version
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["conda_version"] = result.stdout.strip()
        else:
            info["conda_version"] = "Not available"
    except FileNotFoundError:
        info["conda_version"] = "Not installed"

    # Package versions - check availability first
    packages = ["edsger", "networkit", "graph-tool", "scipy", "numpy", "pandas"]
    available_packages = []

    for package in packages:
        try:
            if package == "edsger":
                import edsger

                info[f"{package}_version"] = getattr(edsger, "__version__", "dev")
                available_packages.append(package)
            elif package == "networkit":
                import networkit as nk

                info[f"{package}_version"] = nk.__version__
                available_packages.append(package)
            elif package == "graph-tool":
                # Graph-tool is not available on Windows
                if platform.system() == "Windows":
                    info[f"{package}_version"] = "Not available on Windows"
                else:
                    import graph_tool as gt

                    info[f"{package}_version"] = gt.__version__
                    available_packages.append(package)
            elif package == "scipy":
                import scipy

                info[f"{package}_version"] = scipy.__version__
                available_packages.append(package)
            elif package == "numpy":
                import numpy as np

                info[f"{package}_version"] = np.__version__
                available_packages.append(package)
            elif package == "pandas":
                import pandas as pd

                info[f"{package}_version"] = pd.__version__
                available_packages.append(package)
        except ImportError:
            info[f"{package}_version"] = "Not installed"
        except Exception as e:
            info[f"{package}_version"] = f"Error: {str(e)}"

    info["available_packages"] = available_packages
    return info


def run_benchmark(library, repeat=5, data_dir=None, network_name="USA"):
    """Run dijkstra_dimacs.py for a specific library."""
    print(f"\n[BENCHMARK] Running {library} benchmark ({repeat} iterations)...")

    # Get the project root directory (parent of scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # For SciPy, we need to run Edsger with -c flag multiple times to get SciPy timing
    # (SciPy only runs once per dijkstra_dimacs.py call when using -c)
    if library == "SciPy":
        cmd = [
            sys.executable,
            "dijkstra_dimacs.py",
            "-n",
            network_name,
            "-l",
            "E",  # Run Edsger but with -c to get SciPy comparison
            "-r",
            "1",  # Run only 1 iteration of Edsger per call
            "-c",
        ]
    else:
        cmd = [
            sys.executable,
            "dijkstra_dimacs.py",
            "-n",
            network_name,
            "-l",
            library,
            "-r",
            str(repeat),
        ]

    # Add data directory argument if provided
    if data_dir:
        cmd.extend(["-d", data_dir])

    # Set environment to include project root in Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    try:
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        print(f"[DEBUG] Working directory: {os.getcwd()}")
        print(f"[DEBUG] Project root: {project_root}")

        times = []
        total_start_time = time.time()

        # For SciPy, we need to run the command multiple times since SciPy only runs once per call
        iterations = repeat if library == "SciPy" else 1

        for iteration in range(iterations):
            if library == "SciPy":
                print(f"[DEBUG] SciPy iteration {iteration + 1}/{iterations}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )

            print(f"[DEBUG] Return code: {result.returncode}")
            if iteration == 0:  # Only print lengths for first iteration to avoid spam
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

            # Map library codes to full names used in output
            library_names = {
                "E": "Edsger",
                "NK": "NetworKit",
                "GT": "graph-tool",
                "SciPy": "SciPy",
            }

            expected_name = library_names.get(library, library)

            for line in lines:
                if library == "SciPy":
                    # For SciPy, look for the pattern from the check_result section
                    # "SciPy Dijkstra - Elapsed time: 1.2345 s" (no trial number)
                    if "SciPy Dijkstra - Elapsed time:" in line:
                        parts = line.split("Elapsed time:")
                        if len(parts) > 1:
                            time_str = parts[1].strip().split()[0]
                            try:
                                elapsed_time = float(time_str)
                                times.append(elapsed_time)
                                break  # Only one SciPy timing per run
                            except ValueError:
                                continue
                else:
                    # For other libraries, look for the pattern with trial numbers
                    # "Edsger Dijkstra 1/5 - Elapsed time:   2.5410 s"
                    if (
                        f"{expected_name} Dijkstra" in line
                        and "Elapsed time:" in line
                        and "/" in line
                    ):
                        parts = line.split("Elapsed time:")
                        if len(parts) > 1:
                            time_str = parts[1].strip().split()[0]
                            try:
                                elapsed_time = float(time_str)
                                times.append(elapsed_time)
                            except ValueError:
                                continue

        total_end_time = time.time()

        if not times:
            print(f"[WARNING] Could not parse timing data for {library}")
            print(f"[DEBUG] Expected pattern: '{expected_name} Dijkstra'")
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
            "total_duration": total_end_time - total_start_time,
        }

    except subprocess.TimeoutExpired:
        print(f"[ERROR] {library} benchmark timed out!")
        return None
    except Exception as e:
        print(f"[ERROR] {library} benchmark failed with exception: {e}")
        return None


def save_results_json(results, system_info, network_name="USA"):
    """Save detailed results to OS-specific JSON file."""
    os_name = platform.system().lower()  # linux, windows, darwin
    output_file = f"benchmark_dimacs_{network_name}_{os_name}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "os": os_name,
        "network": network_name,
        "system_info": system_info,
        "benchmark_results": [r for r in results if r is not None],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[DATA] Saved detailed results to: {output_file}")
    return output_file


def main():
    """Main benchmarking function."""
    # Determine default data directory based on OS
    if platform.system() == "Windows":
        # Check common Windows paths
        if os.path.exists(
            r"C:\Users\fpacu\Documents\Workspace\Edsger\data\DIMACS_road_networks"
        ):
            default_data_dir = (
                r"C:\Users\fpacu\Documents\Workspace\Edsger\data\DIMACS_road_networks"
            )
        else:
            default_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "DIMACS_road_networks",
            )
    else:
        default_data_dir = "/home/francois/Data/DIMACS_road_networks/"

    # Use environment variable if set, otherwise use OS-specific default
    default_data_dir = os.getenv("DIMACS_DATA_DIR", default_data_dir)

    # Parse command line arguments
    parser = ArgumentParser(
        description="OS-specific benchmark comparison for Dijkstra implementations"
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="data_dir",
        help="Data folder with network sub-folders",
        metavar="TXT",
        type=str,
        required=False,
        default=default_data_dir,
    )
    parser.add_argument(
        "-n",
        "--network",
        dest="network_name",
        help="network name, must be 'NY', 'BAY', 'COL', 'FLA', 'NW', "
        + "'NE', 'CAL', 'LKS', 'E', 'W', 'CTR', 'USA' (default: USA)",
        metavar="TXT",
        type=str,
        required=False,
        default="USA",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        dest="repeat",
        help="Number of benchmark iterations per library (default: 5)",
        metavar="INT",
        type=int,
        required=False,
        default=5,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DIJKSTRA ALGORITHM PERFORMANCE BENCHMARK")
    print(f"OS: {platform.system()}")
    print(f"Network: {args.network_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Iterations per library: {args.repeat}")
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
    print(f"[INFO] OS: {system_info['os_name']}")
    print(f"[INFO] Conda: {system_info['conda_version']}")

    # Define libraries to benchmark based on what's available
    all_libraries = ["E", "NK", "GT", "SciPy"]
    libraries = []

    # Check which libraries are available
    for lib in all_libraries:
        lib_name = {
            "E": "edsger",
            "NK": "networkit",
            "GT": "graph-tool",
            "SciPy": "scipy",
        }[lib]
        if lib_name in system_info["available_packages"]:
            libraries.append(lib)
        else:
            print(f"[INFO] Skipping {lib} - {lib_name} not available")

    if not libraries:
        print("[ERROR] No libraries available for benchmarking!")
        sys.exit(1)

    print(f"\n[INFO] Running benchmarks with {args.repeat} iterations each...")
    print(f"[INFO] Libraries to benchmark: {', '.join(libraries)}")

    # Run benchmarks
    results = []
    for lib in libraries:
        result = run_benchmark(lib, args.repeat, args.data_dir, args.network_name)
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

    # Save detailed results
    json_file = save_results_json(successful_results, system_info, args.network_name)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {json_file}")
    print("\nTo create comparison plots, run:")
    print("  python plot_benchmark_comparison.py")


if __name__ == "__main__":
    main()
