#!/usr/bin/env python3
"""
Plotting script that combines benchmark results from different operating systems.

Reads benchmark_dimacs_USA_*.json files and creates comparison plots showing
performance across different libraries and operating systems.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob


def clean_version(version_str):
    """Clean up version string to remove commit hashes and other metadata."""
    if not version_str or version_str == "?" or "Not" in version_str:
        return version_str

    # Handle various graph-tool version formats
    # "2.97 (commit 4f33d8da, )" -> "2.97"
    # "2.97+git.abc123" -> "2.97"
    if "(" in version_str:
        version_str = version_str.split("(")[0].strip()
    if "+" in version_str:
        version_str = version_str.split("+")[0].strip()

    return version_str


def extract_processor_model(processor_str, os_name):
    """Extract a clean processor model name from the system info."""
    import re

    if not processor_str:
        return None

    # Try to extract Intel Core model (e.g., i7-14700K, i9-12900H)
    match = re.search(r"i[3579]-\d{4,5}[A-Z]*", processor_str)
    if match:
        return match.group(0)

    # For Windows, the processor string is like "Intel64 Family 6 Model 183..."
    # Map known model numbers to processor names
    model_mapping = {
        "Model 183": "i7-14700K",  # Raptor Lake Refresh
        "Model 154": "i9-12900H",  # Alder Lake
    }
    for model_id, name in model_mapping.items():
        if model_id in processor_str:
            return name

    # Fallback for known OS/processor combinations where platform.processor()
    # only returns architecture (e.g., "x86_64" on Linux)
    os_processor_fallback = {
        "linux": "i9-12900H",  # Alder Lake
        "windows": "i7-14700K",  # Raptor Lake Refresh
    }
    if processor_str in ("x86_64", "AMD64"):
        return os_processor_fallback.get(os_name)

    return None


def load_benchmark_results():
    """Load all available benchmark result files."""
    results = {}

    # Find all benchmark JSON files
    json_files = glob.glob("benchmark_dimacs_*_*.json")

    if not json_files:
        print("[WARNING] No benchmark result files found!")
        print(
            "Expected files like: benchmark_dimacs_USA_linux.json, benchmark_dimacs_USA_windows.json"
        )
        return results

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                os_name = data["os"]
                results[os_name] = data
                print(f"[INFO] Loaded results from {json_file} (OS: {os_name})")
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file}: {e}")

    return results


def create_comparison_plot(
    all_results, output_file="dijkstra_benchmark_comparison.png"
):
    """Create comparison plot showing results across different OS."""

    if not all_results:
        print("[ERROR] No results to plot!")
        return

    # Prepare data structure
    libraries = ["E", "NK", "GT", "SciPy"]
    library_names = {
        "E": "Edsger",
        "NK": "NetworKit",
        "GT": "Graph-tool",
        "SciPy": "SciPy",
    }

    # Colors for different OS
    os_colors = {
        "linux": "#4477AA",  # Blue
        "windows": "#EE6677",  # Red
        "darwin": "#228833",  # Green (for macOS)
    }

    # Collect data by library and OS
    plot_data = {}
    for lib in libraries:
        plot_data[lib] = {}
        for os_name, results in all_results.items():
            for bench_result in results["benchmark_results"]:
                if bench_result["library"] == lib:
                    plot_data[lib][os_name] = {
                        "min_time": bench_result["min_time"],
                        "avg_time": bench_result["avg_time"],
                        "std_time": bench_result["std_time"],
                        "version": results["system_info"].get(
                            f"{library_names[lib].lower()}_version", "?"
                        ),
                    }

    # Filter out empty libraries
    plot_data = {lib: data for lib, data in plot_data.items() if data}

    if not plot_data:
        print("[ERROR] No valid benchmark data found!")
        return

    # Create figure
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Prepare bar positions
    libraries_to_plot = list(plot_data.keys())
    os_list = sorted(
        set(os for lib_data in plot_data.values() for os in lib_data.keys())
    )

    x = np.arange(len(libraries_to_plot))
    width = 0.8 / len(os_list)  # Width of bars

    # Plot bars for each OS
    for i, os_name in enumerate(os_list):
        positions = x + (i - len(os_list) / 2 + 0.5) * width
        min_times = []
        std_times = []
        labels = []

        for lib in libraries_to_plot:
            if os_name in plot_data[lib]:
                min_times.append(plot_data[lib][os_name]["min_time"])
                std_times.append(plot_data[lib][os_name]["std_time"])
                version = clean_version(plot_data[lib][os_name]["version"])
                labels.append(f"{library_names[lib]} {version}")
            else:
                min_times.append(0)  # Library not available on this OS
                std_times.append(0)
                labels.append(f"{library_names[lib]} N/A")

        # Plot bars
        bars = ax.bar(
            positions,
            min_times,
            width,
            label=os_name.capitalize(),
            color=os_colors.get(os_name, "#808080"),
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        # Add error bars
        ax.errorbar(
            positions,
            min_times,
            yerr=std_times,
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=1.5,
            alpha=0.7,
        )

        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only label if library was benchmarked
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std_times[j] + 0.05,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

    # Styling
    ax.set_xlabel("Library", fontsize=14, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=14, fontweight="bold")

    # Adjust title based on number of OS
    if len(os_list) > 1:
        title = "Dijkstra Algorithm Performance on USA Road Network\n(23.9M vertices, 57.7M edges) - Multi-OS Comparison"
    else:
        title = f"Dijkstra Algorithm Performance on USA Road Network\n(23.9M vertices, 57.7M edges) - {os_list[0].capitalize()}"

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([library_names[lib] for lib in libraries_to_plot], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # Add grid
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    # Improve layout
    y_max = max(
        max(
            plot_data[lib][os]["min_time"] + plot_data[lib][os]["std_time"]
            for os in plot_data[lib]
        )
        for lib in plot_data
        if plot_data[lib]
    )
    ax.set_ylim(0, y_max * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add system info for each OS
    info_texts = []
    for os_name, results in all_results.items():
        python_version = results["system_info"]["python_version"]
        processor_str = results["system_info"].get("processor", "")
        processor_model = extract_processor_model(processor_str, os_name)
        if processor_model:
            info_texts.append(
                f"{os_name.capitalize()}: {processor_model}, Python {python_version}"
            )
        else:
            info_texts.append(f"{os_name.capitalize()}: Python {python_version}")

    info_text = " • ".join(info_texts) + f" • {datetime.now().strftime('%Y-%m-%d')}"
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

    # Save plot
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"\n[PLOT] Saved comparison plot to: {output_file}")

    # Also save to docs assets folder if it exists
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


def create_single_os_plot(results, os_name, output_file=None):
    """Create a plot for a single OS (similar to original benchmark_comparison.py)."""

    if output_file is None:
        output_file = f"dijkstra_benchmark_comparison_{os_name}.png"

    system_info = results["system_info"]
    benchmark_results = results["benchmark_results"]

    if not benchmark_results:
        print(f"[ERROR] No benchmark results for {os_name}!")
        return

    # Prepare data
    libraries = [r["library"] for r in benchmark_results]
    min_times = [r["min_time"] for r in benchmark_results]
    std_times = [r["std_time"] for r in benchmark_results]

    # Create figure
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Colors
    colors = {"E": "#4477AA", "NK": "#EE6677", "GT": "#228833", "SciPy": "#CCBB44"}

    # Library names with versions
    library_names = {
        "E": f"Edsger {system_info.get('edsger_version', '?')}",
        "NK": f"NetworKit {system_info.get('networkit_version', '?')}",
        "GT": f"Graph-tool {clean_version(system_info.get('graph-tool_version', '?'))}",
        "SciPy": f"SciPy {system_info.get('scipy_version', '?')}",
    }

    bar_colors = [colors.get(lib, "#808080") for lib in libraries]
    bar_labels = [library_names.get(lib, lib) for lib in libraries]

    # Create bars
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

    # Add error bars
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
        f"Dijkstra Algorithm Performance on USA Road Network ({os_name.capitalize()})\n(23.9M vertices, 57.7M edges)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # Add grid
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

    # Improve layout
    ax.set_ylim(0, max(min_times) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add system info
    info_text = f"Python {system_info['python_version']} • {system_info['platform']} • {datetime.now().strftime('%Y-%m-%d')}"
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

    # Save plot
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"[PLOT] Saved {os_name} plot to: {output_file}")

    return output_file


def main():
    """Main function to create comparison plots."""
    print("=" * 80)
    print("BENCHMARK COMPARISON PLOTTER")
    print("=" * 80)

    # Load all available benchmark results
    all_results = load_benchmark_results()

    if not all_results:
        print("\n[ERROR] No benchmark results found!")
        print("Please run benchmark_comparison_os.py first to generate result files.")
        sys.exit(1)

    print(
        f"\n[INFO] Found results for {len(all_results)} operating system(s): {', '.join(all_results.keys())}"
    )

    # Always create comparison plot, even for single OS
    print("\n[INFO] Creating comparison plot...")
    create_comparison_plot(all_results)

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    main()
