"""
Plot mesh quality histograms from evaluate_mesh CSV output.

Usage:
    python plot_mesh_quality.py mesh_hist
    python plot_mesh_quality.py mesh_hist --output quality.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_histogram(filepath):
    """Load evaluate_mesh histogram CSV file."""
    bin_low, bin_high, count = [], [], []
    title = ""
    meta = ""

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# evaluate_mesh histogram:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("# config:"):
                meta = line[2:]
            elif line.startswith("#") or line.startswith("bin_low"):
                continue
            else:
                parts = line.split(",")
                if len(parts) >= 3:
                    bin_low.append(float(parts[0]))
                    bin_high.append(float(parts[1]))
                    count.append(int(parts[2]))

    return {
        "title": title,
        "meta": meta,
        "bin_low": np.array(bin_low),
        "bin_high": np.array(bin_high),
        "count": np.array(count),
    }


# Display name and unit for each metric
METRIC_INFO = {
    "element_size": ("Element Size", "m"),
    "dt_cfl": ("CFL dt", "s"),
    "ppw": ("Points Per Wavelength", ""),
    "aspect_ratio": ("Aspect Ratio", ""),
    "jacobian_det": ("Jacobian det(J)", ""),
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot mesh quality histograms from evaluate_mesh")
    parser.add_argument("prefix",
                        help="Histogram file prefix (e.g., mesh_hist)")
    parser.add_argument("--output", default=None,
                        help="Output image file (default: <prefix>_quality.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Find all CSV files with the prefix
    pattern = f"{args.prefix}_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching '{pattern}'")
        sys.exit(1)

    print(f"Found {len(files)} histogram files:")
    for f in files:
        print(f"  {f}")

    # Determine grid layout
    n = len(files)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, filepath in enumerate(files):
        ax = axes[i]
        data = load_histogram(filepath)

        # Extract metric name from filename
        basename = os.path.basename(filepath)
        metric_key = basename.replace(args.prefix + "_", "").replace(".csv", "")
        display_name, unit = METRIC_INFO.get(metric_key, (metric_key, ""))

        centers = (data["bin_low"] + data["bin_high"]) / 2
        widths = data["bin_high"] - data["bin_low"]

        ax.bar(centers, data["count"], width=widths * 0.9,
               color="steelblue", edgecolor="white", linewidth=0.3)

        xlabel = f"{display_name}" + (f" [{unit}]" if unit else "")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Element count", fontsize=10)
        ax.set_title(display_name, fontsize=11, fontweight="bold")

        # Add statistics text
        total = data["count"].sum()
        mean = np.average(centers, weights=data["count"])
        ax.text(0.97, 0.95,
                f"N={total}\nmean={mean:.3g}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Mesh Quality Report", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    output = args.output or f"{args.prefix}_quality.png"
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
