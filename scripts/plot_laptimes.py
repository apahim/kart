"""Plot official lap times from a race."""

import sys
import matplotlib.pyplot as plt
from load_data import load_laptimes


def plot_laptimes(csv_path):
    df = load_laptimes(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["lap"], df["seconds"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Lap Times")

    # Highlight best lap
    best_idx = df["seconds"].idxmin()
    ax.bar(df.loc[best_idx, "lap"], df.loc[best_idx, "seconds"],
           color="green", edgecolor="black", label=f"Best: {df.loc[best_idx, 'time']}")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_laptimes.py <path/to/laptimes.csv>")
        sys.exit(1)
    plot_laptimes(sys.argv[1])
