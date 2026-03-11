"""Regenerate all dashboards and the index.

Usage:
    python scripts/generate_all.py data/races/2026-03-08-Kiltorcan/
    python scripts/generate_all.py --all
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dashboard import main as generate_dashboard
from scripts.generate_index import main as generate_index


def main():
    parser = argparse.ArgumentParser(description="Generate all kart racing dashboards")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("race_dir", nargs="?", help="Path to a single race directory")
    group.add_argument("--all", action="store_true", help="Regenerate dashboards for all races")
    args = parser.parse_args()

    data_dir = "data/races"

    if args.all:
        race_dirs = sorted(
            d for d in glob.glob(os.path.join(data_dir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "race.yaml"))
        )
        if not race_dirs:
            print(f"No race directories with race.yaml found in {data_dir}/")
            sys.exit(1)
        for race_dir in race_dirs:
            print(f"\n=== {os.path.basename(race_dir)} ===")
            generate_dashboard(race_dir)
    else:
        generate_dashboard(args.race_dir)

    print("\n=== Index page ===")
    generate_index()

    print("\nDone. Outputs in docs/")


if __name__ == "__main__":
    main()
