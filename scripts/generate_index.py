"""Generate the root index.html listing all race dashboards.

Usage:
    python scripts/generate_index.py
"""

import os
import sys
import glob

import yaml
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    data_dir = "data/races"
    races = []

    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue

        name = os.path.basename(race_dir)
        dashboard_path = os.path.join("races", name, "dashboard.html")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_dashboard = os.path.join(repo_root, "docs", dashboard_path)
        if not os.path.exists(abs_dashboard):
            continue

        race_info = {"name": name, "dashboard_path": dashboard_path}

        meta_path = os.path.join(race_dir, "race.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f) or {}
            race_info["track"] = meta.get("track", name)
            race_info["date"] = str(meta.get("date", ""))
            if meta.get("session_type"):
                race_info["session_type"] = meta["session_type"]

        summary_path = os.path.join(race_dir, "summary_generated.yaml")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = yaml.safe_load(f) or {}
            race_info["best_lap"] = summary.get("best_lap", {}).get("time")
            race_info["total_laps"] = summary.get("total_laps")

        races.append(race_info)

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("index.html.j2")

    html = template.render(races=races)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "docs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Index written to {output_path} ({len(races)} race(s))")


if __name__ == "__main__":
    main()
