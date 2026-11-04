"""Analyze the separate mobile-scheduler exports: python -m analysis.mobility."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

RUN = re.compile(r"seed(?P<seed>\d+)_algo(?P<algorithm>\d+)_D.*_source(?P<sourceChanges>True|False|true|false)_episodes\.csv$")
PROTOCOL = re.compile(r"_dt[^_]+_train[^_]+_length[^_]+")
KEYS = ["algorithm", "diffusion", "tolerance", "beta", "sourceChanges", "protocol"]
METRICS = ["roundsPerNodeSecond", "absoluteError", "normalizedError", "toleratedError", "withinTolerance",
           "reachableFraction", "recoveryCensored", "rewardRate"]
NAMES = {0: "Mobile GNN", 1: "Mobile MLP", 2: "Heuristic", **{i: f"Fixed {i-2}s" for i in range(3, 8)},
         8: "Static-trained MLP", 9: "Static-trained GNN", 10: "Fixed 0.5s"}


def load_episodes(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*_episodes.csv")):
        match = RUN.fullmatch(path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        protocol = PROTOCOL.search(path.name)
        frame["protocol"] = protocol.group().lstrip("_") if protocol else "unspecified"
        frame["seed"] = int(match["seed"])
        frame["algorithm"] = int(match["algorithm"])
        frame["sourceChanges"] = match["sourceChanges"].lower() == "true"
        frames.append(frame.loc[~frame["training"].astype(bool)])
    if not frames:
        raise ValueError(f"No mobile episode exports in {directory}")
    return pd.concat(frames, ignore_index=True)


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    """Average within each seed first: episodes from the same trained policy are not independent runs."""
    per_seed = episodes.groupby(KEYS + ["seed"], dropna=False)[METRICS].mean().reset_index()
    records = []
    for key, group in per_seed.groupby(KEYS, dropna=False):
        record = dict(zip(KEYS, key))
        record["seeds"] = len(group)
        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            record[metric] = values.mean() if len(values) else np.nan
            record[f"{metric}_ci95"] = (t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
                                        if len(values) > 1 else np.nan)
        records.append(record)
    return pd.DataFrame(records)


def plot(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Each panel holds tolerance, diffusion and source regime fixed; beta traces the trade-off.
    for (diffusion, tolerance, sources, protocol), group in summary.groupby(["diffusion", "tolerance", "sourceChanges", "protocol"]):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for algorithm, points in group.groupby("algorithm"):
            points = points.sort_values("beta")
            label = NAMES.get(algorithm, str(algorithm))
            axes[0].plot(points.roundsPerNodeSecond, points.toleratedError, "o-", label=label)
            axes[1].plot(points.roundsPerNodeSecond, points.withinTolerance, "o-", label=label)
        for axis in axes:
            axis.set_xlabel("Rounds per node per second")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Mean normalized error beyond tolerance")
        axes[1].set_ylabel("Fraction of time within tolerance")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize="small")
        figure.suptitle(f"D={diffusion} m²/s; tolerance={tolerance} m; source changes={sources}")
        name = f"mobility_D{diffusion}_tol{tolerance}_source{sources}_{protocol}"
        figure.savefig(output / f"{name}.pdf")
        figure.savefig(output / f"{name}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=Path("build/exports/mobility"))
    parser.add_argument("--out", type=Path, default=Path("charts/archive/mobility"))
    parser.add_argument("--pareto", action="store_true",
                        help="Render paper-style Pareto/beta figures; require all five betas and four seeds")
    arguments = parser.parse_args()
    episodes = load_episodes(arguments.exports)
    if arguments.pareto:
        from .mobility_pareto import prepare
        prepare(episodes)  # Validate completeness before publishing any output.
    summary = summarize(episodes)
    arguments.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(arguments.out / "evaluation_episodes.csv", index=False)
    summary.to_csv(arguments.out / "summary.csv", index=False)
    # Never average successful recoveries as though censored episodes had recovered.
    episodes[KEYS + ["seed", "episode", "recoverySeconds", "recoveryCensored"]].to_csv(
        arguments.out / "recovery.csv", index=False)
    if arguments.pareto:
        from .mobility_pareto import render
        render(episodes, arguments.out)
    else:
        plot(summary, arguments.out)
    print(f"Saved {len(summary)} configurations to {arguments.out}")


if __name__ == "__main__":
    main()
