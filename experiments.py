# experiments.py - the three experiments. Saves exp*.csv and exp*.png.
#   python experiments.py --quick     (small run, ~3 min, to check everything works)
#   python experiments.py             (full run)
import csv
import random
import signal
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from puzzle_gen import make_puzzle
from reader import load_all, make_pools, read_mosaic
from csp import init_domains, ac3, solve, counters
from search import search, solvable

SEEDS = [1, 2, 3, 4, 5]
PER_SEED = 20            # puzzles per seed per setting (experiments 1 and 3)
PER_SEED_E2 = 4         # puzzles per seed per setting (experiment 2)
MISREADS = [0, 1, 2]
GIVENS = [25, 30, 35]
TIMEOUT = 10            # seconds allowed for one plain-backtracking run
RANDOM_BUDGET = 60      # CSP tests given to the random-correction baseline
if "--quick" in sys.argv:
    SEEDS, PER_SEED, PER_SEED_E2 = [1, 2], 2, 2


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)       # Linux only


def mean_std(per_seed):
    v = np.array(per_seed, dtype=float)
    v = v[~np.isnan(v)]
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def seed_mean(values):
    return float(np.mean(values)) if values else float("nan")


def random_correction(read, rng, truth):
    """Baseline: test the reading as is, then try random 1-2 cell changes."""
    sol = solvable(read)
    if sol is not None:
        return sol == truth
    cells = [(r, c) for r in range(9) for c in range(9) if read[r][c]]
    for _ in range(RANDOM_BUDGET):
        g = [row[:] for row in read]
        for (r, c) in rng.sample(cells, rng.choice([1, 2])):
            g[r][c] = rng.choice([d for d in range(1, 10) if d != read[r][c]])
        sol = solvable(g)
        if sol is not None:
            return sol == truth
    return False


def exp13():
    nn, x_test, y_test = load_all()
    pools = make_pools(nn, x_test, y_test)
    out = {}
    for level in MISREADS:
        keys = ["ucs_exact", "ucs_nodes", "ucs_cost", "greedy_exact", "greedy_nodes",
                "greedy_cost", "none_exact", "random_exact"]
        per_seed = {k: [] for k in keys}
        for seed in SEEDS:
            tmp = {k: [] for k in keys}
            for i in range(PER_SEED):
                s = seed * 1000 + level * 100 + i
                puzzle, truth = make_puzzle(s, 30)
                read, probs, _ = read_mosaic(puzzle, pools, nn, x_test, random.Random(s), level)
                for m in ("ucs", "greedy"):
                    res = search(read, probs, m)
                    tmp[m + "_exact"].append(1.0 if res and res["solution"] == truth else 0.0)
                    if res:
                        tmp[m + "_nodes"].append(res["expanded"])
                        tmp[m + "_cost"].append(res["cost"])
                sol = solvable(read)
                tmp["none_exact"].append(1.0 if sol is not None and sol == truth else 0.0)
                tmp["random_exact"].append(
                    1.0 if random_correction(read, random.Random(s + 1), truth) else 0.0)
            for k in keys:
                per_seed[k].append(seed_mean(tmp[k]))
            print(f"  exp1/3: misreads={level} seed={seed} done", flush=True)
        out[level] = {k: mean_std(v) for k, v in per_seed.items()}
    return out


def run_bt(grid, mode):
    counters["assignments"] = 0
    counters["backtracks"] = 0
    g = [row[:] for row in grid]
    timed_out = False
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        dom = init_domains(g)
        if mode == "bt+fc+ac3":
            ac3(dom)
        solve(g, dom, use_fc=(mode != "bt"))
    except Timeout:
        timed_out = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return counters["assignments"], counters["backtracks"], timed_out


def exp2():
    modes = ["bt", "bt+fc", "bt+fc+ac3"]
    out = {}
    for givens in GIVENS:
        per = {m: {"assign": [], "back": [], "timeouts": 0} for m in modes}
        for seed in SEEDS:
            tmp = {m: {"a": [], "b": []} for m in modes}
            for i in range(PER_SEED_E2):
                puzzle, _ = make_puzzle(seed * 1000 + givens * 10 + i, givens)
                for m in modes:
                    a, b, to = run_bt(puzzle, m)
                    tmp[m]["a"].append(a)
                    tmp[m]["b"].append(b)
                    per[m]["timeouts"] += int(to)
            for m in modes:
                per[m]["assign"].append(seed_mean(tmp[m]["a"]))
                per[m]["back"].append(seed_mean(tmp[m]["b"]))
            print(f"  exp2: givens={givens} seed={seed} done", flush=True)
        out[givens] = {m: {"assign": mean_std(per[m]["assign"]), "back": mean_std(per[m]["back"]),
                           "timeouts": per[m]["timeouts"]} for m in modes}
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("saved", path)


def grouped_bar(ax, labels, series, ylabel, title, log=False):
    w = 0.8 / len(series)
    x = np.arange(len(labels))
    for i, (name, (m, s)) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, m, w, yerr=s, label=name, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend()


def main():
    t0 = time.time()
    print("Experiments 1 and 3 (search methods, NN vs baselines)...")
    r13 = exp13()
    print("Experiment 2 (BT vs BT+FC vs BT+FC+AC-3)...")
    r2 = exp2()

    # ---- tables
    rows = []
    for lv in MISREADS:
        d = r13[lv]
        for m in ("ucs", "greedy"):
            rows.append([lv, m, *d[m + "_nodes"], *d[m + "_cost"], *d[m + "_exact"]])
    write_csv("exp1_search.csv", ["injected_misreads", "method", "nodes_mean", "nodes_std",
              "cost_mean", "cost_std", "recovery_mean", "recovery_std"], rows)

    rows = []
    for lv in MISREADS:
        d = r13[lv]
        rows.append([lv, *d["ucs_exact"], *d["none_exact"], *d["random_exact"]])
    write_csv("exp3_nn_vs_baselines.csv", ["injected_misreads", "nn_guided_mean", "nn_guided_std",
              "no_correction_mean", "no_correction_std", "random_mean", "random_std"], rows)

    rows = []
    for g in GIVENS:
        for m, d in r2[g].items():
            rows.append([g, m, *d["assign"], *d["back"], d["timeouts"]])
    write_csv("exp2_csp.csv", ["givens", "method", "assignments_mean", "assignments_std",
              "backtracks_mean", "backtracks_std", "timeouts"], rows)

    # ---- plots
    labels = [str(lv) for lv in MISREADS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grouped_bar(axes[0], labels, {m.upper(): ([r13[lv][m + "_nodes"][0] for lv in MISREADS],
                [r13[lv][m + "_nodes"][1] for lv in MISREADS]) for m in ("ucs", "greedy")},
                "nodes expanded", "Exp 1: nodes expanded (solved cases)")
    grouped_bar(axes[1], labels, {m.upper(): ([r13[lv][m + "_exact"][0] for lv in MISREADS],
                [r13[lv][m + "_exact"][1] for lv in MISREADS]) for m in ("ucs", "greedy")},
                "exact recovery rate", "Exp 1: exact recovery rate")
    for ax in axes:
        ax.set_xlabel("injected misreads")
    fig.savefig("exp1_search.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    names = {"ucs_exact": "NN-guided (UCS)", "none_exact": "no correction", "random_exact": "random correction"}
    grouped_bar(ax, labels, {n: ([r13[lv][k][0] for lv in MISREADS], [r13[lv][k][1] for lv in MISREADS])
                for k, n in names.items()}, "exact recovery rate", "Exp 3: NN-guided vs baselines")
    ax.set_xlabel("injected misreads")
    fig.savefig("exp3_nn_vs_baselines.png", dpi=120, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, yl in ((axes[0], "assign", "assignments (log)"), (axes[1], "back", "backtracks (log)")):
        grouped_bar(ax, [str(g) for g in GIVENS],
                    {m: ([r2[g][m][key][0] for g in GIVENS], [r2[g][m][key][1] for g in GIVENS])
                     for m in ("bt", "bt+fc", "bt+fc+ac3")}, yl, "Exp 2: " + yl, log=True)
        ax.set_xlabel("number of givens")
    fig.savefig("exp2_csp.png", dpi=120, bbox_inches="tight")

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} minutes.")
    print("Plain-BT timeouts per setting:", {g: r2[g]["bt"]["timeouts"] for g in GIVENS})


if __name__ == "__main__":
    main()