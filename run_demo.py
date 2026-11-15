# run_demo.py - the integrated demo.
#   python run_demo.py                      (default: seed 1, 30 givens, 2 misreads)
#   python run_demo.py --misreads 3 --seed 5
#   python run_demo.py --no-nn              (system with the correction step OFF)
import argparse
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from puzzle_gen import make_puzzle
from reader import load_all, make_pools, read_mosaic
from csp import init_domains, ac3
from search import search, solvable, least_confident, apply


def draw(ax, grid, title, colors=None):
    colors = colors or {}
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(10):
        w = 2.5 if i % 3 == 0 else 0.5
        ax.plot([0, 9], [i, i], "k", lw=w)
        ax.plot([i, i], [0, 9], "k", lw=w)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                ax.text(c + 0.5, r + 0.5, str(grid[r][c]), ha="center", va="center",
                        fontsize=14, color=colors.get((r, c), "black"))


def show(grid):
    for row in grid:
        print("  " + " ".join(str(v) if v else "." for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--no-nn", action="store_true", help="skip the NN-guided correction step")
    args = ap.parse_args()

    nn, x_test, y_test = load_all()
    pools = make_pools(nn, x_test, y_test)
    puzzle, solution = make_puzzle(args.seed, args.givens)
    rng = random.Random(args.seed + 100)
    read, probs, hard_cells = read_mosaic(puzzle, pools, nn, x_test, rng, args.misreads)

    wrong = {(r, c) for r in range(9) for c in range(9) if read[r][c] != puzzle[r][c]}
    print("1) INPUT: puzzle as read by the neural network (. = blank)")
    show(read)
    print(f"   cells misread (hidden from the system): {len(wrong)}")

    print("2) AC-3 on the reading:")
    conflict_free = ac3(init_domains(read))
    print("   no conflict found" if conflict_free else "   CONFLICT found: the reading cannot be right")

    if args.no_nn:
        res = solvable(read)
        print("3) NN correction is OFF:", "solved" if res else "puzzle is unsolvable, no way to fix it")
        return

    print("3) SEARCH (UCS) over corrections, goal test = CSP (AC-3 + BT + FC):")
    res = search(read, probs, "ucs")
    if res is None:
        print("   UNREADABLE. Least confident cells:", least_confident(probs))
        fig, ax = plt.subplots(figsize=(5, 5))
        draw(ax, read, "Unreadable puzzle", {c: "red" for c in wrong})
        fig.savefig("demo.png", dpi=120, bbox_inches="tight")
        return

    for cell, old, new, cost in res["corrections"]:
        print(f"   cell {cell}: {old} -> {new}  (cost {cost:.3f})")
    print(f"   nodes expanded={res['expanded']} generated={res['generated']} total cost={res['cost']:.3f}")
    print("4) SOLUTION:")
    show(res["solution"])
    print("   matches the true solution:", res["solution"] == solution)

    corrected = apply(read, frozenset((cell, new) for cell, _, new, _ in res["corrections"]))
    fixed = {cell for cell, _, _, _ in res["corrections"]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    draw(axes[0], read, "1. NN reading (red = misread)", {c: "red" for c in wrong})
    draw(axes[1], corrected, "2. After search corrections (green)", {c: "green" for c in fixed})
    filled = {(r, c): "tab:blue" for r in range(9) for c in range(9) if corrected[r][c] == 0}
    draw(axes[2], res["solution"], "3. CSP solution (blue = filled by CSP)", filled)
    fig.savefig("demo.png", dpi=120, bbox_inches="tight")
    print("plot saved to demo.png")


if __name__ == "__main__":
    main()