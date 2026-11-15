# search.py - find the most likely set of digit corrections that makes the puzzle solvable.
import heapq
import math
import numpy as np
from csp import init_domains, ac3, solve, get_neighbours


def solvable(grid):
    """CSP test: AC-3 first (catches equal digits in a unit), then backtracking + FC.
    Returns the solved grid, or None."""
    dom = init_domains(grid)
    if not ac3(dom):
        return None
    g = [row[:] for row in grid]
    return solve(g, dom, use_fc=True)


def conflicts(grid):
    """Number of pairs of equal digits in the same row/column/box (greedy heuristic)."""
    n = 0
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v:
                for nb in get_neighbours(r, c):
                    if nb > (r, c) and grid[nb[0]][nb[1]] == v:
                        n += 1
    return n


def build_actions(read, probs, n_cells=12, n_alts=2):
    """Allowed corrections: (cell, new_digit, cost). Only the n_cells least-confident
    cells, and only their n_alts most likely alternative digits.
    cost = log(p_read) - log(p_new), always >= 0 because 'read' was the most likely digit."""
    cells = sorted(probs, key=lambda cell: probs[cell].max())[:n_cells]
    actions = []
    for cell in cells:
        p = probs[cell]
        a = read[cell[0]][cell[1]]
        count = 0
        for k in np.argsort(-p):
            b = int(k) + 1
            if b == a:
                continue
            cost = math.log(max(p[a - 1], 1e-12)) - math.log(max(p[b - 1], 1e-12))
            actions.append((cell, b, max(cost, 0.0)))
            count += 1
            if count == n_alts:
                break
    return actions


def apply(read, state):
    g = [row[:] for row in read]
    for (r, c), d in state:
        g[r][c] = d
    return g


def least_confident(probs, k=5):
    return sorted(probs, key=lambda cell: probs[cell].max())[:k]


def search(read, probs, method="ucs", max_corr=2, n_cells=12, n_alts=2):
    """method: 'ucs' (order by cost so far) or 'greedy' (order by remaining conflicts).
    State = frozenset of (cell, new_digit). Goal = the CSP can solve the corrected puzzle.
    Returns a dict, or None if nothing within max_corr corrections works."""
    actions = build_actions(read, probs, n_cells, n_alts)
    action_cost = {(cell, b): cost for cell, b, cost in actions}
    stats = {"expanded": 0, "generated": 1}

    def priority(g, state):
        if method == "ucs":
            return (g,)
        return (conflicts(apply(read, state)), g)

    start = frozenset()
    heap = [(priority(0.0, start), 0, 0.0, start)]
    tie = 1
    seen = set()
    while heap:
        _, _, g, state = heapq.heappop(heap)
        if state in seen:
            continue
        seen.add(state)
        stats["expanded"] += 1

        solution = solvable(apply(read, state))        # goal test = CSP
        if solution is not None:
            log = [(cell, read[cell[0]][cell[1]], d, action_cost[(cell, d)])
                   for cell, d in sorted(state)]
            return {"corrections": log, "cost": g, "solution": solution, **stats}

        if len(state) >= max_corr:
            continue
        changed = {cell for cell, _ in state}
        for cell, b, cost in actions:
            if cell in changed:
                continue
            new_state = state | {(cell, b)}
            if new_state in seen:
                continue
            stats["generated"] += 1
            heapq.heappush(heap, (priority(g + cost, new_state), tie, g + cost, new_state))
            tie += 1
    return None


if __name__ == "__main__":
    import random
    from puzzle_gen import make_puzzle
    from reader import load_all, make_pools, read_mosaic

    nn, x_test, y_test = load_all()
    pools = make_pools(nn, x_test, y_test)
    puzzle, solution = make_puzzle(seed=1, givens=30)

    for n_hard in (1, 2):
        rng = random.Random(7)
        read, probs, hard_cells = read_mosaic(puzzle, pools, nn, x_test, rng, n_hard)
        print(f"\n=== {n_hard} injected misread(s) ===")
        for method in ("ucs", "greedy"):
            res = search(read, probs, method)
            if res is None:
                print(method, "-> unreadable; least confident cells:", least_confident(probs))
            else:
                exact = res["solution"] == solution
                print(f"{method:6s} expanded={res['expanded']} generated={res['generated']} "
                      f"cost={res['cost']:.3f} exact_solution={exact}")
                for cell, old, new, cost in res["corrections"]:
                    print(f"        cell {cell}: {old} -> {new}  (cost {cost:.3f})")