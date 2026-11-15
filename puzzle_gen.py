# puzzle_gen.py - generate Sudoku puzzles with a UNIQUE solution, reproducibly.
import random
from csp import is_consistent


def candidates(grid, r, c):
    return [d for d in range(1, 10) if is_consistent(grid, r, c, d)]


def most_constrained(grid):
    """Empty cell with the fewest candidates, or None if the grid is full."""
    best = None
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                cand = candidates(grid, r, c)
                if best is None or len(cand) < len(best[2]):
                    best = (r, c, cand)
    return best


def full_grid(rng):
    """Random complete valid grid, by randomised backtracking."""
    grid = [[0] * 9 for _ in range(9)]

    def fill():
        best = most_constrained(grid)
        if best is None:
            return True
        r, c, cand = best
        rng.shuffle(cand)
        for d in cand:
            grid[r][c] = d
            if fill():
                return True
            grid[r][c] = 0
        return False

    fill()
    return grid


def count_solutions(grid, limit=2):
    """Count solutions, stopping once `limit` is reached."""
    best = most_constrained(grid)
    if best is None:
        return 1
    r, c, cand = best
    total = 0
    for d in cand:
        grid[r][c] = d
        total += count_solutions(grid, limit - total)
        grid[r][c] = 0
        if total >= limit:
            break
    return total


def make_puzzle(seed, givens=30):
    """Return (puzzle, solution). Removes cells while the solution stays unique."""
    rng = random.Random(seed)
    solution = full_grid(rng)
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    filled = 81
    for r, c in cells:
        if filled <= givens:
            break
        old = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions(puzzle) != 1:
            puzzle[r][c] = old      # removing it broke uniqueness, put it back
        else:
            filled -= 1
    return puzzle, solution


if __name__ == "__main__":
    puzzle, solution = make_puzzle(seed=1, givens=30)
    print("givens:", sum(1 for row in puzzle for v in row if v))
    for row in puzzle:
        print(row)
    # same seed must give the same puzzle
    assert make_puzzle(1, 30)[0] == puzzle
    print("same seed -> same puzzle: OK")