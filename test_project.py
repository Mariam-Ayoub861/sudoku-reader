# test_project.py - 6 normal, 3 edge, 2 impossible cases.  Run: python test_project.py
import unittest
import numpy as np
from csp import get_neighbours, init_domains, ac3, solve, counters
from puzzle_gen import make_puzzle, count_solutions
from search import search, solvable

CLASSIC = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]
CLASSIC_FIRST_ROW = [5, 3, 4, 6, 7, 8, 9, 1, 2]


def run(grid, fc):
    counters["assignments"] = 0
    counters["backtracks"] = 0
    g = [row[:] for row in grid]
    res = solve(g, init_domains(g), use_fc=fc)
    return res, counters["assignments"]


def synthetic_probs(read):
    """Fake NN output: 95% sure of every read digit."""
    probs = {}
    for r in range(9):
        for c in range(9):
            if read[r][c]:
                p = np.full(9, 0.05 / 8)
                p[read[r][c] - 1] = 0.95
                probs[(r, c)] = p
    return probs


class NormalCases(unittest.TestCase):
    def test_1_every_cell_has_20_neighbours(self):
        for r in range(9):
            for c in range(9):
                n = get_neighbours(r, c)
                self.assertEqual(len(n), 20)
                self.assertNotIn((r, c), n)

    def test_2_initial_domains(self):
        d = init_domains(CLASSIC)
        self.assertEqual(d[(0, 0)], {5})
        self.assertEqual(d[(0, 2)], {1, 2, 4})

    def test_3_backtracking_solves_classic_puzzle(self):
        res, _ = run(CLASSIC, fc=False)
        self.assertEqual(res[0], CLASSIC_FIRST_ROW)

    def test_4_forward_checking_same_answer_fewer_assignments(self):
        res_bt, a_bt = run(CLASSIC, fc=False)
        res_fc, a_fc = run(CLASSIC, fc=True)
        self.assertEqual(res_bt, res_fc)
        self.assertLess(a_fc, a_bt)

    def test_5_generator_unique_and_reproducible(self):
        puzzle, _ = make_puzzle(2, 30)
        self.assertEqual(count_solutions([row[:] for row in puzzle]), 1)
        self.assertEqual(make_puzzle(2, 30)[0], puzzle)

    def test_6_search_known_answer(self):
        """Small example with a known answer: one misread cell, the NN is
        50% sure of the wrong digit and 45% sure of the right one."""
        puzzle, solution = make_puzzle(3, 30)
        for r in range(9):
            givens = [(c, puzzle[r][c]) for c in range(9) if puzzle[r][c]]
            if len(givens) >= 2:
                break
        (c1, true_digit), (_, other_digit) = givens[0], givens[1]
        read = [row[:] for row in puzzle]
        read[r][c1] = other_digit                    # duplicate digit in the row
        probs = synthetic_probs(read)
        p = np.full(9, 0.05 / 7)
        p[other_digit - 1] = 0.50
        p[true_digit - 1] = 0.45
        probs[(r, c1)] = p
        for method in ("ucs", "greedy"):
            res = search(read, probs, method)
            self.assertIsNotNone(res)
            self.assertEqual([(cell, new) for cell, _, new, _ in res["corrections"]],
                             [((r, c1), true_digit)])
            self.assertEqual(res["solution"], solution)


class EdgeCases(unittest.TestCase):
    def test_7_already_complete_grid(self):
        _, solution = make_puzzle(5, 30)
        res, assignments = run(solution, fc=True)
        self.assertEqual(res, solution)
        self.assertEqual(assignments, 0)

    def test_8_single_blank_cell(self):
        _, solution = make_puzzle(5, 30)
        grid = [row[:] for row in solution]
        grid[4][4] = 0
        res, _ = run(grid, fc=True)
        self.assertEqual(res[4][4], solution[4][4])

    def test_9_empty_grid_is_still_solved(self):
        res, _ = run([[0] * 9 for _ in range(9)], fc=True)
        self.assertIsNotNone(res)
        for row in res:
            self.assertEqual(sorted(row), list(range(1, 10)))


class ImpossibleCases(unittest.TestCase):
    def test_10_cell_with_no_possible_digit(self):
        grid = [[0] * 9 for _ in range(9)]
        grid[0] = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        grid[1][8] = 9                               # cell (0,8) now has no candidate
        self.assertFalse(ac3(init_domains(grid)))
        self.assertIsNone(solvable(grid))
        print("\n   -> NO SOLUTION: cell (0,8) has no possible digit")

    def test_11_too_many_conflicts_to_fix_is_unreadable(self):
        grid = [[0] * 9 for _ in range(9)]
        grid[0][0] = grid[0][1] = 5                  # three separate duplicate pairs,
        grid[1][3] = grid[1][4] = 7                  # but only 2 corrections allowed
        grid[2][6] = grid[2][7] = 3
        self.assertIsNone(search(grid, synthetic_probs(grid), "ucs", max_corr=2))
        print("\n   -> NO SOLUTION: unreadable within 2 corrections")


if __name__ == "__main__":
    unittest.main(verbosity=2)