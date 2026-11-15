# reader.py - build a mosaic of MNIST images for a puzzle, inject HARD digits, read it with the NN.
import random
import numpy as np
import joblib

BLANK_THRESHOLD = 5.0   # total brightness below this = blank cell (simple rule, not the NN)


def load_all():
    d = np.load("data/mnist_1to9.npz")
    nn = joblib.load("data/nn.joblib")
    return nn, d["x_test"], d["y_test"]


def make_pools(nn, x_test, y_test):
    """For each digit: 'easy' = any test image, 'hard' = test images the NN misreads.
    (If a digit has no misread image, use its 10 least-confident images.)"""
    probs = nn.predict_proba(x_test)
    pred = nn.classes_[probs.argmax(axis=1)]
    easy, hard = {}, {}
    for d in range(1, 10):
        idx = np.where(y_test == d)[0]
        easy[d] = idx
        wrong = idx[pred[idx] != d]
        if len(wrong) == 0:
            wrong = idx[np.argsort(probs[idx, d - 1])[:10]]
        hard[d] = wrong
    return easy, hard


def read_mosaic(puzzle, pools, nn, x_test, rng, n_hard=0):
    """Return (read_grid, probs, hard_cells).
    read_grid : 9x9 digits the NN read (0 = blank)
    probs     : {(r, c): array of 9 probabilities for digits 1..9} for non-blank cells
    hard_cells: set of cells where a hard image was injected"""
    easy, hard = pools
    givens = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c]]
    hard_cells = set(rng.sample(givens, n_hard))

    images = np.zeros((9, 9, 784), dtype=np.float32)      # blank cell = all zeros
    for (r, c) in givens:
        d = puzzle[r][c]
        pool = hard[d] if (r, c) in hard_cells else easy[d]
        images[r, c] = x_test[pool[rng.randrange(len(pool))]]

    read = [[0] * 9 for _ in range(9)]
    probs = {}
    for r in range(9):
        for c in range(9):
            img = images[r, c]
            if img.sum() < BLANK_THRESHOLD:                # blank detected by brightness
                continue
            p = nn.predict_proba(img.reshape(1, -1))[0]
            probs[(r, c)] = p
            read[r][c] = int(p.argmax()) + 1
    return read, probs, hard_cells


if __name__ == "__main__":
    from puzzle_gen import make_puzzle
    from csp import init_domains, ac3

    nn, x_test, y_test = load_all()
    pools = make_pools(nn, x_test, y_test)
    puzzle, solution = make_puzzle(seed=1, givens=30)
    rng = random.Random(1)

    for n_hard in (0, 2):
        read, probs, hard_cells = read_mosaic(puzzle, pools, nn, x_test, rng, n_hard)
        wrong = [(r, c, puzzle[r][c], read[r][c]) for r in range(9) for c in range(9)
                 if read[r][c] != puzzle[r][c]]
        print(f"n_hard={n_hard}: misread cells (row, col, true, read) = {wrong}")
        print("   AC-3 finds no conflict:", ac3(init_domains(read)))