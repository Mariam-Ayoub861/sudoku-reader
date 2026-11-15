# Handwritten Sudoku Reader and Solver (P7)

A neural network reads a Sudoku written in handwritten digits (MNIST). Sometimes it misreads a digit and the puzzle breaks. The system finds the most likely misread digits, corrects them and solves the puzzle.

## How the three parts work together
1. **Neural network** (scikit-learn MLP) classifies each digit image and gives 9 probabilities.
2. **AC-3** detects a conflict in the reading (for example two equal digits in a row).
3. **Search** (UCS or Greedy) tries sets of corrections, at most 2. Cost of changing a cell = log(p_read) - log(p_new), never negative.
4. **CSP** (Backtracking, Forward Checking, AC-3, each switchable) tests every corrected puzzle. The first solvable one wins.
5. If nothing works within the limit: the puzzle is reported as unreadable, with the least confident cells.

## Environment setup
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run
Download https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz (about 11 MB) and save it as data/mnist.npz, then:

    python make_data.py        # drop digit 0, scale to 0-1, split (seed 42)
    python train_nn.py         # NN + logistic regression baseline, accuracy, ECE, confusion matrix
    python run_demo.py         # integrated demo, saves demo.png
    python run_demo.py --no-nn # same puzzle with the correction step OFF
    python run_demo.py --seed 5 --misreads 3
    python test_project.py     # 11 tests: 6 normal, 3 edge, 2 impossible
    python experiments.py      # three experiments, saves exp*.csv and exp*.png (about 3 minutes)

## Files
- csp.py: neighbours, domains, backtracking, forward checking, revise, AC-3 (with counters)
- make_data.py, train_nn.py: data and model
- puzzle_gen.py: seeded puzzle generator (unique solution)
- reader.py: mosaic builder and NN reading, injects hard digits
- search.py: UCS and Greedy correction search
- run_demo.py, experiments.py, test_project.py, check_csp.py

## Data dictionary
| Item | Real or generated | How |
|---|---|---|
| MNIST images and labels (digits 1-9) | Real | LeCun et al.; digit 0 removed; train 44077 / val 10000 / test 9020 |
| Sudoku puzzles and solutions | Generated | randomised backtracking, cells removed while the solution stays unique; seeded |
| 9x9 mosaics | Generated | each given cell gets a real MNIST test image of that digit; blank cells are all-zero images |
| Blank-cell detection | Rule | total brightness below 5, not the NN |
| Hard injected digits | Generated | test images of the right digit that the trained NN misreads |

## Limitations
- A correction that makes the puzzle solvable is not always the original puzzle (uniqueness goal not implemented).
- At most 2 corrections; A* and temperature scaling are not implemented (optional extensions).
- Puzzles and mosaics are generated, not photographed real-world Sudoku.
- The MLP and the logistic regression both stop at the iteration limit.
