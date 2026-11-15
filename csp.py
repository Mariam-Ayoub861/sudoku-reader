# csp.py - Sudoku as a CSP
# grid: list of 9 lists, 9 ints each, 0 = empty

def get_neighbours(r, c):
    """Return the set of (row, col) cells that share a row, column or box with (r, c).
    Must NOT include (r, c) itself. Result size should always be 20."""
    # TODO: row peers, column peers, box peers
    neighbours = set()
    box_row = 3 * (r // 3)
    box_col = 3 * (c // 3)

    for col in range(9):
        neighbours.add((r, col))
    for row in range(9):
        neighbours.add((row, c))
    for row in range(box_row, box_row+3):
        for col in range(box_col, box_col+3):
                neighbours.add((row, col))
    neighbours.remove((r,c))
    
    # HINT: the box's top-left corner is (3*(r//3), 3*(c//3))
    return neighbours


def init_domains(grid):
    """Return dict {(r, c): set of digits} .
    Given cell -> its domain is just {value}.
    Empty cell -> digits 1-9 minus any digit already in a neighbour."""
    # TODO
    domains = {}
    for r in range(9):
        for c in range(9):
            # case 1 or case 2 here
            if grid[r][c] != 0:
                domains[(r,c)] = {grid[r][c]}
            else:
                x = set(range(1,10))
                for nb in get_neighbours(r,c):
                    val = grid[nb[0]][nb[1]]
                    if val != 0:
                        x.discard(val)
                domains[(r,c)] = x
    return domains

def is_consistent(grid, r, c, d):
    """Can digit d go in (r, c) without equalling any neighbour's value?"""
    # TODO
    for nb in get_neighbours(r, c):
        if grid[nb[0]][nb[1]] == d:
            return False
    return True


def pick_cell(grid, domains):
    """Choose the next empty cell to fill."""
    # TODO: start simple (first empty cell).
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return (row,col)
    return None
    # Later upgrade: the empty cell with the smallest domain (MRV).


counters = {"assignments": 0, "backtracks": 0}


def solve(grid, domains):
    """Plain backtracking. Return the solved grid, or None if no solution."""
    # TODO:
    # 1. pick an empty cell; if none left -> solved, return grid
    cell = pick_cell(grid, domains)
    if cell is None:
        return grid
    r,c = cell
    # 2. for each digit in its domain:
    #      - if consistent: assign it (count it), recurse
    #      - if the recursion succeeded, return the result
    #      - otherwise undo the assignment (count a backtrack)
    for digit in domains[cell]:
        if is_consistent(grid,r,c,digit):
            grid[r][c] = digit
            counters["assignments"] += 1
            result =  solve(grid, domains)
            if result is not None:
                return result
            else:
                grid[r][c] = 0
                counters["backtracks"] += 1
        
    # 3. no digit worked -> return None
    return None

counters["assignments"] = 0
counters["backtracks"] = 0

grid = [[5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9]]
d = init_domains(grid)
assert d[(0, 0)] == {5}
assert d[(0, 2)] == {1, 2, 4}

for r in range(9):
    for c in range(9):
        n = get_neighbours(r, c)
        assert len(n) == 20 and (r, c) not in n

assert is_consistent(grid, 0, 2, 4) == True
assert is_consistent(grid, 0, 2, 5) == False   # 5 is already in row 0
assert is_consistent(grid, 0, 2, 3) == False   # 3 is already in row 0
assert is_consistent(grid, 0, 2, 8) == False   # 8 is already in the box

assert pick_cell(grid, d) == (0, 2)
full = [[1]*9 for _ in range(9)]
assert pick_cell(full, None) == None

grid_copy = [row[:] for row in grid]
result = solve(grid_copy, init_domains(grid_copy))
assert result is not None
assert result[0] == [5,3,4,6,7,8,9,1,2]
print(counters)