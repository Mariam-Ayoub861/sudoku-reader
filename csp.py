def get_neighbours(r, c):
    neighbours = set()
    box_row = 3 * (r // 3)
    box_col = 3 * (c // 3)

    for col in range(9):
        neighbours.add((r, col))

    for row in range(9):
        neighbours.add((row, c))

    for row in range(box_row, box_row + 3):
        for col in range(box_col, box_col + 3):
            neighbours.add((row, col))

    neighbours.remove((r, c))
    return neighbours


def init_domains(grid):
    domains = {}

    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                domains[(r, c)] = {grid[r][c]}
            else:
                x = set(range(1, 10))

                for nb in get_neighbours(r, c):
                    val = grid[nb[0]][nb[1]]
                    if val != 0:
                        x.discard(val)

                domains[(r, c)] = x

    return domains


def is_consistent(grid, r, c, d):
    for nb in get_neighbours(r, c):
        if grid[nb[0]][nb[1]] == d:
            return False

    return True


def pick_cell(grid, domains):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return (row, col)

    return None


def forward_check(grid, domains, r, c, digit):
    new = {cell: set(vals) for cell, vals in domains.items()}

    for nb in get_neighbours(r, c):
        if grid[nb[0]][nb[1]] == 0:
            new[nb].discard(digit)

            if not new[nb]:
                return None

    return new


counters = {"assignments": 0, "backtracks": 0}


def solve(grid, domains):
    cell = pick_cell(grid, domains)

    if cell is None:
        return grid

    r, c = cell

    for digit in domains[cell]:
        if is_consistent(grid, r, c, digit):
            grid[r][c] = digit
            counters["assignments"] += 1

            result = solve(grid, domains)

            if result is not None:
                return result

            grid[r][c] = 0
            counters["backtracks"] += 1

    return None


counters["assignments"] = 0
counters["backtracks"] = 0

grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

d = init_domains(grid)

assert d[(0, 0)] == {5}
assert d[(0, 2)] == {1, 2, 4}

for r in range(9):
    for c in range(9):
        n = get_neighbours(r, c)
        assert len(n) == 20 and (r, c) not in n

assert is_consistent(grid, 0, 2, 4) == True
assert is_consistent(grid, 0, 2, 5) == False
assert is_consistent(grid, 0, 2, 3) == False
assert is_consistent(grid, 0, 2, 8) == False

assert pick_cell(grid, d) == (0, 2)

full = [[1] * 9 for _ in range(9)]
assert pick_cell(full, None) == None

grid_copy = [row[:] for row in grid]
result = solve(grid_copy, init_domains(grid_copy))

assert result is not None
assert result[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]

print(counters)

new = forward_check(grid, d, 0, 2, 2)

assert new is not None
assert 2 not in new[(0, 3)]
assert 2 in d[(0, 3)]