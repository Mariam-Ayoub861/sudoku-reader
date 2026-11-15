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

def revise(domains, x, y):
    changed = False

    for v in list(domains[x]):
        if not any(w != v for w in domains[y]):
            domains[x].remove(v)
            changed = True

    return changed

def ac3(domains):
    queue = []
    for x in domains:
        for y in get_neighbours(x[0], x[1]):
            queue.append((x, y))

    while queue:
        x, y = queue.pop(0)
        if revise(domains, x, y):
            if not domains[x]:
                return False
            for z in get_neighbours(x[0], x[1]):
                if z != y:
                    queue.append((z, x))

    return True 


counters = {"assignments": 0, "backtracks": 0}


def solve(grid, domains, use_fc=False):
    cell = pick_cell(grid, domains)

    if cell is None:
        return grid

    r, c = cell

    for digit in domains[cell]:
        if is_consistent(grid, r, c, digit):
            if use_fc:
                new_domains = forward_check(grid, domains, r, c, digit)
                if new_domains is None:
                    continue
            else:
                new_domains = domains

            grid[r][c] = digit
            counters["assignments"] += 1

            result = solve(grid, new_domains, use_fc)

            if result is not None:
                return result

            grid[r][c] = 0
            counters["backtracks"] += 1

    return None


