from csp import *
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

for fc in (False, True):
    counters["assignments"] = 0
    counters["backtracks"] = 0
    g = [row[:] for row in grid]
    res = solve(g, init_domains(g), use_fc=fc)
    assert res[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]
    print("FC =", fc, counters)

t = {(0, 0): {1, 2}, (0, 1): {1}}
assert revise(t, (0, 0), (0, 1)) == True
assert t[(0, 0)] == {2}
assert revise(t, (0, 0), (0, 1)) == False   # nothing left to remove

t2 = {(0, 0): {1, 2}, (0, 1): {1, 2}}
assert revise(t2, (0, 0), (0, 1)) == False  # both can still differ

dd = init_domains(grid)
assert ac3(dd) == True            # the normal puzzle has no conflict

bad = [row[:] for row in grid]
bad[0][2] = 5                     # a second 5 in row 0, next to the given 5
assert ac3(init_domains(bad)) == False