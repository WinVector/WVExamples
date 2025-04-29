import numpy as np
import pandas as pd
import sympy as sp
from scipy import sparse


# define some symbolic utility functions


def safe_subs(expr, target, val):
    """Substitute, even if expr is a constant"""
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.subs(target, val)


def keep_low_degree_terms(poly, *, degree_bound: int):
    """Keep all terms of polynomial no larger than degree bound"""
    p2 = poly.as_poly(gens=sp.symbols("x y"))
    return sum(
        [
            coef * sp.Symbol("x") ** x * sp.Symbol("y") ** y
            for (x, y), coef in zip(p2.monoms(), p2.coeffs())
            if x + y <= degree_bound
        ],
        0,
    )


def convert_coefs_to_float(poly):
    """Convert polynomial coefficients to floating point"""
    p2 = poly.as_poly(gens=sp.symbols("x y"))
    return sum(
        [
            coef.n() * sp.Symbol("x") ** x * sp.Symbol("y") ** y
            for (x, y), coef in zip(p2.monoms(), p2.coeffs())
        ],
        0,
    )


def make_transition_matrices(k: int):
    k = int(k)
    assert k > 2
    keys = {k: str(k) for k in np.linspace(0, 1, num=k)}


def make_discrete_matrices(k: int):
    keys = np.linspace(0, 1, num=k)
    key_strings = [str(k) for k in keys]
    # key_strings
    states_2 = []
    states_3 = []
    index_map = {}
    for nm in ["F", "G"]:
        for x_i in range(k - 1):
            for y_i in range(x_i + 1, k):
                key = f"{nm}({key_strings[x_i]},{key_strings[y_i]})"
                index_map[key] = len(states_2)
                states_2.append(key)
    for nm in ["A", "B", "C", "D", "E"]:
        for x_i in range(k - 2):
            for y_i in range(x_i + 1, k - 1):
                for z_i in range(y_i + 1, k):
                    key = f"{nm}({key_strings[x_i]},{key_strings[y_i]},{key_strings[z_i]})"
                    index_map[key] = len(states_3)
                    states_3.append(key)
    deletion_map = np.zeros(shape=(len(states_2), len(states_3)), dtype=float)
    insertion_map = np.zeros(shape=(len(states_3), len(states_2)), dtype=float)
    # copy transition rules from pp. 302, 303 of journal article
    for x_i in range(k - 1):
        for y_i in range(x_i + 1, k):
            key = f"({key_strings[x_i]},{key_strings[y_i]})"
            idx_F = index_map[f"F{key}"]
            idx_G = index_map[f"G{key}"]
            for z_i in range(x_i):
                insertion_map[
                    index_map[
                        f"A({key_strings[z_i]},{key_strings[x_i]},{key_strings[y_i]})"
                    ]
                ][idx_F] = 1 / (k - 2)
                insertion_map[
                    index_map[
                        f"C({key_strings[z_i]},{key_strings[x_i]},{key_strings[y_i]})"
                    ]
                ][idx_G] = 1 / (k - 2)
            for z_i in range(x_i + 1, y_i):
                insertion_map[
                    index_map[
                        f"B({key_strings[x_i]},{key_strings[z_i]},{key_strings[y_i]})"
                    ]
                ][idx_F] = 1 / (k - 2)
                insertion_map[
                    index_map[
                        f"D({key_strings[x_i]},{key_strings[z_i]},{key_strings[y_i]})"
                    ]
                ][idx_G] = 1 / (k - 2)
            for z_i in range(y_i + 1, k):
                insertion_map[
                    index_map[
                        f"C({key_strings[x_i]},{key_strings[y_i]},{key_strings[z_i]})"
                    ]
                ][idx_F] = 1 / (k - 2)
                insertion_map[
                    index_map[
                        f"E({key_strings[x_i]},{key_strings[y_i]},{key_strings[z_i]})"
                    ]
                ][idx_G] = 1 / (k - 2)
    for x_i in range(k - 2):
        for y_i in range(x_i + 1, k - 1):
            for z_i in range(y_i + 1, k):
                key = f"({key_strings[x_i]},{key_strings[y_i]},{key_strings[z_i]})"
                d1 = f"({key_strings[y_i]},{key_strings[z_i]})"
                d2 = f"({key_strings[x_i]},{key_strings[z_i]})"
                d3 = f"({key_strings[x_i]},{key_strings[y_i]})"
                idx = index_map[f"A{key}"]
                deletion_map[index_map[f"F{d1}"]][idx] = 1 / 3
                deletion_map[index_map[f"F{d2}"]][idx] = 1 / 3
                deletion_map[index_map[f"F{d3}"]][idx] = 1 / 3
                idx = index_map[f"B{key}"]
                deletion_map[index_map[f"F{d1}"]][idx] = 1 / 3
                deletion_map[index_map[f"F{d2}"]][idx] = 1 / 3
                deletion_map[index_map[f"G{d3}"]][idx] = 1 / 3
                idx = index_map[f"C{key}"]
                deletion_map[index_map[f"G{d1}"]][idx] = 1 / 3
                deletion_map[index_map[f"F{d2}"]][idx] = 1 / 3
                deletion_map[index_map[f"F{d3}"]][idx] = 1 / 3
                idx = index_map[f"D{key}"]
                deletion_map[index_map[f"G{d1}"]][idx] = 1 / 3
                deletion_map[index_map[f"G{d2}"]][idx] = 1 / 3
                deletion_map[index_map[f"G{d3}"]][idx] = 1 / 3
                idx = index_map[f"E{key}"]
                deletion_map[index_map[f"G{d1}"]][idx] = 1 / 3
                deletion_map[index_map[f"G{d2}"]][idx] = 1 / 3
                deletion_map[index_map[f"G{d3}"]][idx] = 1 / 3
    # pd.DataFrame(deletion_map, index=states_2, columns=states_3)
    # pd.DataFrame(insertion_map, index=states_3, columns=states_2)
    return (
        sparse.csr_matrix(deletion_map),
        sparse.csr_matrix(insertion_map),
        states_2,
        states_3,
    )


def solve_for_F(*, deletion_map, insertion_map, states_2):
    two_to_two_map = deletion_map @ insertion_map
    # pd.DataFrame(two_to_two_map.toarray(), index=states_2, columns=states_2)
    # solve for the limiting distribution
    p = two_to_two_map - np.identity(two_to_two_map.shape[1])
    del two_to_two_map
    p = np.append(p, [[1] * p.shape[1]], axis=0)
    soln = np.linalg.lstsq(p, np.zeros(shape=(p.shape[0],)) + 1, rcond=None)[0]
    del p
    p_F = 0
    for i, st in enumerate(states_2):
        if st.startswith("F"):
            p_F += soln[i]
    return p_F


def initial_state(rng):
    while True:
        v1 = rng.random()
        v2 = rng.random()
        if v1 != v2:
            break
    if v1 > v2:
        v1, v2 = v2, v1
    return (rng.choice(["F", "G"]), v1, v2)


def insert_node(state, *, rng):
    symbol, x, y = state
    while True:
        z = rng.random()  # uniform [0, 1]
        if (z != x) and (z != y):
            break
    if z < x:
        return ("A" if symbol == "F" else "C", z, x, y)
    if z < y:
        return ("B" if symbol == "F" else "D", x, z, y)
    return ("C" if symbol == "F" else "E", x, y, z)


deletion_table = {
    ("A", "x"): "F", ("A", "y"): "F", ("A", "z"): "F",
    ("B", "x"): "F", ("B", "y"): "F", ("B", "z"): "G",
    ("C", "x"): "G", ("C", "y"): "F", ("C", "z"): "F",
    ("D", "x"): "G", ("D", "y"): "G", ("D", "z"): "G",
    ("E", "x"): "G", ("E", "y"): "G", ("E", "z"): "G",
}


def delete_node(state, *, rng):
    symbol, x, y, z = state
    to_delete = rng.choice(["x", "y", "z"])
    v1, v2 = y, z  # delete x
    if to_delete == "y":
        v1, v2 = x, z
    elif to_delete == "z":
        v1, v2 = x, y
    return (deletion_table[(symbol, to_delete)], v1, v2)
