import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from time import perf_counter
import tabulate
from rk_methods import *
from imex_methods import *

@dataclass
class TestResult:
    name: str
    Tl: float
    Tr: float
    N: int
    exec_time: float
    h: float
    t: np.ndarray
    y: np.ndarray
    error: float = 0

def measure_method(method, name, y0, Tl, Tr, f_exact, target_error, N_ = 2000, N_min = 2000, N_max = 10000) -> TestResult:
    def f_impl(n):
        n = min(max(int(n), 1), N_max)
        h, t, y = method(y0, Tl, Tr, n)
        return target_error - np.linalg.norm(y - f_exact(t), np.inf)
    
    if abs(f_impl(N_)) < target_error / 10:
        N = N_
    else:
        l_val = f_impl(N_min)
        r_val = f_impl(N_max)

        if l_val > 0:
            N = N_min
        elif r_val < 0:
            N = N_max
        else:
            N = int(brentq(f_impl, N_min, N_max, xtol=1))

    start = perf_counter()
    h, t, y = method(y0, Tl, Tr, N)
    end = perf_counter()
    error = np.linalg.norm(y - f_exact(t), np.inf)
    return TestResult(
        name,
        Tl,
        Tr,
        N,
        end - start,
        h,
        t,
        y,
        error
    )

def get_table_data(results) -> str:
    data = []
    for result in results:
        data.append([result.name, f'{result.exec_time:.5f}', f'{result.h:.5f}', f'{result.error:.5f}'])
    return data

def create_mesasurement(f, g, A, A_, b, b_, c, Tl, Tr, f_exact, errors, G = None, N_min = 2000, N_max = 10000, verbose = True):
    data = None
    header = ['Name', 'Exec time (s)', 'Step Size', 'Error']
    for error in errors:
        
        if verbose:
            print(f'Calculating for error: {error}')
        
        if G is None:
            starting_method = lambda y0, Tl, Tr, N: IMEX(f, ImplicitSolver(g), y0, A, A_, b, b_, c, Tl, Tr, N) 
        else:
            starting_method = lambda y0, Tl, Tr, N: IMEX(f, LinearImplicitSolver(G), y0, A, A_, b, b_, c, Tl, Tr, N)

        test_solve = measure_method(starting_method, 'Test', f_exact(Tl), Tl, Tr, f_exact, error, N_min, N_min, N_max)
        N_ = test_solve.N

        f_ = lambda t, x: f(t, x) + g(t, x)
        c_ = [0] + c
        methods = [
            (lambda y0, Tl, Tr, N: ERK(f_, y0, A_, b_, c_, Tl, Tr, N), 'ERK'),
            (lambda y0, Tl, Tr, N: IRK(f_, y0, A, b, c, Tl, Tr, N), 'IRK'),
            (lambda y0, Tl, Tr, N: IMEX(f, ImplicitSolver(g), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX'),
        ]
        if G is not None:
            methods += [
                (lambda y0, Tl, Tr, N: IMEX(f, LinearImplicitSolver(G), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX Lin'),
                (lambda y0, Tl, Tr, N: IMEX(f, LinearImplicitSolverLU(G), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX LU')
            ]
        
        results = []
        for method, name in methods:
            if verbose:
                print(f'Measuring method: {name}')
            results.append(measure_method(method, name, f_exact(Tl), Tl, Tr, f_exact, error, N_, N_min, N_max))

        data_ = get_table_data(results)

        if data is None:
            data = data_
        else:
            for i in range(len(data)):
                data[i] = data[i] + data_[i][1:]
                header += ['Exec time (s)', 'Step Size', 'Error']

    table = tabulate.tabulate(data, headers=header, tablefmt='html')
    return table