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

def measure_method(method, name, y0, Tl, Tr, f_exact, target_error, Ns) -> TestResult:
    def f_impl(n):
        h, t, y = method(y0, Tl, Tr, n)
        return target_error - np.linalg.norm(y - f_exact(t), np.inf)

    ## Calculate optimal log(N) as integer
    lo = -1
    hi = len(Ns) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        error_diff = f_impl(Ns[mid])
        if not np.isnan(error_diff) and error_diff > 0:
            hi = mid
        else:
            lo = mid

    N = Ns[hi]

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

def create_mesasurement(f, g, A, A_, b, b_, c, Tl, Tr, f_exact, errors, G = None, Ns = [2**12], verbose = True):
    data = None
    header = ['Name', 'Exec time (s)', 'Step Size', 'Error']
    for error in errors:
        
        if verbose:
            print(f'Calculating for error: {error}')
    
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
            results.append(measure_method(method, name, f_exact(Tl), Tl, Tr, f_exact, error, Ns))

        data_ = get_table_data(results)

        if data is None:
            data = data_
        else:
            for i in range(len(data)):
                data[i] = data[i] + data_[i][1:]
                header += ['Exec time (s)', 'Step Size', 'Error']

    table = tabulate.tabulate(data, headers=header, tablefmt='html')
    return table

def create_mesasurement_lin(f, g, G, A, A_, b, b_, c, Tl, Tr, f_exact, errors, Ns = [2**12], verbose = True):
    data = None
    header = ['Name', 'Exec time (s)', 'Step Size', 'Error']
    for error in errors:
        
        if verbose:
            print(f'Calculating for error: {error}')

        f_ = lambda t, x: f(t, x) + g(t, x)
        c_ = [0] + c
        methods = [
            (lambda y0, Tl, Tr, N: ERK(f_, y0, A_, b_, c_, Tl, Tr, N), 'ERK'),
            #(lambda y0, Tl, Tr, N: IMEX(f, ImplicitSolver(g), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX'),
            (lambda y0, Tl, Tr, N: IMEX(f, LinearImplicitSolver(G), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX Lin'),
            (lambda y0, Tl, Tr, N: IMEX(f, LinearImplicitSolverLU(G), y0, A, A_, b, b_, c, Tl, Tr, N), 'IMEX LU')
        ]
        
        results = []
        for method, name in methods:
            if verbose:
                print(f'Measuring method: {name}')
            results.append(measure_method(method, name, f_exact(Tl), Tl, Tr, f_exact, error, Ns))

        data_ = get_table_data(results)

        if data is None:
            data = data_
        else:
            for i in range(len(data)):
                data[i] = data[i] + data_[i][1:]
                header += ['Exec time (s)', 'Step Size', 'Error']

    table = tabulate.tabulate(data, headers=header, tablefmt='html')
    return table