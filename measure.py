import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from time import perf_counter
import tabulate

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
