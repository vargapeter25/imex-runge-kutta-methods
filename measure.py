import numpy as np
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

def measure_method(method, name, y0, Tl, Tr, N, f_exact) -> TestResult:
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

def get_table(results) -> str:
    data = []
    for result in results:
        data.append([result.name, f'{result.exec_time:.5f}', f'{result.h:.5f}', f'{result.error:.5f}'])

    table = tabulate.tabulate(data, headers=['Name', 'Exec time (s)', 'Step Size', 'Error'], tablefmt='html')
    return table
