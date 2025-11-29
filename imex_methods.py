import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import lu_factor, lu_solve
from typing import Callable, Any

class ImplicitSolver():
    def __init__(self, f):
        self.f = lambda t, u: np.asarray(f(t, *u))
    
    def __call__(self, t, x) -> np.ndarray:
        return self.f(t, x)

    # solve for x = g(t, x * alpha + beta)
    def solve_for(self, t: float, alpha: float, beta: np.ndarray, base: np.ndarray) -> Any:
        f_impl = lambda x: x - self.f(t, alpha * x + beta)
        return fsolve(f_impl, base)


class LinearImplicitSolver(ImplicitSolver):
    def __init__(self, G):
        self.G = np.asarray(G)

    def __call__(self, t, x) -> np.ndarray:
        return self.G @ x

    def solve_for(self, t: float, alpha: float, beta: np.ndarray, base: np.ndarray) -> Any:
        return np.linalg.solve(np.eye(*np.shape(self.G)) - self.G * alpha, self.G @ beta)  


class LinearImplicitSolverLU(ImplicitSolver):
    def __init__(self, G):
        self.G = np.asarray(G)
        self.LUs = []
        self.alphas = np.array([])

    def get_LU(self, alpha: float):
        i = np.argmax(np.isclose(self.alphas, alpha)) if len(self.alphas) > 0 else -1
        if i == -1 or not np.isclose(self.alphas[i], alpha):
            lu, piv = lu_factor(np.eye(*np.shape(self.G)) - self.G * alpha)
            self.LUs.append((lu, piv))
            self.alphas = np.append(self.alphas, [alpha])
            return self.LUs[-1]
        return self.LUs[i]

    def __call__(self, t, x) -> np.ndarray:
        return self.G @ x

    def solve_for(self, t: float, alpha: float, beta: np.ndarray, base: np.ndarray) -> Any:
        return np.asarray(lu_solve(self.get_LU(alpha), self.G @ beta))


def IMEX(f: Callable[[float, np.ndarray], np.ndarray], g: ImplicitSolver, u0, A, A_, b, b_, c, Tl, Tr, N):
    """ Numerical solver for IVP on a uniform grid. Uses IMEX DIRK method.

    Uses IMEX solver for the `h = f + g` ODE, where `f` is solved with ERK method and `g` is solved with DIRK.
    Based on the `ImplicitSolver` type it can be more efficient in sepcial cases.

    :param f: Explicit discretization is used for this part. `f` must have args in order (t, u).
    :param g: Implicit discretization is used for this part. `g` must be an instance of an `ImplicitSolver`.
    :param u0: The initial value at `Tl`.
    :param A: The `a` coefficients of the Butcher tableau for the DIRK system.
    :param A_: The `a` coefficients of the Butcher tableau for the ERK system.
    :param b: The `b` coefficients of the Butcher tableau for the DIRK system.
    :param b_: The `b` coefficients of the Butcher tableau for the ERK system.
    :param c: The `c` coefficients of the Butcher tableaus. 
    :param Tl: The beginning of the time intervall.
    :param Tr: The end of the time intervall.
    :param N: The number of iterations.
    :return: The numercial solution in the form `h, t, y`, step size, grid and values in order.
    """

    # Initialize variables
    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    b_ = np.asarray(b_)
    A_ = np.asarray(A_)
    single_var = False

    # Check for errors
    if not np.all(np.isclose(np.sum(A, axis=1), c)) or not np.all(np.isclose(np.sum(A_, axis=1), np.insert(c, 0, 0))):
        raise ValueError('c values must be same as the values in the A and A_ tables.')

    if isinstance(u0, float) or isinstance(u0, int):
        single_var = True
        u0 = [u0]

    # Initialize starting values
    h = (Tr - Tl) / N
    n = len(u0)
    s = len(c)
    t = np.linspace(Tl, Tr, N + 1)
    y = np.zeros((N + 1, n))
    y[0] = np.asarray(u0)

    f_ = lambda t, u: np.asarray(f(t, *u))
    
    for i in range(0, N):
        tn = t[i]
        yn = y[i]

        def calc_kernels():
            k = np.zeros((n, s))
            k_ = np.zeros((n, s + 1))
            k_[:, 0] = f_(tn, yn)
            implicit_solver_base = g(tn, yn)

            for i in range(s):
                k[:, i] = g.solve_for(tn + h * c[i], h * A[i, i], yn + h * (
                    np.dot(k[:, :i], A[i, :i]) + 
                    np.dot(k_[:, :i + 1], A_[i + 1, :i + 1])
                ), implicit_solver_base)

                # Calculating explicit part
                k_[:, i + 1] = f_(tn + h * c[i], yn + h * (
                        np.dot(k[:, :i + 1], A[i, :i + 1]) +
                        np.dot(k_[:, :i + 1], A_[i + 1, :i + 1])
                ))

            return k, k_

        k, k_ = calc_kernels()
        y[i + 1] = yn + h * (np.dot(k, b) + np.dot(k_, b_))

    if single_var:
        y = y[:, 0]

    return h, t, y
