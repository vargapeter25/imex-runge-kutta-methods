import numpy as np
from scipy.optimize import fsolve

def ERK(f, u0, A, b, c, Tl, Tr, N):
    """ Numerical solver for IVP on a uniform grid. Assumes that the RK method is explicit.

    Uses the RK method to solve the IVP (`f`, `u0`, `Tl`) on the uniform grid defined by `Tl`, `Tr` and `N`.

    :param f: The function defining the ODE. `f` must have args in order (t, u).
    :param u0: The initial value at `Tl`.
    :param A: The `a` coefficients of the Butcher tableau.
    :param b: The `b` coefficients of the Butcher tableau.
    :param c: The `c` coefficients of the Butcher tableau.
    :param Tl: The beginning of the time intervall.
    :param Tr: The end of the time intervall.
    :param N: The number of iterations.
    :return: The numercial solution in the form `h, t, y`, step size, grid and values in order.
    """

    # Initializing variables
    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    single_var = False

    if not np.all(np.isclose(np.sum(A, axis=1), c)):
        raise ValueError('c values must be same as the values in the A table.')

    # Handling one dimensional case
    if isinstance(u0, float) or isinstance(u0, int):
        single_var = True
        u0 = [u0]

    # Set inital values
    h = (Tr - Tl) / N
    n = len(u0)
    s = len(c)
    t = np.linspace(Tl, Tr, N + 1)
    y = np.zeros((N + 1, n))
    y[0] = np.asarray(u0)
    
    for i in range(0, N):
        k = np.zeros((s, n))
        tn = t[i]
        yn = y[i]
        for j in range(s):
            args = yn + h * A[j, :j] @ k[:j]
            k[j] = np.asarray(f(tn + h * c[j], args))

        y[i + 1] = yn + h * np.dot(b, k)

    if single_var:
        y = y[:, 0]

    return h, t, y

def IRK(f, u0, A, b, c, Tl, Tr, N):
    """ Numerical solver for IVP on a uniform grid. Uses general implicit solver.

    Uses the RK method to solve the IVP (`f`, `u0`, `Tl`) on the uniform grid defined by `Tl`, `Tr` and `N`.

    :param f: The function defining the ODE. `f` must have args in order (t, u).
    :param u0: The initial value at `Tl`.
    :param A: The `a` coefficients of the Butcher tableau.
    :param b: The `b` coefficients of the Butcher tableau.
    :param c: The `c` coefficients of the Butcher tableau.
    :param Tl: The beginning of the time intervall.
    :param Tr: The end of the time intervall.
    :param N: The number of iterations.
    :return: The numercial solution in the form `h, t, y`, step size, grid and values in order.
    """

    # Initializing variables
    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    single_var = False

    if not np.all(np.isclose(np.sum(A, axis=1), c)):
        raise ValueError('c values must be same as the values in the A table.')

    # Handling one dimensional case
    if isinstance(u0, float) or isinstance(u0, int):
        single_var = True
        u0 = [u0]

    # Set inital values
    h = (Tr - Tl) / N
    n = len(u0)
    s = len(c)
    t = np.linspace(Tl, Tr, N + 1)
    y = np.zeros((N + 1, n))
    y[0] = np.asarray(u0)

    for i in range(0, N):
        k = np.zeros((s, n))
        tn = t[i]
        yn = y[i]

        def calc_k(k):
            k = np.reshape(k, (s, n))
            r = np.zeros((s, n))
            for j in range(s):
                r[j] = np.asarray(f(tn + h * c[j], (yn + h * A[j, :] @ k[:])))

            return np.reshape(r, s * n)

        base = np.repeat([np.asarray(f(tn, yn))], s)

        k = np.reshape(fsolve(lambda k: calc_k(k) - k, base), (s, n))
        y[i + 1] = yn + h * np.dot(b, k)

    if single_var:
        y = y[:, 0]

    return h, t, y