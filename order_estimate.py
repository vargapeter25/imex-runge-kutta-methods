import numpy as np
from typing import Any, Union

def order_from_fine_grid(method: Any, u0: Union[np.ndarray, float], Tl: float, Tr: float, steps: list[int], fine_steps: int, norm_ord : Any = np.inf):
    """ Estimating the order of consistency for numerical solution based on a fine grid solution.
    Calculates the `E(h_i)` from based on `steps` and a fine grid defined by `fine_step`, and calculates the estimate based on adjacent values.
        
    :param method: The method callable with `y_0`, `t_0`, `t_f` and `N` for solving the ODE.
    :param y_0: The initial value at `t_0`. Must be an `np.array`.
    :param t_0: The beginning of the time intervall.
    :param t_f: The end of the time intervall.
    :param steps: The number of steps per calculations used in the estimates.
    :param fine_step: The number of steps in the fine grid. 
    :param norm_ord: The norm used for the approximation. It uses `numpy.norm`.
    :return: An array of the estimates where each row corresponds to one of the coordinates.
    """

    for N in steps:
        if fine_steps % N != 0:
            raise ValueError("Step size must divide the 'fine_steps'.")

    # Calculating solution for fine grid
    _, _, y_fine = method(u0, Tl, Tr, fine_steps)

    errors = []
    hs = []
    for N in steps:
        h, t, y = method(u0, Tl, Tr, N)
        # Estimate from fine grid
        exact = y_fine[::fine_steps // N]
        error = np.linalg.norm(y - exact, axis=0, ord=norm_ord)
        # Discrete p-norm | if p is not inf, than
        if isinstance(norm_ord, int):
            error *= h ** (1. / norm_ord)
        
        hs.append(h)
        errors.append(error)

    result = []
    for i in range(len(errors) - 1):
        result.append(np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1]))

    return np.array(np.array(result).transpose())