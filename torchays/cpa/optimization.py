from typing import Any, Dict, Tuple

import numpy as np
from scipy import optimize


def lineprog(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray = None,
    b_eq: np.ndarray = None,
    x0: np.ndarray = None,
    method: str = "highs",
    bounds: Any = (None, None),
    options: Dict = {
        "maxiter": 100,
        # "disp": True,
    },
):
    return optimize.linprog(c, a_ub, b_ub, a_eq, b_eq, method=method, options=options, bounds=bounds, x0=x0)


def cheby_ball(funcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Find the largest Euclidean ball inside the halfspaces a @ x + b >= 0.

    Minimize -r subject to -a @ x + r * ||a_i|| <= b and -a @ x <= b.
    Return a center only when the solver succeeds with a positive radius.
    """
    a, b_ub = funcs[:, :-1], funcs[:, -1]
    c = np.negative(np.r_[np.zeros(np.shape(a)[1]), 1])
    norm = np.sqrt(np.sum(a * a, axis=1))
    # -ax+r||a|| <= b
    r_ub = np.c_[-a, norm]
    # -ax <= b
    a_ub = np.c_[-a, np.zeros_like(b_ub)]
    # Stack the ball-containment and center-feasibility constraints.
    a_ub = np.concatenate([r_ub, a_ub])
    b_ub = np.concatenate([b_ub, b_ub])
    sol = lineprog(c, a_ub, b_ub)
    if sol.success:
        x, r = sol.x[:-1], sol.x[-1]
        if r > 0 and x is not None:
            return x, r, sol.success
    return None, None, sol.success


def lineprog_intersect(
    func: np.ndarray,
    pn_funcs: np.ndarray,
    x0: np.ndarray,
    bounds: Any = None,
) -> bool:
    """Test whether a hyperplane meets the constrained domain from x0's side.

    Minimize the signed affine function using homogeneous coordinates with the
    final coordinate fixed at 1. A nonpositive optimum accepts intersection,
    including contact with the boundary.
    """
    c, b = func[:-1], func[-1]
    x0_r = c @ x0 + b
    if x0_r == 0:
        return True
    sign = 1 if x0_r > 0 else -1
    a_eq = np.r_[np.zeros_like(c), 1]
    a_eq = np.expand_dims(a_eq, axis=0)
    b_eq = np.ones(1)
    a_ub, b_ub = -pn_funcs, np.zeros(pn_funcs.shape[0])
    sol = lineprog(
        func * sign,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds=bounds,
    )
    if not sol.success or sol.fun is None:
        return False
    if sol.fun <= 0 and sol.slack.all() >= 0:
        return True
    return False
