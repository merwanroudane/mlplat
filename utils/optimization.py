"""Educational optimisers implemented from scratch (sections 16 and 71).

All functions are pure NumPy, bounded in iterations, and return the full
trajectory so labs can animate it. They are for teaching; production code
uses the solvers inside scikit-learn / SciPy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------- surfaces
@dataclass(frozen=True)
class Surface:
    name: str
    f: Callable[[Array, Array], Array]           # vectorised f(w1, w2)
    grad: Callable[[Array], Array]               # gradient at w (shape (2,))
    hess: Callable[[Array], Array]               # Hessian at w (2x2)
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    minimum: tuple[float, float]
    convex: bool


def quadratic(a: float = 1.0, b: float = 10.0) -> Surface:
    """Ill-conditioned bowl f = ½(a·w1² + b·w2²); condition number b/a."""
    return Surface(
        f"Quadratic bowl (κ = {b / a:g})",
        lambda w1, w2: 0.5 * (a * w1 ** 2 + b * w2 ** 2),
        lambda w: np.array([a * w[0], b * w[1]]),
        lambda w: np.diag([a, b]),
        (-4, 4), (-3, 3), (0.0, 0.0), True)


def rosenbrock() -> Surface:
    """Non-convex banana valley; global minimum at (1, 1)."""
    return Surface(
        "Rosenbrock (non-convex)",
        lambda w1, w2: (1 - w1) ** 2 + 100 * (w2 - w1 ** 2) ** 2,
        lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2), 200 * (w[1] - w[0] ** 2)]),
        lambda w: np.array([[2 - 400 * (w[1] - 3 * w[0] ** 2), -400 * w[0]], [-400 * w[0], 200]]),
        (-2, 2), (-1, 3), (1.0, 1.0), False)


def two_minima() -> Surface:
    """A double-well surface: the starting point decides which minimum is reached."""
    return Surface(
        "Double well (two local minima)",
        lambda w1, w2: (w1 ** 2 - 1) ** 2 + 0.3 * w1 + 0.5 * w2 ** 2,
        lambda w: np.array([4 * w[0] * (w[0] ** 2 - 1) + 0.3, w[1]]),
        lambda w: np.array([[12 * w[0] ** 2 - 4, 0.0], [0.0, 1.0]]),
        (-2, 2), (-2, 2), (-1.04, 0.0), False)


SURFACES: dict[str, Callable[[], Surface]] = {
    "quadratic": quadratic, "rosenbrock": rosenbrock, "double_well": two_minima,
}


# ----------------------------------------------------- full-batch methods
def gradient_descent(grad: Callable[[Array], Array], w0: Array, lr: float, n_iter: int,
                     momentum: float = 0.0, noise: float = 0.0, seed: int = 0, clip: float = 1e6) -> Array:
    """(Optionally noisy / momentum) gradient descent. Returns path of shape (k+1, d).

    ``noise`` adds N(0, noise²) to each gradient to mimic stochastic gradients.
    Stops early if the iterate explodes (divergence is part of the lesson).
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    v = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(int(n_iter)):
        g = grad(w) + (rng.normal(scale=noise, size=w.shape) if noise > 0 else 0.0)
        v = momentum * v - lr * g
        w = w + v
        path.append(w.copy())
        if not np.all(np.isfinite(w)) or np.abs(w).max() > clip:
            break
    return np.array(path)


def adam(grad: Callable[[Array], Array], w0: Array, lr: float, n_iter: int, beta1: float = 0.9,
         beta2: float = 0.999, eps: float = 1e-8) -> Array:
    """Adam (Kingma & Ba, 2015) — shown as a concept for the deep-learning bridge."""
    w = np.asarray(w0, dtype=float).copy()
    m = np.zeros_like(w)
    s = np.zeros_like(w)
    path = [w.copy()]
    for t in range(1, int(n_iter) + 1):
        g = grad(w)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g ** 2
        w = w - lr * (m / (1 - beta1 ** t)) / (np.sqrt(s / (1 - beta2 ** t)) + eps)
        path.append(w.copy())
    return np.array(path)


def newton(grad: Callable[[Array], Array], hess: Callable[[Array], Array], w0: Array, n_iter: int,
           damping: float = 1.0) -> Array:
    """(Damped) Newton's method: w ← w − damping · H⁻¹ ∇f."""
    w = np.asarray(w0, dtype=float).copy()
    path = [w.copy()]
    for _ in range(int(n_iter)):
        H = hess(w)
        try:
            step = np.linalg.solve(H, grad(w))
        except np.linalg.LinAlgError:
            break
        w = w - damping * step
        path.append(w.copy())
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
            break
    return np.array(path)


# ------------------------------------------------ data-driven least squares
def make_linear_data(n: int = 200, w_true: tuple[float, float] = (2.0, -1.0), noise: float = 0.5,
                     seed: int = 0) -> tuple[Array, Array]:
    """y = w1·x + w2 + ε  (w2 is the intercept) — used for SGD vs batch GD."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = w_true[0] * x + w_true[1] + rng.normal(scale=noise, size=n)
    return np.c_[x, np.ones(n)], y


def mse_loss_surface(X: Array, y: Array) -> Callable[[Array, Array], Array]:
    """Vectorised MSE(w1, w2) over a grid for the design X = [x, 1]."""
    def f(W1: Array, W2: Array) -> Array:
        pred = W1[..., None] * X[:, 0] + W2[..., None] * X[:, 1]
        return np.mean((y - pred) ** 2, axis=-1)
    return f


def sgd_linear(X: Array, y: Array, w0: Array, lr: float, n_epochs: int, batch_size: int | None = None,
               seed: int = 0, decay: float = 0.0) -> tuple[Array, Array]:
    """Mini-batch SGD on MSE. batch_size=None → full batch. Returns (path, loss per update)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = n if not batch_size else int(batch_size)
    w = np.asarray(w0, dtype=float).copy()
    path, losses = [w.copy()], [float(np.mean((y - X @ w) ** 2))]
    step = 0
    for _ in range(int(n_epochs)):
        idx = rng.permutation(n)
        for start in range(0, n, bs):
            b = idx[start:start + bs]
            g = -2 * X[b].T @ (y[b] - X[b] @ w) / len(b)
            eta = lr / (1 + decay * step)
            w = w - eta * g
            step += 1
            path.append(w.copy())
            losses.append(float(np.mean((y - X @ w) ** 2)))
            if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
                return np.array(path), np.array(losses)
    return np.array(path), np.array(losses)


def ols_closed_form(X: Array, y: Array) -> Array:
    """Normal equations via lstsq (numerically safer than an explicit inverse)."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ------------------------------------------------------ coordinate descent
def soft_threshold(z: Array | float, t: float) -> Array | float:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_coordinate_descent(X: Array, y: Array, alpha: float, n_sweeps: int = 100, tol: float = 1e-8) -> tuple[Array, list[Array]]:
    """Cyclic coordinate descent for (1/2n)||y − Xw||² + α||w||₁ (the objective scikit-learn's Lasso uses).

    Assumes centred y and columns of X (no intercept). Returns (w, history per sweep).
    """
    n, p = X.shape
    w = np.zeros(p)
    col_sq = (X ** 2).sum(axis=0) / n
    r = y - X @ w
    hist = [w.copy()]
    for _ in range(int(n_sweeps)):
        w_old = w.copy()
        for j in range(p):
            if col_sq[j] == 0:
                continue
            r += X[:, j] * w[j]
            rho = X[:, j] @ r / n
            w[j] = soft_threshold(rho, alpha) / col_sq[j]
            r -= X[:, j] * w[j]
        hist.append(w.copy())
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w, hist


def learning_rate_limit(hess: Array) -> float:
    """For a quadratic, GD converges iff 0 < lr < 2/λ_max."""
    return 2.0 / float(np.max(np.linalg.eigvalsh(hess)))
