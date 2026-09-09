"""Kitaev chain solver — BdG and Majorana representations."""

import numpy as np
from scipy.linalg import eigvalsh

_VALID_REPRESENTATIONS = frozenset({"bdg", "majorana"})
_VALID_BOUNDARIES = frozenset({"obc", "pbc"})


def _validate_inputs(N: int, representation: str, boundary: str) -> tuple[str, str]:
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    rep = representation.lower()
    bnd = boundary.lower()
    if rep not in _VALID_REPRESENTATIONS:
        raise ValueError(
            f"representation must be 'bdg' or 'majorana', got {representation!r}"
        )
    if bnd not in _VALID_BOUNDARIES:
        raise ValueError(f"boundary must be 'obc' or 'pbc', got {boundary!r}")
    return rep, bnd


def _build_bdg(N: int, mu: float, delta: float, t: float, boundary: str) -> np.ndarray:
    """Build H_BdG (2N x 2N, real symmetric) in Nambu ordering."""
    H = np.zeros((2 * N, 2 * N), dtype=np.float64)

    # A block (top-left): on-site and hopping
    np.fill_diagonal(H[:N, :N], -mu)
    if N > 1:
        j = np.arange(N - 1)
        H[j, j + 1] = -t
        H[j + 1, j] = -t

    # B block (top-right): pairing
    B = np.zeros((N, N), dtype=np.float64)
    if N > 1:
        j = np.arange(N - 1)
        B[j, j + 1] = delta
        B[j + 1, j] = -delta

    if boundary == "pbc" and N >= 2:
        H[0, N - 1] = H[N - 1, 0] = -t
        B[N - 1, 0] = delta
        B[0, N - 1] = -delta

    H[:N, N:] = B
    H[N:, :N] = -B
    H[N:, N:] = -H[:N, :N]

    return H


def _build_majorana_antisymmetric(
    N: int, mu: float, delta: float, t: float, boundary: str
) -> np.ndarray:
    """Build A_M (2N x 2N, real antisymmetric) in interleaved Majorana ordering."""
    A = np.zeros((2 * N, 2 * N), dtype=np.float64)

    j = np.arange(N)
    A[2 * j, 2 * j + 1] = -mu
    A[2 * j + 1, 2 * j] = mu

    if N > 1:
        j = np.arange(N - 1)
        A[2 * j, 2 * j + 3] = -(t + delta)
        A[2 * j + 3, 2 * j] = t + delta
        A[2 * j + 1, 2 * j + 2] = t - delta
        A[2 * j + 2, 2 * j + 1] = -(t - delta)

    if boundary == "pbc" and N >= 2:
        A[2 * N - 2, 1] = -(t + delta)
        A[1, 2 * N - 2] = t + delta
        A[2 * N - 1, 0] = t - delta
        A[0, 2 * N - 1] = -(t - delta)

    return A


def build_hamiltonian(
    N: int,
    mu: float,
    delta: float,
    t: float,
    representation: str,
    boundary: str,
) -> np.ndarray:
    """Return the 2N x 2N matrix to diagonalize (H_BdG or i*A_M)."""
    rep, bnd = _validate_inputs(N, representation, boundary)
    if rep == "bdg":
        return _build_bdg(N, mu, delta, t, bnd)
    A = _build_majorana_antisymmetric(N, mu, delta, t, bnd)
    return (1j * A).astype(np.complex128)


def spectrum(
    N: int,
    mu: float,
    delta: float,
    t: float = 1.0,
    representation: str = "bdg",
    boundary: str = "obc",
) -> dict:
    """Diagonalize the Kitaev chain and return the energy spectrum."""
    H = build_hamiltonian(N, mu, delta, t, representation, boundary)
    full = eigvalsh(H)
    E = full[N:].copy()
    E_gs = -0.5 * E.sum()
    return {"full": full, "E": E, "E_gs": E_gs}
