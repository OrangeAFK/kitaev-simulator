"""Four Kitaev-chain Hamiltonians with fixed hopping t = 1.

Majorana labels a = 2, ..., 2N+1 map to array indices a - 2.
The Majorana form is H = (i/4) gamma^T A gamma with A real antisymmetric.
A term i*alpha*gamma_a*gamma_b (a < b) contributes A[a,b] = 2*alpha,
A[b,a] = -2*alpha.
"""

from __future__ import annotations

import numpy as np


def _majorana_index(label: int, N: int) -> int:
    """Map Majorana label a in {2, ..., 2N+1} (with PBC wrap) to 0..2N-1."""
    # Labels run 2 .. 2N+1; PBC wraps 2N+2 -> 2, 2N+3 -> 3.
    a = label
    if a == 2 * N + 2:
        a = 2
    elif a == 2 * N + 3:
        a = 3
    return a - 2


def _set_A(A: np.ndarray, a: int, b: int, alpha: float, N: int) -> None:
    """Add contribution of i*alpha*gamma_a*gamma_b to antisymmetric A."""
    ia = _majorana_index(a, N)
    ib = _majorana_index(b, N)
    if ia == ib:
        return
    # Ensure we write A[min,max] consistently with a < b in label sense after wrap
    A[ia, ib] += 2.0 * alpha
    A[ib, ia] -= 2.0 * alpha


def build_majorana_obc(N: int, mu: float, delta: float) -> np.ndarray:
    """Real antisymmetric A for OBC Majorana Hamiltonian (2N x 2N)."""
    A = np.zeros((2 * N, 2 * N), dtype=float)
    # -(i mu / 2) sum_j gamma_{2j} gamma_{2j+1}  => alpha = -mu/2
    for j in range(1, N + 1):
        _set_A(A, 2 * j, 2 * j + 1, -0.5 * mu, N)
    # (i/2) sum_{j=1}^{N-1} [(1-Delta) g_{2j} g_{2j+3} - (1+Delta) g_{2j+1} g_{2j+2}]
    for j in range(1, N):
        _set_A(A, 2 * j, 2 * j + 3, 0.5 * (1.0 - delta), N)
        _set_A(A, 2 * j + 1, 2 * j + 2, -0.5 * (1.0 + delta), N)
    return A


def build_majorana_pbc(N: int, mu: float, delta: float) -> np.ndarray:
    """Real antisymmetric A for PBC Majorana Hamiltonian (2N x 2N)."""
    A = np.zeros((2 * N, 2 * N), dtype=float)
    for j in range(1, N + 1):
        _set_A(A, 2 * j, 2 * j + 1, -0.5 * mu, N)
    for j in range(1, N + 1):
        _set_A(A, 2 * j, 2 * j + 3, 0.5 * (1.0 - delta), N)
        _set_A(A, 2 * j + 1, 2 * j + 2, -0.5 * (1.0 + delta), N)
    return A


def _build_fermionic_blocks(
    N: int, mu: float, delta: float, pbc: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Normal block h and pairing block Delta_mat for BdG."""
    h = np.zeros((N, N), dtype=complex)
    dmat = np.zeros((N, N), dtype=complex)
    for j in range(N):
        h[j, j] = mu
    n_bonds = N if pbc else N - 1
    for j in range(n_bonds):
        j1 = j
        j2 = (j + 1) % N
        # hopping t=1: - (c_j^dagger c_{j+1} + h.c.)
        h[j1, j2] += -1.0
        h[j2, j1] += -1.0
        # pairing: Delta (c_j c_{j+1} + c_{j+1}^dagger c_j^dagger)
        # antisymmetric pairing matrix: Delta_mat[j,j+1] = -Delta, Delta_mat[j+1,j] = +Delta
        dmat[j1, j2] += -delta
        dmat[j2, j1] += delta
    return h, dmat


def build_fermionic_obc(N: int, mu: float, delta: float) -> np.ndarray:
    """BdG matrix H_BdG for OBC second-quantized Hamiltonian (2N x 2N)."""
    h, dmat = _build_fermionic_blocks(N, mu, delta, pbc=False)
    top = np.block([[h, dmat]])
    bottom = np.block([[-dmat.conj(), -h.T]])
    return np.vstack([top, bottom])


def build_fermionic_pbc(N: int, mu: float, delta: float) -> np.ndarray:
    """BdG matrix H_BdG for PBC second-quantized Hamiltonian (2N x 2N)."""
    h, dmat = _build_fermionic_blocks(N, mu, delta, pbc=True)
    top = np.block([[h, dmat]])
    bottom = np.block([[-dmat.conj(), -h.T]])
    return np.vstack([top, bottom])


def count_majorana_bonds(A: np.ndarray, N: int, pbc: bool) -> int:
    """Count inter-site Majorana bond pairs with nonzero coupling (sanity helper)."""
    # Bond terms involve (2j, 2j+3) or (2j+1, 2j+2) for j = 1..n_bonds
    n_bonds = N if pbc else N - 1
    count = 0
    for j in range(1, n_bonds + 1):
        ia = _majorana_index(2 * j, N)
        ib = _majorana_index(2 * j + 3, N)
        ic = _majorana_index(2 * j + 1, N)
        id_ = _majorana_index(2 * j + 2, N)
        if abs(A[ia, ib]) > 1e-14 or abs(A[ic, id_]) > 1e-14:
            count += 1
    return count
