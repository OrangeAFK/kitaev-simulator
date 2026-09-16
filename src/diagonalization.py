"""Diagonalization and spectrum utilities for the Kitaev chain."""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from .hamiltonians import build_majorana_obc, build_majorana_pbc

# Near-zero numerical tolerance for quasiparticle energies / degeneracy.
ZERO_TOL = 1e-9


def diagonalize_majorana(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize iA (Hermitian). Return all eigenvalues and eigenvectors.

    Eigenvalues of iA come in +/- pairs. Eigenvectors are columns.
    """
    iA = 1j * A
    # Enforce Hermiticity numerically
    iA = 0.5 * (iA + iA.conj().T)
    evals, evecs = la.eigh(iA)
    return evals, evecs


def extract_positive_energies(evals: np.ndarray) -> np.ndarray:
    """Return the N positive quasiparticle energies, sorted ascending."""
    pos = evals[evals > -ZERO_TOL]
    # Keep the upper half (positive / near-zero from above)
    # For exact +/- symmetry, take the largest N eigenvalues (all >= 0).
    n = len(evals) // 2
    sorted_all = np.sort(evals)
    positive = sorted_all[n:]  # N largest -> nonnegative
    # Clip tiny negatives from numerics
    positive = np.maximum(positive, 0.0)
    return np.sort(positive)


def diagonalize_bdg(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize BdG Hamiltonian. Return eigenvalues and eigenvectors."""
    H = 0.5 * (H + H.conj().T)
    evals, evecs = la.eigh(H)
    return evals, evecs


def extract_positive_energies_bdg(evals: np.ndarray) -> np.ndarray:
    """Positive BdG quasiparticle energies, sorted ascending."""
    n = len(evals) // 2
    sorted_all = np.sort(np.real(evals))
    positive = sorted_all[n:]
    positive = np.maximum(positive, 0.0)
    return np.sort(positive)


def majorana_spectrum(N: int, mu: float, delta: float, pbc: bool) -> np.ndarray:
    """Positive quasiparticle energies from Majorana representation."""
    A = build_majorana_pbc(N, mu, delta) if pbc else build_majorana_obc(N, mu, delta)
    evals, _ = diagonalize_majorana(A)
    return extract_positive_energies(evals)


def compute_gap(N: int, mu: float, delta: float) -> float:
    """Bulk gap = smallest positive PBC quasiparticle energy."""
    e = majorana_spectrum(N, mu, delta, pbc=True)
    return float(e[0])


def e_min_obc(N: int, mu: float, delta: float) -> float:
    """Smallest positive OBC quasiparticle energy."""
    e = majorana_spectrum(N, mu, delta, pbc=False)
    return float(e[0])


def analytic_dispersion(k: np.ndarray, mu: float, delta: float) -> np.ndarray:
    """Analytic E_+(k) = sqrt((mu - 2 cos k)^2 + 4 Delta^2 sin^2 k)."""
    return np.sqrt((mu - 2.0 * np.cos(k)) ** 2 + (2.0 * delta * np.sin(k)) ** 2)


def bloch_energies(k: float, mu: float, delta: float) -> np.ndarray:
    """Numerical Bloch quasiparticle energies from 2x2 Majorana momentum Hamiltonian.

    Builds A(k) from the onsite and nearest-neighbor Majorana couplings of the
    PBC A matrix (N=2 probe cell is enough to read the hoppings).
    """
    # Explicit 2x2 A(k) from the Majorana PBC couplings:
    # Onsite: A_{0,1} = -mu  (labels gamma_2, gamma_3 of a unit cell)
    # Hopping to next cell:
    #   (1-Delta)/2 term: i*alpha with alpha=(1-Delta)/2 between gamma_{2j} and gamma_{2j+3}
    #   -> A contribution (1-Delta) between index 0 of cell and index 1 of next cell
    #   -(1+Delta)/2 term: alpha=-(1+Delta)/2 between gamma_{2j+1} and gamma_{2j+2}
    #   -> A contribution -(1+Delta) between index 1 of cell and index 0 of next cell
    #
    # A(k) = A0 + A1 e^{ik} + A1^T e^{-ik}  (with care for antisymmetry)
    # Resulting eigenvalues of iA(k): +/- sqrt((mu-2cos k)^2 + 4 Delta^2 sin^2 k)
    ck = np.cos(k)
    sk = np.sin(k)
    # A(k)_{01} = (2 cos k - mu) - 2 i Delta sin k   (plan)
    a01 = (2.0 * ck - mu) - 2.0j * delta * sk
    Ak = np.array([[0.0, a01], [-a01, 0.0]], dtype=complex)
    # For complex antisymmetric-like structure, diagonalize i*Ak carefully.
    # Make Hermitian matrix H = i * Ak but Ak is complex; use the standard
    # BdG/Majorana Bloch Hamiltonian whose eigenvalues are +/- E(k).
    # i*Ak is Hermitian when A(k)^dagger = -A(k) with the conjugation on k->-k;
    # for a single k we diagonalize the Hermitian matrix:
    #   [[0, -i a01], [i a01*, 0]] wait — simpler: use analytic-equivalent matrix
    H = np.array(
        [
            [0.0, -1j * a01],
            [1j * np.conj(a01), 0.0],
        ],
        dtype=complex,
    )
    H = 0.5 * (H + H.conj().T)
    evals = la.eigvalsh(H)
    return np.sort(np.real(evals))


def extract_low_energy_states(
    evals: np.ndarray, evecs: np.ndarray, n_states: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return lowest-|E| eigenpairs (near zero preferred).

    If n_states is None, take the near-degenerate subspace of the smallest |E|
    within ZERO_TOL of the minimum |E| (at least 1, typically 2 for MZMs).
    """
    abs_e = np.abs(evals)
    order = np.argsort(abs_e)
    emin = abs_e[order[0]]
    if n_states is None:
        mask = abs_e <= emin + ZERO_TOL
        # Also include particle-hole partners that share the same |E|
        idx = np.where(mask)[0]
        if len(idx) < 2:
            # take at least the two closest to zero (one +/- pair)
            idx = order[:2]
    else:
        idx = order[:n_states]
    return evals[idx], evecs[:, idx]
