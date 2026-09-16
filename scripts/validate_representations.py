#!/usr/bin/env python3
"""Validate all SPEC section-17 checklist items. Exit nonzero on failure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagonalization import (  # noqa: E402
    ZERO_TOL,
    analytic_dispersion,
    bloch_energies,
    compute_gap,
    diagonalize_bdg,
    diagonalize_majorana,
    e_min_obc,
    extract_positive_energies,
    extract_positive_energies_bdg,
    majorana_spectrum,
)
from src.hamiltonians import (  # noqa: E402
    build_fermionic_obc,
    build_fermionic_pbc,
    build_majorana_obc,
    build_majorana_pbc,
    count_majorana_bonds,
)
from src.wavefunctions import (  # noqa: E402
    compute_localization_length,
    edge_concentration,
    lowest_subspace_weights_majorana,
)

REPORT: list[str] = []
FAILURES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    REPORT.append(line)
    print(line)
    if not ok:
        FAILURES += 1


def spectrum_agreement(N: int, mu: float, delta: float) -> tuple[float, float]:
    A_o = build_majorana_obc(N, mu, delta)
    H_o = build_fermionic_obc(N, mu, delta)
    A_p = build_majorana_pbc(N, mu, delta)
    H_p = build_fermionic_pbc(N, mu, delta)

    em_o = extract_positive_energies(diagonalize_majorana(A_o)[0])
    ef_o = extract_positive_energies_bdg(diagonalize_bdg(H_o)[0])
    em_p = extract_positive_energies(diagonalize_majorana(A_p)[0])
    ef_p = extract_positive_energies_bdg(diagonalize_bdg(H_p)[0])
    return float(np.max(np.abs(em_o - ef_o))), float(np.max(np.abs(em_p - ef_p)))


def main() -> int:
    # --- Matrix properties ---
    N, mu, delta = 20, 0.5, 0.8
    for name, builder, pbc in [
        ("OBC", build_majorana_obc, False),
        ("PBC", build_majorana_pbc, True),
    ]:
        A = builder(N, mu, delta)
        check(f"{name} A is real", np.isrealobj(A) and np.allclose(A.imag if np.iscomplexobj(A) else 0, 0))
        check(f"{name} A antisymmetric", np.allclose(A + A.T, 0), f"max|A+A.T|={np.max(np.abs(A+A.T)):.2e}")
        iA = 1j * A
        check(f"{name} iA Hermitian", np.allclose(iA, iA.conj().T))
        expected_bonds = N if pbc else N - 1
        bonds = count_majorana_bonds(A, N, pbc)
        check(f"{name} bond count == {expected_bonds}", bonds == expected_bonds, f"got {bonds}")

    # --- Spectrum agreement ---
    for N, delta, mu in [(20, 1.0, 0.0), (20, 0.7, 1.3)]:
        err_o, err_p = spectrum_agreement(N, mu, delta)
        check(
            f"OBC spectrum agreement N={N}, mu={mu}, Delta={delta}",
            err_o < 1e-10,
            f"max|Ec-Eg|={err_o:.3e}",
        )
        check(
            f"PBC spectrum agreement N={N}, mu={mu}, Delta={delta}",
            err_p < 1e-10,
            f"max|Ec-Eg|={err_p:.3e}",
        )

    # --- Special point (μ,Δ)=(0,1): unpaired γ_2, γ_{2N+1} ---
    N = 20
    A = build_majorana_obc(N, 0.0, 1.0)
    evals, evecs = diagonalize_majorana(A)
    n_zero = int(np.sum(np.abs(evals) < 1e-8))
    check("(mu,Delta)=(0,1) OBC has two zero modes", n_zero == 2, f"n_zero={n_zero}")

    # Kernel projector support
    zero_idx = np.where(np.abs(evals) < 1e-8)[0]
    proj = np.sum(np.abs(evecs[:, zero_idx]) ** 2, axis=1)
    # Labels: index 0 -> gamma_2, index 2N-1 -> gamma_{2N+1}
    support = proj.copy()
    support[0] = 0.0
    support[-1] = 0.0
    check(
        "zero-mode support on gamma_2 and gamma_{2N+1} only",
        np.max(support) < 1e-10 and proj[0] > 0.4 and proj[-1] > 0.4,
        f"proj[g2]={proj[0]:.4f}, proj[g2N+1]={proj[-1]:.4f}, max_other={np.max(support):.2e}",
    )

    # Nonzero A entries only between (2j+1, 2j+2)
    bad = False
    for i in range(2 * N):
        for j in range(i + 1, 2 * N):
            if abs(A[i, j]) < 1e-12:
                continue
            a, b = i + 2, j + 2
            # expect odd,even consecutive: (3,4), (5,6), ...
            if not (a % 2 == 1 and b == a + 1):
                bad = True
    check("(mu,Delta)=(0,1) A only pairs gamma_{2j+1} gamma_{2j+2}", not bad)

    # --- Bloch vs analytic ---
    ks = np.linspace(-np.pi, np.pi, 101)
    bloch_err = 0.0
    for k in ks:
        e_num = bloch_energies(float(k), 0.0, 1.0)
        e_an = analytic_dispersion(np.array([k]), 0.0, 1.0)[0]
        bloch_err = max(bloch_err, abs(e_num[1] - e_an), abs(e_num[0] + e_an))
    check("Bloch vs analytic dispersion", bloch_err < 1e-12, f"max err={bloch_err:.3e}")

    # --- Gap closure at μ=±2 ---
    N = 100
    gap_m2 = compute_gap(N, -2.0, 1.0)
    gap_p2 = compute_gap(N, 2.0, 1.0)
    gap_0 = compute_gap(N, 0.0, 1.0)
    gap_3 = compute_gap(N, 3.0, 1.0)
    check("gap closes at mu=-2", gap_m2 < 1e-6, f"gap={gap_m2:.3e}")
    check("gap closes at mu=+2", gap_p2 < 1e-6, f"gap={gap_p2:.3e}")
    check("gap finite at mu=0", gap_0 > 0.5, f"gap={gap_0:.3e}")
    check("gap finite at mu=3", gap_3 > 0.5, f"gap={gap_3:.3e}")

    # --- OBC E_min topological vs trivial ---
    emin_topo = e_min_obc(N, 0.0, 1.0)
    emin_triv = e_min_obc(N, 2.5, 1.0)
    check("OBC E_min tiny for |mu|<2", emin_topo < 1e-6, f"Emin={emin_topo:.3e}")
    check("OBC E_min O(gap) for |mu|>2", emin_triv > 0.1, f"Emin={emin_triv:.3e}")

    # --- PBC no edge localization ---
    site_pbc, _, _ = lowest_subspace_weights_majorana(N, 0.0, 1.0, pbc=True)
    site_obc, _, _ = lowest_subspace_weights_majorana(N, 0.0, 1.0, pbc=False)
    check(
        "PBC lowest state not edge-localized",
        edge_concentration(site_pbc) < 0.3,
        f"edge_frac={edge_concentration(site_pbc):.3f}",
    )
    check(
        "OBC lowest state edge-localized for |mu|<2",
        edge_concentration(site_obc) > 0.8,
        f"edge_frac={edge_concentration(site_obc):.3f}",
    )

    # --- Localization length grows toward transition ---
    xi_0 = compute_localization_length(N, 0.0, 1.0)
    xi_15 = compute_localization_length(N, 1.5, 1.0)
    xi_19 = compute_localization_length(N, 1.9, 1.0)
    xi_triv = compute_localization_length(N, 2.5, 1.0)
    check("xi defined at mu=0", xi_0 is not None, f"xi={xi_0}")
    check("xi defined at mu=1.5", xi_15 is not None, f"xi={xi_15}")
    check(
        "xi grows toward mu->2",
        xi_0 is not None and xi_19 is not None and xi_19 > xi_0,
        f"xi(0)={xi_0}, xi(1.9)={xi_19}",
    )
    check("xi undefined in trivial phase mu=2.5", xi_triv is None, f"xi={xi_triv}")

    # --- Finite-size splitting decays with N ---
    # Use mu near the transition so E_min is resolvable over a modest N window.
    Ns = np.array([10, 12, 14, 16, 18, 20])
    emins = np.array([e_min_obc(int(n), 0.0, 1.0) for n in [20, 40, 60, 80]])
    emins_soft = np.array([e_min_obc(int(n), 1.5, 1.0) for n in Ns])
    decreasing = all(emins_soft[i] > emins_soft[i + 1] for i in range(len(emins_soft) - 1))
    valid = emins_soft > 0
    if np.sum(valid) >= 2:
        slope = np.polyfit(Ns[valid], np.log(emins_soft[valid]), 1)[0]
    else:
        slope = 0.0
    check(
        "E_min(N) decreases (mu=1.5, Delta=1)",
        decreasing and slope < 0,
        f"E={emins_soft}, slope={slope:.3e}",
    )
    check(
        "E_min ~ 0 at exact solvable point for N=20..80",
        np.all(emins < 1e-8),
        f"E={emins}",
    )

    # --- mu grid includes +/-2 ---
    # (n-1) must be divisible by 6 so that step divides 1.0 on [-3, 3].
    mu_grid = np.linspace(-3.0, 3.0, 1003)
    check("mu grid contains -2", np.any(np.isclose(mu_grid, -2.0)))
    check("mu grid contains +2", np.any(np.isclose(mu_grid, 2.0)))

    # --- No configurable t (smoke: builders accept only N,mu,delta) ---
    import inspect

    for fn in [build_majorana_obc, build_majorana_pbc, build_fermionic_obc, build_fermionic_pbc]:
        params = list(inspect.signature(fn).parameters)
        check(f"{fn.__name__} has no t parameter", "t" not in params, f"params={params}")

    # Write report
    report_path = ROOT / "validation_report.txt"
    report_path.write_text("\n".join(REPORT) + f"\n\nFailures: {FAILURES}\n", encoding="utf-8")
    print(f"\nWrote {report_path} ({FAILURES} failures)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
