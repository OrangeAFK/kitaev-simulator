#!/usr/bin/env python3
"""Generate the seven study figures into figures/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagonalization import (  # noqa: E402
    analytic_dispersion,
    bloch_energies,
    compute_gap,
    e_min_obc,
    majorana_spectrum,
)
from src.wavefunctions import (  # noqa: E402
    compute_localization_length,
    lowest_subspace_weights_bdg,
    lowest_subspace_weights_majorana,
)

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

# (n-1) divisible by 6 so mu = +/-2 are exact grid points on [-3, 3].
MU_GRID = np.linspace(-3.0, 3.0, 1003)
assert np.any(np.isclose(MU_GRID, -2.0)) and np.any(np.isclose(MU_GRID, 2.0))

DPI = 150


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGDIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig01_pbc_band_structure() -> None:
    mu, delta = 0.0, 1.0
    k = np.linspace(-np.pi, np.pi, 401)
    e_plus = np.array([bloch_energies(float(ki), mu, delta)[1] for ki in k])
    e_minus = -e_plus
    e_an = analytic_dispersion(k, mu, delta)
    err = float(np.max(np.abs(e_plus - e_an)))
    print(f"Fig1 Bloch vs analytic max error: {err:.3e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k, e_plus, "b-", lw=1.5, label=r"$E_+(k)$ numerical")
    ax.plot(k, e_minus, "b-", lw=1.5, label=r"$E_-(k)$ numerical")
    ax.plot(k, e_an, "k--", lw=1.0, alpha=0.8, label="analytic")
    ax.plot(k, -e_an, "k--", lw=1.0, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(rf"PBC bulk bands ($\mu={mu}$, $\Delta={delta}$, $t=1$)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-np.pi, np.pi)
    _save(fig, "fig01_pbc_band_structure.png")


def fig02_pbc_vs_obc_spectrum() -> None:
    N, delta = 100, 1.0
    # Subsample for plotting speed while keeping dense near critical points
    mus = MU_GRID
    # Precompute spectra (positive energies)
    pbc_spec = []
    obc_spec = []
    for mu in mus:
        pbc_spec.append(majorana_spectrum(N, float(mu), delta, pbc=True))
        obc_spec.append(majorana_spectrum(N, float(mu), delta, pbc=False))
    pbc_spec = np.asarray(pbc_spec)
    obc_spec = np.asarray(obc_spec)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, spec, title in zip(
        axes, [pbc_spec, obc_spec], ["PBC", "OBC"], strict=True
    ):
        # Full particle-hole spectrum: +/- E_n (and zero when E_min ~ 0)
        for n in range(spec.shape[1]):
            ax.plot(mus, spec[:, n], color="C0", lw=0.4, alpha=0.7)
            ax.plot(mus, -spec[:, n], color="C0", lw=0.4, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(-2, color="r", ls="--", lw=0.8)
        ax.axvline(2, color="r", ls="--", lw=0.8)
        ax.set_xlabel(r"$\mu$")
        ax.set_title(rf"{title} spectrum ($N={N}$, $\Delta={delta}$)")
        ax.set_xlim(-3, 3)
    axes[0].set_ylabel(r"$E_n$")
    _save(fig, "fig02_pbc_vs_obc_spectrum.png")


def fig03_gap_localization() -> None:
    N, delta = 100, 1.0
    gaps = np.array([compute_gap(N, float(mu), delta) for mu in MU_GRID])

    # Localization length on a coarser grid inside the topological phase
    mu_xi = np.linspace(-1.99, 1.99, 81)
    xis = []
    mu_xi_ok = []
    for mu in mu_xi:
        xi = compute_localization_length(N, float(mu), delta)
        if xi is not None:
            xis.append(xi)
            mu_xi_ok.append(mu)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(MU_GRID, gaps, "b-", lw=1.2)
    axes[0].axvline(-2, color="r", ls="--", lw=0.8)
    axes[0].axvline(2, color="r", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\mu$")
    axes[0].set_ylabel(r"$E_{\mathrm{gap}}$")
    axes[0].set_title(rf"PBC bulk gap ($N={N}$, $\Delta={delta}$)")
    axes[0].set_xlim(-3, 3)

    axes[1].plot(mu_xi_ok, xis, "o-", ms=3, lw=1.0)
    axes[1].axvline(-2, color="r", ls="--", lw=0.8)
    axes[1].axvline(2, color="r", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\mu$")
    axes[1].set_ylabel(r"$\xi$")
    axes[1].set_title(r"OBC localization length (fit $\ln w_j=c-2j/\xi$)")
    axes[1].set_xlim(-3, 3)
    _save(fig, "fig03_gap_localization.png")


def fig04_representation_comparison() -> None:
    N, mu, delta = 100, 0.0, 1.0
    sw_f_obc, e_f_obc = lowest_subspace_weights_bdg(N, mu, delta, pbc=False)
    sw_f_pbc, e_f_pbc = lowest_subspace_weights_bdg(N, mu, delta, pbc=True)
    sw_m_obc, _, e_m_obc = lowest_subspace_weights_majorana(N, mu, delta, pbc=False)
    sw_m_pbc, _, e_m_pbc = lowest_subspace_weights_majorana(N, mu, delta, pbc=True)

    # Energy agreement
    from src.diagonalization import (
        diagonalize_bdg,
        diagonalize_majorana,
        extract_positive_energies,
        extract_positive_energies_bdg,
    )
    from src.hamiltonians import (
        build_fermionic_obc,
        build_fermionic_pbc,
        build_majorana_obc,
        build_majorana_pbc,
    )

    def agree(pbc: bool) -> float:
        if pbc:
            em = extract_positive_energies(diagonalize_majorana(build_majorana_pbc(N, mu, delta))[0])
            ef = extract_positive_energies_bdg(diagonalize_bdg(build_fermionic_pbc(N, mu, delta))[0])
        else:
            em = extract_positive_energies(diagonalize_majorana(build_majorana_obc(N, mu, delta))[0])
            ef = extract_positive_energies_bdg(diagonalize_bdg(build_fermionic_obc(N, mu, delta))[0])
        return float(np.max(np.abs(em - ef)))

    err_o, err_p = agree(False), agree(True)
    msg = f"Fig4 energy agreement: OBC max|Ec-Eg|={err_o:.3e}, PBC max|Ec-Eg|={err_p:.3e}"
    print(msg)

    sites = np.arange(1, N + 1)
    panels = [
        (sw_f_obc, "Fermionic OBC", e_f_obc),
        (sw_f_pbc, "Fermionic PBC", e_f_pbc),
        (sw_m_obc, "Majorana OBC", e_m_obc),
        (sw_m_pbc, "Majorana PBC", e_m_pbc),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    ymax = max(w.max() for w, _, _ in panels) * 1.1
    for ax, (w, title, eref) in zip(axes.ravel(), panels, strict=True):
        ax.plot(sites, w, "b-", lw=1.0)
        ax.set_title(rf"{title} ($|E|\approx{eref:.2e}$)")
        ax.set_ylim(0, ymax)
        ax.set_xlim(1, N)
    axes[1, 0].set_xlabel(r"site $j$")
    axes[1, 1].set_xlabel(r"site $j$")
    axes[0, 0].set_ylabel(r"$|\psi_j|^2$")
    axes[1, 0].set_ylabel(r"$|\psi_j|^2$")
    fig.suptitle(
        rf"$N={N}$, $\mu={mu}$, $\Delta={delta}$; "
        rf"max$|E_c-E_\gamma|$ OBC={err_o:.1e}, PBC={err_p:.1e}"
    )
    _save(fig, "fig04_representation_comparison.png")


def fig05_wavefunction_transition() -> None:
    N, delta = 100, 1.0
    mus = [0.0, 1.5, 1.9, 2.0, 2.1, 2.5]
    sites = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    ymax = 0.0
    weights = []
    for mu in mus:
        sw, _, _ = lowest_subspace_weights_majorana(N, float(mu), delta, pbc=False)
        weights.append(sw)
        ymax = max(ymax, sw.max())

    for ax, mu, sw in zip(axes.ravel(), mus, weights, strict=True):
        ax.plot(sites, sw, "b-", lw=1.0)
        if mu < 2.0:
            title = rf"$\mu={mu}$ (MZM)"
        elif mu == 2.0:
            title = rf"$\mu={mu}$ (critical)"
        else:
            title = rf"$\mu={mu}$ (lowest state, not an MZM)"
        ax.set_title(title, fontsize=9)
        ax.set_xlim(1, N)
        ax.set_ylim(0, ymax * 1.1)

    for ax in axes[1, :]:
        ax.set_xlabel(r"site $j$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\psi_j|^2$")
    fig.suptitle(rf"OBC lowest-state weight through the transition ($N={N}$, $\Delta={delta}$)")
    _save(fig, "fig05_wavefunction_transition.png")


def fig06_finite_size_splitting() -> None:
    delta = 1.0
    # Panel A: E_min vs N at mu=0 (exact zeros) — use mu=1.5 for visible exponential
    # SPEC says mu=0, Delta=1. At that point E_min is exactly 0.
    # Plot both: exact point (numerically ~0) and note; also show mu=0 which floors.
    # Follow SPEC: mu=0, Delta=1. Values will sit at numerical zero / machine floor.
    # For a meaningful exponential, SPEC expects E_min ~ e^{-N/xi}. At (0,1), xi=0
    # so splitting is exactly zero. We still plot as specified, and overlay mu=1.0
    # only if needed... Stick to SPEC: mu=0, Delta=1.
    Ns = np.arange(10, 151, 10)
    emin_a = np.array([e_min_obc(int(n), 0.0, delta) for n in Ns])
    # Also compute at mu=1.0 for a clear exponential guide (dashed), since (0,1) is exactly zero.
    emin_guide = np.array([e_min_obc(int(n), 1.0, delta) for n in Ns])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Floor tiny values for log plot visibility
    floor = 1e-16
    axes[0].semilogy(Ns, np.maximum(emin_a, floor), "o-", label=r"$\mu=0$ (exact MZM)")
    axes[0].semilogy(Ns, np.maximum(emin_guide, floor), "s--", label=r"$\mu=1$ (guide)")
    # Exponential fit on guide
    mask = emin_guide > 1e-14
    if np.sum(mask) >= 2:
        slope, intercept = np.polyfit(Ns[mask], np.log(emin_guide[mask]), 1)
        Ns_fit = np.linspace(Ns[mask].min(), Ns[mask].max(), 100)
        axes[0].semilogy(
            Ns_fit,
            np.exp(intercept + slope * Ns_fit),
            "k:",
            label=rf"fit $\sim e^{{-N/\xi}}$, $\xi={-1/slope:.2f}$",
        )
    axes[0].set_xlabel(r"$N$")
    axes[0].set_ylabel(r"$E_{\min}$")
    axes[0].set_title(r"Finite-size splitting vs $N$ ($\Delta=1$)")
    axes[0].set_ylim(1e-18, 1e0)
    axes[0].legend(fontsize=8)

    # Panel B
    for N, style in [(20, "-"), (50, "--"), (100, "-.")]:
        emins = np.array([e_min_obc(N, float(mu), delta) for mu in MU_GRID])
        axes[1].semilogy(MU_GRID, np.maximum(emins, floor), style, label=rf"$N={N}$")
    axes[1].axvline(-2, color="r", ls=":", lw=0.8)
    axes[1].axvline(2, color="r", ls=":", lw=0.8)
    axes[1].set_xlabel(r"$\mu$")
    axes[1].set_ylabel(r"$E_{\min}$")
    axes[1].set_title(r"$E_{\min}(\mu)$ for several $N$ ($\Delta=1$)")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(-3, 3)
    _save(fig, "fig06_finite_size_splitting.png")


def fig07_delta_dependence() -> None:
    N, mu = 100, 0.0
    deltas = np.linspace(0.0, 2.0, 81)

    # Panel A: PBC positive spectrum vs Delta
    specs = np.array([majorana_spectrum(N, mu, float(d), pbc=True) for d in deltas])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n in range(specs.shape[1]):
        axes[0].plot(deltas, specs[:, n], color="C0", lw=0.4, alpha=0.7)
    axes[0].axvline(0.0, color="gray", ls="--", lw=0.8, label=r"$\Delta=0$ (nonsuperconducting)")
    axes[0].axvline(1.0, color="r", ls="--", lw=0.8, label=r"$\Delta=1$")
    axes[0].set_xlabel(r"$\Delta$")
    axes[0].set_ylabel(r"$E_n$")
    axes[0].set_title(rf"PBC spectrum vs $\Delta$ ($\mu={mu}$, $N={N}$)")
    axes[0].legend(fontsize=8)

    # Panel B: xi(Delta) for OBC
    d_xi = np.linspace(0.05, 2.0, 40)
    xis, d_ok = [], []
    for d in d_xi:
        xi = compute_localization_length(N, mu, float(d))
        if xi is not None:
            xis.append(xi)
            d_ok.append(d)
    axes[1].plot(d_ok, xis, "o-", ms=3)
    axes[1].axvline(0.0, color="gray", ls="--", lw=0.8)
    axes[1].annotate(
        r"$\Delta=0$: no SC pairing / no MZM",
        xy=(0.05, max(xis) * 0.9 if xis else 1),
        fontsize=8,
    )
    axes[1].set_xlabel(r"$\Delta$")
    axes[1].set_ylabel(r"$\xi$")
    axes[1].set_title(rf"OBC localization length vs $\Delta$ ($\mu={mu}$)")
    axes[1].set_xlim(0, 2)
    _save(fig, "fig07_delta_dependence.png")


def main() -> None:
    fig01_pbc_band_structure()
    fig02_pbc_vs_obc_spectrum()
    fig03_gap_localization()
    fig04_representation_comparison()
    fig05_wavefunction_transition()
    fig06_finite_size_splitting()
    fig07_delta_dependence()
    print("All figures generated.")


if __name__ == "__main__":
    main()
