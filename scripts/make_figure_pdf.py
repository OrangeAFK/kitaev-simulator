#!/usr/bin/env python3
"""Combine the figures into a single printable PDF (two figures per page)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "kitaev_figures.pdf"

FIGURES = [
    ("fig01_pbc_band_structure.png", "Figure 1 - PBC bulk band structure"),
    ("fig02_pbc_vs_obc_spectrum.png", "Figure 2 - PBC vs OBC spectrum across mu"),
    ("fig03_gap_localization.png", "Figure 3 - Bulk gap and localization length"),
    ("fig04_representation_comparison.png", "Figure 4 - Representation / eigenstate comparison"),
    ("fig05_wavefunction_transition.png", "Figure 5 - OBC wavefunction through the transition"),
    ("fig06_finite_size_splitting.png", "Figure 6 - Finite-size Majorana splitting"),
    ("fig07_delta_dependence.png", "Figure 7 - Pairing-strength dependence"),
]

PAGESIZE = (8.5, 11.0)  # US Letter portrait, inches


def _draw_slot(fig: plt.Figure, slot: int, name: str, title: str) -> None:
    """Place one figure in the top (slot=0) or bottom (slot=1) half of the page."""
    # Vertical layout: title bar + image area for each half-page.
    y0 = 0.52 if slot == 0 else 0.04
    height = 0.42
    fig.text(0.5, y0 + height + 0.02, title, ha="center", va="bottom", fontsize=11)
    ax = fig.add_axes((0.06, y0, 0.88, height))
    ax.imshow(plt.imread(FIGDIR / name))
    ax.axis("off")


def main() -> None:
    missing = [name for name, _ in FIGURES if not (FIGDIR / name).exists()]
    if missing:
        raise SystemExit(
            "Missing figures: " + ", ".join(missing) + "\nRun scripts/generate_figures.py first."
        )

    pairs = [FIGURES[i : i + 2] for i in range(0, len(FIGURES), 2)]
    with PdfPages(OUT) as pdf:
        pdf.infodict()["Title"] = "Numerical Kitaev Chain Study - Figures"

        for page_figs in pairs:
            fig = plt.figure(figsize=PAGESIZE)
            fig.text(
                0.5,
                0.015,
                r"$t=1$ everywhere; transition at $\mu=\pm2$",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )
            for slot, (name, title) in enumerate(page_figs):
                _draw_slot(fig, slot, name, title)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {OUT} ({len(pairs)} pages, {len(FIGURES)} figures)")


if __name__ == "__main__":
    main()
