"""지도·곡선 그리기 (원본 노트북의 show_step12_with_contour, energy 그래프 정리)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def map_with_contours(ax, grid, extent, title="", cmap="turbo", nlevels=15, vmin=None, vmax=None,
                      boundary=None, unit=""):
    """컬러맵 + 등고선(%.1f). boundary=(x, y) 를 주면 흰 선으로 겹친다."""
    im = ax.imshow(grid, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    finite = grid[np.isfinite(grid)]
    if finite.size and not np.isclose(finite.min(), finite.max()):
        ny, nx = grid.shape
        xx, yy = np.meshgrid(np.linspace(extent[0], extent[1], nx), np.linspace(extent[3], extent[2], ny))
        cs = ax.contour(xx, yy, grid, levels=np.linspace(finite.min(), finite.max(), nlevels),
                        colors="k", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    if boundary is not None:
        ax.plot(boundary[0], boundary[1], color="white", lw=2)
    ax.set_title(title)
    ax.ticklabel_format(style="plain", useOffset=False)
    cb = plt.colorbar(im, ax=ax, shrink=0.9)
    if unit:
        cb.set_label(unit)
    return im


def energy_plot(curves, side="upper"):
    """임계값 선택용: 남는 에너지(%) 와 상관계수를 경계값에 따라 그린다."""
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if side == "upper":
        ea, eb, cc, label = curves.upper_energy_a, curves.upper_energy_b, curves.upper_corr, "lower bound x  (band [x, 1])"
    else:
        ea, eb, cc, label = curves.lower_energy_a, curves.lower_energy_b, curves.lower_corr, "upper bound x  (band [-1, x])"
    ax1.plot(curves.bounds, ea, "-o", ms=3, label="retained power A (%)")
    ax1.plot(curves.bounds, eb, "-s", ms=3, label="retained power B (%)")
    ax1.set_xlabel(label)
    ax1.set_ylabel("retained power (%)")
    ax2 = ax1.twinx()
    ax2.plot(curves.bounds, cc, "k--", label="correlation after filtering")
    ax2.set_ylabel("correlation")
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig
