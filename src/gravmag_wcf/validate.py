"""결과 검증: 독립적으로 만든 기존 모델과 비교 (원본 Validation Code v1.0 정리).

- corr_safe(): Pearson·Spearman 상관계수 (값이 부족하거나 상수면 NaN)
- polygon_mask(): BLN 경계선 안쪽만 비교할 때 쓰는 마스크
- resample_to_grid(): 다른 격자(예: 3D 밀도 모델의 깊이 단면)를 지표 격자 위로 보간
- depthwise_correlation(): 3D 모델의 깊이별 단면과 지표 지도의 상관을 깊이마다 계산
- sample_profile(): 지도에서 측선을 따라 값을 뽑는다
- best_shift(): 측선 두 개를 서로 밀어 가며 상관이 최대가 되는 이동량을 찾는다
  (논문: 측선 비교 상관 0.87~0.95 → 위치 보정 후 0.95~0.98)
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr, spearmanr


def corr_safe(a: np.ndarray, b: np.ndarray):
    """(pearson_r, pearson_p, spearman_rho, spearman_p, n)"""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    if n < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, n
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    return float(pr), float(pp), float(sr), float(sp), n


def grid_axes(extent: Sequence[float], shape: Tuple[int, int]):
    """격자 좌표축. y 는 보간용 오름차순으로 돌려준다."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = shape
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)


def polygon_mask(extent, shape, poly_x, poly_y) -> np.ndarray:
    """origin="upper" 격자에서 폴리곤 안쪽 True."""
    x, y_asc = grid_axes(extent, shape)
    xx, yy = np.meshgrid(x, y_asc[::-1])
    inside = Path(np.column_stack([poly_x, poly_y])).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    return inside.reshape(shape)


def resample_to_grid(src: np.ndarray, src_x: np.ndarray, src_y_asc: np.ndarray,
                     extent, shape) -> np.ndarray:
    """src(행=y 오름차순) 를 extent/shape 격자(origin="upper") 위로 선형 보간."""
    interp = RegularGridInterpolator((src_y_asc, src_x), src, bounds_error=False, fill_value=np.nan)
    x, y_asc = grid_axes(extent, shape)
    xx, yy = np.meshgrid(x, y_asc[::-1])
    return interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(shape)


def depthwise_correlation(index_map: np.ndarray, extent, volume: np.ndarray, depths: Iterable[float],
                          vol_x: np.ndarray, vol_y_asc: np.ndarray, mask: np.ndarray | None = None):
    """volume[z, y, x] 의 깊이별 단면과 index_map 의 상관을 계산.

    반환: depths, pearson, spearman, n  (모두 배열)
    """
    depths = np.asarray(list(depths), float)
    pr, sr, ns = [], [], []
    for iz in range(len(depths)):
        sl = resample_to_grid(volume[iz], vol_x, vol_y_asc, extent, index_map.shape)
        a, b = index_map, sl
        if mask is not None:
            a, b = a[mask], b[mask]
        r, _, s, _, n = corr_safe(a, b)
        pr.append(r)
        sr.append(s)
        ns.append(n)
    return depths, np.array(pr), np.array(sr), np.array(ns)


def sample_profile(grid: np.ndarray, extent, p0, p1, n: int = 300):
    """p0→p1 직선 측선을 따라 n 개 점을 뽑는다. 반환: 거리(m), 값"""
    x_axis, y_asc = grid_axes(extent, grid.shape)
    interp = RegularGridInterpolator((y_asc, x_axis), grid[::-1], bounds_error=False, fill_value=np.nan)
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return dist, interp(np.column_stack([ys, xs]))


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.zeros_like(v) if np.isclose(lo, hi) else (v - lo) / (hi - lo)


def best_shift(ref: np.ndarray, target: np.ndarray, max_shift: int):
    """target 을 −max_shift..+max_shift 칸 밀어 가며 ref 와의 Pearson 상관이 최대인 이동량.

    반환: (최적 이동 칸 수, 최대 상관, 이동 없이 잰 상관)
    """
    ref = np.asarray(ref, float)
    target = np.asarray(target, float)
    base = corr_safe(ref, target)[0]
    best = (0, base)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b = ref[s:], target[:len(target) - s]
        else:
            a, b = ref[:s], target[-s:]
        r = corr_safe(a, b)[0]
        if np.isfinite(r) and (not np.isfinite(best[1]) or r > best[1]):
            best = (s, r)
    return best[0], best[1], base
