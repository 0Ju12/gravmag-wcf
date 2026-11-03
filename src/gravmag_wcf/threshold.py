"""임계값을 감으로 정하지 않기 위한 도구.

- energy_curves(): CC(k) 구간의 경계를 옮겨 가며 '남는 신호 에너지(%)'와
  '필터링 후 두 데이터의 상관계수'를 함께 계산한다 (원본 노트북 Step 04: Energy Analysis).
  상관은 높이면서 에너지는 너무 버리지 않는 지점을 고르는 데 쓴다.
  논문: SLFI 구간 0.5~1.0 → 원래 에너지의 약 절반 유지, 상관계수 0.85
- knee_point(): 곡선의 양 끝을 잇는 직선에서 가장 멀리 떨어진 점(꺾이는 점).
  논문에서 상향 연속 높이(1,500 m)를 고른 방법.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.fft import ifft2

from .wcf import EPS, correlation_spectrum, prepare


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class EnergyCurves:
    bounds: np.ndarray
    # 상한을 옮길 때: 구간 [-1, x]  (DLFI 쪽, 음의 상관 성분)
    lower_energy_a: np.ndarray
    lower_energy_b: np.ndarray
    lower_corr: np.ndarray
    # 하한을 옮길 때: 구간 [x, 1]  (SLFI 쪽, 양의 상관 성분)
    upper_energy_a: np.ndarray
    upper_energy_b: np.ndarray
    upper_corr: np.ndarray


def energy_curves(A: np.ndarray, B: np.ndarray, cck: np.ndarray, step: float = 0.1) -> EnergyCurves:
    bounds = np.round(np.arange(-1.0, 1.0 + 1e-9, step), 2)
    PA, PB = np.abs(A) ** 2, np.abs(B) ** 2
    tA, tB = PA.sum() + EPS, PB.sum() + EPS

    def run(make_mask: Callable[[float], np.ndarray]):
        ea, eb, cc = [], [], []
        for x in bounds:
            m = make_mask(x)
            ea.append((PA * m).sum() / tA * 100.0)
            eb.append((PB * m).sum() / tB * 100.0)
            cc.append(_corr(ifft2(A * m).real, ifft2(B * m).real))
        return np.array(ea), np.array(eb), np.array(cc)

    la, lb, lc = run(lambda x: (cck >= -1.0) & (cck <= x))
    ua, ub, uc = run(lambda x: (cck >= x) & (cck <= 1.0))
    return EnergyCurves(bounds, la, lb, lc, ua, ub, uc)


def knee_point(x: Sequence[float], y: Sequence[float]) -> int:
    """곡선 양 끝을 잇는 직선(현)에서 수직 거리가 가장 먼 점의 인덱스."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = p1 - p0
    norm = np.hypot(*d) + EPS
    dist = np.abs(d[0] * (y - p0[1]) - d[1] * (x - p0[0])) / norm
    return int(np.argmax(dist))


def choose_continuation_height(magnetic: np.ndarray, gravity_by_height: dict) -> tuple:
    """높이별로 상향 연속한 중력 격자와 자력 격자의 상관계수를 계산하고 꺾이는 높이를 고른다.

    gravity_by_height: {높이(m): 격자}
    반환: (선택한 높이, 높이 배열, 상관계수 배열)
    """
    heights = np.array(sorted(gravity_by_height))
    corrs = []
    for h in heights:
        a, b = prepare(magnetic), prepare(gravity_by_height[h])
        cck, A, B = correlation_spectrum(a, b)
        corrs.append(_corr(ifft2(A).real, ifft2(B).real))
    corrs = np.array(corrs)
    return float(heights[knee_point(heights, corrs)]), heights, corrs
