"""파수 상관 필터링(Wavenumber Correlation Filtering)과 지표(SLFI·DLFI).

처리 흐름 (석사 학위논문 3장, 원본 노트북의 FOURIER2D PIPELINE):

    1. 정규화        normalize()          z-score × 10  → 단위가 다른 두 데이터를 비교 가능하게
    2. 평균 제거      remove_mean()
    3. 크기 맞추기    center_crop_common()  두 격자를 같은 크기로 가운데 정렬
    4. CC(k)         correlation_spectrum() CC(k) = cos(θ_A(k) − θ_B(k))
    5. 지표          slfi_dlfi()
         SLFI = IFFT(A·M⁺) + IFFT(B·M⁺),   M⁺ = 구간 ∩ (CC(k) > 0)   → 기반암처럼 함께 움직이는 성분
         DLFI = IFFT(B·M⁻) − IFFT(A·M⁻),   M⁻ = 구간 ∩ (CC(k) < 0)   → 암염처럼 반대로 움직이는 성분

A 는 자력(DRTP), B 는 중력(상향 연속한 부게 이상 또는 그 미분) 입니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifft2

EPS = 1e-12


def normalize(data: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """z-score 정규화 후 scale 배 (원본 노트북과 동일하게 ×10)."""
    return (data - np.nanmean(data)) / (np.nanstd(data) + EPS) * scale


def remove_mean(data: np.ndarray) -> np.ndarray:
    return data - np.nanmean(data)


def center_crop_common(*arrays: np.ndarray):
    """여러 격자를 가장 작은 공통 크기로 가운데 잘라 맞춘다."""
    ny = min(a.shape[0] for a in arrays)
    nx = min(a.shape[1] for a in arrays)
    out = []
    for a in arrays:
        y0 = (a.shape[0] - ny) // 2
        x0 = (a.shape[1] - nx) // 2
        out.append(a[y0:y0 + ny, x0:x0 + nx])
    return out


def prepare(grid: np.ndarray) -> np.ndarray:
    """정규화 → 평균 제거 → NaN 을 0 으로 (FFT 입력용)."""
    g = remove_mean(normalize(grid))
    return np.nan_to_num(g, nan=0.0)


def correlation_spectrum(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """두 격자의 파수별 상관계수 CC(k) 와 각 스펙트럼을 돌려준다.

    CC(k) = cos(Δθ_k) — 진폭은 보지 않고 위상 차이만 본다.
    +1 이면 같은 방향, −1 이면 반대 방향, 0 이면 무관.
    """
    a, b = center_crop_common(a, b)
    A, B = fft2(a), fft2(b)
    cck = np.cos(np.angle(A) - np.angle(B))
    return cck, A, B


def band_mask(cck: np.ndarray, cc_min: float, cc_max: float) -> np.ndarray:
    lo, hi = sorted((cc_min, cc_max))
    return (cck >= lo) & (cck <= hi)


@dataclass
class LFIResult:
    slfi: np.ndarray
    dlfi: np.ndarray
    combined: np.ndarray
    slfi_mask: np.ndarray
    dlfi_mask: np.ndarray


def slfi_dlfi(A: np.ndarray, B: np.ndarray, cck: np.ndarray,
              cc_min: float = -1.0, cc_max: float = 1.0) -> LFIResult:
    """구간 [cc_min, cc_max] 안에서 양(+)의 상관 성분으로 SLFI, 음(−)의 상관 성분으로 DLFI 를 만든다.

    논문에서 쓴 구간: SLFI 0.5 ≤ CC(k) ≤ 1.0, DLFI −1.0 ≤ CC(k) ≤ −0.5
    → slfi_dlfi(A, B, cck, 0.5, 1.0).slfi,  slfi_dlfi(A, B, cck, -1.0, -0.5).dlfi
    """
    union = band_mask(cck, cc_min, cc_max)
    m_pos = union & (cck > 0)
    m_neg = union & (cck < 0)
    slfi = ifft2(A * m_pos).real + ifft2(B * m_pos).real
    dlfi = ifft2(B * m_neg).real - ifft2(A * m_neg).real
    return LFIResult(slfi, dlfi, slfi + dlfi, m_pos, m_neg)


def filtered(A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크한 스펙트럼을 공간 영역으로 되돌린다."""
    return ifft2(A * mask).real


def retained_power(A: np.ndarray, mask: np.ndarray) -> float:
    """마스크 후 남는 신호 에너지 비율(%)."""
    P = np.abs(A) ** 2
    return float(np.sum(P * mask) / (np.sum(P) + EPS) * 100.0)


def upward_continue(grid: np.ndarray, dx: float, dy: float, height: float) -> np.ndarray:
    """주파수 영역 상향 연속: exp(−|k|·h) 를 곱한다.

    원본 연구에서는 상향 연속한 격자(예: 1,500 m)를 입력으로 받았고, 이 함수는
    합성 예제와 높이 선택(choose_continuation_height) 실험을 위해 같은 연산을 다시 구현한 것입니다.
    """
    ny, nx = grid.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    g = np.nan_to_num(grid - np.nanmean(grid))
    return ifft2(fft2(g) * np.exp(-k * height)).real + np.nanmean(grid)
