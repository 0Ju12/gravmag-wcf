"""합성 데이터로 전체 흐름을 재현하는 예제 (실제 연구 데이터는 포함하지 않음).

가상의 지하 구조
  - 기반암 융기(basement high): 무겁고 자성이 있음 → 중력 +, 자력 +   (같은 방향)
  - 암염 돔(salt dome):        가볍고 자성이 없음 → 중력 −, 자력은 주변 대비 상대적으로 + (반대 방향)

실행:  python examples/synthetic_demo.py   →  examples/output/ 에 그림 저장
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gravmag_wcf as gw

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# ---- 1. 합성 격자 만들기 (50 km × 50 km, 250 m 간격) ----
n, dx = 200, 250.0
x = np.arange(n) * dx
y = np.arange(n) * dx
X, Y = np.meshgrid(x, y[::-1])
extent = [x.min(), x.max(), y.min(), y.max()]

def blob(cx, cy, sx, sy):
    return np.exp(-(((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))

basement = 1.0 * blob(15e3, 30e3, 9e3, 14e3) + 0.6 * blob(38e3, 12e3, 7e3, 6e3)
salt = blob(30e3, 32e3, 4e3, 4e3)
regional = 0.3 * (X / X.max())

gravity = 10 * basement - 10 * salt + 2 * regional + rng.normal(0, 0.15, X.shape)
magnetic = 80 * basement + 60 * salt + rng.normal(0, 1.5, X.shape)

# ---- 2. 상향 연속 높이 고르기 (꺾이는 점) ----
heights = [0, 250, 500, 750, 1000, 1500, 2000, 3000]
grav_by_h = {h: gw.upward_continue(gravity, dx, dx, h) for h in heights}
h_sel, hs, cs = gw.choose_continuation_height(magnetic, grav_by_h)
print(f"선택한 상향 연속 높이: {h_sel:.0f} m")
g_up = grav_by_h[h_sel]

# ---- 3. CC(k) ----
A_in, B_in = gw.prepare(magnetic), gw.prepare(g_up)
cck, A, B = gw.correlation_spectrum(A_in, B_in)

# ---- 4. 임계값 근거 곡선 ----
curves = gw.energy_curves(A, B, cck, step=0.1)
i05 = int(np.argmin(np.abs(curves.bounds - 0.5)))
print(f"SLFI 구간 [0.5, 1.0]: 남는 에너지 A={curves.upper_energy_a[i05]:.1f}% "
      f"B={curves.upper_energy_b[i05]:.1f}%, 필터 후 상관={curves.upper_corr[i05]:.2f}")
gw.plotting.energy_plot(curves, "upper").savefig(OUT / "energy_slfi.png", dpi=120)

# ---- 5. SLFI · DLFI ----
slfi = gw.slfi_dlfi(A, B, cck, 0.5, 1.0).slfi
dlfi = gw.slfi_dlfi(A, B, cck, -1.0, -0.5).dlfi

# ---- 6. 검증: 정답 구조와 비교 ----
r_b = gw.corr_safe(slfi, basement)[0]
r_s = gw.corr_safe(-dlfi, salt)[0]  # DLFI 가 낮은 곳 = 암염 (논문 Fig. 11b 와 같은 방향)
print(f"SLFI vs 기반암 정답 r = {r_b:.3f}")
print(f"−DLFI vs 암염 정답  r = {r_s:.3f}")

d, prof_slfi = gw.sample_profile(slfi, extent, (2e3, 30e3), (48e3, 30e3))
_, prof_true = gw.sample_profile(basement, extent, (2e3, 30e3), (48e3, 30e3))
shift, r_best, r0 = gw.best_shift(prof_true, prof_slfi, max_shift=10)
print(f"측선 비교 r = {r0:.3f} → 위치 보정({shift}칸) 후 {r_best:.3f}")

# ---- 그림 ----
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
gw.plotting.map_with_contours(axs[0, 0], gravity, extent, "Synthetic gravity (raw)")
gw.plotting.map_with_contours(axs[0, 1], magnetic, extent, "Synthetic magnetic")
axs[0, 2].imshow(np.fft.fftshift(cck), cmap="seismic", vmin=-1, vmax=1)
axs[0, 2].set_title("CC(k), centered (fftshift)")
gw.plotting.map_with_contours(axs[1, 0], slfi, extent, f"SLFI  (r={r_b:.2f} vs true basement)")
gw.plotting.map_with_contours(axs[1, 1], dlfi, extent, f"DLFI  (low = salt, r={r_s:.2f})")
axs[1, 2].plot(d / 1e3, gw.validate.minmax(prof_true), label="true basement")
axs[1, 2].plot(d / 1e3, gw.validate.minmax(prof_slfi), label="SLFI")
axs[1, 2].set_xlabel("distance (km)")
axs[1, 2].legend()
axs[1, 2].set_title("Profile comparison (min-max)")
fig.tight_layout()
fig.savefig(OUT / "synthetic_overview.png", dpi=110)
print(f"그림 저장: {OUT}")
