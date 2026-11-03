import numpy as np
import pytest

import gravmag_wcf as gw


def _synthetic(seed=0, n=128, dx=250.0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * dx
    X, Y = np.meshgrid(x, x[::-1])

    def blob(cx, cy, s):
        return np.exp(-(((X - cx) / s) ** 2 + ((Y - cy) / s) ** 2))

    basement = blob(10e3, 18e3, 6e3) + 0.6 * blob(24e3, 8e3, 4e3)
    salt = blob(20e3, 20e3, 2.5e3)
    grav = 10 * basement - 10 * salt + rng.normal(0, 0.1, X.shape)
    mag = 80 * basement + 60 * salt + rng.normal(0, 1.0, X.shape)
    return grav, mag, basement, salt, dx


def test_cck_range_and_identity():
    a = np.random.default_rng(1).normal(size=(32, 32))
    cck, A, B = gw.correlation_spectrum(a, a)
    assert np.allclose(cck, 1.0)
    cck, _, _ = gw.correlation_spectrum(a, -a)
    assert np.allclose(cck, -1.0)


def test_full_band_reconstructs_inputs():
    a = gw.prepare(np.random.default_rng(2).normal(size=(40, 40)))
    b = gw.prepare(np.random.default_rng(3).normal(size=(40, 40)))
    cck, A, B = gw.correlation_spectrum(a, b)
    full = np.ones_like(cck, dtype=bool)
    assert np.allclose(gw.filtered(A, full), a)
    assert gw.retained_power(A, full) == pytest.approx(100.0)


def test_slfi_recovers_basement():
    grav, mag, basement, salt, dx = _synthetic()
    g = gw.upward_continue(grav, dx, dx, 1500)
    cck, A, B = gw.correlation_spectrum(gw.prepare(mag), gw.prepare(g))
    slfi = gw.slfi_dlfi(A, B, cck, 0.5, 1.0).slfi
    dlfi = gw.slfi_dlfi(A, B, cck, -1.0, -0.5).dlfi
    assert gw.corr_safe(slfi, basement)[0] > 0.9
    assert gw.corr_safe(-dlfi, salt)[0] > 0.3


def test_energy_curves_monotonic():
    grav, mag, *_ = _synthetic()
    cck, A, B = gw.correlation_spectrum(gw.prepare(mag), gw.prepare(grav))
    c = gw.energy_curves(A, B, cck, step=0.25)
    assert np.all(np.diff(c.upper_energy_a) <= 1e-9)   # 하한을 올릴수록 남는 에너지는 줄어든다
    assert np.all(np.diff(c.lower_energy_a) >= -1e-9)


def test_knee_point():
    x = np.linspace(0, 10, 50)
    y = 1 - np.exp(-x)
    assert 5 <= gw.knee_point(x, y) <= 15


def test_best_shift_finds_offset():
    t = np.linspace(0, 6 * np.pi, 200)
    ref = np.sin(t)
    target = np.roll(ref, 5)
    s, r_best, r0 = gw.best_shift(ref, target, max_shift=10)
    assert abs(s) == 5 and r_best > r0


def test_io_roundtrip(tmp_path):
    grid = np.arange(12, dtype=float).reshape(3, 4)
    extent = [0, 300, 0, 200]
    xyz = gw.grid_to_xyz(grid, extent)
    p = tmp_path / "g.dat"
    np.savetxt(p, xyz)
    back, ext = gw.read_grid(str(p))
    assert np.allclose(back, grid) and ext == pytest.approx(extent)
