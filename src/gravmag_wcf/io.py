"""격자 데이터 입출력.

원본 노트북(Correlation Code v5.1)의 읽기 함수를 정리한 것입니다.
- ASC: 첫 줄 헤더에 xmin xmax ymin ymax ... ncols nrows 가 들어 있는 격자 파일
- XYZ(.dat/.txt/.xyz): x y z 세 열로 된 규칙 격자 점 목록
- BLN: Surfer 경계선 파일 (첫 줄 점 개수, 이후 x y)

모든 격자는 (ny, nx) 배열과 extent=[xmin, xmax, ymin, ymax] 로 돌려줍니다.
행 0 이 북쪽(ymax)인 imshow(origin="upper") 기준입니다.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

Extent = List[float]


def read_asc_grid(path: str) -> Tuple[np.ndarray, Extent]:
    """헤더 한 줄 + 값 나열 형식의 ASC 격자를 읽는다."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) < 8:
            raise ValueError(f"ASC 헤더 값이 8개보다 적습니다: {len(header)}")
        xmin, xmax, ymin, ymax = map(float, header[:4])
        ncols, nrows = int(header[6]), int(header[7])
        values = np.array([float(v) for line in f for v in line.split()])

    expected = nrows * ncols
    if values.size > expected:
        values = values[:expected]
    elif values.size < expected:  # 모자라면 NaN 으로 채움
        values = np.concatenate([values, np.full(expected - values.size, np.nan)])
    return values.reshape(nrows, ncols), [xmin, xmax, ymin, ymax]


def read_xyz_grid(path: str) -> Tuple[np.ndarray, Extent]:
    """x y z 점 목록을 규칙 격자로 바꾼다. 빠진 점이 있으면 오류."""
    xyz = None
    for delim in (None, ",", "\t", ";"):
        try:
            xyz = np.loadtxt(path, delimiter=delim)
            break
        except ValueError:
            continue
    if xyz is None or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"x y z 3열 형식이 아닙니다: {path}")

    x, y, z = xyz.T
    xu, yu = np.unique(x), np.unique(y)
    if z.size != xu.size * yu.size:
        raise ValueError(f"규칙 격자가 아닙니다: {z.size} != {xu.size}x{yu.size}")
    order = np.lexsort((x, -y))  # 북쪽 행부터, 행 안에서는 서→동
    return z[order].reshape(yu.size, xu.size), [xu.min(), xu.max(), yu.min(), yu.max()]


def read_grid(path: str) -> Tuple[np.ndarray, Extent]:
    """확장자로 형식을 골라 읽는다."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".asc":
        return read_asc_grid(path)
    if ext in (".dat", ".txt", ".xyz"):
        return read_xyz_grid(path)
    try:
        return read_xyz_grid(path)
    except ValueError:
        return read_asc_grid(path)


def crop_grid(grid: np.ndarray, extent: Sequence[float], crop: Sequence[float]) -> Tuple[np.ndarray, Extent]:
    """crop=(xmin, xmax, ymin, ymax) 범위만 잘라낸다."""
    xmin, xmax, ymin, ymax = extent
    cx0, cx1, cy0, cy1 = crop
    if cx0 < xmin or cx1 > xmax or cy0 < ymin or cy1 > ymax:
        raise ValueError("자를 범위가 데이터 범위를 벗어납니다")

    ny, nx = grid.shape
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymax, ymin, ny)  # origin="upper"
    xi = np.where((xs >= cx0) & (xs <= cx1))[0]
    yi = np.where((ys >= cy0) & (ys <= cy1))[0]
    if xi.size == 0 or yi.size == 0:
        raise ValueError("자른 범위에 격자점이 없습니다")

    out = grid[yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1]
    return out, [xs[xi[0]], xs[xi[-1]], ys[yi[-1]], ys[yi[0]]]


def read_bln(path: str, close: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Surfer BLN 경계선을 읽는다 (첫 번째 폴리곤만)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0].replace(",", " ").split()[0])
    pts = np.array([[float(v) for v in ln.replace(",", " ").split()[:2]] for ln in lines[1:1 + n]])
    xb, yb = pts[:, 0], pts[:, 1]
    if close and (xb[0] != xb[-1] or yb[0] != yb[-1]):
        xb, yb = np.r_[xb, xb[0]], np.r_[yb, yb[0]]
    return xb, yb


def grid_to_xyz(grid: np.ndarray, extent: Sequence[float]) -> np.ndarray:
    """격자를 (N, 3) x y z 배열로 펼친다. 엑셀·CSV 저장용."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = grid.shape
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymax, ymin, ny))
    return np.column_stack([xx.ravel(), yy.ravel(), grid.ravel()])
