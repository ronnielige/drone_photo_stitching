"""WGS84 <-> UTM 横轴墨卡托投影（纯 numpy，无外部依赖），与 EPSG:32650 一致。"""
import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2 - F)
K0 = 0.9996
DEG = np.pi / 180.0


def _series(k, e2):
    e4, e6, e8 = e2 ** 2, e2 ** 3, e2 ** 4
    return (1.0
            + e2 / 4 + 3 * e2 ** 2 / 64 + 5 * e2 ** 3 / 256
            - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * k
            + (15 * e4 / 256 + 45 * e6 / 1024) * k ** 2
            - (35 * e6 / 3072) * k ** 3)


def latlon_to_utm(lat, lon, zone=50, north=True):
    """输入角度，返回 (E, N)"""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lam = lon * DEG
    phi = lat * DEG
    lam0 = (zone * 6 - 183) * DEG

    ep2 = E2 / (1 - E2)
    N = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    Aa = np.cos(phi) * (lam - lam0)

    M = A * ((1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256) * phi
             - (3 * E2 / 8 + 3 * E2 ** 2 / 32 + 45 * E2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * E2 ** 2 / 256 + 45 * E2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * E2 ** 3 / 3072) * np.sin(6 * phi))

    Cm = _series(0, E2)
    k = 0.0
    # 使用标准级数展开
    e2, e4, e6 = E2, E2 ** 2, E2 ** 3
    x = (K0 * N * (Aa + (1 - T + C) * Aa ** 3 / 6
                   + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * Aa ** 5 / 120))
    y = (K0 * (M + N * np.tan(phi) * (Aa ** 2 / 2
                                       + (5 - T + 9 * C + 4 * C ** 2) * Aa ** 4 / 24
                                       + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * Aa ** 6 / 720)))
    E = x + 500000.0
    Nn = y if north else y + 10000000.0
    return E, Nn


def utm_to_latlon(E, N, zone=50, north=True):
    E = np.asarray(E, dtype=np.float64) - 500000.0
    N = np.asarray(N, dtype=np.float64)
    ep2 = E2 / (1 - E2)
    e1 = (1 - np.sqrt(1 - E2)) / (1 + np.sqrt(1 - E2))
    M = N / K0
    mu = M / (A * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    C1 = ep2 * np.cos(phi1) ** 2
    T1 = np.tan(phi1) ** 2
    N1 = A / np.sqrt(1 - E2 * np.sin(phi1) ** 2)
    R1 = A * (1 - E2) / (1 - E2 * np.sin(phi1) ** 2) ** 1.5
    D = E / (N1 * K0)
    lat = phi1 - (N1 * np.tan(phi1) / R1) * (D ** 2 / 2
                                             - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
                                             + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120) / np.cos(phi1)
    lam0 = (zone * 6 - 183) * DEG
    return lat / DEG, lon / DEG + lam0 / DEG


if __name__ == "__main__":
    # 与大疆报告角点对比校验
    print("TL", latlon_to_utm(30.053162147487896, 117.6412719324784))
    print("expect ~ (561816.86, 3324854.01)")
    print("BR", latlon_to_utm(30.04788468688969, 117.64974802570119))
