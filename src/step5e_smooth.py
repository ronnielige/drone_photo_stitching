# -*- coding: utf-8 -*-
"""Step5e: DSM 保边平滑（Perona-Malik 各向异性扩散）。

问题：plane-sweep 在植被/弱纹理上生成的深度存在 0.3~3m 高频噪声，
       hillshade 呈"雪花"状；正射时在偏斜视角下放大成横向米级抖动 -> 建筑/树木边缘熔化。
方案：Perona-Malik 扩散（噪声梯度被扩散掉、建筑边缘梯度被保留）
       + 小核中值去残刺。目标：曲面光滑且建筑边缘仍在原位。
"""
import os
import sys
import numpy as np
from scipy.ndimage import median_filter, distance_transform_edt
from common import WORK, load_json, save_json, log

N_ITER = int(os.environ.get("PM_ITER", "80"))
DT = float(os.environ.get("PM_DT", "0.25"))
K = float(os.environ.get("PM_K", "6.0"))


def perona_malik(z, n_iter=N_ITER, dt=DT, k=K):
    z = np.ascontiguousarray(z, dtype=np.float32)
    for i in range(n_iter):
        gx = np.zeros_like(z)
        gy = np.zeros_like(z)
        gx[:, 1:-1] = z[:, 2:] - z[:, :-2]
        gy[1:-1, :] = z[2:, :] - z[:-2, :]
        c = np.exp(-((gx * gx + gy * gy) / (k * k)))
        d = np.zeros_like(z)
        d[:, 1:-1] += (c[:, 1:-1] + c[:, 2:]) * 0.5 * (z[:, 2:] - z[:, 1:-1])
        d[:, 1:-1] += (c[:, 1:-1] + c[:, :-2]) * 0.5 * (z[:, :-2] - z[:, 1:-1])
        d[1:-1, :] += (c[1:-1, :] + c[2:, :]) * 0.5 * (z[2:, :] - z[1:-1, :])
        d[1:-1, :] += (c[1:-1, :] + c[:-2, :]) * 0.5 * (z[:-2, :] - z[1:-1, :])
        z += dt * d
        if (i + 1) % 20 == 0:
            log(f"  PM 迭代 {i+1}/{n_iter}  粗糙度(std of ∇) "
                f"{np.std(z[:,1:]-z[:,:-1]):.4f} m/px")
    return z


def roughness(z):
    return float(np.median(np.abs(z[:, 1:] - z[:, :-1])))


def main():
    p = os.path.join(WORK, "dsm_v2.npy")
    z = np.load(p).astype(np.float32)
    m = load_json(os.path.join(WORK, "dsm_meta_v2.json"))
    nan0 = int(np.isnan(z).sum())
    log(f"输入 DSM {z.shape} NaN {nan0} ({nan0/z.size*100:.2f}%) "
        f"高程 [{np.nanmin(z):.2f},{np.nanmax(z):.2f}] 中位 {np.nanmedian(z):.2f}")
    log(f"输入粗糙度 |Δz| 中位 {roughness(z):.4f} m/0.1m")

    # NaN 填充（最近有效）
    if nan0:
        bad = np.isnan(z)
        _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
        z = z[iy, ix]
        log(f"NaN 已填充 -> 范围 [{z.min():.2f},{z.max():.2f}]")

    zs = perona_malik(z)
    log(f"PM 后粗糙度 |Δz| 中位 {roughness(zs):.4f} m/0.1m")
    zs = median_filter(zs, size=3)
    log(f"中值(3) 后粗糙度 {roughness(zs):.4f}, 高程 [{zs.min():.2f},{zs.max():.2f}]")

    np.save(p, zs.astype(np.float32))
    m["z_min"] = float(zs.min())
    m["z_max"] = float(zs.max())
    m["z_median"] = float(np.median(zs))
    m["smoothed"] = f"perona-malik iter={N_ITER} dt={DT} K={K} + median3"
    save_json(os.path.join(WORK, "dsm_meta_v2.json"), m)
    log("已保存 dsm_v2.npy (保边平滑)")


if __name__ == "__main__":
    main()
