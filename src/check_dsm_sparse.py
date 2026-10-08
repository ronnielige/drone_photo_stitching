# -*- coding: utf-8 -*-
"""用 BA 稀疏三维点检验 DSM 自洽性（稀疏点重投影残差 0.76px，可信）。"""
import os
import sys
import numpy as np
from scipy.ndimage import map_coordinates
from common import WORK, load_json

X = np.load(os.path.join(WORK, "ba_X.npy"))
print("稀疏点", X.shape, "z 分位", np.round(np.nanpercentile(X[:, 2], [1, 10, 50, 90, 99]), 2))

for tag, zn, mn in (("v2(平滑后)", "dsm_v2.npy", "dsm_meta_v2.json"),
                    ("v1(旧)", "dsm.npy", "dsm_meta.json")):
    z = np.load(os.path.join(WORK, zn))
    m = load_json(os.path.join(WORK, mn))
    g = m["gsd"]
    fx = (X[:, 0] - m["x0"]) / g - 0.5
    fy = (m["y0"] - X[:, 1]) / g - 0.5
    inb = (fx >= 0) & (fx < z.shape[1] - 1) & (fy >= 0) & (fy < z.shape[0] - 1)
    v = map_coordinates(z, [fy[inb], fx[inb]], order=1, mode="nearest")
    d = v - X[inb, 2]
    fin = np.isfinite(d)
    d = d[fin]
    print(f"\n[{tag}] 参与 {fin.sum()}/{len(X)}")
    print(f"  高程差 (DSM - 稀疏点): 中位 {np.median(d):+.3f}m  mean {d.mean():+.3f}m")
    print(f"  |差| 分位 50/90/99: {np.percentile(np.abs(d),50):.3f} / "
          f"{np.percentile(np.abs(d),90):.3f} / {np.percentile(np.abs(d),99):.3f} m")
    print(f"  std {d.std():.3f}  |  差 >1m 占比 {(np.abs(d)>1).mean()*100:.1f}%  "
          f">3m 占比 {(np.abs(d)>3).mean()*100:.1f}%")
    # 按“稀疏点是否在地面”分层（以局部 p10 为地面）
    gz = np.percentile(X[:, 2], 15)
    gl = np.abs(X[:, 2] - gz) < 1.5
    print(f"  近地面点({gl[inb][fin].sum()}): 差 std {d[gl[inb][fin]].std():.3f}m "
          f"中位 {np.median(d[gl[inb][fin]]):+.3f}")
