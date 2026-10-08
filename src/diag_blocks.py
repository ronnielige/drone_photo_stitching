# -*- coding: utf-8 -*-
"""诊断：在指定地理窗内，统计「取胜视图」的色调差异 —— 直接量化硬切换色块。
输出每个在该窗内取胜的视图：占用像素、放置后的均值 BGR、其增益。
用法: python diag_blocks.py E0 N0 E1 N1
"""
import os
import sys
import json
import numpy as np

ROOT = r"D:/lr/project/无人机航拍拼接"
WORK = os.path.join(ROOT, "mysolution", "work")
OUT = os.path.join(ROOT, "mysolution", "out")


def main():
    E0, N0, E1, N1 = (float(sys.argv[i]) for i in range(1, 5))
    om = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd, x0, y0 = om["gsd"], om["x0"], om["y0"]
    c0, c1 = int((E0 - x0) / gsd), int((E1 - x0) / gsd)
    r0, r1 = int((y0 - N1) / gsd), int((y0 - N0) / gsd)
    rgb = np.load(os.path.join(OUT, "ortho_rgb.npy"), mmap_mode="r")[r0:r1, c0:c1]
    vid = np.load(os.path.join(OUT, "ortho_vidx.npy"), mmap_mode="r")[r0:r1, c0:c1]
    val = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")[r0:r1, c0:c1]
    gains = np.load(os.path.join(WORK, "gains.npy"))
    print(f"窗 {E1-E0:.0f}x{N1-N0:.0f}m -> {rgb.shape[1]}x{rgb.shape[0]}px 有效 {val.mean()*100:.1f}%")

    v = vid[val]
    uniq, cnt = np.unique(v, return_counts=True)
    order = np.argsort(-cnt)
    print(f"\n该窗内取胜视图数 {len(uniq)}  (前 12 个)")
    print(f"{'view':>5} {'占%':>6} {'放置后BGR均值':>22} {'增益RGB':>20}")
    tones = []
    for k in order[:12]:
        u = uniq[k]
        m = val & (vid == u)
        col = rgb[m].astype(np.float32)
        tone = col.mean(axis=0)
        tones.append((u, tone, cnt[k]))
        print(f"{int(u):>5} {cnt[k]/len(v)*100:>5.1f}% "
              f"({tone[0]:6.1f},{tone[1]:6.1f},{tone[2]:6.1f})   "
              f"({gains[u,0]:.3f},{gains[u,1]:.3f},{gains[u,2]:.3f})")
    # 相邻取胜视图之间的色调落差
    print("\n相邻（边界相接）视图的色调差：")
    g = vid.astype(np.int32)
    pairs = set()
    for ax in (0, 1):
        a = np.take(g, np.arange(g.shape[ax] - 1), axis=ax)
        b = np.take(g, np.arange(1, g.shape[ax]), axis=ax)
        m = val.take(np.arange(val.shape[ax] - 1), axis=ax) & \
            val.take(np.arange(1, val.shape[ax]), axis=ax) & (a != b)
        for u1, u2 in zip(a[m], b[m]):
            pairs.add((min(u1, u2), max(u1, u2)))
    tonemap = {int(u): t for u, t, _ in tones}
    rows = []
    for u1, u2 in pairs:
        if u1 in tonemap and u2 in tonemap:
            d = np.abs(tonemap[u1] - tonemap[u2])
            rows.append((float(d.max()), u1, u2, d))
    rows.sort(reverse=True)
    for dmax, u1, u2, d in rows[:10]:
        print(f"  #{u1:>3} | #{u2:>3}  ΔBGR=({d[0]:5.1f},{d[1]:5.1f},{d[2]:5.1f})  max {dmax:.1f}")
    if rows:
        arr = np.array([r[0] for r in rows])
        print(f"\n相接视图对数 {len(rows)}; 通道最大差 中位 {np.median(arr):.1f} "
              f"p90 {np.percentile(arr,90):.1f} max {arr.max():.1f} (0-255)")


if __name__ == "__main__":
    main()
