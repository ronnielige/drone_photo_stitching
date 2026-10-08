"""最终评估：在双方有效区内，量化清晰度、颜色一致性、接缝色块、几何一致性。
用带 rasterio 的 venv 运行。
"""
import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
WORK = os.path.join(ROOT, "mysolution", "work")
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")


def to_bgr(x):
    return x[:, :, ::-1]


def stats(a, m=None):
    if m is not None:
        v = a[m]
    else:
        v = a.reshape(-1, a.shape[-1])
    return dict(mean=round(float(v.mean()), 1), std=round(float(v.std()), 1),
                p2=round(float(np.percentile(v, 2)), 1),
                p98=round(float(np.percentile(v, 98)), 1))


def sharp_bgr(a, m):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap[m].var()), float(np.abs(lap[m]).mean())


def tile_seam(img, tile, k=6):
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    dg = []
    for x in range(tile, w - 1, tile):
        dg.append(np.abs(f[:, x - k:x].mean(axis=(0, 1)) - f[:, x:x + k].mean(axis=(0, 1))).mean())
    for y in range(tile, h - 1, tile):
        dg.append(np.abs(f[y - k:y, :].mean(axis=(0, 1)) - f[y:y + k, :].mean(axis=(0, 1))).mean())
    rng = np.random.default_rng(0)
    dr = []
    xs = rng.choice(np.arange(k, w - k - 1), size=min(300, max(len(dg), 1)), replace=False)
    for x in xs:
        dr.append(np.abs(f[:, x - k:x].mean(axis=(0, 1)) - f[:, x:x + k].mean(axis=(0, 1))).mean())
    return float(np.mean(dg)), float(np.max(dg)), float(np.mean(dr))


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json")))
    with rasterio.open(MINE) as dm, rasterio.open(DJI) as dd:
        print(f"我的: {dm.width}x{dm.height} GSD {dm.transform.a:.4f} "
              f"压缩 {dm.profile.get('compress')} 金字塔 {dm.overviews(1)}")
        print(f"大疆: {dd.width}x{dd.height} GSD {dd.transform.a:.4f} "
              f"波段 {dd.count} 压缩 {dd.profile.get('compress')} 金字塔 {dd.overviews(1)}")
        gsd_m, gsd_d = dm.transform.a, dd.transform.a
        x0, y0 = om["x0"], om["y0"]
        x1, y1 = x0 + om["w"] * gsd_m, y0 - om["h"] * gsd_m
        ox0, oy0 = max(x0, dd.transform.c), min(y0, dd.transform.f)
        ox1, oy1 = min(x1, dd.transform.c + dd.width * gsd_d), \
            max(y1, dd.transform.f - dd.height * gsd_d)
        # 统一到 8cm 网格做统计（兼顾两边）
        gg = 0.08
        gw = int((ox1 - ox0) / gg)
        gh = int((oy0 - oy1) / gg)
        print(f"统一对比网格 {gw}x{gh} @8cm")
        mine = dm.read(window=from_bounds(ox0, oy1, ox1, oy0, dm.transform),
                       out_shape=(3, gh, gw),
                       resampling=rasterio.enums.Resampling.average)
        mine = np.moveaxis(mine, 0, 2)[:, :, ::-1].astype(np.uint8)   # RGB->BGR
        dji = dd.read(window=from_bounds(ox0, oy1, ox1, oy0, dd.transform),
                      out_shape=(min(4, dd.count), gh, gw),
                      resampling=rasterio.enums.Resampling.average)
        alpha = dji[3] if dd.count == 4 else np.full((gh, gw), 255)
        dji = np.moveaxis(dji[:3], 0, 2)[:, :, ::-1].astype(np.uint8)
    valid_d = alpha > 128
    valid_m = np.max(mine, axis=2) > 3
    both = valid_m & valid_d
    print(f"有效区: 我 {valid_m.mean()*100:.1f}%  大疆 {valid_d.mean()*100:.1f}%  "
          f"交集 {both.mean()*100:.1f}%")

    print("\n--- 颜色统计（仅双方有效区）---")
    sm = {c: stats(mine[:, :, c], both) for c in range(3)}
    sd = {c: stats(dji[:, :, c], both) for c in range(3)}
    name = ["B", "G", "R"]
    print(f"{'通道':<6}{'我的 mean/std':>18}{'大疆 mean/std':>18}{'亮度差':>10}")
    for c in range(3):
        print(f"{name[c]:<6}{sm[c]['mean']:>10.1f}/{sm[c]['std']:<7.1f}"
              f"{sd[c]['mean']:>10.1f}/{sd[c]['std']:<7.1f}"
              f"{sm[c]['mean']-sd[c]['mean']:>10.1f}")

    print("\n--- 清晰度（同一 8cm 网格，同一有效区）---")
    vm, lpm = sharp_bgr(mine, both)
    vd, lpd = sharp_bgr(dji, both)
    print(f"拉普拉斯方差   我的 {vm:.1f}  大疆 {vd:.1f}   比值 {vm/max(vd,1e-6):.2f}x")
    print(f"平均梯度能量   我的 {lpm:.3f}  大疆 {lpd:.3f}   比值 {lpm/max(lpd,1e-6):.2f}x")

    print("\n--- 接缝/色块检测 ---")
    for tag, im, tl in (("我的(1024正射块)", mine, int(1024 * 0.05 / gg)),
                        ("大疆(参考)", dji, int(1024 * 0.05 / gg))):
        a, mx, base = tile_seam(im, tl)
        print(f"{tag:<20} tile边界色差 mean {a:.2f} max {mx:.2f} | 随机基线 {base:.2f} "
              f"| 比值 {a/max(base,1e-6):.2f}x")

    print("\n--- 几何一致性 ---")
    gm = cv2.cvtColor(mine, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gd = cv2.cvtColor(dji, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gm = gm[both]
    gd = gd[both]
    corr = np.corrcoef(gm, gd)[0, 1]
    print(f"灰度相关系数 {corr:.4f}  (1.0=完全一致)")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    # 全局 SSIM 近似（均值/方差/协方差形式）
    mm, md = gm.mean(), gd.mean()
    sm_, sd_ = gm.std(), gd.std()
    cov = ((gm - mm) * (gd - md)).mean()
    ssim = ((2 * mm * md + c1) * (2 * cov + c2)) / \
           ((mm ** 2 + md ** 2 + c1) * (sm_ ** 2 + sd_ ** 2 + c2))
    print(f"结构相似度 SSIM(灰度) {ssim:.4f}")

    res = dict(grid_ggsd=gg, valid_mine=float(valid_m.mean()), valid_dji=float(valid_d.mean()),
               overlap=float(both.mean()),
               stats_mine={name[c]: sm[c] for c in range(3)},
               stats_dji={name[c]: sd[c] for c in range(3)},
               sharp_var=[vm, vd], sharp_grad=[lpm, lpd],
               corr=float(corr), ssim=float(ssim))
    with open(os.path.join(OUT, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已保存 final_metrics.json")


if __name__ == "__main__":
    main()
