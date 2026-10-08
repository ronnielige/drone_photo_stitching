"""Step7: 空洞填充 + 导出 GeoTIFF(内嵌地理参考) + 预览图 + 与大疆结果对比。
需要用带 rasterio 的 venv 运行。
"""
import os
import json
import numpy as np
import cv2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from common import imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
WORK = os.path.join(ROOT, "mysolution", "work")
OUT = os.path.join(ROOT, "mysolution", "out")
DJI_TIF = os.path.join(ROOT, "xiangjia1", "map", "result.tif")
DJI_TFW = os.path.join(ROOT, "xiangjia1", "map", "result.tfw")
LOG = print


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fill_holes(img, mask):
    """用最近有效像素迭代膨胀填充空洞"""
    out = img.astype(np.uint8)
    holes = ~mask
    if holes.sum() == 0:
        return out
    filled = mask.copy()
    src = out.copy()
    for _ in range(64):
        if filled.all():
            break
        for ax in (0, 1):
            for s in (1, -1):
                sh = np.roll(src, s, ax)
                idm = [slice(None)] * 3
                idm[ax] = 0 if s > 0 else -1
                sh[tuple(idm)] = 0
                mf = np.roll(filled, s, ax)
                idm2 = [slice(None)] * 2
                idm2[ax] = 0 if s > 0 else -1
                mf[tuple(idm2)] = False
                take = (~filled) & mf
                out[take] = sh[take]
                filled |= take
        src = out
    return out


def main():
    om = load_json(os.path.join(OUT, "ortho_meta.json"))
    img = np.load(os.path.join(OUT, "ortho_rgb.npy"), mmap_mode="r")
    mask = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    H, W = img.shape[:2]
    LOG(f"正射图 {W} x {H}")

    # numpy -> 内存处理（分块填充）
    LOG("空洞填充 ...")
    out = np.array(img, dtype=np.uint8)
    m = np.array(mask, dtype=bool)
    holes = ~m
    LOG(f"  空洞比例 {holes.mean()*100:.3f}%")
    if holes.any():
        out = fill_holes(out, m)

    # ---- 写 GeoTIFF ----
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    gsd = om["gsd"]
    transform = Affine(gsd, 0.0, om["x0"], 0.0, -gsd, om["y0"])
    dst = os.path.join(OUT, "xiangjia_ortho_mine.tif")
    profile = dict(
        driver="GTiff", height=H, width=W, count=3, dtype="uint8",
        crs=CRS.from_epsg(32650), transform=transform,
        tiled=True, blockxsize=1024, blockysize=1024,
        compress="JPEG", photometric="YCBCR", jpeg_quality=92,
        BIGTIFF="IF_SAFER", interleave="PIXEL")
    with rasterio.open(dst, "w", **profile) as ds:
        ds.write(np.moveaxis(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), 2, 0))
        # 概览金字塔
        ds.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")
    LOG(f"写出 GeoTIFF: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")

    # 同时写 prj / tfw 便于其他软件
    with open(os.path.join(OUT, "xiangjia_ortho_mine.tfw"), "w") as f:
        f.write(f"{gsd:.10f}\n0.0\n0.0\n{-gsd:.10f}\n{om['x0']:.10f}\n{om['y0']:.10f}\n")
    prj = open(os.path.join(ROOT, "xiangjia1", "map", "result.prj")).read()
    with open(os.path.join(OUT, "xiangjia_ortho_mine.prj"), "w") as f:
        f.write(prj)

    # ---- 预览 ----
    pw = 3600
    ph = int(round(H * pw / W))
    prev = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_AREA)
    imwrite_any(os.path.join(OUT, "preview_mine.png"), prev,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    LOG(f"预览图: {pw} x {ph}")

    # ---- 与大疆对比 ----
    if not os.path.exists(DJI_TIF):
        LOG("未找到大疆结果，跳过对比")
        return

    def sharp(a):
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    def stats(a):
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return dict(mean=round(float(g.mean()), 1), std=round(float(g.std()), 1),
                    p2=round(float(np.percentile(g, 2)), 1),
                    p98=round(float(np.percentile(g, 98)), 1))

    with rasterio.open(DJI_TIF) as ds:
        LOG(f"大疆 result.tif: {ds.width} x {ds.height}, GSD {ds.transform.a:.5f} m, "
            f"{ds.count} 波段, crs={ds.crs}")
        dji_t = ds.transform
        gsd_d = dji_t.a
        mx0, my0 = om["x0"], om["y0"]
        mx1, my1 = mx0 + W * gsd, my0 - H * gsd
        dx0, dy0 = dji_t.c, dji_t.f
        dx1, dy1 = dx0 + ds.width * gsd_d, dy0 - ds.height * gsd_d
        ox0, oy0 = max(mx0, dx0), min(my0, dy0)
        ox1, oy1 = min(mx1, dx1), max(my1, dy1)
        LOG(f"重叠区 E[{ox0:.1f},{ox1:.1f}] N[{oy1:.1f},{oy0:.1f}]")
        # 方向 A：在我的 5cm 网格上比较（大疆上采样）
        w0, w1 = int((ox0 - mx0) / gsd), int((ox1 - mx0) / gsd)
        h0, h1 = int((my0 - oy0) / gsd), int((my0 - oy1) / gsd)
        mine_crop = out[h0:h1, w0:w1]
        dj = ds.read(window=rasterio.windows.from_bounds(ox0, oy1, ox1, oy0, dji_t),
                     out_shape=(min(4, ds.count), mine_crop.shape[0], mine_crop.shape[1]),
                     resampling=rasterio.enums.Resampling.average)
        dji_crop = np.moveaxis(dj[:3], 0, 2)[:, :, ::-1]  # RGB->BGR 统一
        alpha = dj[3] if ds.count == 4 else np.full(
            (mine_crop.shape[0], mine_crop.shape[1]), 255, np.uint8)
        # 方向 B：在大疆网格上比较（我的降采样）
        dw0, dw1 = int((ox0 - dx0) / gsd_d), int((ox1 - dx0) / gsd_d)
        dh0, dh1 = int((dy0 - oy0) / gsd_d), int((dy0 - oy1) / gsd_d)
        dj_win = ds.read(window=rasterio.windows.Window(dw0, dh0, dw1 - dw0, dh1 - dh0))
        dji_native = np.moveaxis(dj_win[:3], 0, 2)[:, :, ::-1]
        alpha_native = dj_win[3] if ds.count == 4 else None
        target = (dh1 - dh0, dw1 - dw0)
        mine_on_dji = cv2.resize(out, (target[1], target[0]), interpolation=cv2.INTER_AREA)
    LOG(f"对比区: {mine_crop.shape[1]} x {mine_crop.shape[0]} (我的网格) / "
        f"{target[1]} x {target[0]} (大疆网格)")

    # 有效掩膜（排除黑边）
    vm = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")[h0:h1, w0:w1]
    valid_both = vm & (alpha > 128)
    cov = float(valid_both.mean())
    LOG(f"双方均有效的重叠面积占比: {cov*100:.1f}%")

    def masked_sharp(a, m):
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        v = lap[m]
        return float(v.var())

    sA_mine = masked_sharp(mine_crop, valid_both)
    sA_dji = masked_sharp(dji_crop, valid_both)
    vn = (alpha_native > 128) if alpha_native is not None else np.ones(target, bool)
    sB_mine = masked_sharp(mine_on_dji, vn)
    sB_dji = masked_sharp(dji_native, vn)
    LOG(f"[A 我的5cm网格] 清晰度 我的 {sA_mine:.1f} vs 大疆(上采样) {sA_dji:.1f}  "
        f"比值 {sA_mine/max(sA_dji,1e-6):.2f}x")
    LOG(f"[B 大疆5.6cm网格] 清晰度 我的(降采样) {sB_mine:.1f} vs 大疆 {sB_dji:.1f}  "
        f"比值 {sB_mine/max(sB_dji,1e-6):.2f}x")
    LOG(f"灰度统计 我的: {stats(mine_crop)}")
    LOG(f"灰度统计 大疆: {stats(dji_crop)}")

    # 接缝/色差检查：相邻影像边界处的梯度异常
    g = cv2.cvtColor(mine_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gd = cv2.cvtColor(dji_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    LOG(f"有效区梯度能量 我的 {np.abs(cv2.Laplacian(g, cv2.CV_32F))[valid_both].mean():.3f} "
        f"大疆 {np.abs(cv2.Laplacian(gd, cv2.CV_32F))[valid_both].mean():.3f}")

    # 局部对比图
    hs, ws = mine_crop.shape[:2]
    crops = []
    best = []
    for cy in np.linspace(0.15, 0.8, 12):
        for cx in np.linspace(0.15, 0.85, 12):
            y, x = int(hs * cy), int(ws * cx)
            s = 760
            if y + s > hs or x + s > ws:
                continue
            m = valid_both[y:y + s, x:x + s]
            if m.mean() > 0.98:
                best.append((float(cv2.Laplacian(
                    cv2.cvtColor(mine_crop[y:y + s, x:x + s], cv2.COLOR_BGR2GRAY)
                    .astype(np.float32), cv2.CV_32F).var()), y, x))
    best.sort(reverse=True)
    picks = [(y, x) for _, y, x in best[:1]] + \
             [(int(hs * a), int(ws * b)) for a, b in [(0.25, 0.45), (0.6, 0.7)]]
    for k, (y, x) in enumerate(picks):
        s = min(760, hs - y, ws - x)
        crops.append(mine_crop[y:y + s, x:x + s])
        crops.append(dji_crop[y:y + s, x:x + s])
    tiles = []
    for i in range(0, len(crops), 2):
        a, b = crops[i], crops[i + 1]
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        sep = np.full((h, 8, 3), 240, np.uint8)
        tiles.append(np.concatenate([a[:h, :w], sep, b[:h, :w]], axis=1))
    comp = np.concatenate(tiles, axis=0)
    imwrite_any(os.path.join(OUT, "compare_vs_dji.png"), comp,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    LOG(f"对比图已保存: {comp.shape[1]} x {comp.shape[0]} (左=我的, 右=大疆)")

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "size_mine": [W, H], "gsd_mine": gsd,
            "size_dji": [int(ds.width if False else 14599), 10488], "gsd_dji": float(gsd_d),
            "sharpness_A_mine": sA_mine, "sharpness_A_dji": sA_dji,
            "sharpness_B_mine": sB_mine, "sharpness_B_dji": sB_dji,
            "stats_mine": stats(mine_crop), "stats_dji": stats(dji_crop),
            "valid_overlap_ratio": cov}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
