# -*- coding: utf-8 -*-
"""通用导出：把 out/*.npy 正射结果写成带地理参考的 GeoTIFF + tfw/prj + 预览图。

用法: python export_tif.py <src_npy> <out_tif_路径> [meta_json]
meta_json 缺省用 out/ortho_meta.json（或同名 *_meta3.json 自动匹配）。
需要 venv(rasterio)。
"""
import os
import sys
import json
import time
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, DJI_DIR, EPSG

PRJ = os.path.join(DJI_DIR, "map", "result.prj")   # 大疆工程坐标系文件（可选，缺失则回退默认 EPSG）


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    mj = sys.argv[3] if len(sys.argv) > 3 else (
        os.path.join(OUT, "ortho_meta3.json")
        if "rgb3" in os.path.basename(src) and
        os.path.exists(os.path.join(OUT, "ortho_meta3.json"))
        else os.path.join(OUT, "ortho_meta.json"))
    om = json.load(open(mj, encoding="utf-8"))
    gsd, x0, y0, W, H = om["gsd"], om["x0"], om["y0"], om["w"], om["h"]
    t0 = time.time()
    img = np.load(src, mmap_mode="r")
    assert img.shape[0] == H and img.shape[1] == W, f"形状不符 {img.shape} vs {H}x{W}"
    print(f"源 {src} {W}x{H} @{gsd*100:.2f}cm  meta={os.path.basename(mj)}")

    transform = Affine(gsd, 0.0, x0, 0.0, -gsd, y0)
    profile = dict(driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                   crs=CRS.from_string(EPSG), transform=transform,
                   tiled=True, blockxsize=1024, blockysize=1024,
                   compress="JPEG", photometric="YCBCR", jpeg_quality=92,
                   BIGTIFF="IF_SAFER", interleave="PIXEL")
    # 分条写入，避免整图 RGB 转置的峰值内存
    STRIP = 2048
    with rasterio.open(dst, "w", **profile) as ds:
        for y in range(0, H, STRIP):
            y1 = min(y + STRIP, H)
            blk = np.ascontiguousarray(img[y:y1])[:, :, ::-1]      # BGR->RGB
            ds.write(np.moveaxis(blk, 2, 0), window=rasterio.windows.Window(0, y, W, y1 - y))
        ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")
    print(f"写出 {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")

    base = os.path.splitext(dst)[0]
    with open(base + ".tfw", "w") as f:
        f.write(f"{gsd:.10f}\n0.0\n0.0\n{-gsd:.10f}\n{x0:.10f}\n{y0:.10f}\n")
    if os.path.exists(PRJ):
        open(base + ".prj", "w").write(open(PRJ).read())
    else:
        # 无大疆参考工程时，回退到管线配置的坐标系（EPSG:32650 = WGS84 / UTM 50N）
        open(base + ".prj", "w").write(CRS.from_string(EPSG).wkt)

    # 预览（无效区标白，便于看覆盖范围）
    pw = 3600
    step = max(1, H // 3000)
    small = np.ascontiguousarray(img[::step, ::step])
    ph = int(round(small.shape[0] * pw / small.shape[1]))
    prev = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_AREA)
    mk = np.load(os.path.join(OUT, "ortho_valid.npy"), mmap_mode="r")
    msmall = np.ascontiguousarray(mk[::step, ::step]).astype(np.uint8) * 255
    mprev = cv2.resize(msmall, (pw, ph), interpolation=cv2.INTER_NEAREST) > 128
    prev[~mprev] = 255
    pv = os.path.join(OUT, "preview_" + os.path.splitext(os.path.basename(dst))[0] + ".png")
    cv2.imencode(".png", prev, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])[1].tofile(pv)
    print(f"预览 {pv}  {prev.shape[1]}x{prev.shape[0]}   用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
