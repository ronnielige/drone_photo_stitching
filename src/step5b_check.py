"""DSM 质量检查：统计 + 高程彩色可视化。"""
import os
import numpy as np
import cv2
from common import WORK, OUT, log, load_json, imwrite_any

dm = load_json(os.path.join(WORK, "dsm_meta.json"))
dsm = np.load(os.path.join(WORK, "dsm.npy"))
conf = np.load(os.path.join(WORK, "dsm_conf.npy"))
log(f"DSM {dsm.shape} @ {dm['gsd']}m, Z [{np.nanmin(dsm):.2f}, {np.nanmax(dsm):.2f}]")
zs = dsm.ravel()
for p in (1, 5, 25, 50, 75, 95, 99):
    log(f"  p{p}: {np.nanpercentile(zs, p):.2f} m")
log(f"置信度(累积观测权重): mean {conf.mean():.2f}, 有效格点 {(conf>0).mean()*100:.1f}%")

# 高程彩色可视化
zmin, zmax = np.nanpercentile(dsm, 2), np.nanpercentile(dsm, 98)
n = ((dsm - zmin) / max(zmax - zmin, 1e-6) * 255).clip(0, 255).astype(np.uint8)
col = cv2.applyColorMap(n, cv2.COLORMAP_TURBO)
scale = 3
h, w = col.shape[:2]
small = cv2.resize(col, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
imwrite_any(os.path.join(OUT, "dsm_preview.png"), small)
log(f"DSM 预览 -> dsm_preview.png ({small.shape[1]}x{small.shape[0]}), "
    f"色标 {zmin:.1f}-{zmax:.1f} m")

# 与大疆 DSM 对比（如果可读）
dji_dsm = os.path.join(os.path.dirname(WORK), "xiangjia1", "map", "dsm.tif")
if os.path.exists(dji_dsm):
    try:
        import rasterio
        from rasterio.transform import Affine
        with rasterio.open(dji_dsm) as ds:
            log(f"大疆 DSM: {ds.width} x {ds.height}, GSD {ds.transform.a:.4f} m")
            t = ds.transform
            # 我的 DSM 地理范围
            x0, y0, g = dm["x0"], dm["y0"], dm["gsd"]
            x1, y1 = x0 + dsm.shape[1] * g, y0 - dsm.shape[0] * g
            win = rasterio.windows.from_bounds(x0, y1, x1, y0, t)
            a = ds.read(1, window=win, out_shape=(dsm.shape[0], dsm.shape[1]),
                        resampling=rasterio.enums.Resampling.bilinear)
            a = a.astype(np.float32)
            a[a <= 0] = np.nan
            m = ~np.isnan(a) & ~np.isnan(dsm)
            if m.sum() > 1000:
                d = (dsm[m] - a[m])
                log(f"与大疆 DSM 对比 (n={m.sum()}): 偏差 mean {d.mean():.2f}m, "
                    f"|d| p50 {np.percentile(np.abs(d),50):.2f}m, "
                    f"p90 {np.percentile(np.abs(d),90):.2f}m, std {d.std():.2f}m")
                log(f"  大疆 DSM Z 范围 [{np.nanmin(a):.1f}, {np.nanmax(a):.1f}], "
                    f"中位 {np.nanmedian(a):.1f}")
    except Exception as e:
        log(f"大疆 DSM 对比失败: {e}")
