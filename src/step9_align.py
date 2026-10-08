"""几何配准校验与校正：用 SIFT 匹配测量自研图与大疆图的平移/旋转/缩放，
若存在系统性偏移则对输出 GeoTIFF 的地理变换做刚性校正。
venv 运行（需要 rasterio + cv2）。
"""
import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")


def grab(dd, cx, cy, half, g, nband=3):
    win = from_bounds(cx - half, cy - half, cx + half, cy + half, dd.transform)
    gh = gw = int(2 * half / g)
    if dd.count == 4 and nband == 3:
        a = dd.read(indexes=(1, 2, 3), window=win, out_shape=(3, gh, gw),
                    resampling=rasterio.enums.Resampling.average)
    else:
        a = dd.read(window=win, out_shape=(nband, gh, gw),
                    resampling=rasterio.enums.Resampling.average)
    return np.moveaxis(a, 0, 2)[:, :, ::-1]   # RGB->BGR


def measure(mine_bgr, dji_bgr):
    g1 = cv2.cvtColor(mine_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(dji_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in (x[:2] for x in knn if len(x) >= 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0, maxIters=8000)
    inl = inl.ravel().astype(bool)
    if inl.sum() < 20:
        return None
    dx = float(np.median(p2[inl, 0] - p1[inl, 0]))
    dy = float(np.median(p2[inl, 1] - p1[inl, 1]))
    scale = float(M[0, 0] ** 2 + M[1, 0] ** 2) ** 0.5
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return dict(n_match=len(good), n_inlier=int(inl.sum()), dx_px=dx, dy_px=dy,
                scale=scale, rot_deg=rot)


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json")))
    with rasterio.open(MINE) as dm, rasterio.open(DJI) as dd:
        gsd_m, gsd_d = dm.transform.a, dd.transform.a
        x0, y0 = om["x0"], om["y0"]
        x1, y1 = x0 + om["w"] * gsd_m, y0 - om["h"] * gsd_m
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        g = 0.10
        results = []
        for (ox, oy) in [(0, 0), (-300, -300), (300, 300), (-300, 300), (300, -300)]:
            ccx, ccy = cx + ox, cy + oy
            half = 350
            mine = grab(dd if False else dm, ccx, ccy, half, g)
            dji = grab(dd, ccx, ccy, half, g)
            r = measure(mine.astype(np.uint8), dji.astype(np.uint8))
            if r:
                r["center"] = (ccx, ccy)
                r["dx_m"] = r["dx_px"] * g
                r["dy_m"] = r["dy_px"] * g   # 像素 y 向下 = 南向
                results.append(r)
                print(f"@({ccx:.0f},{ccy:.0f}) 匹配 {r['n_inlier']}/{r['n_match']} "
                      f"位移 E {r['dx_m']:+.3f}m S {r['dy_m']:+.3f}m "
                      f"scale {r['scale']:.5f} rot {r['rot_deg']:+.4f}deg")
        if not results:
            print("匹配失败")
            return
        dx = float(np.median([r["dx_m"] for r in results]))
        dy = float(np.median([r["dy_m"] for r in results]))
        sc = float(np.median([r["scale"] for r in results]))
        rt = float(np.median([r["rot_deg"] for r in results]))
        print(f"\n中位: E {dx:+.3f}m, S {dy:+.3f}m, scale {sc:.5f}, rot {rt:+.4f}deg")
        shift = (dx ** 2 + dy ** 2) ** 0.5
        print(f"位移大小 {shift:.2f} m")

        # 应用刚性校正（仅平移；scale/rot 接近 1/0 时不处理）
        if shift > 0.5:
            with rasterio.open(MINE, "r+") as ds:
                t = ds.transform
                # 影像内容相对大疆偏移 (dx 东, dy 南) -> 将地理原点平移
                # 若我的地物出现在大疆的东边 dx_m，需要把我的地理坐标向西移 dx
                new_t = Affine(t.a, t.b, t.c - dx, t.d, t.e, t.f + dy)
                ds.transform = new_t
                print("已更新 GeoTIFF transform:")
                print("  旧:", t)
                print("  新:", new_t)
            om["x0"] -= dx
            om["y0"] += dy
            json.dump(om, open(os.path.join(OUT, "ortho_meta.json"), "w"),
                      ensure_ascii=False, indent=1)
            # 重写 tfw
            with open(os.path.join(OUT, "xiangjia_ortho_mine.tfw"), "w") as f:
                f.write(f"{gsd_m:.10f}\n0.0\n0.0\n{-gsd_m:.10f}\n"
                        f"{om['x0']:.10f}\n{om['y0']:.10f}\n")
        else:
            print("偏移小于 0.5m，无需校正")

        # 复测相关性
        with rasterio.open(MINE) as dm2:
            mine = grab(dm2, cx, cy, 350, 0.10)
        dji = grab(dd, cx, cy, 350, 0.10)
        g1 = cv2.cvtColor(mine.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(dji.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        print(f"校正后灰度相关系数: {np.corrcoef(g1.ravel(), g2.ravel())[0,1]:.4f}")


if __name__ == "__main__":
    main()
