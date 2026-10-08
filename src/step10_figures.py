"""生成交付用对比图：全图并排 + 细节区域对比（我的 vs 大疆）。venv 运行。"""
import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import imread_any, imwrite_any

ROOT = r"D:/lr/project/无人机航拍拼接"
OUT = os.path.join(ROOT, "mysolution", "out")
MINE = os.path.join(OUT, "xiangjia_ortho_mine.tif")
DJI = os.path.join(ROOT, "xiangjia1", "map", "result.tif")


def read_win(ds, x0, y0, x1, y1, w, h, bands=None):
    """(x0,y0)=西南角, (x1,y1)=东北角"""
    idx = bands or tuple(range(1, 4))
    a = ds.read(indexes=idx, window=from_bounds(x0, y0, x1, y1, ds.transform),
                out_shape=(len(idx), h, w),
                resampling=rasterio.enums.Resampling.average)
    return np.moveaxis(a, 0, 2)[:, :, ::-1]   # ->BGR


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (30, 30, 30), -1)
    cv2.putText(img, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    om = json.load(open(os.path.join(OUT, "ortho_meta.json")))
    with rasterio.open(MINE) as dm, rasterio.open(DJI) as dd:
        gsd_m = dm.transform.a
        # ---- 1) 全图并排 ----
        pw = 1400
        m_prev = cv2.resize(imread_any(os.path.join(OUT, "preview_mine.png")),
                            (pw, int(pw * om["h"] / om["w"])), interpolation=cv2.INTER_AREA)
        # 大疆整图
        dw = dd.width
        d_prev = dd.read(indexes=(1, 2, 3), out_shape=(3, int(pw * dd.height / dd.width), pw),
                         resampling=rasterio.enums.Resampling.average)
        d_prev = np.moveaxis(d_prev, 0, 2)[:, :, ::-1]
        h = max(m_prev.shape[0], d_prev.shape[0])
        canvas = np.full((h, pw * 2 + 12, 3), 255, np.uint8)
        canvas[:d_prev.shape[0], :pw] = d_prev
        canvas[:m_prev.shape[0], pw + 12:] = m_prev
        canvas = label(canvas, "LEFT: DJI Terra 5.62cm   |   RIGHT: Mine 5.0cm")
        imwrite_any(os.path.join(OUT, "fig_full_compare.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("fig_full_compare.jpg", canvas.shape)

        # ---- 2) 细节对比（三个典型区域）----
        spots = [
            ("Village & road", 562350, 3324600, 170),
            ("Terraced fields", 562150, 3324500, 200),
            ("Blue-roof building", 562230, 3324400, 120),
        ]
        tiles = []
        for name, cx, cy, half in spots:
            w = h = int(2 * half / 0.05)
            m = read_win(dm, cx - half, cy - half, cx + half, cy + half, w, h)
            d = read_win(dd, cx - half, cy - half, cx + half, cy + half,
                         int(2 * half / dd.transform.a), int(2 * half / dd.transform.a))
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
            sep = np.full((h, 8, 3), 240, np.uint8)
            pair = np.concatenate([m, sep, d], axis=1)
            sc = 1450.0 / pair.shape[1]
            pair = cv2.resize(pair, (1450, int(pair.shape[0] * sc)),
                              interpolation=cv2.INTER_AREA)
            pair = label(pair, f"{name}  (L=DJI, R=Mine)")
            tiles.append(pair)
            print("detail", name, pair.shape)
        W = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, W - t.shape[1], cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        comp = np.concatenate(tiles, axis=0)
        imwrite_any(os.path.join(OUT, "fig_detail_compare.jpg"), comp,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("fig_detail_compare.jpg", comp.shape)


if __name__ == "__main__":
    main()
