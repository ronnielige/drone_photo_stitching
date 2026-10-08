"""快速客观评估：tile 色块检测、清晰度、与大疆结果的差异。"""
import os
import numpy as np
import cv2
import json
from common import OUT, WORK, load_json, imread_any, imwrite_any

DJI_TIF = r"D:/lr/project/无人机航拍拼接/xiangjia1/map/result.tif"


def lat_tile_analysis(img, tile):
    """检测 tile 网格边界上的颜色跳变是否显著高于随机位置"""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    k = 8
    # tile 边界处的左右块均值差
    diffs_grid = []
    for x in range(tile, w - 1, tile):
        a = f[:, max(0, x - k):x].mean(axis=(0, 1))
        b = f[:, x:x + k].mean(axis=(0, 1))
        diffs_grid.append(np.abs(a - b).mean())
    for y in range(tile, h - 1, tile):
        a = f[max(0, y - k):y, :].mean(axis=(0, 1))
        b = f[y:y + k, :].mean(axis=(0, 1))
        diffs_grid.append(np.abs(a - b).mean())
    # 随机位置基线
    rng = np.random.default_rng(0)
    diffs_rand = []
    for x in rng.choice(np.arange(k, w - k - 1), size=min(200, len(diffs_grid)), replace=False):
        a = f[:, x - k:x].mean(axis=(0, 1))
        b = f[:, x:x + k].mean(axis=(0, 1))
        diffs_rand.append(np.abs(a - b).mean())
    return float(np.mean(diffs_grid)), float(np.percentile(diffs_grid, 90)), \
        float(np.mean(diffs_rand)), diffs_grid


def main():
    q = imread_any(os.path.join(OUT, "quick_overview.png"))
    print("我的概览:", q.shape)
    tile = 256  # 1024 正射像素 @5cm -> 0.2m 网格下为 256
    mean_g, p90_g, mean_r, dg = lat_tile_analysis(q, tile)
    print(f"色块检测(tile={tile}px): 网格边界色差 mean {mean_g:.2f} p90 {p90_g:.2f} | "
          f"随机基线 {mean_r:.2f} | 比值 {mean_g/max(mean_r,1e-6):.2f}x")
    print(f"  最大边界色差 {max(dg):.2f}")

    g = q.astype(np.float32)
    print(f"通道均值 BGR: {g.mean(axis=(0,1)).round(1)}  标准差: {g.std(axis=(0,1)).round(1)}")

    mine_p = cv2.resize(q, (1000, int(1000 * q.shape[0] / q.shape[1])),
                        interpolation=cv2.INTER_AREA)
    imwrite_any(os.path.join(OUT, "view_overview.jpg"), mine_p,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # ---- 与大疆对比（同一 0.2m 网格）----
    try:
        import rasterio
        from rasterio.windows import from_bounds
        om = load_json(os.path.join(WORK, "grid.json"))
        gsd = om["gsd"]
        x0, y0 = om["x0"], om["y0"]
        x1, y1 = x0 + om["w"] * gsd, y0 - om["h"] * gsd
        with rasterio.open(DJI_TIF) as ds:
            a = ds.read(window=from_bounds(x0, y1, x1, y0, ds.transform),
                        out_shape=(3, q.shape[0], q.shape[1]),
                        resampling=rasterio.enums.Resampling.average)
            al = ds.read(4, window=from_bounds(x0, y1, x1, y0, ds.transform),
                         out_shape=(q.shape[0], q.shape[1]),
                         resampling=rasterio.enums.Resampling.nearest)
        dji = np.moveaxis(a, 0, 2)[:, :, ::-1]   # RGB -> BGR
        valid = al > 128
        v = (np.max(q, axis=2) > 0) & valid
        d = (g - dji.astype(np.float32))[v]
        print(f"与大疆 DOM 逐像素颜色差: mean {d.mean():.2f} |abs| mean "
              f"{np.abs(d).mean():.2f} p90 {np.percentile(np.abs(d),90):.2f}")
        print(f"  逐通道偏差 BGR: {d.mean(axis=0).round(2)}")
        # 结构相似度 (降采样后)
        Gray = lambda x: cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        m1 = cv2.resize(Gray(q), (1200, 918), interpolation=cv2.INTER_AREA).astype(np.float32)
        m2 = cv2.resize(Gray(dji), (1200, 918), interpolation=cv2.INTER_AREA).astype(np.float32)
        cov = ((m1 - m1.mean()) * (m2 - m2.mean())).mean()
        s1, s2 = m1.std(), m2.std()
        c1, c2 = (0.01 * 255) ** 2, (0.01 * 255) ** 2
        ssim = ((2 * m1.mean() * m2.mean() + c1) * (2 * cov + c2)) / \
               ((m1.mean() ** 2 + m2.mean() ** 2 + c1) * (s1 ** 2 + s2 ** 2 + c2))
        print(f"  与 DJI 的 SSIM(灰度加权近似): {ssim:.4f}")
        dj_sm = cv2.resize(dji, (1000, int(1000 * q.shape[0] / q.shape[1])),
                           interpolation=cv2.INTER_AREA)
        imwrite_any(os.path.join(OUT, "view_dji_dom.jpg"), dj_sm,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        # 双向: 大疆的 tile 色块检测
        dg_mean, dg_p90, dg_rand, _ = lat_tile_analysis(dji, tile)
        print(f"  大疆自身色块基线: 网格边界 {dg_mean:.2f} / 随机 {dg_rand:.2f} "
              f"= {dg_mean/max(dg_rand,1e-6):.2f}x")
    except Exception as e:
        print("大疆对比失败:", e)

    # ---- crop 清晰度（应接近原始影像）----
    for k in range(4):
        c = imread_any(os.path.join(OUT, f"quick_crop{k}.png"))
        if c is None:
            continue
        gg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        f = gg.astype(np.float32)
        lap = cv2.Laplacian(f, cv2.CV_32F)
        # 2.5cm 超采样 -> 折算到 5cm 网格再评价
        f5 = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA)
        print(f"crop{k}: @{2.5}cm sharp {lap.var():.1f} | 折算5cm sharp "
              f"{cv2.Laplacian(f5, cv2.CV_32F).var():.1f}")


if __name__ == "__main__":
    main()
