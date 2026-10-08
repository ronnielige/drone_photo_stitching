import os, sys, json, time
import numpy as np
import cv2
import tifffile

OUT = r"D:\lr\project\无人机航拍拼接\mysolution\out"
DJI = r"D:\lr\project\无人机航拍拼接\xiangjia1\map\result.tif"
MINE = os.path.join(OUT, "mosaic2_geotiff.tif")

def read_bgr(path):
    a = tifffile.imread(path)
    if a.shape[2] == 4:
        a = a[..., :3]
    return a[..., ::-1]  # RGB -> BGR, alpha dropped

def geo(path):
    tf = tifffile.TiffFile(path)
    p = tf.pages[0]
    sc = p.tags['ModelPixelScaleTag'].value
    tp = p.tags['ModelTiepointTag'].value
    tf.close()
    return float(sc[0]), float(tp[3]), float(tp[4]), p.shape[1], p.shape[0]

def clarity_metrics(gray, mask, tile=256):
    """per-tile Laplacian variance on valid tiles"""
    H, W = gray.shape
    vals = []
    for r in range(0, H - tile, tile):
        for c in range(0, W - tile, tile):
            t = gray[r:r + tile, c:c + tile]
            m = mask[r:r + tile, c:c + tile]
            if m.mean() < 0.98:
                continue
            vals.append(cv2.Laplacian(t, cv2.CV_32F).var())
    return np.array(vals)

def main():
    t0 = time.time()
    g_m, x_m, y_m, w_m, h_m = geo(MINE)
    g_d, x_d, y_d, w_d, h_d = geo(DJI)
    print("mine: %dx%d gsd=%.4f  tl=(%.1f, %.1f)" % (w_m, h_m, g_m, x_m, y_m))
    print("dji : %dx%d gsd=%.4f  tl=(%.1f, %.1f)" % (w_d, h_d, g_d, x_d, y_d))

    mine = read_bgr(MINE)
    dji = read_bgr(DJI)

    # extents (UTM)
    me0, me1 = x_m, x_m + w_m * g_m
    mn1, mn0 = y_m - h_m * g_m, y_m      # mine: n0 top, n1 bottom
    de0, de1 = x_d, x_d + w_d * g_d
    dn1, dn0 = y_d - h_d * g_d, y_d

    # overlap window in UTM
    oe0, oe1 = max(me0, de0), min(me1, de1)
    on0, on1 = max(mn0, dn0), min(mn1, dn1)
    print("overlap: E %.1f..%.1f  N %.1f..%.1f (%.0f x %.0f m)" %
          (oe0, oe1, on1, on0, oe1 - oe0, on0 - on1))

    def to_rc(x, y, g, x0, y0):
        c = int(round((x - x0) / g)); r = int(round((y0 - y) / g))
        return r, c

    r0m, c0m = to_rc(oe0, on0, g_m, x_m, y_m)
    r1m, c1m = to_rc(oe1, on1, g_m, x_m, y_m)
    r0d, c0d = to_rc(oe0, on0, g_d, x_d, y_d)
    r1d, c1d = to_rc(oe1, on1, g_d, x_d, y_d)

    hm, wm = r1m - r0m, c1m - c0m
    hd, wd = r1d - r0d, c1d - c0d
    hm2 = wm2 = min(hm, hd, 4096); wd2 = min(wm, wd, 4096)
    A = mine[r0m:r0m + hm2, c0m:c0m + wd2]
    B = dji[r0d:r0d + hm2, c0d:c0d + wd2]

    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    ma = ga > 0; mb = gb > 0
    both = ma & mb
    print("overlap sample %dx%d, both-valid %.1f%%" % (hm2, wd2, 100 * both.mean()))

    # color sanity: correlation of my R vs dji R (should be high if channels correct)
    ar = A[..., 2].astype(np.float32); br = B[..., 2].astype(np.float32)
    ab = A[..., 0].astype(np.float32); bb = B[..., 0].astype(np.float32)
    m = both & (ga > 20)
    def corr(x, y):
        x = x[m].ravel(); y = y[m].ravel()
        xm = x - x.mean(); ym = y - y.mean()
        return float(xm @ ym / (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-9))
    print("channel corr: mineR~djiR %.3f  mineR~djiB %.3f" % (corr(ar, br), corr(ar, bb)))

    va = clarity_metrics(ga, ma)
    vb = clarity_metrics(gb, mb)
    print("clarity (Laplacian var): mine med %.0f std %.0f | dji med %.0f std %.0f" %
          (np.median(va), va.std(), np.median(vb), vb.std()))

    # coverage over my full block
    cov_mine = float((mine.sum(2) > 0).mean())
    cov_dji = float((dji.sum(2) > 0).mean())
    print("coverage: mine %.1f%% | dji %.1f%%" % (100 * cov_mine, 100 * cov_dji))

    # side-by-side full-res crops at shared geo points
    rng = np.random.default_rng(7)
    crops = []
    n_try = 0
    while len(crops) < 6 and n_try < 300:
        n_try += 1
        ex = rng.uniform(oe0 + 60, oe1 - 60)
        ny = rng.uniform(on1 + 60, on0 - 60)
        r_m, c_m = to_rc(ex, ny, g_m, x_m, y_m)
        r_d, c_d = to_rc(ex, ny, g_d, x_d, y_d)
        hC = 500; wC = 500
        if r_m + hC >= mine.shape[0] or c_m + wC >= mine.shape[1]:
            continue
        if r_d + hC >= dji.shape[0] or c_d + wC >= dji.shape[1]:
            continue
        Am = mine[r_m:r_m + hC, c_m:c_m + wC]
        Bm = dji[r_d:r_d + hC, c_d:c_d + wC]
        if Am.sum() == 0 or Bm.sum() == 0:
            continue
        # scale mine (0.050) to dji (0.0562) for fair side-by-side? keep native, label GSD
        side = np.concatenate([Am, np.full((hC, 12, 3), 255, np.uint8), Bm], axis=1)
        cv2.imwrite(os.path.join(OUT, "cmp2_crop%d.jpg" % len(crops)), side,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        crops.append((float(ex), float(ny)))
    print("saved %d comparison crops" % len(crops))

    json.dump({
        "gsd_mine": g_m, "gsd_dji": g_d,
        "cov_mine": cov_mine, "cov_dji": cov_dji,
        "clarity_mine_med": float(np.median(va)), "clarity_mine_std": float(va.std()),
        "clarity_dji_med": float(np.median(vb)), "clarity_dji_std": float(vb.std()),
        "chcorr_R": corr(ar, br), "chcorr_RB": corr(ar, bb),
        "crops": crops,
    }, open(os.path.join(OUT, "metrics2.json"), "w"), indent=1)
    print("verify2 done (%.0fs)" % (time.time() - t0))

if __name__ == "__main__":
    main()
