"""Step3: SIFT 特征提取（并行）+ 邻接对几何验证匹配。"""
import os
import sys
import numpy as np
import cv2
from common import WORK, log, save_json, load_json, Camera, imread_any

SIFT_MAX = 9000
DOWNSCALE = 2          # 在 1/2 分辨率上提特征
RATIO = 0.80


def _extract(args):
    path, cam_params, downscale, nmax = args
    w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2 = cam_params
    cam = Camera(w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    img = imread_any(path, gray=True)
    if img is None:
        return None
    H0, W0 = img.shape[:2]
    if downscale > 1:
        img = cv2.resize(img, (W0 // downscale, H0 // downscale), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=nmax, contrastThreshold=0.04, edgeThreshold=12, sigma=1.6)
    kp, des = sift.detectAndCompute(img, None)
    if not kp:
        return None
    pts = np.array([k.pt for k in kp], dtype=np.float64) * downscale   # 回到原图像素坐标
    # 去畸变 -> 归一化平面坐标（无畸变相机系）
    pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array([k1, k2, p1, p2, k3])
    und = cv2.undistortPoints(pts_in, K, D, None, K)   # 仍为像素坐标（无畸变）
    und = und.reshape(-1, 2)
    # 归一化坐标
    nx = (und[:, 0] - cx) / fx
    ny = (und[:, 1] - cy) / fy
    scale = np.array([k.size for k in kp], dtype=np.float32) * downscale
    ang = np.array([k.angle for k in kp], dtype=np.float32)
    resp = np.array([k.response for k in kp], dtype=np.float32)
    return (und, nx, ny, scale, ang, resp, des)


def _match_one(args):
    """返回内点匹配 (n,2) 的索引对，及内点数"""
    i, j, fi, fj, ratio = args
    di, dj = fi["des"], fj["des"]
    if di is None or dj is None or len(di) < 8 or len(dj) < 8:
        return np.zeros((0, 2), dtype=np.int32)
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=64))
    try:
        knn = flann.knnMatch(di, dj, k=2)
    except cv2.error:
        return np.zeros((0, 2), dtype=np.int32)
    good = []
    src = []
    dst = []
    for m in knn:
        if len(m) < 2:
            continue
        m1, m2 = m[0], m[1]
        if m1.distance < ratio * m2.distance:
            src.append(m1.queryIdx)
            dst.append(m1.trainIdx)
    if len(src) < 12:
        return np.zeros((0, 2), dtype=np.int32)
    src = np.array(src, dtype=np.int32)
    dst = np.array(dst, dtype=np.int32)
    pi = fi["px"][src].astype(np.float64)   # 无畸变像素坐标
    pj = fj["px"][dst].astype(np.float64)

    best_inl = np.zeros((0, 2), dtype=np.int32)
    # 同时尝试 F 与 H，取内点多者
    for method, thresh in (("F", 3.0), ("H", 5.0)):
        if method == "F":
            M, mask = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, thresh, 0.9995)
            if M is None or M.shape != (3, 3):
                continue
            mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(pi), bool)
            # 对称检验
            M2, mask2 = cv2.findFundamentalMat(pj, pi, cv2.FM_RANSAC, thresh, 0.9995)
            if mask2 is not None:
                mask &= mask2.ravel().astype(bool)
        else:
            M, mask = cv2.findHomography(pi, pj, cv2.RANSAC, thresh, maxIters=5000, confidence=0.9995)
            if M is None:
                continue
            mask = mask.ravel().astype(bool)
        if mask.sum() > len(best_inl):
            best_inl = np.stack([src[mask], dst[mask]], axis=1).astype(np.int32)
        if len(best_inl) > 400:
            break
    return best_inl


def main():
    from concurrent.futures import ProcessPoolExecutor
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam_params = (cj["w"], cj["h"], *cj["init"])

    cache_f = os.path.join(WORK, "feats.npz")
    if os.path.exists(cache_f):
        log("读取缓存特征")
        z = np.load(cache_f, allow_pickle=True)
        feats = [{k: z[f"{i}_{k}"] for k in ("px", "nx", "ny", "scale", "ang", "resp", "des")}
                 for i in range(len(meta))]
    else:
        args = [(m["path"], cam_params, DOWNSCALE, SIFT_MAX) for m in meta]
        feats = []
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
            for r in ex.map(_extract, args, chunksize=2):
                feats.append(dict(px=r[0].astype(np.float32), nx=r[1].astype(np.float32),
                                  ny=r[2].astype(np.float32), scale=r[3], ang=r[4],
                                  resp=r[5], des=r[6]))
        log(f"特征提取完成: 平均 {np.mean([len(f['des']) for f in feats]):.0f} 点/张")
        d = {}
        for i, f in enumerate(feats):
            d[f"{i}_px"] = np.asarray(f["px"], dtype=np.float32)
            d[f"{i}_nx"] = np.asarray(f["nx"], dtype=np.float32)
            d[f"{i}_ny"] = np.asarray(f["ny"], dtype=np.float32)
            d[f"{i}_scale"] = np.asarray(f["scale"])
            d[f"{i}_ang"] = np.asarray(f["ang"])
            d[f"{i}_resp"] = np.asarray(f["resp"])
            d[f"{i}_des"] = np.asarray(f["des"])
        np.savez_compressed(cache_f, **d)

    pairs = load_json(os.path.join(WORK, "pairs.json"))
    cache_m = os.path.join(WORK, "matches.npz")
    if os.path.exists(cache_m):
        log("读取缓存匹配")
        z = np.load(cache_m, allow_pickle=True)
        keys = sorted(z.files, key=lambda s: int(s.split("_")[0]))
        matches = [z[k] for k in keys]
    else:
        margs = [(i, j, feats[i], feats[j], RATIO) for i, j, _ in pairs]
        matches = []
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
            for r in ex.map(_match_one, margs, chunksize=4):
                matches.append(r)
        np.savez_compressed(cache_m, **{f"{k}_m": v for k, v in enumerate(matches)})
        log(f"匹配完成: {len(matches)} 对, 内点中位数 "
            f"{np.median([len(m) for m in matches]):.0f}, "
            f"总匹配 {sum(len(m) for m in matches)}")
        zero = sum(1 for m in matches if len(m) < 20)
        log(f"匹配失败(<20内点)的对数: {zero} / {len(matches)}")


if __name__ == "__main__":
    main()
