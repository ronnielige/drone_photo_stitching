# -*- coding: utf-8 -*-
"""Step5f: 曝光增益均衡（乘性），产出 work/gains.npy。

从特征匹配的「同名点亮度比」中解出每张影像的 R/G/B 乘性增益，
再归一到几何均值 1（不改变整体亮度）。这是 v3 之前的粗粒度色彩均衡；
v3/v4 主链路用更精细的逐帧仿射校正（step4b_radiometry.py 的 radio.npy），
但 step6b 仍要求 gains.npy 存在（作为 radio 缺失时的回退）。

前置：step2(meta/camera/pairs) + step3(feats/matches) + step4(ba_R/ba_C)
产出：work/gains.npy  (n,3) float32
用法：python step5f_gains.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, log, load_json, Camera
from step6_ortho import estimate_gains


def main():
    gc_path = os.path.join(WORK, "gains.npy")
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    n = len(meta)

    z = np.load(os.path.join(WORK, "feats.npz"), allow_pickle=True)
    feats = [{k: z[f"{i}_{k}"] for k in ("px", "nx", "ny", "scale", "ang", "resp", "des")}
             for i in range(n)]
    zm = np.load(os.path.join(WORK, "matches.npz"), allow_pickle=True)
    keys = sorted(zm.files, key=lambda s: int(s.split("_")[0]))
    matches = [zm[k] for k in keys]
    pairs = load_json(os.path.join(WORK, "pairs.json"))

    t0 = time.time()
    gains, ratios = estimate_gains(meta, feats, matches, pairs, cam)
    gains = np.clip(gains, 0.7, 1.45)
    gains = gains / np.exp(np.mean(np.log(gains), axis=0))[None, :]
    np.save(gc_path, gains.astype(np.float32))
    log(f"曝光增益完成 ({time.time()-t0:.0f}s): "
        f"R[{gains[:,0].min():.3f},{gains[:,0].max():.3f}] "
        f"G[{gains[:,1].min():.3f},{gains[:,1].max():.3f}] "
        f"B[{gains[:,2].min():.3f},{gains[:,2].max():.3f}]  -> {gc_path}")


if __name__ == "__main__":
    main()
