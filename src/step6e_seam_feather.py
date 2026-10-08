# -*- coding: utf-8 -*-
"""Step6e: 缝合带羽化 —— 抹平 ~1px BA 残余错位留下的"台阶"。

接缝线优化（step6d）把缝合线挪到了低梯度路径，但相邻影像之间约 1px 的残余
几何错位仍会在缝线上留下一个"刀口"。本步只在缝合带（±DIL 像素）内做 MED×MED
中值，把台阶抹平；**带外一个像素都不动**，因此锐度不受影响。
影响宽度约 (MED-1) 像素 ≈ 0.28m。

用法: python step6e_seam_feather.py
产出 out/ortho_rgb_cut_f.npy
"""
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, log

BLK = 2048
HALO = 16
MED = int(os.environ.get("MED", "5"))
DIL = int(os.environ.get("DIL", "1"))


def main():
    SF = os.path.join(OUT, "ortho_rgb_cut.npy")
    VI = os.path.join(OUT, "ortho_vidx_cut.npy")
    VM = os.path.join(OUT, "ortho_valid.npy")
    DST = os.path.join(OUT, "ortho_rgb_cut_f.npy")
    sf = np.load(SF, mmap_mode="r")
    vi = np.load(VI, mmap_mode="r")
    vm = np.load(VM, mmap_mode="r")
    H, W = sf.shape[:2]
    log(f"缝合带羽化: {W}x{H}, DIL={DIL} MED={MED} (影响宽度 ~{(MED-1)*0.055:.2f}m)")
    out = np.lib.format.open_memmap(DST, mode="w+", dtype=np.uint8, shape=(H, W, 3))
    k = np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)
    t0 = time.time()
    nseam = 0
    for y0 in range(0, H, BLK):
        for x0 in range(0, W, BLK):
            y1, x1 = min(y0 + BLK, H), min(x0 + BLK, W)
            ry0, ry1 = max(0, y0 - HALO), min(H, y1 + HALO)
            rx0, rx1 = max(0, x0 - HALO), min(W, x1 + HALO)
            img = np.ascontiguousarray(sf[ry0:ry1, rx0:rx1])
            v = np.ascontiguousarray(vi[ry0:ry1, rx0:rx1])
            m = np.ascontiguousarray(vm[ry0:ry1, rx0:rx1])
            seam = np.zeros_like(m)
            a_ = (v[:, 1:] != v[:, :-1]) & m[:, 1:] & m[:, :-1]
            seam[:, 1:] |= a_; seam[:, :-1] |= a_
            b_ = (v[1:, :] != v[:-1, :]) & m[1:, :] & m[:-1, :]
            seam[1:, :] |= b_; seam[:-1, :] |= b_
            R = cv2.dilate(seam.astype(np.uint8), k).astype(bool) & m
            nseam += int(R.sum())
            if R.any():
                med = cv2.medianBlur(img, MED)
                o = np.where(R[..., None], med, img)
            else:
                o = img
            cy, cx = y0 - ry0, x0 - rx0
            out[y0:y1, x0:x1] = o[cy:cy + (y1 - y0), cx:cx + (x1 - x0)]
        log(f"  行 {y0//BLK+1} ({time.time()-t0:.0f}s)")
    out.flush()
    del out
    log(f"完成: 羽化像素 {nseam} ({nseam/(H*W)*100:.2f}%), 用时 {time.time()-t0:.0f}s")
    log(f"已写 {DST}")


if __name__ == "__main__":
    main()
