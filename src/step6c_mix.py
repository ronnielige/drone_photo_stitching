# -*- coding: utf-8 -*-
"""Step6c: 低频色调共识混合。

问题：纯硬性单帧选取(v2)虽最锐，但相邻影像间残留的亮度/色调差(雾霭、曝光)
被硬切换放大成「扇形/条带色块」，肉眼极其明显（见 haze_zone.jpg）。

思路（细节=单帧，色调=共识）：
    out = sf + ALPHA * ( LP(multi-view avg) - LP(sf) )
其中 LP 是尺度为 LF_M 米的高斯低通。
 - 高频(细节)完全来自 sf —— 逐像素单帧，不做任何跨帧平均，因此不模糊；
 - 低频(色调)来自多视几何加权平均 —— 该场的权重在空间上连续变化，
   所以色调在空间上连续，块边界消失。
数学上：对阶跃色块，LP(sf) 含阶跃而 LP(avg) 不含 → 差值正好把阶跃抵消，
       块内部不变，只在边界 ±2σ 内做补偿，故无残留台阶。

边界处理：按 HALO 重叠分块，权重归一化（只在有效区做低通），避免空洞污染与瓦片缝。
用法: python step6c_mix.py [ALPHA]
产出 out/ortho_rgb3.npy + out/ortho_meta3.json
"""
import os
import sys
import json
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, WORK

SF = os.environ.get("MIX_SF", os.path.join(OUT, "ortho_rgb.npy"))
AV = os.path.join(OUT, "ortho_avg.npy")
VM = os.path.join(OUT, "ortho_valid.npy")
DST = os.environ.get("MIX_DST", os.path.join(OUT, "ortho_rgb3.npy"))

LF_M = float(os.environ.get("LF_M", "2.5"))       # 低通尺度(米)；实测 2.5m 色块抑制好且不损锐度
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else float(os.environ.get("ALPHA", "1.0"))
BLK = 1536
RADIO = os.path.join(WORK, "radio.npy")


def renormal_gain():
    """仿射辐射校正的 a 均值因裁剪偏离 1，等价于整图被缩放。
    这里除以 per-channel mean(a) 精确还原（=把全局增益锚定回 1，色偏/亮度回到原始水平）。"""
    if not os.path.exists(RADIO):
        return np.ones(3, np.float32)
    r = np.load(RADIO)
    k = r[:, 0, :].mean(0)
    print(f"全局增益锚定: mean(a)={k.round(4)} -> 缩放 {1.0/k.round(4)}")
    return (1.0 / k).astype(np.float32)


def main():
    sf = np.load(SF, mmap_mode="r")
    av = np.load(AV, mmap_mode="r")
    vm = np.load(VM, mmap_mode="r")
    H, W = sf.shape[:2]
    meta = json.load(open(os.path.join(OUT, "ortho_meta.json"), encoding="utf-8"))
    gsd = meta["gsd"]
    sigma = LF_M / gsd
    halo = int(3 * sigma) + 8
    print(f"图像 {W}x{H} @{gsd*100:.2f}cm; 低通 {LF_M:.2f}m -> sigma {sigma:.1f}px; "
          f"ALPHA={ALPHA}; 分块 {BLK} 带 halo {halo}")

    out = np.lib.format.open_memmap(DST, mode="w+", dtype=np.uint8, shape=(H, W, 3))
    kn = renormal_gain().reshape(1, 1, 3)
    t0 = time.time()
    ny = (H + BLK - 1) // BLK
    nx = (W + BLK - 1) // BLK
    k = int(2 * round(3 * sigma) + 1)
    for iy, y0 in enumerate(range(0, H, BLK)):
        for x0 in range(0, W, BLK):
            y1 = min(y0 + BLK, H); x1 = min(x0 + BLK, W)
            ry0 = max(0, y0 - halo); ry1 = min(H, y1 + halo)
            rx0 = max(0, x0 - halo); rx1 = min(W, x1 + halo)
            s = np.ascontiguousarray(sf[ry0:ry1, rx0:rx1]).astype(np.float32) * kn
            a = np.ascontiguousarray(av[ry0:ry1, rx0:rx1]).astype(np.float32) * kn
            w = np.ascontiguousarray(vm[ry0:ry1, rx0:rx1]).astype(np.float32)[..., None]
            ws = cv2.GaussianBlur(w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
            ws = np.maximum(ws.reshape(w.shape) if ws.ndim == 2 else ws, 1e-6)
            ls = cv2.GaussianBlur(s * w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE) / ws
            la = cv2.GaussianBlur(a * w, (k, k), sigma, borderType=cv2.BORDER_REPLICATE) / ws
            o = s + ALPHA * (la - ls)
            o = np.clip(o, 0, 255).astype(np.uint8)
            cy, cx = y0 - ry0, x0 - rx0
            out[y0:y1, x0:x1] = o[cy:cy + (y1 - y0), cx:cx + (x1 - x0)]
        print(f"  行 {iy+1}/{ny} ({time.time()-t0:.0f}s)")
    out.flush()
    del out
    m3 = dict(meta)
    m3.update({"source": "step6c_mix", "lf_m": LF_M, "alpha": ALPHA})
    json.dump(m3, open(os.path.join(OUT, "ortho_meta3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已写 {DST}  用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
