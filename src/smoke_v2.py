# -*- coding: utf-8 -*-
"""DSM v2 单视图冒烟测试：验证屋顶削顶是否修复、单视图耗时。"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, load_json
from step5c_dsm_refine import _sweep_one_v2, Z_SAT

meta = load_json(os.path.join(WORK, "meta.json"))
cj = load_json(os.path.join(WORK, "camera.json"))
cam_p = (cj["w"], cj["h"], *cj["init"])
paths = [m["path"] for m in meta]
Rs = np.load(os.path.join(WORK, "ba_R.npy"))
Cs = np.load(os.path.join(WORK, "ba_C.npy"))
dmeta = dict(load_json(os.path.join(WORK, "dsm_meta.json")))

prior = np.load(os.path.join(WORK, "dsm.npy")).astype(np.float32)
n_sat = int((prior >= Z_SAT).sum())
prior[prior >= Z_SAT] = np.float32(157.0)
dmeta["_z0"] = float(np.nanmedian(prior))
P = os.path.join(WORK, "dsm_prior_v2.npy")
np.save(P, prior)
print(f"先验: 饱和格 {n_sat} ({n_sat/prior.size*100:.2f}%), 中位 {dmeta['_z0']:.2f}")

# 建筑点（诊断中确认的真建筑簇）
E, N = 562543.0, 3324779.0
cands = []
for i in range(len(meta)):
    if -Rs[i][2, 2] <= 0.85:
        continue
    d = np.linalg.norm(Cs[i][:2] - [E, N])
    if d < 250:
        cands.append((d, i))
cands.sort()
ri = cands[0][1]
print(f"参考视图 #{ri}, 水平距 {cands[0][0]:.0f}m, 附近视图 {len(cands)}")

pairs = load_json(os.path.join(WORK, "pairs.json"))
adj = {}
for i, j, ov in pairs:
    adj.setdefault(i, []).append((j, ov))
    adj.setdefault(j, []).append((i, ov))
nb = [j for j, _ in sorted(adj.get(ri, []), key=lambda t: -t[1])[:3]]
print("邻视", nb)

t0 = time.time()
ri2, G, conf = _sweep_one_v2((ri, cam_p, nb, Rs, Cs, P, dmeta, paths))
dt = time.time() - t0
print(f"单视图耗时 {dt:.1f}s -> 117 视图 / 14 进程 ≈ {dt*117/14/60:.1f} min")

p = G.reshape(-1, 3)
cf = conf.reshape(-1)
g = dmeta["gsd"]
m = (cf > 0.30) & (np.abs(p[:, 0] - E) < 15) & (np.abs(p[:, 1] - N) < 15)
z = p[m, 2]
print(f"建筑 30m 窗口: 点数 {m.sum()}, z p10 {np.percentile(z,10):.2f} "
      f"p50 {np.percentile(z,50):.2f} p90 {np.percentile(z,90):.2f} "
      f"p99 {np.percentile(z,99):.2f} max {z.max():.2f}")
xs = np.arange(prior.shape[1]) * g + dmeta["x0"]
ys = dmeta["y0"] - np.arange(prior.shape[0]) * g
wm = (np.abs(xs - E) < 15)[None, :] & (np.abs(ys - N) < 15)[:, None]
zp = prior[wm]
print("  旧 DSM 同窗口: p50 %.2f p90 %.2f p99 %.2f max %.2f  (饱和占比 %.1f%%)" % (
    np.nanpercentile(zp, 50), np.nanpercentile(zp, 90), np.nanpercentile(zp, 99),
    np.nanmax(zp), (zp >= Z_SAT).mean() * 100))
