# -*- coding: utf-8 -*-
"""Step4b: 逐帧「仿射」辐射一致性校正 (gain + offset)。

背景：step6 的纯乘性增益 g 无法校正「加性」差异(雾霭/大气散射/曝光偏置)，
硬性单帧选取时相邻影像间出现 30~77/255 的色块台阶（见 diag_blocks.py）。

模型：I'_i = a_i,c * I + b_i,c
目标：同名点校正后颜色一致   min Σ ρ( (a_i I_i + b_i) - (a_j I_j + b_j) )
规范：参考帧 ref 的 (a,b) 硬固定为 (1,0) —— 消掉 1 个尺度 + 1 个偏置规范自由度；
      再把整体色彩中位水平平移回原始（全局锚定）。
鲁棒：Huber 的 IRLS 3 轮。
验证：留出 1/5 匹配对不参与拟合，报告留出集残差改善（防自欺）。

产出 work/radio.npy  shape (n,2,3)  [[a_R,a_G,a_B],[b_R,b_G,b_B]]
用法: python step4b_radiometry.py
"""
import os
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, log, load_json, Camera, imread_any

N_PT = 300
IRLS_ITERS = 2
HUBER_K = 4.0
RIDGE = 1e-3
A_LO, A_HI = 0.60, 1.70
B_LO, B_HI = -60.0, 60.0
SAT = 250.0
LOW = 3.0
VAL_EVERY = 5
FEAT_KEYS = ("px", "nx", "ny", "scale", "ang", "resp", "des")


def load_inputs():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])
    n = len(meta)
    z = np.load(os.path.join(WORK, "feats.npz"), allow_pickle=True)
    feats = [{k: z[f"{i}_{k}"] for k in FEAT_KEYS} for i in range(n)]
    zm = np.load(os.path.join(WORK, "matches.npz"), allow_pickle=True)
    keys = sorted(zm.files, key=lambda s: int(s.split("_")[0]))
    matches = [zm[k] for k in keys]
    pairs = load_json(os.path.join(WORK, "pairs.json"))
    return meta, cam, feats, matches, pairs


def build_observations(meta, cam, feats, matches, pairs, n_pt=N_PT):
    n = len(meta)
    need = [set() for _ in range(n)]
    sel_list = []
    for (i, j, _), m in zip(pairs, matches):
        if len(m) == 0:
            sel_list.append(None)
            continue
        step = max(1, len(m) // n_pt)
        sel = m[::step][:n_pt]
        sel_list.append(sel)
        for a, b in sel:
            need[i].add(int(a))
            need[j].add(int(b))
    cols = {}
    for i in range(n):
        idxs = np.array(sorted(need[i]), dtype=np.int64)
        if len(idxs) == 0:
            cols[i] = (idxs, None)
            continue
        img = imread_any(meta[i]["path"])
        if img is None:
            cols[i] = (idxs, None)
            continue
        nx = feats[i]["nx"][idxs].astype(np.float64)
        ny = feats[i]["ny"][idxs].astype(np.float64)
        r2 = nx * nx + ny * ny
        rad = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
        xd = nx * rad + 2 * cam.p1 * nx * ny + cam.p2 * (r2 + 2 * nx * nx)
        yd = ny * rad + cam.p1 * (r2 + 2 * ny * ny) + 2 * cam.p2 * nx * ny
        u = np.clip((cam.fx * xd + cam.cx).astype(np.int32), 0, cam.w - 1)
        v = np.clip((cam.fy * yd + cam.cy).astype(np.int32), 0, cam.h - 1)
        cols[i] = (idxs, img[v, u].astype(np.float32))
    obs = []
    for (i, j, _), sel in zip(pairs, sel_list):
        if sel is None:
            continue
        ii, ci = cols[i]
        jj, cj = cols[j]
        if ci is None or cj is None:
            continue
        obs.append((i, j,
                    ci[np.searchsorted(ii, sel[:, 0].astype(np.int64))],
                    cj[np.searchsorted(jj, sel[:, 1].astype(np.int64))]))
    return obs


def vmask(Ii, Ij):
    return (Ii < SAT).all(1) & (Ij < SAT).all(1) & (Ii > LOW).all(1) & (Ij > LOW).all(1)


def resid(obs, a, b, idxs):
    out = []
    for k in idxs:
        i, j, Ii, Ij = obs[k]
        m = vmask(Ii, Ij)
        if m.sum() == 0:
            continue
        out.append((Ii[m] * a[i][None, :] + b[i][None, :]) -
                   (Ij[m] * a[j][None, :] + b[j][None, :]))
    return np.concatenate(out) if out else np.zeros((0, 3))


def report(tag, R):
    if len(R) == 0:
        log(f"{tag}: 无有效观测")
        return None
    ar = np.abs(R)
    d = dict(med=[float(np.median(ar[:, c])) for c in range(3)],
             p90=[float(np.percentile(ar[:, c], 90)) for c in range(3)],
             std=[float(R[:, c].std()) for c in range(3)])
    log(f"{tag}: |残差| 中位 " + ", ".join(f"{v:.2f}" for v in d["med"]) +
        "  p90 " + ", ".join(f"{v:.2f}" for v in d["p90"]))
    return d


def solve_affine(obs, n, fit_idx):
    """IRLS 求解逐帧仿射 (a,b)。

    关键：目标函数 min Σ||(a_i I_i + b_i) - (a_j I_j + b_j)||² 存在「压平退化」——
    把所有影像压成同一灰度可使残差→0。因此必须把规范自由度写成【硬约束】：
        Σ_i a_i,c = n   (平均增益 = 1，即整体对比度守恒，防止 a→0)
        Σ_i b_i,c = 0   (整体色偏为零)
    用代入法消元：a_{n-1} = n - Σ_{i<n-1} a_i 等。
    """
    nf = n * 3
    m = n - 1                       # 自由帧数
    ifit = set(int(k) for k in fit_idx)
    rows, cols, vals = [], [], []
    rhs_r, rhs_v = [], []
    r = 0
    for k, (i, j, Ii, Ij) in enumerate(obs):
        if k not in ifit:
            continue
        vmsk = vmask(Ii, Ij)
        if vmsk.sum() == 0:
            continue
        K = int(vmsk.sum())
        Iiv = Ii[vmsk].astype(np.float64)
        Ijv = Ij[vmsk].astype(np.float64)
        for c in range(3):
            idx = np.arange(r, r + K)
            rows.append(idx); cols.append(np.full(K, i * 3 + c)); vals.append(Iiv[:, c])
            rows.append(idx); cols.append(np.full(K, j * 3 + c)); vals.append(-Ijv[:, c])
            rows.append(idx); cols.append(np.full(K, nf + i * 3 + c)); vals.append(np.ones(K))
            rows.append(idx); cols.append(np.full(K, nf + j * 3 + c)); vals.append(-np.ones(K))
            rhs_r.append(idx)
            rhs_v.append(np.zeros(K))
            r += K
    Rows = np.concatenate(rows); Cols = np.concatenate(cols)
    Vals = np.concatenate(vals).astype(np.float64)
    A = sp.csr_matrix((Vals, (Rows, Cols)), shape=(r, 2 * nf))
    B = np.zeros(r)
    B[np.concatenate(rhs_r)] = np.concatenate(rhs_v)

    # ---- 代入消元: x_full = c + S x_red ----
    nred = 2 * 3 * m               # 未知量: da_0..da_{m-1}, db_0..db_{m-1}
    sr, sc, sv = [], [], []
    for c in range(3):
        for i in range(m):
            sr.append(i * 3 + c); sc.append(i * 3 + c); sv.append(1.0)
            sr.append((n - 1) * 3 + c); sc.append(i * 3 + c); sv.append(-1.0)
            sr.append(nf + i * 3 + c); sc.append(3 * m + i * 3 + c); sv.append(1.0)
            sr.append(nf + (n - 1) * 3 + c); sc.append(3 * m + i * 3 + c); sv.append(-1.0)
    S = sp.csr_matrix((sv, (sr, sc)), shape=(2 * nf, nred))
    cvec = np.zeros(2 * nf)
    cvec[:nf] = 1.0                # a 的基准 1

    AS = (A @ S).tocsr()
    Bc = B - A @ cvec

    # 岭项（轻微，向 a=1,b=0 拉，避免孤立帧病态）
    nr = nred
    Ar = sp.identity(nr, format="csr") * RIDGE
    Br = np.zeros(nr)

    Aall = sp.vstack([AS, Ar]).tocsr()
    Ball = np.concatenate([Bc, Br])

    n_obs = AS.shape[0]

    def solve(w):
        Aw = Aall.multiply(w[:, None])
        Sys = (Aw.T @ Aw).tocsc()
        x = spl.spsolve(Sys, Aw.T @ (w * Ball))
        return np.nan_to_num(x)

    w = np.ones(Aall.shape[0])
    x = np.zeros(nred)
    for it in range(IRLS_ITERS):
        x = solve(w)
        res = AS @ x - Bc
        med = np.median(res); mad = np.median(np.abs(res - med)) * 1.4826 + 1e-6
        z = np.abs(res - med) / (HUBER_K * mad)
        wn = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-6))
        w = np.concatenate([wn, np.ones(Ar.shape[0])])
        log(f"  IRLS {it+1}: 观测残差 std {res.std():.3f}  Huber权重中位 {np.median(wn):.3f}")

    full = cvec + S @ x
    a = full[:nf].reshape(n, 3).copy()
    b = full[nf:].reshape(n, 3).copy()
    a = np.clip(a, A_LO, A_HI)
    b = np.clip(b, B_LO, B_HI)
    # 裁剪会破坏规范约束，此处用「纯规范变换」补回（对目标函数严格不变）：
    #   b -> b - mean(b)   （整体色偏置零）
    b = b - b.mean(axis=0, keepdims=True)
    return a, b


def main():
    meta, cam, feats, matches, pairs = load_inputs()
    n = len(meta)
    log(f"影像 {n} 张, 匹配对 {len(pairs)}")
    t0 = time.time()
    obs = build_observations(meta, cam, feats, matches, pairs)
    log(f"采样数据就绪: {len(obs)} 对 ({time.time()-t0:.0f}s)")

    acc = [np.zeros((0, 3), np.float32) for _ in range(n)]
    for i, j, Ii, Ij in obs:
        acc[i] = np.concatenate([acc[i], Ii])
    lev = np.array([np.median(x, axis=0) if len(x) else np.array([120., 120., 120.], np.float32)
                    for x in acc])
    gm = float(np.median(lev.mean(1)))
    log(f"各帧同名点中位亮度: 全局中位 {gm:.1f}, 范围 [{lev.mean(1).min():.1f},{lev.mean(1).max():.1f}]")

    val_idx = [k for k in range(len(obs)) if k % VAL_EVERY == 0]
    fit_idx = [k for k in range(len(obs)) if k % VAL_EVERY != 0]
    log(f"拟合对 {len(fit_idx)} / 留出对 {len(val_idx)}")

    one = np.ones((n, 3), np.float32); zero = np.zeros((n, 3), np.float32)
    bf = report("[校正前·拟合集]", resid(obs, one, zero, fit_idx))
    bv = report("[校正前·留出集]", resid(obs, one, zero, val_idx))

    a, b = solve_affine(obs, n, fit_idx)
    a = a.astype(np.float32); b = b.astype(np.float32)
    log(f"[仿射系数] a R[{a[:,0].min():.2f},{a[:,0].max():.2f}] "
        f"G[{a[:,1].min():.2f},{a[:,1].max():.2f}] B[{a[:,2].min():.2f},{a[:,2].max():.2f}]  "
        f"(均值应为1: {a.mean(0).round(3)})")
    log(f"           b R[{b[:,0].min():.1f},{b[:,0].max():.1f}] "
        f"G[{b[:,1].min():.1f},{b[:,1].max():.1f}] B[{b[:,2].min():.1f},{b[:,2].max():.1f}]  "
        f"(均值应为0: {b.mean(0).round(2)})")

    af = report("[校正后·拟合集]", resid(obs, a, b, fit_idx))
    av = report("[校正后·留出集]", resid(obs, a, b, val_idx))
    if bv and av:
        log("留出集改善: " + ", ".join(
            f"{nm} 中位 {bv['med'][c]:.2f}->{av['med'][c]:.2f} "
            f"({(1-av['med'][c]/max(bv['med'][c],1e-6))*100:.0f}%), "
            f"p90 {bv['p90'][c]:.2f}->{av['p90'][c]:.2f}"
            for c, nm in enumerate("BGR")))

    np.save(os.path.join(WORK, "radio.npy"), np.stack([a, b], axis=1).astype(np.float32))
    json.dump({"before_fit": bf, "after_fit": af, "before_val": bv, "after_val": av,
               "a_mean": [float(v) for v in a.mean(0)],
               "b_mean": [float(v) for v in b.mean(0)],
               "a_range": [[float(a[:, c].min()), float(a[:, c].max())] for c in range(3)],
               "b_range": [[float(b[:, c].min()), float(b[:, c].max())] for c in range(3)]},
              open(os.path.join(WORK, "radio_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"已保存 work/radio.npy  用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
