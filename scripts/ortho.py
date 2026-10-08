import os, sys, pickle, json, struct, collections, time
import numpy as np
import cv2
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
from config import (CACHE, SRC, get_intrinsics_k, K1, K2, K3, P1, P2, load_meta, to_local,
                    IMG_W, IMG_H, camera_from_gimbal)

ORTHO_GSD = 0.056   # m/px, match DJI
KATCH = 12          # images sampled per tile
IMG_CACHE = 10      # max full-res images kept in RAM at once

def dsm_lookup(meta, E, N):
    """bilinear sample DSM elevation at (E,N) arrays (broadcastable). Returns shape, NaN outside."""
    E, N = np.broadcast_arrays(E, N)
    grid = meta["grid"]; E0 = meta["E0"]; N0 = meta["N0"]; cell = meta["cell"]
    rows, cols = grid.shape
    c = (E - E0) / cell
    r = (N - N0) / cell
    c0 = np.floor(c).astype(np.int64); r0 = np.floor(r).astype(np.int64)
    fc = c - c0; fr = r - r0
    ok = (c0 >= 0) & (c0 < cols-1) & (r0 >= 0) & (r0 < rows-1)
    out = np.zeros(E.shape, np.float64)
    for i, j, w in [(0,0,(1-fr)*(1-fc)), (0,1,(1-fr)*fc), (1,0,fr*(1-fc)), (1,1,fr*fc)]:
        rr = np.clip(r0 + i, 0, rows-1); cc = np.clip(c0 + j, 0, cols-1)
        out += w * np.where(ok, grid[rr, cc], 0.0)
    out[~ok] = np.nan
    return out

def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR

class ImageBank:
    """LRU cache of full-res images, scaled by an exposure gain to homogenize tone."""
    def __init__(self, local, gain):
        self.local = local
        self.gain = gain          # (N,3) per-channel multipliers
        self.cache = collections.OrderedDict()

    def get(self, i):
        img = self.cache.get(i)
        if img is None:
            arr = read_image(os.path.join(SRC, self.local[i]["name"])).astype(np.float32)
            img = arr * self.gain[i][None, None, :]
            self.cache[i] = img
            while len(self.cache) > IMG_CACHE:
                self.cache.popitem(last=False)
        return img

def exposure_factors(local):
    """Per-image per-channel gain to equalize brightness (computed on half-res reads)."""
    N = len(local)
    means = np.zeros((N, 3), np.float32)
    for i in range(N):
        arr = cv2.imdecode(np.fromfile(os.path.join(SRC, local[i]["name"]), dtype=np.uint8),
                           cv2.IMREAD_REDUCED_COLOR_2)
        h, w = arr.shape[:2]
        y0, y1 = int(h*0.25), int(h*0.75); x0, x1 = int(w*0.25), int(w*0.75)
        means[i] = arr[y0:y1, x0:x1].astype(np.float32).reshape(-1, 3).mean(0)
    ref = np.median(means, axis=0)
    gain = np.clip(ref[None, :] / np.maximum(means, 1.0), 0.72, 1.45)
    return gain.astype(np.float32)

def _proj_px(Xp, C, R, K):
    """Xp (P,3) world -> (u,v,valid) in full-res pixels."""
    xc = (Xp - C[None, :]) @ R.T
    z = xc[:, 2]; ok = z > 1e-6
    zz = np.maximum(z, 1e-6)
    x = np.where(ok, xc[:, 0] / zz, 0.0); y = np.where(ok, xc[:, 1] / zz, 0.0)
    r2 = x*x + y*y
    rd = 1 + K1*r2 + K2*r2**2 + K3*r2**3
    u = K[0,0]*x*rd + K[0,2]
    v = K[1,1]*y*rd + K[1,2]
    return u, v, ok

def _sampl(red, u, v, ok):
    """Bilinear sample reduced image at full-res pixel coords (u,v). Returns (P,3), NaN invalid."""
    S = red.shape[0] / IMG_H        # reduced-scale factor
    ur = u * S; vr = v * S
    h, w = red.shape[:2]
    i = np.floor(vr).astype(np.int64); j = np.floor(ur).astype(np.int64)
    fi = np.clip(vr - i, 0, 1); fj = np.clip(ur - j, 0, 1)
    inside = ok & (j >= 0) & (j < w-1) & (i >= 0) & (i < h-1)
    jj = np.clip(j, 0, w-2); ii = np.clip(i, 0, h-2)
    cols = np.stack([
        red[ii, jj, c]*(1-fj)*(1-fi) + red[ii, jj+1, c]*fj*(1-fi) +
        red[ii+1, jj, c]*(1-fj)*fi + red[ii+1, jj+1, c]*fj*fi
        for c in range(3)], 1)
    return np.where(inside[:, None], cols, np.nan)

def exposure_balance(local, Cs, Rs, K, hg, sc=0.25):
    """Overlap-based pairwise radiometric balance. Returns per-image per-channel gain (N,3).
    Solves global per-channel log-gains so overlapping images match, removing flight-line striping."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr
    N = len(local)
    red = {}
    for i in range(N):
        arr = cv2.resize(read_image(os.path.join(SRC, local[i]["name"])).astype(np.float32),
                         (int(round(IMG_W*sc)), int(round(IMG_H*sc))), interpolation=cv2.INTER_AREA)
        red[i] = arr
    cE = np.array([r["E"] for r in local]); cN = np.array([r["N"] for r in local])
    Zg = hg * np.ones(N)
    farr = K[0, 0]
    r_radius = np.hypot(0.5*IMG_W/farr, 0.5*IMG_H/farr) * (Cs[:, 2] - Zg)
    # neighbor pairs
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cE[i]-cE[j], cN[i]-cN[j])
            if d < 2.05 * max(r_radius[i], r_radius[j]):
                edges.append((i, j))
    gain = np.ones((N, 3), np.float32)
    A = lil_matrix((len(edges), N), dtype=np.float64)
    for ch in range(3):
        b = np.zeros(len(edges), np.float64); wgt = np.zeros(len(edges))
        for e, (i, j) in enumerate(edges):
            mE = 0.5*(cE[i]+cE[j]); mN = 0.5*(cN[i]+cN[j])
            xss = mE + np.linspace(-0.35, 0.35, 7)[:, None]*r_radius[i]
            yss = mN + np.linspace(-0.35, 0.35, 7)[None, :]*r_radius[i]
            XX, YY = np.broadcast_arrays(xss, yss)
            Xp = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, hg)], 1).astype(np.float64)
            u_i, v_i, ok_i = _proj_px(Xp, Cs[i], Rs[i], K)
            u_j, v_j, ok_j = _proj_px(Xp, Cs[j], Rs[j], K)
            ci = _sampl(red[i], u_i, v_i, ok_i); cj = _sampl(red[j], u_j, v_j, ok_j)
            both = np.isfinite(ci[:, ch]) & np.isfinite(cj[:, ch])
            if both.sum() < 8:
                continue
            rr = np.median(cj[both, ch] / np.maximum(ci[both, ch], 1.0))
            if not np.isfinite(rr) or rr <= 0:
                continue
            A[e, i] = -1.0; A[e, j] = 1.0
            b[e] = np.log(rr); wgt[e] = min(both.sum(), 500.0)
        # drop zero-weight rows
        rows = wgt > 0
        Ai = A.tocsr()[rows]; bi = b[rows]; wi = wgt[rows]
        # anchor: pin first image to 0
        import scipy.sparse as sp
        Ai = sp.vstack([Ai.multiply(wi[:, None]), sp.csr_matrix([[1.0] + [0.0]*(N-1)])])
        bi = np.concatenate([bi*wi, [0.0]])
        sol = lsqr(Ai, bi, atol=1e-9, btol=1e-9, iter_lim=1000)[0]
        sol = sol - np.median(sol)          # neutral tone
        gain[:, ch] = np.clip(np.exp(sol), 0.62, 1.62)
    return red, gain.astype(np.float32)

def build_poses(N):
    with open(os.path.join(CACHE, "sfm_cam.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["Cs"][:N], d["Rs"][:N]

def project_blend(Xg, C, R, img, K, D):
    """Xg: (H,W,3) ground E,N,Z. Project to image, sample color + center-favouring weight.
    Returns color float32 (H,W,3) and weight (H,W). Speedy: cv2.remap bilinear + float32."""
    H, W, _ = Xg.shape
    Xf = np.ascontiguousarray(Xg, np.float32)
    xc = (Xf - C.reshape(1,1,3).astype(np.float32)) @ R.T.astype(np.float32)
    z = xc[:,:,2]
    okz = z > 1e-6
    x = np.zeros_like(z); y = np.zeros_like(z)
    np.divide(xc[:,:,0], z, out=x, where=okz)
    np.divide(xc[:,:,1], z, out=y, where=okz)
    r2 = x*x + y*y
    radial = 1 + K1*r2 + K2*r2**2 + K3*r2**3
    xd = x*radial + 2*P1*x*y + P2*(r2 + 2*x*x)
    yd = y*radial + 2*P2*x*y + P1*(r2 + 2*y*y)
    u = (K[0,0]*xd + K[0,2]).astype(np.float32)
    v = (K[1,1]*yd + K[1,2]).astype(np.float32)
    img_H, img_W = img.shape[:2]
    inside = okz & (u >= 0) & (u <= img_W-1) & (v >= 0) & (v <= img_H-1)
    mapx = np.where(inside, u, -10000.0).astype(np.float32)
    mapy = np.where(inside, v, -10000.0).astype(np.float32)
    color = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    d2 = (u - K[0,2])**2 + (v - K[1,2])**2
    sigma = 0.55 * min(img_W, img_H)
    cw = (sigma*sigma) / (sigma*sigma + d2)
    cw = np.power(cw, 1.5)
    wmask = np.where(inside, cw, 0.0)
    return color, wmask

def write_geotiff(path, rgb, covmask, g, E_tl, N_tl, gsd=None):
    gsd = g if gsd is None else gsd
    keydir = struct.pack('<16H', 1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1,
                         3072, 0, 1, 32650)
    extratags = [
        (33550, 'd', 3, (gsd, gsd, 0.0), True),
        (33922, 'd', 6, (0, 0, 0, float(E_tl), float(N_tl), 0.0), True),
        (34735, 'H', 16, keydir, True),
    ]
    alpha = np.where(covmask > 0, 255, 0).astype(np.uint8)
    rga = np.concatenate([rgb, alpha[..., None]], axis=-1)
    tifffile.imwrite(path, rga, photometric='rgb', compression='deflate',
                     extratags=extratags)

def main():
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else None
    local, origin = to_local(load_meta())
    if subset is not None:
        local = local[:subset]
    N = len(local)
    Cs, Rs = build_poses(N)
    with open(os.path.join(CACHE, "dsm_meta.json")) as f:
        jm = json.load(f)
    dsm_grid = np.load(os.path.join(CACHE, "dsm.npy"))
    hg = float(np.nanmedian(dsm_grid))
    K = get_intrinsics_k()
    t0 = time.time()
    _red, gain = exposure_balance(local, Cs, Rs, K, hg)
    del _red
    print("exposure balance time %.1fs, hg=%.1f, sample gains:" % (time.time()-t0, hg), np.round(gain[:5], 2).tolist())
    bank = ImageBank(local, gain)

    dsm = {"grid": dsm_grid, "E0": jm["E0"], "N0": jm["N0"], "cell": jm["cell"]}
    rows, cols = dsm["grid"].shape
    Emin, Nmin = dsm["E0"], dsm["N0"]
    Emax = Emin + (cols-1)*dsm["cell"]; Nmax = Nmin + (rows-1)*dsm["cell"]
    g = ORTHO_GSD
    ocols = int(np.ceil((Emax - Emin) / g)) + 1
    orows = int(np.ceil((Nmax - Nmin) / g)) + 1
    print("ortho dims:", orows, "x", ocols, " GSD", g, "n_cam", N)

    cE = np.array([r["E"] for r in local]); cN = np.array([r["N"] for r in local])
    D = np.array([K1, K2, P1, P2, K3], float)

    TILE = 2048
    M = np.zeros((orows, ocols, 3), np.uint8)
    Mw = np.zeros((orows, ocols), np.float32)
    nT = 0
    for r0 in range(0, orows, TILE):
        for c0 in range(0, ocols, TILE):
            r1 = min(r0+TILE, orows); c1 = min(c0+TILE, ocols)
            Ht, Wt = r1-r0, c1-c0
            gx = c0 + np.arange(Wt)[None, :] + 0.5
            gy = r0 + np.arange(Ht)[:, None] + 0.5
            E = Emin + gx*g; N = Nmax - gy*g
            E, N = np.broadcast_arrays(E.astype(np.float32), N.astype(np.float32))
            Z = dsm_lookup(dsm, E, N)
            ok = np.isfinite(Z)
            if not ok.any():
                nT += 1; continue
            Ec = E.mean(); Nc = N.mean()
            dist = np.hypot(cE - Ec, cN - Nc)
            order = np.argsort(dist)
            cands = [int(i) for i in order if dist[i] < 260.0][:KATCH]
            acc = np.zeros((Ht, Wt, 3), np.float32); accw = np.zeros((Ht, Wt), np.float32)
            bestw = np.full((Ht, Wt), -1.0, np.float32)
            bestc = np.zeros((Ht, Wt, 3), np.float32)
            Xg = np.stack([E, N, Z], axis=-1).astype(np.float64)
            for i in cands:
                color, w = project_blend(Xg, Cs[i], Rs[i], bank.get(i), K, D)
                if not (w > 0).any():
                    continue
                acc += color * w[..., None]; accw += w
                upd = w > bestw
                bestc = np.where(upd[..., None], color, bestc)
                bestw = np.maximum(bestw, w)
            cov = accw > 0
            denom = np.where(accw > 0, accw, 1.0)
            color_lo = acc / denom[..., None]
            def nblur(A, mask, ss):
                Mb = cv2.GaussianBlur(mask, (0, 0), ss)
                B = cv2.GaussianBlur(A, (0, 0), ss)
                return np.where((Mb > 1e-4)[..., None], B / np.maximum(Mb, 1e-6)[..., None], 0.0)
            sigma = 30.0
            cmask = cov.astype(np.float32)
            lo = nblur(color_lo, cmask, sigma)
            bestc = np.where(cov[..., None], bestc, 0.0)
            hib = bestc - nblur(bestc, cmask, sigma)
            out = lo + hib
            out[~cov] = 0
            M[r0:r1, c0:c1] = np.clip(out, 0, 255).astype(np.uint8)
            Mw[r0:r1, c0:c1] = accw
            nT += 1
    print("tiles done", nT)
    np.save(os.path.join(CACHE, "ortho_M.npy"), M)
    np.save(os.path.join(CACHE, "ortho_Mw.npy"), Mw)
    cv2.imwrite(os.path.join(r"D:\lr\project\无人机航拍拼接\mysolution\out", "ortho_preview.png"),
                cv2.resize(M, (max(1,ocols//4), max(1,orows//4)), interpolation=cv2.INTER_AREA))
    print("saved preview; coverage px:", int((Mw>0).sum()))
    try:
        E_tl = Emin + origin[0]; N_tl = Nmax + origin[1]
        out_gt = os.path.join(r"D:\lr\project\无人机航拍拼接\mysolution\out", "mosaic_geotiff.tif")
        write_geotiff(out_gt, M, Mw, g, E_tl, N_tl)
        print("GeoTIFF saved:", out_gt)
    except Exception as _e:
        print("geotiff fail", repr(_e))

if __name__ == "__main__":
    main()