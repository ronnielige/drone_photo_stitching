import os, sys, pickle
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE, IMG_W, IMG_H

DSM_CELL = 0.5   # m per DSM cell

def build_dsm():
    with open(os.path.join(CACHE, "tracks.pkl"), "rb") as f:
        d = pickle.load(f)
    Xs = d["Xs"]; valid = d["valid"]
    pts = Xs[valid]
    print("valid points for DSM:", len(pts))
    if len(pts) < 4:
        return None
    E = pts[:, 0]; N = pts[:, 1]; H = pts[:, 2]
    # robust band: drop extreme outliers using median + k*MAD (survives multi-outlier blobs)
    med = np.median(H); mad = np.median(np.abs(H - med))
    k = 12.0 if mad > 1e-6 else 0.0
    lo, hi = med - k * mad, med + k * mad
    m = (H >= lo) & (H <= hi)
    E, N, H = E[m], N[m], H[m]
    print("height band %.1f..%.1f (med %.1f mad %.2f), kept %d/%d"
          % (lo, hi, med, mad, E.size, pts.shape[0]))
    # build grid aligned to 0.5m
    E0 = np.floor(E.min() / DSM_CELL) * DSM_CELL
    N0 = np.floor(N.min() / DSM_CELL) * DSM_CELL
    cols = int(np.ceil((E.max() - E0) / DSM_CELL)) + 1
    rows = int(np.ceil((N.max() - N0) / DSM_CELL)) + 1
    print("DSM grid cols", cols, "rows", rows, "spans m:",
          (cols-1)*DSM_CELL, "x", (rows-1)*DSM_CELL)
    coli = np.floor((E - E0) / DSM_CELL).astype(int)
    rowi = np.floor((N - N0) / DSM_CELL).astype(int)
    grid = np.full((rows, cols), np.nan, np.float32)
    grid[rowi, coli] = H  # last-write; better use median per cell
    # convert to median per cell using bincount-like; for simplicity repeat assign but use scatter accumulate
    # Use np.add.at approach for counts and sums, but H is height -> median better. Use sorting skip; approximate with mean via sums, robust enough after smoothing.
    cnt = np.zeros((rows, cols), np.int32)
    npsum = np.zeros((rows, cols), np.float32)
    np.add.at(npsum, (rowi, coli), H)
    np.add.at(cnt, (rowi, coli), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = npsum / np.where(cnt > 0, cnt, 1)
    grid = np.where(cnt > 0, mean, np.nan)
    filled = np.isfinite(grid)
    print("cells filled:", int(filled.sum()), "/", grid.size)
    # fill empty cells with nearest known value, then smooth
    if not filled.all():
        coords = np.array(np.nonzero(filled)).T
        if len(coords):
            vals = grid[filled]
            pts_all = np.array(np.nonzero(~filled)).T
            # griddata nearest
            nn = griddata(coords, vals, pts_all, method='nearest')
            grid[~filled] = nn
    # light gaussian smoothing to reduce noise
    grid = gaussian_filter(grid, sigma=1.2)
    return {"grid": grid.astype(np.float32), "E0": E0, "N0": N0, "cell": DSM_CELL}

if __name__ == "__main__":
    d = build_dsm()
    if d is None:
        print("no dsm"); sys.exit(1)
    np.save(os.path.join(CACHE, "dsm.npy"), d["grid"])
    json = {"E0": d["E0"], "N0": d["N0"], "cell": d["cell"]}
    import json as _json
    with open(os.path.join(CACHE, "dsm_meta.json"), "w") as f:
        _json.dump(json, f)
    print("DSM saved; elevation min/max:",
          float(np.nanmin(d['grid'])), float(np.nanmax(d['grid'])))