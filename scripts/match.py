import os, json, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import load_meta, to_local, CACHE, SRC, IMG_W, IMG_H

import sys
SCALE = 0.5          # matching resolution scale
MAXFEAT = 8000
NEIGH_RADIUS = 340.0 # meters
NEIGH_K = 14

def load_images_by_index(local, subset=None):
    paths = [os.path.join(SRC, r["name"]) for r in local]
    feats = []          # list per image of (kps_full, desc)
    gimg = {}
    idxlist = range(len(local)) if subset is None else range(subset)
    for i in idxlist:
        p = paths[i]
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        img2 = cv2.resize(img, (int(w*SCALE), int(h*SCALE)), interpolation=cv2.INTER_AREA)
        sift = cv2.SIFT_create(nfeatures=MAXFEAT, contrastThreshold=0.04, edgeThreshold=12)
        kp, des = sift.detectAndCompute(img2, None)
        kp_full = [[k.pt[0]/SCALE, k.pt[1]/SCALE] for k in kp]
        feats.append((kp_full, des))
        gimg[i] = img2
    return feats, gimg

def select_neighbors(local):
    E = np.array([r["E"] for r in local])
    N = np.array([r["N"] for r in local])
    n = len(local)
    neigh = {}
    for i in range(n):
        d = np.hypot(E - E[i], N - N[i])
        idx = np.argsort(d)
        cand = idx[(d[idx] > 1e-6) & (d[idx] < NEIGH_RADIUS)][:NEIGH_K*2]
        # keep every image within radius but cap at NEIGH_K nearest
        cand = idx[(d[idx] > 1e-6)][:NEIGH_K]
        neigh[i] = [int(c) for c in cand]
    num = sum(len(v) for v in neigh.values())
    print("neighbors total pairs:", num, "avg", num/n)
    return neigh

def match_pair(des1, des2):
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return None
    des1 = des1.astype(np.float32); des2 = des2.astype(np.float32)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    pairs = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n2 in pairs:
        if n2 is None:
            continue
        if m.distance < 0.78 * n2.distance:
            good.append((m.queryIdx, m.trainIdx))
    if len(good) < 8:
        return None
    return good

def ransac_h(kp1, kp2, good, thresh_px=5.0):
    # kp1,kp2: full-res keypoint coordinate arrays (N,2). good: list of (qidx,tidx).
    q = np.array([kp1[i] for i, _ in good], float)
    t = np.array([kp2[j] for _, j in good], float)
    H, mask = cv2.findHomography(q, t, cv2.RANSAC, ransacReprojThreshold=thresh_px, maxIters=3000, confidence=0.999)
    if H is None:
        return None
    mask = mask.ravel().astype(bool)
    id_i = np.array([good[k][0] for k in range(len(good))], int)[mask]
    id_j = np.array([good[k][1] for k in range(len(good))], int)[mask]
    return H, id_i, id_j

def main():
    subset = None
    if len(sys.argv) > 1:
        try:
            subset = int(sys.argv[1])
        except Exception:
            subset = None
    local, _ = to_local(load_meta())
    if subset is not None:
        local = local[:subset]
    print(f"processing {'full' if subset is None else subset} images")
    print("loading features ...")
    feats, _ = load_images_by_index(local, subset=subset)
    neigh = select_neighbors(local)
    # save features dims metadata
    np.savez_compressed(os.path.join(CACHE, "feats_meta.npz"))
    # serialize feature keypoints (full-res) and descriptors count
    import pickle
    with open(os.path.join(CACHE, "feats.pkl"), "wb") as f:
        pickle.dump([list(kp) for kp, _ in feats], f)
    with open(os.path.join(CACHE, "descs.pkl"), "wb") as f:
        pickle.dump([d for _, d in feats], f)
    # matching
    pairs = sorted({tuple(sorted((i, j))) for i in neigh for j in neigh[i]})
    print("unique pairs to match:", len(pairs))
    matches = {}  # (i,j) -> (id_i, id_j) keypoint indices (full-res feature indices)
    stats = []
    for k, (i, j) in enumerate(pairs):
        kp_i = feats[i][0]; kp_j = feats[j][0]
        good = match_pair(feats[i][1], feats[j][1])
        if good is None:
            stats.append((i, j, 0, 0)); continue
        res = ransac_h(kp_i, kp_j, good, thresh_px=5.0/SCALE)  # threshold at full-res scale
        if res is None:
            stats.append((i, j, len(good), 0)); continue
        H, id_i, id_j = res
        if len(id_i) < 20:
            stats.append((i, j, len(good), len(id_i))); continue
        matches[(i, j)] = (id_i, id_j)
        stats.append((i, j, len(good), len(id_i)))
    print("pairs with solid matches:", len(matches))
    with open(os.path.join(CACHE, "matches.json"), "w") as f:
        json.dump([[int(i), int(j)] for (i, j) in matches], f)
    # save keypoint indices per pair
    np.savez_compressed(os.path.join(CACHE, "matches.npz"),
                        **{f"{i}_{j}": np.concatenate([a[:, None], b[:, None]], axis=1)
                           for (i, j), (a, b) in matches.items()})
    # stats histogram
    n_gt = [s[3] for s in stats if s[3] > 0]
    print("inlier counts: min", min(n_gt) if n_gt else 0, "med", int(np.median(n_gt)) if n_gt else 0, "max", max(n_gt) if n_gt else 0, "pairs>0", len(n_gt))
    # connectivity
    import collections
    adj = collections.defaultdict(set)
    for (i, j) in matches:
        adj[i].add(j); adj[j].add(i)
    seen = set()
    comps = []
    for s in range(len(local)):
        if s in seen: continue
        st = [s]; seen.add(s); comp = []
        while st:
            x = st.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        comps.append(len(comp))
    print("connected components sizes (sorted):", sorted(comps, reverse=True)[:8])

if __name__ == "__main__":
    main()