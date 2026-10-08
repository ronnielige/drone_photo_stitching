import os, sys, json, pickle, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import load_meta, to_local, CACHE, SRC

MAXFEAT = 24000
NEIGH_K = 12          # nearest neighbors to match against
RATIO = 0.78
F_THRESH = 2.5         # px, epipolar inlier threshold
MIN_INL = 40

def main():
    t00 = time.time()
    local, _ = to_local(load_meta())
    N = len(local)
    print(f"matching {N} images, maxfeat={MAXFEAT}, K={NEIGH_K}", flush=True)

    sift = cv2.SIFT_create(nfeatures=MAXFEAT, contrastThreshold=0.015, edgeThreshold=15)
    kps, descs = [], []
    for i, r in enumerate(local):
        img = cv2.imdecode(np.fromfile(os.path.join(SRC, r["name"]), dtype=np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        pts = np.array([k.pt for k in kp], np.float64)
        kps.append(pts)
        descs.append(des.astype(np.float32))
        if i % 16 == 0:
            print(f"[feat] {i}/{N}: {len(kp)} kps  ({time.time()-t00:.0f}s)", flush=True)
    tot = sum(len(k) for k in kps)
    print(f"features total {tot}, avg {tot/N:.0f}", flush=True)
    with open(os.path.join(CACHE, "feats2.pkl"), "wb") as f:
        pickle.dump(kps, f)

    # pair selection: K nearest neighbors by camera center distance
    E = np.array([r["E"] for r in local]); Nn = np.array([r["N"] for r in local])
    pairs = set()
    for i in range(N):
        d = np.hypot(E - E[i], Nn - Nn[i])
        order = np.argsort(d)
        cnt = 0
        for j in order:
            if j == i:
                continue
            pairs.add((min(i, int(j)), max(i, int(j))))
            cnt += 1
            if cnt >= NEIGH_K:
                break
    pairs = sorted(pairs)
    print(f"pairs to match: {len(pairs)}", flush=True)

    use_usac = int(cv2.__version__.split(".")[0]) >= 4 and hasattr(cv2, "USAC_MAGSAC")
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    matches = {}
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        if len(kps[i]) < 50 or len(kps[j]) < 50:
            continue
        m_ij = flann.knnMatch(descs[i], descs[j], k=2)
        best_ij = {}
        for mm in m_ij:
            if len(mm) < 2:
                continue
            a, b = mm
            if a.distance < RATIO * b.distance:
                best_ij[a.queryIdx] = a.trainIdx
        qidx, tidx = [], []
        for mm in flann.knnMatch(descs[j], descs[i], k=2):
            if len(mm) < 2:
                continue
            a, b = mm
            if a.distance < RATIO * b.distance and a.trainIdx in best_ij \
               and best_ij[a.trainIdx] == a.queryIdx:
                qidx.append(a.trainIdx); tidx.append(a.queryIdx)
        if len(qidx) < MIN_INL:
            continue
        p1 = kps[i][np.array(qidx)]; p2 = kps[j][np.array(tidx)]
        if use_usac:
            F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, F_THRESH, 0.9999, 100000)
        else:
            F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, F_THRESH, 0.9999, 100000)
        if F is None or mask is None:
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < MIN_INL:
            continue
        matches[(i, j)] = (np.array(qidx)[mask], np.array(tidx)[mask])
        if k % 50 == 0:
            el = time.time() - t0
            print(f"[match] {k}/{len(pairs)}  pairs_ok={len(matches)}  ({el:.0f}s, eta {el/max(k,1)*(len(pairs)-k):.0f}s)", flush=True)

    print(f"pairs with solid matches: {len(matches)}", flush=True)
    with open(os.path.join(CACHE, "matches2.json"), "w") as f:
        json.dump([[int(i), int(j)] for (i, j) in matches], f)
    np.savez_compressed(os.path.join(CACHE, "matches2.npz"),
                        **{f"{i}_{j}": np.concatenate([a[:, None], b[:, None]], axis=1)
                           for (i, j), (a, b) in matches.items()})
    inl = [len(v[0]) for v in matches.values()]
    print("inliers/pair: med %d, p10 %d, p90 %d" %
          (int(np.median(inl)), int(np.percentile(inl, 10)), int(np.percentile(inl, 90))), flush=True)
    import collections
    adj = collections.defaultdict(set)
    for (i, j) in matches:
        adj[i].add(j); adj[j].add(i)
    seen = set(); comps = []
    for s in range(N):
        if s in seen:
            continue
        st = [s]; seen.add(s); comp = 0
        while st:
            x = st.pop(); comp += 1
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        comps.append(comp)
    print("connected components:", sorted(comps, reverse=True)[:8], flush=True)
    print("match2 done in %.0fs" % (time.time() - t00), flush=True)

if __name__ == "__main__":
    main()