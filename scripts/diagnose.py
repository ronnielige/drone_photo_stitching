import pickle, json, numpy as np, os
BASE = r"D:\lr\project\无人机航拍拼接"
CACHE = os.path.join(BASE, "mysolution", "cache")

with open(os.path.join(CACHE, "tracks.pkl"), "rb") as f:
    d = pickle.load(f)
Xs, valid = d["Xs"], d["valid"]
pts = Xs[valid]
print("sparse valid points:", len(pts))
print("H stats: min %.1f med %.1f max %.1f std %.2f" % (pts[:,2].min(), np.median(pts[:,2]), pts[:,2].max(), pts[:,2].std()))
E, N = pts[:,0], pts[:,1]
area = (E.max()-E.min())*(N.max()-N.min())
print("extent: %.0f x %.0f m, area %.0f m^2, density %.2f pts/m^2" % (E.max()-E.min(), N.max()-N.min(), area, len(pts)/area))

dsm = np.load(os.path.join(CACHE, "dsm.npy"))
jm = json.load(open(os.path.join(CACHE, "dsm_meta.json")))
print("DSM grid:", dsm.shape, "cell", jm["cell"], "E0", jm["E0"], "N0", jm["N0"])
print("DSM H: min %.2f med %.2f max %.2f" % (dsm.min(), np.median(dsm), dsm.max()))
# roughness: std of gradient magnitude
gy, gx = np.gradient(dsm)
print("DSM roughness (mean |grad|): %.4f m/m" % np.mean(np.hypot(gx, gy)))

with open(os.path.join(CACHE, "sfm_cam.pkl"), "rb") as f:
    c = pickle.load(f)
Cs = c["Cs"]
# baseline between consecutive images along track
order = np.argsort([e for e in Cs[:,0]])
d_consec = []
for k in range(len(Cs)-1):
    a, b = Cs[k], Cs[k+1]
    d_consec.append(np.hypot(a[0]-b[0], a[1]-b[1]))
d_consec = np.array(d_consec)
print("consecutive cam distance: med %.1f m, p10 %.1f, p90 %.1f" % (np.median(d_consec), np.percentile(d_consec,10), np.percentile(d_consec,90)))
# nearest neighbor distance overall (cross-track line spacing)
from scipy.spatial import cKDTree
tree = cKDTree(Cs[:, :2])
dd, _ = tree.query(Cs[:, :2], k=2)
print("NN cam distance (line spacing est): med %.1f m" % np.median(dd[:,1]))
print("altitude: med %.1f" % np.median(Cs[:,2]))
import psutil
print("RAM total %.1f GB, avail %.1f GB" % (psutil.virtual_memory().total/1e9, psutil.virtual_memory().available/1e9))