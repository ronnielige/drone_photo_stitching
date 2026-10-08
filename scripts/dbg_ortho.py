import os, sys, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE, load_meta, to_local, get_intrinsics_k, K1,K2,K3,P1,P2
from ortho import build_poses, dsm_lookup

local, _ = to_local(load_meta())
local = local[:12]
N=12
Cs, Rs = build_poses(N)
print("C0", Cs[0], " R0 zcol", Rs[0])
jm = json.load(open(os.path.join(CACHE,"dsm_meta.json")))
dsm = {"grid": np.load(os.path.join(CACHE,"dsm.npy")), "E0":jm["E0"],"N0":jm["N0"],"cell":jm["cell"]}
print("dsm E0 N0 cell", jm)
print("cam E range", local[0]["E"], local[11]["E"], " N range", local[0]["N"], local[11]["N"])
# tile 0 center
g=0.056
Emin=jm["E0"]; Nmin=jm["N0"]; cell=jm["cell"]
rows,cols=dsm["grid"].shape
Nmax=Nmin+(rows-1)*cell; Emax=Emin+(cols-1)*cell
print("grid bounds E", Emin, Emax, "N", Nmin, Nmax)
# a center point
Ec=(Emin+Emax)/2; Nc=(Nmin+Nmax)/2
Z=dsm_lookup(dsm, np.array([[Ec]],np.float32), np.array([[Nc]],np.float32))
print("Z at center", Z)
# candidate images within 250m of center
ce=[r["E"] for r in local]; cn=[r["N"] for r in local]
d=np.hypot(np.array(ce)-Ec, np.array(cn)-Nc)
print("dists to cam centers from center:", np.round(d,1))
print("min dist", d.min(), "idx", d.argmin())
# test projection of this ground point to nearest cam
from ortho import project_blend
import cv2
from config import SRC
p=os.path.join(SRC, local[int(d.argmin())]["name"])
img=cv2.imdecode(np.fromfile(p,dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
K=get_intrinsics_k()
D=np.array([K1,K2,P1,P2,K3],float)
i=int(d.argmin())
Xg=np.stack([np.array([[Ec]]), np.array([[Nc]]), np.array([[float(Z[0,0])]])], axis=-1)
col,w=project_blend(Xg, Cs[i], Rs[i], img, K, D)
print("weight at center for nearest cam", w[0,0])