import os, sys, json
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE, SRC, load_meta, to_local, get_intrinsics_k, K1,K2,K3,P1,P2
from ortho import build_poses, dsm_lookup, project_blend

N=12
local,_ = to_local(load_meta()); local=local[:N]
Cs,Rs=build_poses(N)
jm=json.load(open(os.path.join(CACHE,"dsm_meta.json")))
dsm={"grid":np.load(os.path.join(CACHE,"dsm.npy")),"E0":jm["E0"],"N0":jm["N0"],"cell":jm["cell"]}
rows,cols=dsm["grid"].shape
g=0.056
Emin=jm["E0"]; Nmin=jm["N0"]
Emax=Emin+(cols-1)*g*0  # placeholder not used
# tile r0=2048,c0=4096
r0,c0=2048,4096; Ht,Wt=2048,2048
gx=c0+np.arange(Wt)[None,:]+0.5; gy=r0+np.arange(Ht)[:,None]+0.5
E=(Emin+gx*g).astype(np.float32); N=(Nmin+(rows-1)*g-gy*g).astype(np.float32)
# fix: Nmax uses DSM cell, not ortho gsd g
Nmax=Nmin+(rows-1)*dsm["cell"]
E=Emin+gx*g; N=Nmax-gy*g
E,N=np.broadcast_arrays(E.astype(np.float32),N.astype(np.float32))
Z=dsm_lookup(dsm,E,N)
print("Z finite frac", np.isfinite(Z).mean())
Xg=np.stack([E,N,Z],axis=-1).astype(np.float64)
for i in [6,5,7,8,9,2]:
    p=os.path.join(SRC, local[i]["name"])
    img=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR).astype(np.float32)
    K=get_intrinsics_k(); D=np.array([K1,K2,P1,P2,K3],float)
    col,w=project_blend(Xg,Cs[i],Rs[i],img,K,D)
    print("cam",i,"Cs=",np.round(Cs[i],1),"maxw=",round(w.max(),3),"W>0 frac=",round((w>0).mean(),4))