"""Step2: 由 GPS/云台姿态构建初始位姿，计算地面 footprint 与邻接关系。"""
import os
import math
import numpy as np
from common import (WORK, OUT, log, save_json, load_json, Camera,
                    R_to_euler, euler_to_R, UTM_ZONE)
from geo import latlon_to_utm

GROUND_MARGIN = 5.0  # 初始地面高度估计的容差


def build_init_pose(meta):
    """返回 N x 6: (omega, phi, kappa, X, Y, Z) 以及对应的 R (N,3,3), C (N,3)"""
    lats = np.array([m["lat"] for m in meta])
    lons = np.array([m["lon"] for m in meta])
    alts = np.array([m["alt"] for m in meta])
    E, N = latlon_to_utm(lats, lons, UTM_ZONE)
    C = np.stack([E, N, alts], axis=1)

    Rs = []
    for m in meta:
        yaw = m["gyaw"]
        pitch = m["gpitch"]   # -90 = 正射朝下
        roll = m["groll"]
        # 机头方向（从北顺时针）
        d = np.array([math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.0])
        # pitch: 0=水平朝前, -90=朝下 -> 光轴
        p = math.radians(pitch + 90.0)  # -90 -> 0 (纯朝下)
        # 光轴在包含 d 的竖直平面内，从 d 向下转 p
        z_cam = np.array([0.0, 0.0, -1.0])
        if abs(math.cos(p) - 1.0) > 1e-9:
            up = np.array([0.0, 0.0, 1.0])
            axis = np.cross(d, up)          # 水平旋转轴（左手侧方向）
            na = np.linalg.norm(axis)
            if na > 1e-9:
                axis = axis / na
                ang = math.radians(pitch + 90.0)   # pitch=-45 -> 光轴向机头方向抬起 45°
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                Rr = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
                z_cam = Rr @ z_cam
            else:
                z_cam = np.array([0.0, 0.0, -1.0])
        y_cam = -d
        y_cam = y_cam - np.dot(y_cam, z_cam) * z_cam
        y_cam = y_cam / np.linalg.norm(y_cam)
        x_cam = np.cross(y_cam, z_cam)
        R = np.stack([x_cam, y_cam, z_cam], axis=1)  # 列向量为相机轴在世界系下的表示
        # roll 绕光轴旋转
        if abs(roll) > 1e-6:
            r = math.radians(roll)
            Rr = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            R = R @ Rr
        Rs.append(R)
    Rs = np.array(Rs)
    return C, Rs


def footprint(R, C, cam, z_ground):
    """近似：把影像 4 角投影到 z=z_ground 平面，返回 4x2 的 (E,N)"""
    w, h = cam.w, cam.h
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    # 像素 -> 归一化（去畸变近似：直接用针孔，角点畸变忽略，误差可接受）
    xn = (corners[:, 0] - cam.cx) / cam.fx
    yn = (corners[:, 1] - cam.cy) / cam.fy
    dirs_cam = np.stack([xn, yn, np.ones(4)], axis=1)
    dirs_w = dirs_cam @ R.T   # R: cam->world 列向量形式 => world = R @ cam
    dirs_w = dirs_w / np.linalg.norm(dirs_w, axis=1, keepdims=True)
    dz = dirs_w[:, 2]
    t = np.where(np.abs(dz) > 1e-6, (z_ground - C[2]) / np.where(np.abs(dz) > 1e-6, dz, 1), 1e5)
    t = np.clip(t, 0.0, 3.0 * max(C[2] - z_ground, 1.0))    # 限制极端倾斜射线
    pts = C[None, :] + t[:, None] * dirs_w
    return pts[:, :2]


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject, clipper):
    """Sutherland-Hodgman 凸多边形裁剪（clipper 逆时针）"""
    out = list(subject)
    n = len(clipper)

    def inside(pt, a, b):
        return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) >= -1e-12

    def inter(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-15:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    for i in range(n):
        a = clipper[i]
        b = clipper[(i + 1) % n]
        inp = out
        out = []
        if not inp:
            break
        for j in range(len(inp)):
            cur = inp[j]
            prv = inp[j - 1]
            if inside(cur, a, b):
                if not inside(prv, a, b):
                    out.append(inter(prv, cur, a, b))
                out.append(cur)
            elif inside(prv, a, b):
                out.append(inter(prv, cur, a, b))
    return out


def _ccw(p):
    x, y = p[:, 0], p[:, 1]
    s = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return p if s > 0 else p[::-1]


def poly_iou(p1, p2):
    a = _ccw(np.asarray(p1, dtype=np.float64))
    b = _ccw(np.asarray(p2, dtype=np.float64))
    inter = _clip(list(map(tuple, a)), list(map(tuple, b)))
    if len(inter) < 3:
        return 0.0
    ai = poly_area(np.array(inter))
    return ai / max(poly_area(a), 1e-9)


def main():
    meta = load_json(os.path.join(WORK, "meta.json"))
    cj = load_json(os.path.join(WORK, "camera.json"))
    cam = Camera(cj["w"], cj["h"], *cj["init"])

    C, Rs = build_init_pose(meta)
    # 地面高度初值：绝对高 - 相对高
    rel = np.array([m["rel_alt"] for m in meta])
    ground0 = float(np.median(C[:, 2] - rel))
    log(f"初始地面高度估计 Z_ground = {ground0:.2f} m (椭球高)")

    # 与 AT 位姿对比（参考）
    atp = os.path.join(WORK, "at_pose.npy")
    if os.path.exists(atp):
        at = np.load(atp, allow_pickle=True)
        atC = at[:, 3:6]
        d = np.linalg.norm(atC - C, axis=1)
        log(f"初始相机中心 vs 大疆AT: 偏差 mean={d.mean():.2f}m max={d.max():.2f}m")
        atR = np.array([euler_to_R(*at[i, :3]) for i in range(len(at))])
        # 角度差
        errs = []
        for i in range(len(at)):
            dR = atR[i].T @ Rs[i]
            ang = math.degrees(math.acos(max(-1, min(1, (np.trace(dR) - 1) / 2))))
            errs.append(ang)
        errs = np.array(errs)
        log(f"初始姿态 vs 大疆AT: 角度差 mean={errs.mean():.2f}deg max={errs.max():.2f}deg")

    fps = [footprint(Rs[i], C[i], cam, ground0) for i in range(len(meta))]
    fps = np.array(fps)
    log(f"footprint 计算完成，单张覆盖约 "
        f"{np.mean([np.ptp(f[:,0]) for f in fps]):.1f} x {np.mean([np.ptp(f[:,1]) for f in fps]):.1f} m")

    # 邻接图（IoU > 0.10），每个影像最多保留 K 个最强邻居
    n = len(meta)
    K = int(os.environ.get("NEIGHBOR_K", "14"))
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = poly_iou(fps[i], fps[j])
            if ov > 0.10:
                cand.append((i, j, ov))
    cand.sort(key=lambda t: -t[2])
    cnt = np.zeros(n, dtype=int)
    pairs = []
    for i, j, ov in cand:
        if cnt[i] < K and cnt[j] < K:
            pairs.append((i, j, ov))
            cnt[i] += 1
            cnt[j] += 1
    log(f"候选对 {len(cand)} -> 保留 {len(pairs)} 对 (K={K}, 每影像邻居数 "
        f"min={cnt.min()} max={cnt.max()})")

    np.save(os.path.join(WORK, "init_C.npy"), C)
    np.save(os.path.join(WORK, "init_R.npy"), Rs)
    np.save(os.path.join(WORK, "fps.npy"), fps)
    save_json(os.path.join(WORK, "pairs.json"), [[i, j, round(o, 4)] for i, j, o in pairs])
    save_json(os.path.join(WORK, "ground0.json"), {"z_ground": ground0})


if __name__ == "__main__":
    main()
