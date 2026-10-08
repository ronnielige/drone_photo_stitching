"""Step1: 解析 DJI 影像 XMP 元数据 + MRK 时间戳，并解析大疆 AT 结果作为参考基准。"""
import os
import re
import glob
import math
import numpy as np
from common import (IMG_DIR, DJI_DIR, WORK, log, save_json, EPSG,
                    Camera, euler_to_R, R_to_euler)

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_xmp(path, max_bytes=400000):
    with open(path, "rb") as f:
        head = f.read(max_bytes)
    i = head.find(b"<x:xmpmeta")
    if i < 0:
        return {}
    j = head.find(b"</x:xmpmeta>", i)
    s = head[i:j if j > 0 else len(head)].decode("utf-8", "replace")
    out = {}
    for key in ("GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "RelativeAltitude",
                "GimbalYawDegree", "GimbalPitchDegree", "GimbalRollDegree",
                "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree",
                "CalibratedFocalLength", "CalibratedOpticalCenterX",
                "CalibratedOpticalCenterY", "DewarpData", "CameraSerialNumber",
                "DroneSerialNumber", "UTCAtExposure", "ShutterType", "FocusDistance"):
        m = re.search(key + r'="([^"]*)"', s)
        if m:
            out[key] = m.group(1)
    m = re.search(r'dc:format="([^"]*)"', s)
    return out


def fnum(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main():
    files = sorted(p for p in glob.glob(os.path.join(IMG_DIR, "*"))
                   if os.path.splitext(p)[1].upper() in (".JPG", ".JPEG", ".TIF", ".TIFF", ".PNG"))
    log(f"发现 {len(files)} 张影像 (目录 {IMG_DIR})")

    # ---- MRK 时间戳 ----
    mrk_files = glob.glob(os.path.join(IMG_DIR, "*.MRK")) or glob.glob(os.path.join(IMG_DIR, "*.mrk"))
    mrk = {}
    if mrk_files:
        with open(mrk_files[0], "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                mrk[idx] = line.rstrip("\n")
        log(f"MRK 行数: {len(mrk)}")

    meta = []
    for i, p in enumerate(files):
        x = parse_xmp(p)
        rec = {
            "idx": i,
            "name": os.path.basename(p),
            "path": p,
            "lat": fnum(x.get("GpsLatitude", "0")),
            "lon": fnum(x.get("GpsLongitude", "0")),
            "alt": fnum(x.get("AbsoluteAltitude", "0")),
            "rel_alt": fnum(x.get("RelativeAltitude", "0")),
            "gyaw": fnum(x.get("GimbalYawDegree", "0")),
            "gpitch": fnum(x.get("GimbalPitchDegree", "-90")),
            "groll": fnum(x.get("GimbalRollDegree", "0")),
            "fyaw": fnum(x.get("FlightYawDegree", "0")),
            "fpitch": fnum(x.get("FlightPitchDegree", "0")),
            "froll": fnum(x.get("FlightRollDegree", "0")),
            "calib_f": fnum(x.get("CalibratedFocalLength", "0")),
            "calib_cx": fnum(x.get("CalibratedOpticalCenterX", "0")),
            "calib_cy": fnum(x.get("CalibratedOpticalCenterY", "0")),
            "dewarp": x.get("DewarpData", ""),
            "mrk": mrk.get(i + 1, ""),
        }
        meta.append(rec)

    # ---- 相机内参：优先用单张影像的 DewarpData（出厂标定）----
    dd = meta[0]["dewarp"]
    # 格式: date;fx,fy,cx,cy,k1,k2,p1,p2,k3
    vals = dd.split(";")[-1].split(",")
    if len(vals) >= 9:
        fx, fy, ox, oy, k1, k2, p1, p2, k3 = [float(v) for v in vals[:9]]
    else:
        raise RuntimeError("无法解析 DewarpData: " + dd)
    log(f"出厂标定: fx={fx:.2f} fy={fy:.2f} cx_off={ox:.2f} cy_off={oy:.2f} "
        f"k1={k1:.5f} k2={k2:.5f} k3={k3:.5f} p1={p1:.7f} p2={p2:.7f}")

    # 大疆 AT 报告中优化后的内参（报告中 Initial 一行）
    at_f, at_cx, at_cy = 3720.60, 2664.36, 1952.64
    at_k1, at_k2, at_k3, at_p1, at_p2 = -0.10893391, 0.00334538, -0.02004984, -0.00062201, -0.00026050

    # 影像尺寸
    from PIL import Image
    im = Image.open(files[0])
    W, H = im.size
    im.close()
    log(f"影像尺寸: {W} x {H}")
    # DewarpData 中的 cx/cy 是相对图像中心的偏移
    cx = W / 2.0 + ox
    cy = H / 2.0 + oy
    log(f"主点(绝对): cx={cx:.2f} cy={cy:.2f}  (AT报告 Initial: 2647.02 / 1969.28)")

    cam_init = Camera(W, H, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    cam_at = Camera(W, H, at_f, at_f, at_cx, at_cy, at_k1, at_k2, at_k3, at_p1, at_p2)

    save_json(os.path.join(WORK, "meta.json"), meta)
    save_json(os.path.join(WORK, "camera.json"), {
        "w": W, "h": H,
        "init": [fx, fy, cx, cy, k1, k2, k3, p1, p2],
        "at_opt": [at_f, at_f, at_cx, at_cy, at_k1, at_k2, at_k3, at_p1, at_p2],
    })

    # ---- 解析大疆 AT 位姿（参考基准）----
    xmlp = os.path.join(DJI_DIR, "AT", "BlocksExchangeUndistortAT_WithoutTiePoints.xml")
    at = parse_at_xml(xmlp) if os.path.exists(xmlp) else []
    if at:
        arr = np.array([a[:6] for a in at], dtype=np.float64)
        np.save(os.path.join(WORK, "at_pose_raw.npy"), arr)
        save_json(os.path.join(WORK, "at_paths.json"), [a[6] for a in at])
        log(f"解析大疆 AT 位姿 {len(at)} 张")
        # AT 影像顺序与文件名顺序不同 -> 按位置做最优匹配建立对应
        from geo import latlon_to_utm
        from scipy.optimize import linear_sum_assignment
        E, N = latlon_to_utm(np.array([m["lat"] for m in meta]),
                             np.array([m["lon"] for m in meta]), 50)
        P = np.stack([E, N], axis=1)
        Q = arr[:, 3:5]
        D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(D)
        dist = D[ri, ci]
        log(f"AT<->影像 对应匹配: 中位距离 {np.median(dist):.2f}m, 最大 {dist.max():.2f}m")
        aligned = np.zeros_like(arr)
        aligned[ri] = arr[ci]
        np.save(os.path.join(WORK, "at_pose.npy"), aligned)
        save_json(os.path.join(WORK, "at_order.json"), [int(c) for c in ci])
        log(f"  AT 位姿样例: OPK=({arr[0,0]:.3f},{arr[0,1]:.3f},{arr[0,2]:.3f}) "
            f"C=({arr[0,3]:.2f},{arr[0,4]:.2f},{arr[0,5]:.2f})")

    # ---- 影像清单顺序确认（大疆 image_list.json 的顺序）----
    il = os.path.join(DJI_DIR, "images", "survey", "image_list.json")
    if os.path.exists(il):
        from common import load_json
        lst = load_json(il)
        order = [os.path.basename(r["origin_path"]) for r in lst]
        save_json(os.path.join(WORK, "dji_order.json"), order)
        same = order == [m["name"] for m in meta]
        log(f"大疆影像顺序与文件名字典序一致: {same}")

    # 统计
    lats = np.array([m["lat"] for m in meta])
    lons = np.array([m["lon"] for m in meta])
    alts = np.array([m["alt"] for m in meta])
    log(f"经纬范围 lat[{lats.min():.6f},{lats.max():.6f}] lon[{lons.min():.6f},{lons.max():.6f}]")
    log(f"绝对高度 alt[{alts.min():.2f},{alts.max():.2f}] mean={alts.mean():.2f}")
    gy = np.array([m["gyaw"] for m in meta])
    gp = np.array([m["gpitch"] for m in meta])
    log(f"云台 yaw[{gy.min():.1f},{gy.max():.1f}] pitch[{gp.min():.1f},{gp.max():.1f}]")


def parse_at_xml(path):
    """解析 BlocksExchange XML -> 每张相机的 (opk_deg[3], center_xyz)"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    out = []
    for pg in root.iter("Photogroup"):
        for ph in pg.iter("Photo"):
            rot = ph.find("Pose/Rotation")
            cen = ph.find("Pose/Center")
            if rot is None or cen is None:
                continue
            om = float(rot.find("Omega").text)
            ph_ = float(rot.find("Phi").text)
            ka = float(rot.find("Kappa").text)
            x = float(cen.find("x").text)
            y = float(cen.find("y").text)
            z = float(cen.find("z").text)
            img = ph.find("ImagePath")
            out.append([om, ph_, ka, x, y, z, img.text if img is not None else ""])
    return out


if __name__ == "__main__":
    main()
