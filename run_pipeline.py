# -*- coding: utf-8 -*-
"""一键跑通主链：SfM → DSM → 正射 → 导出。

用法:
    python run_pipeline.py                # 从头跑完整条链
    python run_pipeline.py --from step5c  # 从指定步骤开始（复用已有中间产物）
    python run_pipeline.py --only step6b  # 只跑某一步
    python run_pipeline.py --list         # 列出全部步骤

路径配置见 src/common.py 顶部注释，或直接用环境变量覆盖：
    DRONE_IMG_DIR    原始影像目录（新采集的数据）
    DRONE_DATA_DIR   数据集根目录
    DRONE_WORK       中间产物目录
    DRONE_OUT        成果目录
    DRONE_UTM_ZONE   UTM 带号（默认 50）
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

from common import IMG_DIR, WORK, OUT, UTM_ZONE, EPSG   # noqa: E402

# (步骤名, 脚本, 说明)
STEPS = [
    ("step1",  "step1_meta.py",        "解析 XMP 元数据（内参 / 位姿 / GPS）"),
    ("step2",  "step2_init.py",        "YPR → 初始旋转 + footprint 邻接图"),
    ("step3",  "step3_features.py",    "SIFT 特征提取 + 匹配"),
    ("step4",  "step4_ba.py",          "鲁棒光束法平差"),
    ("step4b", "step4b_radiometry.py", "逐帧仿射辐射校正"),
    ("step5c", "step5c_dsm_refine.py", "密集匹配 DSM v2（最耗时，约 15 min）"),
    ("step5d", "step5d_despike.py",    "DSM 去尖刺"),
    ("step5e", "step5e_smooth.py",     "DSM 保边平滑"),
    ("step6b", "step6b_ortho.py",      "逐像素单帧选取渲染（约 9 min）"),
    ("step6c", "step6c_mix.py",        "低频色调共识混合"),
    ("export", "export_tif.py",        "导出 GeoTIFF + tfw/prj（需 rasterio）"),
]


def banner():
    print("=" * 68)
    print("无人机航拍拼接管线")
    print(f"  原始影像目录 : {IMG_DIR}")
    print(f"  中间产物目录 : {WORK}")
    print(f"  成果目录     : {OUT}")
    print(f"  投影         : {EPSG}  (UTM zone {UTM_ZONE}N)")
    print("=" * 68)
    if not os.path.isdir(IMG_DIR):
        print(f"\n[错误] 原始影像目录不存在：{IMG_DIR}")
        print("       请用 DRONE_IMG_DIR 指向新采集的影像目录，例如：")
        print(r"         Windows :  set DRONE_IMG_DIR=D:\data\new_flight\raw")
        print('         bash    :  export DRONE_IMG_DIR="/d/data/new_flight/raw"')
        return False
    n = len([f for f in os.listdir(IMG_DIR)
             if f.lower().endswith((".jpg", ".jpeg"))])
    print(f"  影像数量     : {n} 张")
    if n == 0:
        print("[错误] 该目录下没有 .JPG 影像")
        return False
    return True


def run(step, script, desc, args=()):
    print(f"\n{'-' * 68}\n[{step}] {desc}\n{'-' * 68}")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", script, *args], cwd=SRC)
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"\n[失败] {step} 退出码 {p.returncode}，耗时 {dt:.0f}s")
        return False
    print(f"\n[完成] {step}  耗时 {dt:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出全部步骤")
    ap.add_argument("--from", dest="frm", default=None, help="从指定步骤开始")
    ap.add_argument("--only", default=None, help="只跑指定步骤")
    a = ap.parse_args()

    if a.list:
        for i, (s, sc, d) in enumerate(STEPS, 1):
            print(f"  {i:2d}. {s:7s} {sc:24s} {d}")
        return

    if not banner():
        sys.exit(2)

    names = [s for s, _, _ in STEPS]
    idx0, idx1 = 0, len(STEPS)
    if a.only:
        if a.only not in names:
            print(f"[错误] 未知步骤 {a.only}；可用：{', '.join(names)}")
            sys.exit(2)
        idx0 = names.index(a.only)
        idx1 = idx0 + 1
    elif a.frm:
        if a.frm not in names:
            print(f"[错误] 未知步骤 {a.frm}；可用：{', '.join(names)}")
            sys.exit(2)
        idx0 = names.index(a.frm)

    t_all = time.time()
    for i in range(idx0, idx1):
        s, sc, d = STEPS[i]
        args = ()
        if s == "export":
            try:
                import rasterio  # noqa: F401
            except ImportError:
                print("[提示] 当前 Python 没有 rasterio，export 步骤会失败。")
                print("       请改用装了 rasterio 的解释器运行，或 pip install rasterio。")
                sys.exit(1)
            args = (os.path.join(OUT, "ortho_rgb3.npy"),
                    os.path.join(OUT, "xiangjia_ortho_mine.tif"))
        if not run(s, sc, d, args):
            sys.exit(1)

    print(f"\n{'=' * 68}")
    print(f"全部完成，总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"成果：{os.path.join(OUT, 'xiangjia_ortho_mine.tif')}")
    print("=" * 68)


if __name__ == "__main__":
    main()
