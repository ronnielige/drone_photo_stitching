# -*- coding: utf-8 -*-
r"""一键跑主链。

用法（在仓库根目录，或在任意位置用绝对路径）：
    python run_pipeline.py --list              # 只列出步骤，不执行
    python run_pipeline.py                     # 从第 1 步跑到最后
    python run_pipeline.py --from 6            # 从第 6 步开始（复用已有中间产物）
    python run_pipeline.py --only 12           # 只跑第 12 步
    python run_pipeline.py --only 12 13 14     # 只跑 12/13/14 步

换数据集：先设好环境变量（见 README 第 2 节），再跑本脚本。
    set DRONE_IMG_DIR=D:\data\new_flight\raw
    set DRONE_OUT=D:\data\new_flight\out
    set DRONE_WORK=D:\data\new_flight\work
    set DRONE_UTM_ZONE=51
    python run_pipeline.py
"""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

# (编号, 脚本, 启动前要设的环境变量, 说明, 大致耗时)
STEPS = [
    (1,  "step1_meta.py",         {}, "解析 XMP 元数据（GPS/云台/内参）",          "10 s"),
    (2,  "step2_init.py",         {}, "初始位姿 + footprint 邻接图",                "10 s"),
    (3,  "step3_features.py",     {}, "SIFT 特征 + 匹配",                          "~3 min"),
    (4,  "step4_ba.py",           {}, "鲁棒光束法平差（BA）",                       "~1 min"),
    (5,  "step5_dsm.py",          {}, "粗 DSM（0.25m plane-sweep）",               "~2 min"),
    (6,  "step5f_gains.py",       {}, "乘性曝光增益（radio 缺失时的回退）",         "20 s"),
    (7,  "step4b_radiometry.py",  {}, "逐帧仿射辐射校正 radio.npy",                "~3 min"),
    (8,  "step5c_dsm_refine.py",  {}, "DSM v2 精化（0.10m 粗/细两段）",            "~15 min"),
    (9,  "step5d_despike.py",     {}, "DSM 去尖刺（就地更新 dsm_v2.npy）",          "20 s"),
    (10, "step5e_smooth.py",      {}, "DSM 保边平滑（Perona-Malik）",              "30 s"),
    (11, "step6b_ortho.py",       {}, "逐像素单帧选取正射渲染",                     "~19 min"),
    (12, "step6d_seamcut.py",     {}, "接缝线优化（图割）★消除分割线关键",          "~27 min"),
    (13, "step6e_seam_feather.py", {}, "缝合带羽化（±1px 5x5 中值）",             "10 s"),
    (14, "step6c_mix.py",         {"MIX_SF": "out/ortho_rgb_cut_f.npy",
                                   "MIX_DST": "out/ortho_rgb3_cut.npy"},
                                  "低频色调共识混合",                            "~1 min"),
    (15, "export_tif.py",         None, "导出 GeoTIFF（需 rasterio）",             "~2 min"),
]

OUT_NAME = "xiangjia_ortho.tif"   # 第 15 步的输出文件名（放在 DRONE_OUT 下）


def _banner():
    from common import IMG_DIR, OUT, WORK, EPSG, UTM_ZONE     # noqa
    print("=" * 68)
    print(f"  影像目录 : {IMG_DIR}")
    print(f"  中间产物 : {WORK}")
    print(f"  输出目录 : {OUT}")
    print(f"  投影     : {EPSG}  (UTM 带号 {UTM_ZONE})")
    print("=" * 68)
    return IMG_DIR


def _preflight(img_dir):
    """启动前检查，缺东西就给可操作的提示。"""
    if not os.path.isdir(img_dir):
        print(f"\n[错误] 影像目录不存在: {img_dir}")
        print("       请先设置环境变量指向新采集的图片目录，例如：")
        print(r'         set DRONE_IMG_DIR=D:\data\new_flight\raw')
        print(r'         set DRONE_OUT=D:\data\new_flight\out')
        print(r'         set DRONE_WORK=D:\data\new_flight\work')
        return False
    exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    n = sum(1 for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts)
    if n == 0:
        print(f"\n[错误] 影像目录里没有可识别的图片: {img_dir}")
        print("       支持 .JPG/.JPEG/.TIF/.TIFF/.PNG")
        return False
    print(f"  发现 {n} 张影像\n")
    return True


def _need_rasterio():
    try:
        import rasterio  # noqa
        return True
    except Exception:
        print("\n[错误] 第 15 步 export_tif.py 需要 rasterio，当前 Python 没装。")
        print("       请换用装了 rasterio 的 venv，或先 pip install rasterio")
        return False


def run_one(num, script, env_extra, dry=False):
    cmd = [sys.executable, script]
    env = dict(os.environ)
    if env_extra:
        from common import OUT
        for k, v in env_extra.items():
            # 相对路径按 OUT 目录展开
            if v.startswith("out/"):
                v = os.path.join(OUT, v[4:])
            env[k] = v
    print(f"\n{'='*68}\n>>> [{num}] {script}   cwd={SRC}\n{'='*68}", flush=True)
    if dry:
        return 0
    r = subprocess.run(cmd, cwd=SRC, env=env)
    if r.returncode != 0:
        print(f"\n[失败] 第 {num} 步 {script} 退出码 {r.returncode}，已中止。")
        return r.returncode
    return 0


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--list", action="store_true", help="只列出步骤")
    ap.add_argument("--from", dest="frm", type=int, default=1, help="从第 N 步开始")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="只跑指定步")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    a = ap.parse_args()

    if a.list:
        print(f"{'#':>3}  {'脚本':<24} {'耗时':>8}   说明")
        print("-" * 78)
        for num, sc, _, desc, t in STEPS:
            print(f"{num:>3}  {sc:<24} {t:>8}   {desc}")
        print("\n★ 第 12~13 步是消除拼接分割线的关键，不建议跳过。")
        return 0

    sys.path.insert(0, SRC)
    img_dir = _banner()
    if not _preflight(img_dir):
        return 2

    if a.only:
        todo = [s for s in STEPS if s[0] in set(a.only)]
        if not todo:
            print(f"[错误] --only 指定的步骤不存在：{a.only}")
            return 2
    else:
        todo = [s for s in STEPS if s[0] >= a.frm]

    if any(s[0] == 15 for s in todo) and not a.dry_run and not _need_rasterio():
        return 3

    for num, script, env_extra, desc, _ in todo:
        if num == 15:
            from common import OUT
            args = [os.path.join(OUT, "ortho_rgb3_cut.npy"), os.path.join(OUT, OUT_NAME)]
            cmd = [sys.executable, script] + args
            print(f"\n{'='*68}\n>>> [15] export_tif.py   cwd={SRC}\n{'='*68}", flush=True)
            if a.dry_run:
                print("   ", " ".join(cmd))
                continue
            rc = subprocess.run(cmd, cwd=SRC).returncode
            if rc:
                print(f"\n[失败] 第 15 步 export_tif.py 退出码 {rc}")
                return rc
            continue
        rc = run_one(num, script, env_extra, a.dry_run)
        if rc:
            return rc

    if not a.dry_run:
        from common import OUT
        print(f"\n{'='*68}\n全部完成。成果：{os.path.join(OUT, OUT_NAME)}\n{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
