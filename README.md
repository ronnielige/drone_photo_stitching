# 无人机航拍正射拼接（自研管线）

不用大疆智图，从原始 JPG + XMP 元数据出发，自己跑出一条完整正射拼接管线：
**SfM 空三 → 光束法平差 → DSM 重建 → 逐像素单帧选取正射 → 接缝线优化 → 色调共识 → GeoTIFF 出图**。

对标 `xiangjia1/map/result.tif`（大疆智图 2D 建图结果）：

| 指标 | 本管线 (v4) | 大疆智图 | 说明 |
|---|---|---|---|
| 边缘锐度 ↑ | **2.252** | 2.008 | 逐像素单帧选取，无跨帧平均 |
| 缝线"刀口"超出 | **−6.00** | 0（基准） | 负值 = 缝线处梯度低于大疆同位置 |
| 跨线落差超出（中位） | **+1.00** 灰阶 | 0（基准） | |
| 重投影误差 | p50 0.76px / p90 2.04px | 1.003px | |
| 输出 | 16000×12073 @5.50cm | 14599×10488 @5.62cm | |
| 绝对定位差 | 1~3 m | — | 无 GCP/RTK 的固有限制 |

详见 `拼接方案报告.md`。

---

## 1. 环境依赖

| 用途 | 依赖 |
|---|---|
| 主链路 | `numpy` `opencv-python` `scipy` `Pillow` |
| 出图（GeoTIFF） | `rasterio` |
| 坐标转换 | `pyproj`（可选，`geo.py` 自带 UTM 公式兜底） |

```bash
pip install numpy opencv-python scipy Pillow rasterio pyproj
```

> Windows 上建议用独立 venv。本机已有一个可用环境：
> `C:/Users/arc/.workbuddy/binaries/python/envs/default/Scripts/python.exe`

---

## 2. 换一个新采集的数据集怎么跑（重点）

**只需设置环境变量，不用改任何代码。**

### 一次性设置

| 变量 | 含义 | 默认值 |
|---|---|---|
| `DRONE_IMG_DIR` | **航拍原始图片目录（必改）** | `<数据根>/xiangjia1_186m` |
| `DRONE_OUT` | 最终输出目录 | `<仓库>/out` |
| `DRONE_WORK` | 中间产物目录 | `<仓库>/work` |
| `DRONE_DJI_DIR` | 大疆智图工程目录（仅对照评估用，可不存在） | `<数据根>/xiangjia1` |
| `DRONE_DATA_DIR` | 数据根目录（上面几项未单独设置时从它派生） | `<仓库>/..` |
| **`DRONE_UTM_ZONE`** | **UTM 带号，默认 50（东经 114~120°）。换地区务必确认！** | `50` |
| `DRONE_SOUTH` | 南半球设 `1` | `0` |
| `DRONE_EPSG` | 直接指定 EPSG（设了就忽略上面两项） | `EPSG:326<zone>` |

### 三条命令跑完

**Windows (cmd)**

```bat
set DRONE_IMG_DIR=D:\data\new_flight\raw
set DRONE_OUT=D:\data\new_flight\out
set DRONE_WORK=D:\data\new_flight\work
python run_pipeline.py
```

**Windows (PowerShell)**

```powershell
$env:DRONE_IMG_DIR = "D:\data\new_flight\raw"
$env:DRONE_OUT     = "D:\data\new_flight\out"
$env:DRONE_WORK    = "D:\data\new_flight\work"
python run_pipeline.py
```

**Linux / macOS**

```bash
export DRONE_IMG_DIR=/data/new_flight/raw
export DRONE_OUT=/data/new_flight/out
export DRONE_WORK=/data/new_flight/work
python run_pipeline.py
```

`run_pipeline.py` 会先打印将要使用的路径 / 影像数 / 投影带号，缺目录或没图片时给出可操作的提示并退出。

```bash
python run_pipeline.py --list              # 只列步骤
python run_pipeline.py --from 6            # 从第 6 步开始（复用已有中间产物）
python run_pipeline.py --only 12 13 14     # 只跑某几步
python run_pipeline.py --dry-run           # 只打印命令
```

### ⚠️ 换地区必确认的两件事

1. **UTM 带号**。默认 50 只适合东经 114°~120°。
   带号 = `floor(经度/6) + 31`（北半球 EPSG = `32600+带号`）。算错了整幅图坐标全偏。
   启动时会打印 `投影: EPSG:326xx`，**先核对一眼**。
2. **XMP 是否保留**。部分相机/后期流程会剥掉 XMP，没有 GPS/云台角就跑不了 SfM。
   第 1 步会打印经纬范围，为空说明 XMP 丢了。

### 对数据的要求

- **图片**：同一架次、带 DJI XMP 元数据（含 GPS / 云台姿态 / 内参）的 JPG。
  扩展名 `.JPG/.JPEG/.TIF/.TIFF/.PNG` 均可识别（大小写不敏感）。
- **目录**：`DRONE_IMG_DIR` 下直接放图片，无需子目录。
- **可选**：同目录下的 `*.MRK` 时间戳文件（用于时间排序，没有也能跑）。
- **重叠度**：航向 ≥ 70%、旁向 ≥ 60%（DJI M3E 默认设置即可）。
- **相机**：默认内参按 DJI M3E 广角相机标定（`step1_meta.py` 会优先读 XMP 的
  `DewarpData`，读不到才回退到内置值）。**换别的机型必须确认这一步读对了。**

**评估脚本不要照搬**：`step8c_evaluate.py` / `step8d_seam_eval.py` / `hotspot_view.py` 等
需要大疆同区域成果做对照，**主链（1~15 步）不依赖任何大疆产物**。

---

## 3. 完整运行步骤（`run_pipeline.py` 内部顺序）

设当前目录为仓库根，Python 用装了依赖的解释器。

| # | 脚本 | 产物 | 本机耗时（129 张） |
|---|---|---|---|
| 1 | `step1_meta.py` | `work/meta.json` `work/camera.json` | 10 s |
| 2 | `step2_init.py` | `work/init_R.npy` `init_C.npy` `pairs.json` `ground0.json` | 10 s |
| 3 | `step3_features.py` | `work/feats.npz` `work/matches.npz` | 23 s + 2.5 min |
| 4 | `step4_ba.py` | `work/ba_R.npy` `ba_C.npy` `ba_X.npy` | 1 min |
| 5 | `step5_dsm.py` | `work/dsm.npy` `dsm_meta.json` | 2 min |
| 6 | `step5f_gains.py` | `work/gains.npy` | 20 s |
| 7 | `step4b_radiometry.py` | `work/radio.npy` | 3 min |
| 8 | `step5c_dsm_refine.py` | `work/dsm_v2.npy` `dsm_meta_v2.json` | **~15 min** |
| 9 | `step5d_despike.py` | 就地更新 `dsm_v2.npy` | 20 s |
| 10 | `step5e_smooth.py` | 就地更新 `dsm_v2.npy` | 30 s |
| 11 | `step6b_ortho.py` | `out/ortho_rgb.npy` `ortho_avg.npy` `ortho_valid.npy` `ortho_vidx.npy` `ortho_meta.json` | **~19 min** |
| 12 | `step6d_seamcut.py` | `out/ortho_rgb_cut.npy` `ortho_vidx_cut.npy` | **~27 min** |
| 13 | `step6e_seam_feather.py` | `out/ortho_rgb_cut_f.npy` | 10 s |
| 14 | `step6c_mix.py`（带 `MIX_SF`/`MIX_DST`） | `out/ortho_rgb3_cut.npy` | 1 min |
| 15 | `export_tif.py` | `xiangjia_ortho.tif` + `.tfw`/`.prj`/预览图 | 2 min |

> 第 12~13 步是消除拼接"分割线"的关键，**不要跳过**（详见第 5 节）。
>
> 也可以手工分步跑（调试用）。只需要注意第 14 步的环境变量：
> ```bash
> MIX_SF=out/ortho_rgb_cut_f.npy MIX_DST=out/ortho_rgb3_cut.npy python step6c_mix.py
> python export_tif.py out/ortho_rgb3_cut.npy xiangjia_ortho.tif
> ```
> `export_tif.py` **需要 rasterio**，用装了 rasterio 的 venv 跑。

### 只看结果不想等？

第 11 步跑完之后，直接 `export_tif.py out/ortho_rgb.npy` 也能出图，
只是缝线会比较明显（那就是 v2 的水平）。建议至少跑到第 15 步。

---

## 4. 各步骤在做什么

| 步骤 | 作用 | 关键点 |
|---|---|---|
| **step1_meta** | 解析 XMP：GPS、云台 YPR、内参、畸变 | XMP 的 `RelativeAltitude` 是**相对起飞点**，不是地面 |
| **step2_init** | YPR → 初始旋转矩阵；算每张 footprint 建邻接图 | 只对 footprint 相交的影像对做匹配，省 90% 时间 |
| **step3_features** | SIFT（1/2 分辨率，每张 9000 点）+ F/H RANSAC 匹配 | `cv2.setNumThreads(1)` 避免多进程过度订阅 |
| **step4_ba** | 鲁棒 BA：Gauss-Seidel 逐相机、Huber 核、GPS 先验 σ=2m、漂移限制 15m | 不用稀疏求解器，逐相机迭代在大规模下反而稳 |
| **step5_dsm** | 粗 DSM（0.25m 网格 plane-sweep） | 给 step5c 当初值 |
| **step5f_gains** | 乘性曝光增益（同名点亮度比） | 只在没有 `radio.npy` 时被用到 |
| **step4b_radiometry** | 逐帧**仿射**辐射校正 `a·I+b`（Huber-IRLS，硬约束 Σa=n） | 比乘性增益更能吸掉雾霭/偏置差 |
| **step5c_dsm_refine** | DSM v2：0.10m 网格，粗搜 11 层@1.0m + 细搜 13 层@0.1m，逐格最高置信取胜 | 建筑扭曲的根因修复：高程精度从 ~3m 提到 ~0.1m 级 |
| **step5d_despike** | 去高程尖刺 | |
| **step5e_smooth** | Perona-Malik 保边平滑 | 平滑但保住屋檐阶跃 |
| **step6b_ortho** | **逐像素硬性单帧选取**：评分为 `nadir⁴ × 径向衰减`，遮挡沿光线 12 步采样 DSM 剔除；同时输出多视共识层 `ortho_avg.npy` | 这是"不模糊"的根本：每个像素只取一张影像的颜色 |
| **step6d_seamcut** | **接缝线优化（图割）** | 见第 5 节 |
| **step6e_seam_feather** | 缝合带羽化（仅 ±1px 内 5×5 中值） | 抹掉 ~1px 错位台阶 |
| **step6c_mix** | 低频色调共识：`out = sf + (LP(avg) − LP(sf))`，LP=2.5m 高斯 | 细节仍取单帧（不模糊），只把极低频色调换成多视共识 |
| **export_tif** | GeoTIFF（JPEG92 + tiled + 5 级金字塔） | |

---

## 5. 消除"分割线"的核心方法（v4 新增）

### 问题

**逐像素单帧选取**（step6b）保证不模糊，但代价是必然产生缝合线。把整幅视图标签场着色后
（`out/fig_seam_vidx.jpg`）可以看到，缝线**不是**沿地物的曲线，而是**规则的"砖墙"网格**：
每条航带内的影像被切成竖条、航带之间是横贯全区的水平直线。

这类直线与影像内容无关，必然横切房屋、道路；再叠加相邻影像间 ~1px 的 BA 残余错位，
就在缝线上留下肉眼可见的**"刀口"**。`out/fig_hotspots_cut.jpg` 第 1 行是最典型的证据：
一条水平线把一栋房子拦腰切断。

### 方法

**① 接缝线优化（`step6d_seamcut.py`）** —— 把缝合线挪到两视图差异最小的路径上：

```
E = Σ_p D(p, l_p) + Σ_(p,q) V(l_p, l_q)

D(p, a) = λ · (max(s_a, s_b) − s_a)                  偏离更优视图的几何代价
V(a, b) = |col_a − col_b| + 0.8 · min(max(|∇col_a|, |∇col_b|), 150)
                                                      颜色差  +  边缘惩罚
```

**边缘惩罚项是本次的关键**：只比"两视图颜色差"不够 —— 1px 错位在屋檐、田埂这类
**强边缘**上最扎眼，必须显式惩罚，缝线才会主动避开。

实现：对每一对相邻标签在「缝合线 ±`BAND_W`（默认 160px ≈ 8.8m）」的窄带内求解
二值 s-t 最小割（`scipy.sparse.csgraph.maximum_flow`，Dinic，100 万节点约 2.8s），
带外像素标签固定形成天然锚定，最优点自动把缝线"弯进"低代价路径。全图迭代 2 轮。

> 为什么不做全局 alpha-expansion？标签集太大（每块 10~15 个）时求解代价过高；
> 逐对二值割 + 迭代 2 轮在本问题上收敛良好（相邻标签对之间基本解耦）。

**② 缝合带羽化（`step6e_seam_feather.py`）** —— 抹掉残余的 ~1px 错位台阶：
**只在缝合带（±1px）内做 5×5 中值**，带外一个像素都不动。影响宽度 0.28m、占 1.41% 像素。

**③** 仍沿用低频色调共识（`step6c_mix.py`）。

### 效果

| 指标 | v3 | **v4** | 变化 |
|---|---|---|---|
| 缝线"刀口"超出（Sobel p95 相对大疆） | +47.85 | **−6.00** | **转负 → 缝线不再可见** |
| 缝线处 Sobel p95 | 229.4 | **172.5** | −25% |
| 跨线落差超出 中位 / p90（灰阶） | +3.00 / +41.00 | **+1.00 / +37.00** | −67% / −10% |
| `>8 灰阶` 接缝像素占比 | 42.0% | **37.7%** | −10% |
| 边缘锐度 ↑ | 2.267 | 2.252 | −0.7%（几乎无损） |
| 色块边界（相对大疆，同掩膜） | +0.061 | **−0.152** | 改善 |

**代价（如实记录）**：缝线总长增加 90%（155k → 296k 采样像素）——
min-cut 会把直线缝"弯"进低代价路径，路径更长，但每一段都更淡。

目视：`out/fig_seam_hotspots_v4.jpg` / `out/fig_hotspots_cut.jpg` 中
#1 横切房屋的水平线、#2 农田里的水平线、#4 树林里的竖线均**完全消失**；
#3 的青色区块边界明显变柔，但仍有淡淡余痕（属色调残差，非缝线位置问题）。

### 调参

全部通过环境变量，无需改代码：

| 变量 | 默认 | 含义 / 调法 |
|---|---|---|
| `BAND_W` | 160 | 缝线可移动范围(px)。**太小缝线动不了，太大会误改动大片区域**。第一次跑建议 100 |
| `LAM_D` | 800 | 数据项权重。调大 → 更保守（缝线更贴原几何最优）；调小 → 缝线更自由 |
| `EDGE_W` | 0.8 | 边缘惩罚权重。**这是让缝线避开屋檐/田埂的关键** |
| `EDGE_CAP` | 150 | 边缘惩罚上限（防止单点过强） |
| `N_ITER` | 2 | 迭代轮数 |
| `MIN_SEAM` | 40 | 一个标签对最少接缝像素数，低于此不处理 |

快速验证（只跑一个小窗口，秒级出结果）：

```bash
python step6d_seamcut.py win 562121.6 3324780.8 120    # E0 N1 size(米)
```

---

## 6. 目录结构

```
drone_photo_stitching/
├── README.md                 # 本文件
├── run_pipeline.py           # ★一键跑主链（--list / --from / --only / --dry-run）
├── 拼接方案报告.md            # 完整技术报告（含大疆结果分析、指标、踩坑）
├── requirements.txt
├── src/                      # 全部管线代码
│   ├── common.py             # 路径/投影配置(环境变量) / 坐标 / 相机模型 / 中文路径IO
│   ├── geo.py                # 经纬度 → UTM
│   ├── step1_meta.py ~ step6e_seam_feather.py, export_tif.py   # 主链
│   └── step8c_evaluate.py step8d_seam_eval.py seam_scan.py
│       hotspot_view.py cut_eval.py seam_lab.py make_figs_v4.py # 评估/出图
└── out/                      # 关键结果图与指标
    ├── fig_seam_hotspots_v4.jpg   # 【推荐先看】v3 vs v4 vs 大疆 逐热点对照
    ├── fig_hotspots_cut.jpg       # 4 栏：原图 / 优化后 / 缝线位置 / 大疆
    ├── fig_seam_vidx.jpg          # 视图标签场（看"砖墙网格"缝线）
    ├── fig_final_v4_full.jpg      # 成图全览
    ├── fig_cut_eval.jpg           # 窗口级接缝评估曲线
    ├── metrics.json               # 锐度 / 尺寸 / 统计
    └── metrics_seam.json          # 接缝指标 v3 vs v4
```

> 运行时会自动创建 `work/`（中间产物）与使用 `out/`。两者已在 `.gitignore` 里，
> 不会把几百 MB 的 npy 提交上去。

---

## 7. 常见坑（新数据集上容易遇到的）

| 现象 | 原因 / 处理 |
|---|---|
| `cv2.imread` 返回 None | 中文路径。本项目统一用 `common.imread_any`（`np.fromfile`+`imdecode`），别自己调 cv2 |
| 红蓝互换 | OpenCV 是 BGR，rasterio 是 RGB。混用必须显式换 |
| 内存爆掉 | `np.memmap(path,'w+')` 写的是裸二进制，`np.load` 读不了；要写合法 .npy 用 `np.lib.format.open_memmap` |
| 特征提取/渲染极慢 | 多进程里又开 cv2 多线程 = 严重过度订阅。`cv2.setNumThreads(1)` 后 326s → 97s |
| 输出是黑图 | 检查 `work/dsm_v2.npy` 是否存在且范围合理（`dsm_meta_v2.json` 里的 z_min/z_max） |
| `ModuleNotFoundError: rasterio` | 只有 `export_tif.py` 和评估脚本需要它，用装了 rasterio 的 venv |
| 换机型后建筑物扭曲 | `step1_meta.py` 的内参没读对（该机型 XMP 里没有 `DewarpData`），需手动填 `camera.json` |
| 绝对定位差几米 | 正常。单点 GPS 无 GCP/RTK，1~3m 是固有限制 |
