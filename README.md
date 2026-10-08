# 无人机航拍影像拼接（自研管线 · 对标大疆智图）

从 129 张 DJI M3E 航拍照片（含 XMP 元数据）出发，**自建** SfM 光束法平差 → 密集匹配 DSM → 逐像素单帧正射渲染的全流程，
输出与 DJI Terra 同规格的 2D 正射大图。

核心结论：在**边缘锐度**上优于大疆智图约 **10%**，建筑物无扭曲，色调连续无割裂感。

---

## 1. 最终成果

| 项 | 值 |
|---|---|
| 成品 | `out/xiangjia_ortho_mine.tif` |
| 尺寸 / 分辨率 | 16000 × 12073 px @ **5.50 cm/px** |
| 坐标系 | EPSG:32650（WGS84 / UTM 50N） |
| 全图有效覆盖率 | **98.43%** |
| 实际参与影像 | 109 / 129 张（单张最大占比仅 3.2%，分布均匀） |
| 文件 | 94.5 MB · 3 波段 RGB · uint8 · JPEG 压缩 + tiled + 5 级金字塔（2/4/8/16/32） |

配套文件：`out/xiangjia_ortho_mine.prj`、`out/xiangjia_ortho_mine.tfw`（供 GIS 直接读取）。

## 2. 四方对照指标

在大疆覆盖区（重叠掩膜）内统计，`gsd = 0.055 m`：

| 指标 | **v3 单帧+色调共识** | v2 纯单帧 | v1 多视平均 | DJI 智图 |
|---|---|---|---|---|
| 边缘锐度 ↑（感知型） | **2.219** | 2.208 | 2.169 | 2.008 |
| 色块边界超标 ↓（相对大疆） | **0.172** | 0.389 | 0.090 | 0（基准） |
| 跨切换线可见落差 ↓（灰阶） | **+4** | +8 | +2 | 0（基准） |
| 灰度均值 | 117.9 | 118.7 | 119.7 | 117.7 |

> 说明：锐度用**感知型指标** `强边缘处 |∇I| / 9×9 局部对比度`，而非拉普拉斯方差。
> 后者会被噪声误导（本例中大疆的拉普拉斯方差 192 反而最低，显然不代表它最糊）。
> 所有接缝类指标均**以 DJI 在同掩膜的值为基准相减**，扣掉真实地物边缘的干扰。

## 3. 目录结构

```
drone_photo_stitching/
├── src/                     自研管线（v3 主线 + 诊断/评估脚本，46 个 .py）
│   ├── common.py            公共工具：路径常量、旋转/相机模型、中文路径读写
│   ├── step1_meta.py        XMP 元数据解析（内参 + 位姿 + GPS）
│   ├── step2_init.py        YPR → 初始旋转 + footprint 邻接图
│   ├── step3_features.py    SIFT 提取 + F/H RANSAC 匹配
│   ├── step4_ba.py          鲁棒光束法平差（Huber + GPS 先验）
│   ├── step4b_radiometry.py 逐帧仿射辐射校正（增益 + 偏置，硬约束 Σa=n）
│   ├── step5_dsm.py         plane-sweep DSM (v1)
│   ├── step5c_dsm_refine.py DSM v2：0.10m 网格 + 粗/细两段搜索 + 置信度取胜
│   ├── step5d_despike.py    去尖刺
│   ├── step5e_smooth.py     Perona-Malik 保边平滑
│   ├── step6_ortho.py       正射渲染 (v1，多视加权平均)
│   ├── step6b_ortho.py      **逐像素硬性单帧选取渲染器**（v3 核心）
│   ├── step6c_mix.py        **低频色调共识混合**（细节取单帧，极低频色调取多视共识）
│   ├── export_tif.py        GeoTIFF 导出（JPEG92 + tiled + 金字塔）
│   ├── step8c_evaluate.py   四方对照评估（含防自欺的接缝/色块指标）
│   └── ...                  诊断与可视化：diag_*.py / check_*.py / crop_cmp.py / seam_probe.py 等
├── scripts/                 早期原型脚本（sfm / ortho / dsm / match 的迭代版本，不参与主链）
├── work/                    关键中间元数据（均 < 1 MB：内参、位姿、增益、辐射系数、匹配对、日志）
├── out/                     最终成果（GeoTIFF、指标 JSON、对照图）
├── docs/
│   ├── 拼接方案报告.md       完整方案报告
│   └── report_html/         早期 HTML 展示页
├── run_pipeline.py          一键跑通主链（--from / --only / --list）
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. 环境与依赖

- **Python 3.12**（实测 3.12.10 / 3.13 均可）
- 依赖见 `requirements.txt`，实测版本：

| 库 | 实测版本 | 用途 |
|---|---|---|
| numpy | 2.5.3 | 全流程 |
| scipy | 1.18.1 | 稀疏最小二乘（`scipy.sparse.linalg`）、滤波 |
| opencv-python | 5.0.0 | SIFT、remap、模板匹配 |
| Pillow / tifffile | 12.3.0 / 2026.9.20 | 图像读写 |
| rasterio | 1.5.2 | GeoTIFF 读写与导出 |
| pyproj | 3.8.0 | 投影与大地水准面 |
| shapely | 2.2.0 | footprint 几何 |

```bash
pip install -r requirements.txt
```

## 5. 快速开始

**一条命令跑通主链**（推荐）：

```bash
python run_pipeline.py --list          # 先看有哪些步骤
python run_pipeline.py                 # 从头跑：SfM → DSM → 正射 → 导出
python run_pipeline.py --from step6b   # 从某一步开始（复用已有中间产物）
python run_pipeline.py --only export   # 只跑某一步
```

运行时先打印路径 / 影像数 / 投影带号，缺影像目录会给出明确提示；某步失败立即停止并报出该步名称。

**等价的手工分步执行**（调试时用）：

```bash
python src/step1_meta.py
python src/step2_init.py
python src/step3_features.py
python src/step4_ba.py
python src/step4b_radiometry.py
python src/step5c_dsm_refine.py
python src/step5d_despike.py
python src/step5e_smooth.py

python src/step6b_ortho.py        # 逐像素单帧选取（约 9 分钟，16 进程）
python src/step6c_mix.py          # 低频色调共识混合
python src/export_tif.py ../out/ortho_rgb3.npy ../out/xiangjia_ortho_mine.tif

python src/step8c_evaluate.py     # 四方对照评估 → out/metrics_v3.json（需大疆成果，可选）
python src/make_figs.py           # 生成交付对照图 fig_*.jpg（需大疆成果，可选）
```

性能基准（本机 i9-9980XE / 18 核 / 16 worker）：

| 步骤 | 耗时 |
|---|---|
| 特征提取 | 23 s |
| 匹配（1219 对） | 2.5 min |
| BA | 1 min |
| DSM v2 | ~15 min |
| 全图单帧渲染 | 568 s |
| 低频混合 | 56 s |

## 6. 换一个新采集的图片目录怎么跑

### 6.1 三步走

```bash
cd drone_photo_stitching

# ① 指向新数据（Windows PowerShell 用 $env:DRONE_IMG_DIR="..."）
export DRONE_IMG_DIR="/d/data/2026_11_flight/images"

# ② 如测区不在 UTM 50 带，改带号（见 6.3）
export DRONE_UTM_ZONE=51

# ③ 跑
python run_pipeline.py
```

成果落在 `out/xiangjia_ortho_mine.tif`（附 `.tfw` / `.prj`），中间产物落在 `work/`。**不用改任何代码。**

### 6.2 可配置项

所有路径集中在 `src/common.py` 顶部，**环境变量优先，其次用默认值**：

| 环境变量 | 含义 | 默认值 |
|---|---|---|
| `DRONE_IMG_DIR` | 原始航拍影像目录（`*.JPG`） | `<DATA>/xiangjia1_186m` |
| `DRONE_DATA_DIR` | 数据集根目录 | 仓库根 |
| `DRONE_DJI_DIR` | 大疆智图工程目录（**可选**，仅供评估对照） | `<DATA>/xiangjia1` |
| `DRONE_WORK` | 中间产物目录 | `<仓库根>/work` |
| `DRONE_OUT` | 成果目录 | `<仓库根>/out` |
| `DRONE_UTM_ZONE` | UTM 带号（北半球） | `50` → EPSG:32650 |

不想用环境变量，就直接改 `src/common.py` 里 `_cfg(...)` 的默认值（该文件第 8 行起有注释说明）。

### 6.3 换地区必须确认的两件事

1. **投影带号**。默认 `zone 50` → EPSG:32650，覆盖东经 114°~120°（华南）。带号公式 `zone = floor(lon / 6) + 31`：
   上海 121°E → zone 51（EPSG:32651）；北京 116°E → zone 50；成都 104°E → zone 48；乌鲁木齐 87°E → zone 45。
   设 `DRONE_UTM_ZONE` 即可，`geo.py` / `step1` / `step2` / GeoTIFF 输出会全部跟随。
2. **原始影像必须保留大疆原始 XMP**。管线从每张 JPG 的 XMP 读相机内参、GPS 与云台 YPR；
   用微信/看图软件"另存为"过会丢 XMP。务必用采集的原始文件。

### 6.4 数据要求

| 项 | 要求 |
|---|---|
| 机型 | 按 DJI M3E 内参解析；其他机型需核对 `step1_meta.py` 的标定字段 |
| 格式 | `*.JPG`，含 XMP（`drone-dji:*` 命名空间）|
| 重叠度 | 建议航向 ≥ 70%、旁向 ≥ 60% |
| 数量 | 无硬上限，DSM 与渲染耗时随张数近似线性增长 |

### 6.5 只跑到某一步

```bash
python run_pipeline.py --from step5c   # 复用已有 step1~4b 产物，只重跑 DSM 及之后
python run_pipeline.py --only step6b   # 只重渲染（调了渲染参数时最常用）
```

### 6.6 评估脚本不要照搬（重要）

`step8b/8c_evaluate.py`、`make_figs.py`、`crop_cmp.py`、`seam_probe.py`、`diag_*.py` 这些是
**针对本项目大疆成果做四方对照**的专用脚本，内部写死了大疆 `result.tif` 与本项目 v1/v2 中间版本。
`step7_export.py` 同理，属 v1 旧版导出（现用 `export_tif.py`）；`scripts/` 下的原型脚本路径也未做可配置化。
换新数据集后不要直接跑它们 —— 它们只用于复现本 README 第 2 节的对照结论。
**主链（step1 → export）不依赖大疆任何产物，可独立跑完。**

### 6.7 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `原始影像目录不存在` | `DRONE_IMG_DIR` 未设或路径写错 |
| step1 报"发现 0 张影像" | 目录里不是 `.JPG`，或扩展名被改掉 |
| step3 匹配对很少 | 影像重叠不足，或 XMP 丢失导致初始位姿偏差过大 |
| step6b 报"缺少 dsm_v2.npy" | 先跑 `python run_pipeline.py --from step5c` |
| export 报没有 rasterio | export 需要单独装 rasterio（见 `requirements.txt`）|
| 成果套不上地图 | 带号不对（见 6.3），或该批数据无 GPS 固定解 |

## 7. 已知不足（不掩饰）

1. **切换线处有细微跳变**。跨切换线中位落差比大疆高约 +4 灰阶，Sobel p95 高约 25。
   根因是相邻帧 BA 残差约 1 px 的错位在硬切换线上切过地物形成细微断口；
   低频色调共识只能压「色块」，压不掉「断口」。**彻底解决需图割（graph-cut）接缝线优化。**
2. **噪声高于大疆**（拉普拉斯方差 826 vs 192）。自研保留了更多原始纹理——这正是「放大后清楚」的来源，代价是观感不如大疆干净。
3. **绝对定位差 1~3 m**（粗尺度实测约 +1.3 m / +2.0 m）。两套结果均无 GCP / RTK；大疆把位姿强绑 GPS，自研 BA 由影像网主导。
   内部拼接一致性不受影响（重投影 p50 = 0.76 px ≈ 地面 4 cm）。

## 8. 数据说明与体积提示

- 仓库**不包含**原始航拍影像（`xiangjia1_186m/`，约 1.5 GB）与大疆智图成品（`xiangjia1/map/result.tif`）。
- 仓库包含最终成品 `out/xiangjia_ortho_mine.tif`（**94.5 MB**）。GitHub / Gitee 对单文件超过 50 MB 会警告、超过 100 MB 会拒收，
  该文件处于临界区。若不便入库，建议二选一：
  - 使用 **Git LFS**：`git lfs track "out/*.tif"`，再 `git add .gitattributes`；
  - 或在 `.gitignore` 中取消 `# out/*.tif` 一行的注释改为忽略，改以 Release 附件形式分发。
- `cache/`（约 2.9 GB）与 `work/` 中的大数组（`dsm_v2.npy`、`feats.npz` 等，合计约 0.7 GB）为运行时中间产物，**不入库**，可由管线重新生成。
