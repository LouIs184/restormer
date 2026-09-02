# 投影域（post-log sinogram）训练适配方案 v2 —— 方案 B（按视角）+ yml 域切换

> 依据 `D:\ImageReconstruction\develop\AICT-code` 已跑通的投影域做法改写。
> 目标：在 restormer 框架里通过 **yml 的 `domain` 开关** 一键切换「图像域」/「投影域」训练。
> 数据格式见 [DATASET_FORMAT.md](./DATASET_FORMAT.md)。

---

## 〇、结论速览

- **切片方式**：方案 B —— 每个视角的 2D 探测器平面 `(128, 512)` 当作一张图，与 AICT-code 的 `proj_reader` 完全一致。
- **域切换**：把 `Dataset_CTImage` 升级为统一的 `Dataset_CT`，yml 里 `domain: image`（图像域）/ `domain: proj`（投影域），只改这一行 + 对应归一化参数。
- **投影域 recipe**（AICT 已验证）：`proj_norm_clip_max: 0.28`、`proj_crop_rows: 16`（128−32=96 行）、全 720 视角、**禁用几何增强**、可加大 batch。
- **关键差异**：投影域裁剪后高度只有 96，**渐进式 patch 尺寸上限从 256 降到 96**；归一化从 HU 换成 `clip(x, 0, 0.28)/0.28`。

---

## 一、方案 B：按视角取探测器平面

### 1.1 数据流

```
proj_*.raw  (720, 128, 512) = (views, rows, channels)
    │ 按视角索引 v (0..719)
    ▼
proj[v]  (128, 512)   —— 第 v 个视角的 2D 探测器平面
    │ 裁上下黑边（AICT: crop_rows=16 → (96, 512)）
    │ clip(x, 0, 0.28)/0.28 归一化 → [0,1]
    ▼
单通道输入图 (1, 96, 512)，直接喂 Restormer
```

输入 `proj_pbi_...` 与目标 `proj_no_pbi` **逐视角对齐**（DATASET_FORMAT 第三、四节已确认），训练样本对即 `(proj_pbi[v], proj_no_pbi[v])`。

### 1.2 样本量

| 域 | 每病人样本 | train 10 病人 | val 3 病人 |
|----|-----------|--------------|------------|
| 图像域 | 158 切片 | 1580 | 474 |
| **投影域** | **720 视角** | **7200** | **2160** |

投影域样本量约是图像域的 4.5 倍，且每样本 `(96,512)` 面积只有 512² 切片的约 1/5，**训练每 epoch 更快、显存更省**。

### 1.3 尺寸约束：Restormer 比 DenseUNet2d 宽松

| 网络 | 下采样方式 | 要求输入 H,W 能被 | 96 行能否用 |
|------|-----------|------------------|-------------|
| DenseUNet2d（AICT） | 4 次 MaxPool stride=2 | 16 整除 | ✅ 96÷16=6 |
| **Restormer（本仓库）** | 3 次 PixelUnshuffle(2) | **8 整除** | ✅ 96÷8=12 |

Restormer 只需 2³=8 整除（`restormer_arch.py` 的 `down1_2/2_3/3_4`），验证 `pad_test` 的 `window_size=8` 也只需 8 对齐。所以 **`crop_rows=16` 得到 96 行对 Restormer 完全兼容**，且 96 同时满足 16 对齐——将来若想换回 DenseUNet 也不冲突。

---

## 二、yml 域切换设计

核心：**一个统一 dataset 类 `Dataset_CT`，读 yml 里的 `domain` 字段分派**。`domain` 键属于 `datasets.train` / `datasets.val` 段，框架会原样传给 dataset 的 `opt`。

### 2.1 切换方式

| 想训练 | 改 yml 哪里 |
|--------|-------------|
| 图像域 | `type: Dataset_CT` + `domain: image` + `txt: ./train_img.txt` + HU 参数 + `gt_size` 256 |
| **投影域** | `type: Dataset_CT` + `domain: proj` + `txt: ./train_proj_img.txt` + 投影参数 + `gt_size` 96 |

网络、优化器、损失、调度器**完全不用动**（都是单通道 2D 回归任务）。

### 2.2 图像域 ↔ 投影域 yml 段对照

```yaml
datasets:
  train:
    type: Dataset_CT
    domain: proj              # ← 唯一的域开关：'image' | 'proj'
    txt: ./train_proj_img.txt #    image 时换成 ./train_img.txt

    # --- 图像域参数（domain: image 时生效，与现配置一致）---
    hu_min: -1000
    hu_max: 1000
    geometric_augs: true      # image 开随机翻转+90°旋转；proj 必须关

    # --- 投影域参数（domain: proj 时生效，同 AICT-code proj_reader）---
    proj_norm_clip_max: 0.28  # 归一化：clip(x,0,0.28)/0.28
    proj_crop_rows: 16        # 裁上下黑边：128 - 2×16 = 96 行

    # --- 渐进式训练：patch 必须 ≤ 样本短边 ---
    #   image: gt_size 256, gt_sizes [128,160,192,256]
    #   proj:  gt_size 96,  gt_sizes [48,64,80,96]   ← 裁剪后 H=96
    mini_batch_sizes: [4, 3, 2, 1]
    iters: [30000, 24000, 18000, 18000]
    gt_size: 96
    gt_sizes: [48, 64, 80, 96]
```

> ⚠️ **最重要的一条**：`gt_sizes` 里的最大值必须 ≤ 裁剪后的高度 96。若沿用图像域的 256，`padding()` 会把 96 行 reflect 补到 256，模型就训练在**大部分是补出来的人造数据**上。见 §4.1。

---

## 三、代码实现

### 3.1 统一 dataset：`basicsr/data/ct_image_dataset.py` → 改成 `Dataset_CT`

把现有 `Dataset_CTImage` 原地升级为 `Dataset_CT`（图像域行为保持不变，只是增强加了 `domain == 'image'` 门控）。`basicsr/data/__init__.py` 自动扫描 `*_dataset.py` 注册类名，无需改注册表。

```python
## 双能相位 CT 数据集：通过 yml 的 domain 一键切换 图像域/投影域
## domain = 'image' -> 读 rec_*.raw 切片 (512,512)，HU 归一化
## domain = 'proj'  -> 读 proj_*.raw 视角 (128,512)，clip 归一化（同 AICT proj_reader）
## 索引文件每行: <输入raw> <目标raw> <索引>（image=切片号z，proj=视角号v）
import locale
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding


class Dataset_CT(data.Dataset):
    def __init__(self, opt):
        super(Dataset_CT, self).__init__()
        self.opt = opt
        self.phase = opt['phase']          # 'train' / 'val'，框架自动注入
        self.scale = opt['scale']          # 固定为 1
        txt = opt['txt']
        try:
            with open(txt, encoding='utf-8') as f:
                self.paired = f.readlines()
        except UnicodeDecodeError:
            with open(txt, encoding=locale.getpreferredencoding()) as f:
                self.paired = f.readlines()

        # ---- 域开关（yml 里切换）----
        self.domain = opt.get('domain', 'image')      # 'image' | 'proj'

        # 图像域参数
        self.hu_min = float(opt.get('hu_min', -1000.0))
        self.hu_range = float(opt.get('hu_max', 1000.0)) - self.hu_min
        # 投影域参数（同 AICT-code proj_reader）
        self.proj_norm_clip_max = float(opt.get('proj_norm_clip_max', 0.28))
        self.proj_crop_rows = int(opt.get('proj_crop_rows', 0))

        self.geometric_augs = opt.get('geometric_augs', True)
        self._vol_cache = {}

    def _volume(self, path):
        if path not in self._vol_cache:
            shape = (160, 512, 512) if self.domain == 'image' else (720, 128, 512)
            self._vol_cache[path] = np.memmap(
                path, dtype=np.float32, mode='r', shape=shape)
        return self._vol_cache[path]

    def _norm(self, x):
        if self.domain == 'image':
            # 图像域：HU 线性归一化 (x+1000)/2000
            return np.clip((x - self.hu_min) / self.hu_range, 0, 1).astype(np.float32)
        else:
            # 投影域：AICT proj_reader 的 clip 归一化
            return np.clip(x, 0, self.proj_norm_clip_max) / self.proj_norm_clip_max

    def __getitem__(self, index):
        line = self.paired[index].strip()
        inp_path, gt_path, idx = line.split()
        idx = int(idx)

        inp_vol = self._volume(inp_path)
        gt_vol = self._volume(gt_path)

        if self.domain == 'image':
            img_lq, img_gt = inp_vol[idx], gt_vol[idx]        # (512,512) 切片
        else:
            img_lq, img_gt = inp_vol[idx], gt_vol[idx]        # (128,512) 视角
            if self.proj_crop_rows > 0:                        # 裁上下黑边
                img_lq = img_lq[self.proj_crop_rows:-self.proj_crop_rows, :]
                img_gt = img_gt[self.proj_crop_rows:-self.proj_crop_rows, :]

        img_lq = np.expand_dims(self._norm(img_lq), axis=2)   # (H,W,1)
        img_gt = np.expand_dims(self._norm(img_gt), axis=2)

        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_lq, img_gt = padding(img_lq, img_gt, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                self.scale, None)
            if self.geometric_augs and self.domain == 'image':
                # 投影域禁 90° 旋转（交换视角轴/通道轴，物理无意义）
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': inp_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paired)
```

改动相对 `Dataset_CTImage` 只有 3 处：
1. `domain` 分派 memmap 形状 `(160,512,512)` / `(720,128,512)`；
2. `_norm` 按域选 HU 线性 或 clip 归一化；
3. `proj_crop_rows` 裁黑边 + 增强门控 `domain == 'image'`。

> 注：AICT-code 的 `proj_reader` 用 `np.fromfile(..., offset=view_idx*128*512*4)` 按视角 seek 读取。这里沿用本仓库的 `np.memmap`（随机访问单视角同样高效，页缓存负责），语义完全一致。若想 1:1 照搬 AICT 的 seek 方式也完全可以。

### 3.2 索引文件：`prepare_ct_proj_data.py`（新写）

对照 `prepare_ct_data.py`：换 proj 文件、第三列改视角号、写全 720 视角（AICT 做法）。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成投影域数据索引 train_proj_img.txt / valid_proj_img.txt（视角 v=0..719）。
病人划分与 prepare_ct_data.py 一致（train 10 次 / valid 3 次）。
每行格式：<输入.raw绝对路径> <目标.raw绝对路径> <视角号>

用法:
    python prepare_ct_proj_data.py
    python prepare_ct_proj_data.py --data-root /root/autodl-tmp/联影双能相位数据2080 --output ./
"""

import argparse, os

INP = "proj_pbi_fs_dec_blur_phase_100000.raw"   # 输入（含伪影）
GT  = "proj_no_pbi.raw"                          # 目标（干净）
NUM_VIEWS = 720

TRAIN = [ ...同 prepare_ct_data.py 的 TRAIN... ]
VALID = [ ...同 prepare_ct_data.py 的 VALID... ]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/root/autodl-tmp/联影双能相位数据2080")
    p.add_argument("--output", "-o", default="./")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for fname, patients in [("train_proj_img.txt", TRAIN),
                            ("valid_proj_img.txt", VALID)]:
        with open(os.path.join(args.output, fname), "w", encoding="utf-8") as f:
            for pat in patients:
                inp = os.path.join(args.data_root, pat, INP)
                gt  = os.path.join(args.data_root, pat, GT)
                for v in range(NUM_VIEWS):
                    f.write(f"{inp} {gt} {v}\n")
        print(f"已生成 {fname}: {len(patients)} 病人 x {NUM_VIEWS} 视角 "
              f"= {len(patients) * NUM_VIEWS} 行")

if __name__ == "__main__":
    main()
```

### 3.3 完整投影域 yml：`Options/CT_ProjectionDomain_Restormer.yml`

```yaml
name: CT_ProjectionDomain_Restormer
model_type: ImageCleanModel
scale: 1
num_gpu: 1

datasets:
  train:
    name: TrainSet
    type: Dataset_CT
    domain: proj                  # ← 域开关
    txt: ./train_proj_img.txt

    proj_norm_clip_max: 0.28      # 投影域归一化（同 AICT proj_reader）
    proj_crop_rows: 16            # 128 - 2×16 = 96 行
    geometric_augs: false         # 投影域禁 90° 旋转

    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 4         # 须 = max(mini_batch_sizes)=4（渐进式阶段从这里子采样）

    # 渐进式训练：gt_sizes 上限 ≤ 96（裁剪后高度）
    mini_batch_sizes: [4, 3, 2, 1]
    iters: [30000, 24000, 18000, 18000]
    gt_size: 96
    gt_sizes: [48, 64, 80, 96]

  val:
    name: ValSet
    type: Dataset_CT
    domain: proj
    txt: ./valid_proj_img.txt
    proj_norm_clip_max: 0.28
    proj_crop_rows: 16
    io_backend:
      type: disk

network_g:
  type: Restormer
  inp_channels: 1     # 单通道视角平面；双能可设 2（见 §4.5）
  out_channels: 1
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: WithBias
  dual_pixel_task: False

# ---- path / train / val / logger / dist_params 与图像域 yml 相同 ----
# train.pixel_opt 保持 L1Loss（AICT 投影域基线用的是 MSE，想换就改成 MSELoss）
```

> **切回图像域** = 把 `domain: proj` → `domain: image`、`txt` 换 `train_img.txt`、去掉 `proj_*` 参数、恢复 `hu_min/hu_max`、`gt_size`/`gt_sizes` 换回 `256/[128,160,192,256]`、`geometric_augs: true`。其余不动。

### 3.4 推理脚本：`test_ct_proj.py`（在仓库，产出 .raw + 指标）

见仓库根目录 `test_ct_proj.py`：读 proj 文件、逐视角推理、裁黑边，产出 **`<病人>_pred/input/gt.raw`** 三种 `.raw`（input/gt 已按 `proj_crop_rows` 裁剪、与 pred 同形状）+ 终端 **MSE/RMSE/MAE/PSNR/SSIM**（原始值域，同 AICT-code infer.py）。

- `proj_crop_rows` / `proj_norm_clip_max` **默认从训练 yml 读取**，保证与训练一致（可用 `--crop_rows` / `--proj_clip_max` 覆盖）。
- 脚本**不含重建**。要图像域结果，把 `<病人>_pred.raw` 接你的重建流程（正确几何见 §4.4）。
- `--data_root` 默认是服务器路径 `/root/autodl-tmp/联影双能相位数据2080`；本地（WSL/Windows）推理必须显式传 `--data_root` 指向本地数据目录。

```bash
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Restormer/models/net_g_latest.pth \
    --data_root /root/autodl-tmp/联影双能相位数据2080 \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1
```

---

## 四、必须注意的坑

### 4.1 渐进式 patch 尺寸必须 ≤ 裁剪后高度 96（最重要）

框架的 `padding()` 会把样本 reflect 补到 `gt_size` 再随机裁剪。投影域裁剪后 H=96，若 `gt_sizes` 用到 256，96 行会被补到 256，**训练样本大部分是人造反射边**。所以：

```
投影域：gt_size: 96,  gt_sizes: [48, 64, 80, 96]   # 全部 ≤ 96 且是 8 的倍数
图像域：gt_size: 256, gt_sizes: [128, 160, 192, 256]  # 不变
```

48/64/80/96 都是 8 的倍数，Restormer 的 3 次下采样无奇偶问题。换 DenseUNet 时 48/80 不是 16 的倍数，但用 Restormer 无碍。

### 4.2 归一化：`clip(x, 0, 0.28)/0.28`，全局固定

- 这是 **AICT 已验证** 的方案，全数据集统计范围约 `[-0.066, 0.276]`，clip 到 `[0, 0.28]` 再缩放。
- 负值（校准/双能分解残留）被 clip 成 0，输入和目标的负区都归 0，loss 无惩罚，无碍。
- **不要**逐样本 min-max 归一化：会破坏 Restormer 全局残差 `out=net(x)+x` 的可加性（post-log 伪影增量本就可加，见 DATASET_FORMAT 第 4 条）。

### 4.3 增强：禁 90° 旋转；mixup 可选

- `random_augmentation` 含 90° 旋转，对 sinogram **物理无效**（交换「视角轴」与「通道轴」）。投影域必须 `geometric_augs: false`。
- 翻转（翻转通道方向/视角顺序）可选，但 AICT 投影域配置**没开任何几何增强**，先照抄。
- mixup：框架的 `Mixing_Augment` 是 batch 级混合，混合两个视角的 sinogram，物理无害，可选（AICT 的 proj 配置没开）。

### 4.4 重建环节（脚本外）的几何参数

- sinogram 域 PSNR 只用于训练监控；**最终验收必须把 `<病人>_pred.raw` 重建到图像域**与 `rec_no_pbi.raw` 比 PSNR/SSIM。`test_ct_proj.py` 只产出 sinogram，**不含重建**。
- 数据是扇/锥束，若用 skimage `iradon` 只做平行束近似、仅限快速定性。精确几何（源距、探测器间距、扇角）从 **AICT-code 的重建代码** 获取，或用 `astra-toolbox` / `pytorch-fbp`。

### 4.5 通道数与损失

- **通道数**：当前数据集每病人一套 `proj_*.raw`，单能 `inp=1/out=1`。双能若有两个能级的 proj 文件，可 `inp=2/out=2` 拼通道维，归一化对两能级用同一 `clip_max`。
- **损失**：本仓库用 L1（`pixel_opt.type: L1Loss`），post-log 噪声近似高斯，L1 是合理起点。AICT 投影域基线用的是 **MSE**（`mse_weight: 1.0`），想在 restormer 里换就改 `pixel_opt.type: MSELoss` —— 这也是一个 yml 开关。

### 4.6 首尾视角

AICT 确认 **不需要剔除**，用全部 720 视角（`range(0, 720)`）。DATASET_FORMAT 提示首尾可能有机械边界伪影，若实测有坏帧，可改用 `range(1, 719)` 并重新生成索引，dataset 无需改动。

---

## 五、改动清单

- [x] `basicsr/data/ct_image_dataset.py`：`Dataset_CTImage` → 升级为 `Dataset_CT`（加 `domain` 分派，见 §3.1）
- [x] 把现有 `Options/CT_ImageDomain_Restormer.yml` 的 `type` 改为 `Dataset_CT`、`domain: image`（验证图像域回归）
- [x] 新建 `prepare_ct_proj_data.py`（在服务器跑它生成 `train_proj_img.txt / valid_proj_img.txt`）
- [x] 新建 `Options/CT_ProjectionDomain_Restormer.yml`（见 §3.3）
- [x] 新建 `test_ct_proj.py`（投影域推理，产出 `<病人>_pred/input/gt.raw` + 指标，见 §3.4）
- [ ] 用 `analyze_proj_range.py`（可移植自 AICT-code）复核 `proj_norm_clip_max` 与黑边行数
- [ ] 从 AICT-code 获取 FBP 几何参数，做最终图像域验收

---

## 附录：与 AICT-code 的对应关系

| 项目 | AICT-code（已跑通） | restormer（本次改造） |
|------|--------------------|----------------------|
| 切片方式 | `proj_reader` 按视角 `(128,512)` | `Dataset_CT` 按视角 `(128,512)` |
| 索引文件 | `train_proj_img.txt`：`<inp> <gt> <view>` | 同格式，`prepare_ct_proj_data.py` 生成 |
| 归一化 | `clip(x, 0, norm_clip_max=0.28)/0.28` | `proj_norm_clip_max: 0.28`，语义一致 |
| 黑边裁剪 | `crop_rows: 16` → `(96,512)` | `proj_crop_rows: 16` → `(96,512)` |
| 几何增强 | 无 | `geometric_augs: false` |
| 视角范围 | 全 720 | 全 720 |
| 模型 | DenseUNet2d（需 16 对齐） | Restormer（需 8 对齐，更宽松） |
| 损失 | MSE（基线）/ 复合 TM 损失 | L1（`pixel_opt` 可切 MSELoss） |
