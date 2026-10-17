# 适配西门子 PBI 投影数据（`ximenzi_pbi_data`）

> 状态：**已实施**。
> 本文档记录改动内容、参数取值依据（均为实测），以及在新数据集上的运行步骤。
> 相关文档：[5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md)（投影域方案）、[4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md)（参数含义）、[2-EXECUTION.md](2-EXECUTION.md)（执行流程）。

---

## 〇、结论

**只改 yml 不够，必须改代码。** 根因是投影数据形状从 `(720,128,512)` 变成 `(1000,500,500)`，而形状原先硬编码在 `ct_image_dataset.py` 里，yml 无法覆盖。

本次改动的核心思路是**把形状/裁剪量参数化**，从此换数据集只改 yml：

| 文件 | 类型 | 改动 |
|------|------|------|
| `basicsr/data/ct_image_dataset.py` | 修改 | 新增 `proj_shape` / `image_shape` / `proj_crop_cols`；修掉 `[r:-r]` 隐患；`proj_norm_clip_max` 留空直接报错 |
| `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml` | **新增** | 新数据集配置（`clip_max` 待标定） |
| `prepare_ximenzi_proj_data.py` | **新增** | 生成 train/valid/test 三个索引 txt |
| `test_ct_proj.py` | 修改 | 形状/视角数/裁剪全部从 yml 推导，新增 `--txt` |
| `analyze_proj_range.py` | **新增** | 投影域真值域统计，标定 `clip_max` + 复核裁剪边界 |

**未改动**：网络、损失、优化器、调度器、`Options/CT_ProjectionDomain_Restormer.yml`（联影配置仍可用）、`prepare_ct_proj_data.py`、图像域整条线。

---

## 一、实测的新旧数据差异

以下为对本地数据实测所得，不是从文档推断的。

| | 旧（联影双能相位） | 新（西门子 PBI） |
|---|---|---|
| 目录 | 13 个 `病人ID_studyID_960+_AXIAL_...` | `01` … `30` |
| 输入文件 | `proj_pbi_fs_dec_blur_phase_100000.raw` | `proj_pbi_fs_dec_blur_phase.raw`（**无** `_100000`） |
| 目标文件 | `proj_no_pbi.raw` | 同 |
| 形状 | `(720, 128, 512)` = 视角×行×道 | `(1000, 500, 500)` |
| 值域 | −0.0034 ~ **0.1977** | −0.0385 ~ **0.0939** |
| 全零边框 | 无 | **有**（见 §二） |
| 重建数据 | 有 `rec_*.raw` | **无** |

### 1.1 存储顺序（坑）

新数据集 `README.md` 写的形状是 `(500, 500, 1000)`，**这句有误导性**。实测把文件按 **`(1000, 500, 500)`** 解析、取第 `k` 片，与 `gt/gt_k.raw` **逐字节完全相同**（已验证 k = 0,1,2,100,250,500,750,999）。

真实布局是**连续 1000 个 500×500 平面**，最外层才是视角号。若照 README 字面 `reshape(500,500,1000)` 去读，拿到的是完全错乱的数据。

> 建议顺手修正外部数据目录 `ximenzi_pbi_data/README.md` 的那句话（由你决定）。

### 1.2 新数据没有重建数据

`01/` 下只有 `proj_*.raw` 和投影切片 `gt/`、`lq/`，README 也只描述投影。
→ **`CT_ImageDomain_Restormer.yml` 在新数据上不可用**，只有投影域能训。

### 1.3 索引 txt 不在 git 里

`.gitignore` 含 `*_img.txt` / `*_proj.txt`，索引文件需在服务器上跑 prepare 脚本生成。

---

## 二、裁剪量：`crop_rows: 122` / `crop_cols: 0`

对病人 01 全部 1000 个视角逐像素统计：

```
目标(gt) 非零包围盒：行[118, 381] 恒成立，列[45, 454] 仅在部分视角成立
行方向：包围盒外能量 = 0.000000%   ← 完全无信号
列方向：各视角投影足迹宽度不同（旋转展开），全 1000 视角并集为列[0, 499]
```

**行可以裁，列不能裁**——列方向的边界随视角变化（视角 1 是 `[44,455]`、视角 2 是 `[43,456]`…，旋转中物体的投影足迹在探测器上扫过），不存在一个对全部视角都无损的固定列裁剪值。列方向裁掉 `[0,44]∪[455,499]` 会损失 0.51% 的目标能量，故取 `crop_cols: 0`。

行方向的候选值（8 / 16 整除要求下）：

| crop_rows | 行数 | 8 整除 | 16 整除 | 能量损失 |
|-----------|------|--------|---------|---------|
| 118 | 264 | Y | n | 0.0002% |
| **122** | **256** | **Y** | **Y** | **1.74%** |
| 126 | 248 | Y | n | 4.00% |
| 128 | 244 | **n** | n | 5.17% |

取 **122 → 256 行**（256 = 16×16）。注意 `500−2×128 = 244` **连 8 整除都不满足**，所以原始设想的 128 不可用；122 是从 128 向下取的第一个合格值，同时把损失从 5.17% 降到 1.74%。

**最终平面：256 × 500。**

> 若要完全无损可改 `crop_rows: 114`（→272 行，损失 0.0002%，同样 16 整除），代价是每样本多 6% 算力。

---

## 三、整除约束（为什么是 8 和 16）

| 网络 | 下采样方式 | 要求 |
|------|-----------|------|
| **Restormer（本仓库）** | 3 次 `PixelUnshuffle(2)`（`down1_2/2_3/3_4`） | **8 整除** |
| DenseUNet2d（AICT 基线） | 4 次 `MaxPool stride=2` | 16 整除 |

本次仍按 **16 整除**取值（256、`gt_sizes` 全为 16 的倍数），以便将来换回 DenseUNet 做基线对比时不冲突——这也是当初联影配置取 96 的思路。

**注意区分两个尺寸**：

- **训练 patch**（`gt_size`）**必须** 8 整除 —— `paired_random_crop` 会显式裁出 `gt_size×gt_size`，且要求 `平面边长 >= gt_size`，否则抛 `ValueError: LQ (...) is smaller than patch size`。
- **整个平面**（256×500）**不需要**整除 —— 验证集走 `pad_test`，会 reflect 补齐到 `window_size`(=8) 的倍数再裁回；500 会被补成 504。训练时也只从平面里裁 patch。

---

## 四、归一化：`proj_norm_clip_max` 必须重新标定

投影域归一化是单侧截断 `norm(x) = clip(x, 0, clip_max) / clip_max`，**只用上界**。

联影的 `0.28` 是针对联影数据标的（实测联影 max = 0.1977，留约 1.4× 余量）。若新数据沿用 0.28：

```
0.0939 / 0.28 = 0.335   →  归一化值全挤在 [0, 0.335]，动态范围浪费约 2/3
```

**所以 yml 里 `proj_norm_clip_max` 留空（`~`），不会静默套用旧值**——`Dataset_CT` 读到 `None` 会直接报错并提示先跑标定脚本。

标定方式：

```bash
python analyze_proj_range.py --data-root /mnt/tm/ximenzi_pbi_data
```

脚本会输出 min/max/分位数、各候选 `clip_max` 的截断像素占比，并给出建议值；把建议值填进 yml 的 `train` 和 `val` 两处（两处必须一致）。

> 病人 01（隔 50 张抽样）实测：目标 max ≈ 0.0946，99.99% 分位 ≈ 0.0886。**全 20 个病人的真值需在服务器上跑脚本确认后再定稿。**

> ⚠️ `clip_max` 变了，归一化尺度就变了（yml 里 `use_image: false`，验证 PSNR 直接在归一化域算），**所以新数据的验证 PSNR 不能和联影旧实验的数值横向比较**。要跨数据集比，只能看 `test_ct_proj.py` 输出的原始值域指标。

---

## 五、数据划分

20 个病人（`01`~`20`），按**病人**划分（同一病人的全部视角只出现在一个集合）：

| 集合 | 病人 | 样本数（每人 1000 视角） |
|------|------|------------------------|
| train | `01`~`15` | 15000 |
| valid | `16`~`18` | 3000 |
| test | `19`~`20` | 2000 |

`prepare_ximenzi_proj_data.py` 默认生成这个划分，也可用 `--train/--valid/--test` 覆盖（会检查同一病人不跨集合）。

---

## 六、运行步骤

### 1. 生成索引（服务器上）

```bash
cd /mnt/zhujingping/restormer          # 3090 服务器上的部署目录（见 .vscode/sftp.json 的 remotePath）
python prepare_ximenzi_proj_data.py --data-root /mnt/tm/ximenzi_pbi_data --output ./
# 产出 train_ximenzi_proj.txt / valid_ximenzi_proj.txt / test_ximenzi_proj.txt
# 脚本会校验每个 .raw 恰为 1,000,000,000 字节，形状不符直接报错
```

### 2. 标定归一化尺度

```bash
python analyze_proj_range.py --data-root /mnt/tm/ximenzi_pbi_data
# 关键看三处输出：
#   ② 推荐 proj_norm_clip_max   → 填进 yml
#   ③ 目标非零包围盒            → 确认全体病人是否都是 行[118,381]
#   ⑤ 首尾切片退化检查          → 若报退化则需剔除相应视角
```

把推荐值填入 `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml` 的 `train` 和 `val` 两处。

### 3. 冒烟测试

```bash
python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml
# 前 200 iter 确认：
#   - 日志 patch 尺寸为 96×96（渐进式第一阶段）
#   - 无 "LQ (...) is smaller than patch size"
#   - val 能跑通
```

### 4. 正式训练 / 推理

```bash
python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml
# 产物: experiments/CT_ProjectionDomain_Ximenzi_Restormer/（与联影实验目录隔离）

# 若环境有 cuDNN 冲突（CUDNN_STATUS_NOT_SUPPORTED / illegal memory access），
# 命令前加 `env -u LD_LIBRARY_PATH`，见 7-TROUBLESHOOTING.md 案例 2
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Ximenzi_Restormer/models/net_g_latest.pth \
    --config Options/CT_ProjectionDomain_Ximenzi_Restormer.yml \
    --txt ./test_ximenzi_proj.txt
# 产出 results/ct_proj/<病人>_{pred,input,gt}.raw（1000×256×496）
# 脚本无进度输出，2 病人 × 1000 视角约需 10~25 分钟，属正常
```

---

## 七、参数取值与理由（新 yml 相对旧 proj yml 的差异）

| 字段 | 旧（联影） | 新（西门子） | 理由 |
|------|-----------|-------------|------|
| `proj_shape` | `[720,128,512]`（隐式） | `[1000,500,500]` | 实测形状 |
| `proj_crop_rows` | 16 | **122** | §二 |
| `proj_crop_cols` | —（无此字段） | **0** | 列边界随视角变化，无法固定裁剪 |
| `proj_norm_clip_max` | 0.28 | **待标定** | §四 |
| `gt_size` | 96 | **192** | ≤256 且 8 整除；取 192 而非 256 以保留纵向随机裁剪余地 |
| `gt_sizes` | `[48,64,80,96]` | `[96,128,160,192]` | 平面变大，等比放大 |
| `mixup` | false | false | 样本 15000，够多 |
| `total_iter` | 90000 | 90000 | 15000/4 = 3750 iter/epoch → 约 24 epoch（旧配置约 50 epoch），可酌情调大 |
| `val_freq` | 1000 | **5000** | 验证次数 90 → 18。验证跑全尺寸前向，是主要耗时来源 |
| `val_sample_ratio` | —（无此字段） | **0.1** | 验证样本 3000 → 300。**必须配合 `val_freq` 一起调**，否则验证会占掉 18 h |
| 网络/损失/优化器/调度器 | — | **不变** | — |

> **关于验证开销**：这是新旧配置差异里最容易被忽略的一项。联影那次跑完 90000 iter 的实测日志显示，9.53 h 里有 **5.00 h 是验证**（90 次 × 3.4 min）——验证一直是时间大头。西门子的单次验证像素量是联影的 **3.6 倍**（单样本 `256×496` vs `96×512`，样本数 3000 vs 2160），不调的话验证总耗时约 18 h。调完两个参数后降到 ~22 min。完整分析见 [4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md) §6.1。

---

## 八、向后兼容性

改动全部走 `opt.get(key, 旧默认值)`，**旧 yml 不加任何新字段行为不变**：

- 不传 `proj_shape` → 默认 `(720,128,512)`（联影形状）
- 不传 `image_shape` → 默认 `(160,512,512)`
- 不传 `proj_crop_cols` → 默认 `0`
- `proj_norm_clip_max` 的校验只在 `domain: proj` 下触发，图像域不受影响

已在真实数据上验证：联影投影域仍得 `96×512`，联影图像域仍得 `512×512`，与改动前一致。

### 顺手修掉的一个隐患

原代码裁剪写作 `img_lq[r:-r, :]`，在 `r=0` 时会得到**空数组**（`[0:-0]` → `[0:0]`）。原先靠外层 `if r > 0` 侥幸绕过；现在改成显式上下界 `img_lq[r:h-r, c:w-c]`，`r=0` 时自然退化为 `[0:h]`。

---

## 九、已知未验证项

- **Restormer 前向在 256×500 上的实际跑通**未在本机验证（本机缺 `einops`/`lmdb`，非代码问题）。已确认的仅是：`pad_test` 会把 500 补到 504，且 `gt_size=192` 是 8 整除。建议按 §六.3 先跑冒烟测试。
- **全 20 个病人的非零包围盒一致性**未验证（本机只有病人 01）。若某个病人边界明显不同，说明探测器/准直有差异，`crop_rows: 122` 需重新评估——§六.2 的脚本输出③会直接暴露这一点。
