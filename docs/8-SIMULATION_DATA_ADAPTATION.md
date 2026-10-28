# 适配仿真数据（`simulation_change_params_20heads`）

> 目标数据集：titan 服务器 `/mnt/no3/tengmeng/simulation_change_params_20heads`（20 个人头 × 5 个剂量档，共 21 GB）。
> 本文记录**实测数据格式**与**让现有项目跑通它所需的改动**。
> 相关：[6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md)（上一套数据）、[7-TROUBLESHOOTING.md](7-TROUBLESHOOTING.md)（环境问题）。

---

## 〇、结论速览

**不需要改任何 Python 代码。** 上一轮把 `Dataset_CT` 参数化之后，形状 / 裁剪 / 归一化尺度全部由 yml 提供，换数据集只需要：

1. 新增一个 **prepare 索引脚本**（这份数据是两级目录 `<头>/<剂量>/`，现有脚本假设一级）
2. 新增一个 **yml**（改 4 个几何/归一化参数）

但有 **3 个参数绝对不能照抄 ximenzi 的配置**，必须按本数据重新标定（§二）：

| 参数 | ximenzi 的值 | 本数据取值 | 说明 |
|------|-------------|-----------|------|
| `proj_shape` | `[1000,500,500]` | **`[100,500,500]`** | 照抄直接读错 / 报错 |
| `proj_norm_clip_max` | 0.12 | **0.17** | 照抄会截断 3~4% 的亮区像素 |
| `proj_crop_rows` | 122 | **122（相同）** | 有意裁掉 FFT 不可靠的上下边界，见 §2.2 |

---

## 一、数据实测事实

### 1.1 目录结构（与之前不同：两级）

```
simulation_change_params_20heads/
├── 01/
│   ├── dose_1e4/          ← PhotonNumber = 1E+04
│   │   ├── proj_no_pbi.raw                    （GT，100,000,000 B）
│   │   ├── proj_pbi_fs_dec_blur_phase.raw     （Input，100,000,000 B）
│   │   ├── config.jsonc                       （仿真参数）
│   │   └── experiment_params.txt
│   ├── dose_1e5/ … dose_1e8/
├── 02/ … 20/
```

- **20 个人头**（`01`~`20`）× **5 个剂量档**（`1e4 / 1e5 / 1e6 / 1e7 / 1e8` 光子数）
- 文件名与之前完全一致 → **不需要处理文件名差异**

### 1.2 形状：实测是 `(100, 500, 500)`

**坑**：口头描述是 `(500, 500, 100)`，但按这个顺序 `reshape` 会得到完全错乱的数据。

实测判据（病人 01，100 个平面）：
- 按 `(100,500,500)` 解析 → 各平面均值几乎相同（std = **2.9e-5**），说明这是同一物体 100 个角度的投影
- 按 `(500,500,100)` 解析 → 各"平面"均值差异很大（std = **2.4e-2**）

**真实存储 = 连续 100 个 500×500 平面，最外层是视角号。**

```python
x = np.fromfile("proj_no_pbi.raw", dtype=np.float32).reshape(100, 500, 500)
# x[v] 是第 v 个视角的 500×500 探测器平面
```

→ yml 里写 **`proj_shape: [100, 500, 500]`**（`(视角数, H, W)`）。

### 1.3 数值范围与分布

来自 `config.jsonc`：`OutputFileForm: "post_log_images"`，与之前一样是对数域。

| | 实测 |
|---|---|
| GT max（全局，18 个可用头） | **0.162280**（头 08） |
| Input max | 0.161797 |
| Input min | **−0.058639** |
| GT 精确零占比 | 38.6% ~ 48.7%（外围无信号区） |
| GT 负值占比 | ≈ 2.6% |
| Input 负值占比 | ≈ 27.5% |

**各头的最大值差别很大**（4 倍跨度）：

| 头 | 14 | 10 | 08 | 03 | 01 |
|---|---|---|---|---|---|
| GT max | 0.041 | 0.086 | **0.162** | 0.162 | 0.156 |

→ 归一化上界必须覆盖**全局最大**，不能被某个小头像拉低。

### 1.4 GT 在 5 个剂量档下是同一个文件

```
01/dose_1e4/proj_no_pbi.raw  md5 = 2dfa4df0f397e8ce
01/dose_1e5/proj_no_pbi.raw  md5 = 2dfa4df0f397e8ce
01/dose_1e6/proj_no_pbi.raw  md5 = 2dfa4df0f397e8ce
01/dose_1e7/proj_no_pbi.raw  md5 = 2dfa4df0f397e8ce
01/dose_1e8/proj_no_pbi.raw  md5 = 2dfa4df0f397e8ce   ← 完全相同
```

只有 **Input** 随剂量变化（5 个 md5 各不相同）。

**但剂量带来的降质差异很小**（头 01）：

| 剂量 | mean\|input−gt\| | RMSE |
|---|---|---|
| 1e4 | 0.001124 | 0.003304 |
| 1e5 | 0.000773 | 0.003233 |
| 1e6 | 0.000670 | 0.003226 |
| 1e7 | 0.000643 | 0.003225 |
| 1e8 | 0.000637 | 0.003225 |

光子数增加 10000 倍，RMSE 只降了 **2.4%**——说明**降质主要不是光子噪声**，而是相位/束硬化处理本身带来的伪影。5 个剂量档之间高度冗余（但仍可作为轻度数据增强）。

### 1.5 内容边界

GT 非零包围盒（18 个头并集）：**行 `[91, 408]`，列 `[0, 499]`**。

各行不同（与之前的 ximenzi 数据同类型）：

| 头 | 04 | 08 | 10 | 01 | 05 |
|---|---|---|---|---|---|
| 行包围盒 | [95,404] | [92,407] | [101,398] | [118,381] | [113,386] |

列方向所有头都铺满 `[0,499]` → **不能按内容边界裁列**（与之前结论一致）。

### 1.6 ⚠️ 数据集不完整

**头 `19` 和 `20` 的文件缺失/大小异常**，实际可用 **92 / 100 卷**（`01`~`18` 完整）。

→ 原定的「20 头按 15/3/2 划分」**不可用**，见 §四。

---

## 二、必须重新标定的参数

### 2.1 `proj_norm_clip_max`：0.12 → **0.17**（最关键）

沿用 ximenzi 的 0.12 会截断大量亮区像素：

| 头 | >0.12 占比 | >0.15 | >0.16 | GT max |
|---|---|---|---|---|
| 01 | **3.07%** | 0.0008% | 0.0000% | 0.15619 |
| 03 | **4.04%** | 0.0055% | 0.0001% | 0.16198 |
| 08 | **1.86%** | 0.0112% | 0.0001% | 0.16228 |
| 05 | 0.06% | 0 | 0 | 0.12812 |

全局 max = **0.16228** → 取 **0.17**（覆盖全部 + 约 4.8% 余量，动态范围利用率 95.5%）。

> 上线前建议在服务器上用 `analyze_proj_range.py` 复核一遍（见 §五.2）。注意该脚本默认 `--shape 1000,500,500`，这份数据要传 `--shape 100,500,500`。

### 2.2 `proj_crop_rows`：**122**（与 ximenzi 相同）

这份数据的上下内容比 ximenzi 宽得多（包围盒并集到行 `[91,408]`）。按**内容边界**计算的能量损失：

| crop_rows | 行数 | 头04 损失 | 头08 损失 | 头10 损失 | 头01 损失 |
|---|---|---|---|---|---|
| 90 | 320 | 0.000% | 0.000% | 0.000% | 0.000% |
| 98 | 304 | 0.86% | 2.57% | 0.00% | 0.00% |
| 106 | 288 | 4.04% | 6.46% | 1.51% | 0.00% |
| 114 | 272 | 7.68% | 10.73% | 4.58% | 0.00% |
| 118 | 264 | 9.63% | 12.97% | 6.26% | 0.00% |
| **122** | **256** | **11.63%** | **15.26%** | **8.04%** | 1.74% |
| 130 | 240 | 15.85% | 19.93% | 11.95% | 6.37% |

**结论：取 122，接受 11~15% 的能量损失。**

理由：**仿真数据与 ximenzi 一样，上下边界的信号受相位处理中 FFT 的影响、本身不可靠，属于主动裁掉**，不是按内容边界裁剪。所以上表的"损失"是无法避免的取舍，而不是配置失误。

> 如果将来要无损裁剪，最小可行值是 **90**（→320 行，对全部头 0 损失），代价是每个样本多 25% 计算量。

### 2.3 `proj_crop_cols`：**2**（与之前相同）

列方向所有头都铺满 `[0,499]`，不能按内容边界裁；裁 2 列纯粹为把宽度凑成 16 整除（500 → 496），代价可忽略。

### 2.4 `gt_size`

裁剪后平面 **256 × 496**，短边 256 → `gt_size` 上限 256（需 8 整除）。

沿用 **`gt_size: 192` / `gt_sizes: [96,128,160,192]`** 即可（保守，且与现有配置一致）。想用更大 patch 可提到 256。

---

## 三、需要新增的文件

### 3.1 `prepare_sim_proj_data.py`

现有 `prepare_ximenzi_proj_data.py` 假设目录是 `<root>/<病人>/<文件>`，这份数据是 `<root>/<头>/<剂量>/<文件>`，且「病人」应由 **(头, 剂量)** 组合而成。建议新写一个，结构如下：

```python
ROOT   = "/mnt/no3/tengmeng/simulation_change_params_20heads"
INP    = "proj_pbi_fs_dec_blur_phase.raw"
GT     = "proj_no_pbi.raw"
NUM_VIEWS = 100
SHAPE  = (100, 500, 500)
EXPECT_BYTES = 100_000_000
DOSES  = ["dose_1e4", "dose_1e5", "dose_1e6", "dose_1e7", "dose_1e8"]

# 按头划分（同一头的 5 个剂量必须落在同一集合，避免泄漏）
TRAIN_HEADS = [f"{i:02d}" for i in range(1, 15)]   # 01~14
VALID_HEADS = ["15", "16"]
TEST_HEADS  = ["17", "18"]
# 19/20 数据不完整，暂不纳入

# 每个 (head, dose) 产出一条"卷"，每卷 NUM_VIEWS 行
# 行格式不变: <输入.raw绝对路径> <目标.raw绝对路径> <视角号>
```

产出：`train_sim_proj.txt` / `valid_sim_proj.txt` / `test_sim_proj.txt`。

**必须校验**：每个文件恰为 100,000,000 字节；缺文件/大小不符直接报错（这份数据确实有不完整的目录，见 §1.6）。

### 3.2 `Options/CT_ProjectionDomain_Sim_Restormer.yml`

以 `CT_ProjectionDomain_Ximenzi_Restormer.yml` 为模板，**只改这些字段**：

```yaml
name: CT_ProjectionDomain_Sim_Restormer      # ← 新实验名，与之前实验目录隔离

datasets:
  train:
    txt: ./train_sim_proj.txt
    proj_shape: [100, 500, 500]   # ← (视角数, H, W)
    proj_crop_rows: 122           # ← 256 行（主动裁掉 FFT 不可靠的上下边界）
    proj_crop_cols: 2             # ← 496 列
    proj_norm_clip_max: 0.17      # ← 必须重新标定
    gt_size: 192
    gt_sizes: [96, 128, 160, 192]
    # 其余（geometric_augs: false / batch 4 / iters / mini_batch_sizes）不变
  val:
    txt: ./valid_sim_proj.txt
    proj_shape: [100, 500, 500]
    proj_crop_rows: 122
    proj_crop_cols: 2
    proj_norm_clip_max: 0.17
    val_sample_ratio: 0.2

val:
  val_freq: !!float 5000
```

**代码一行都不用改。**

---

## 四、数据划分

按**头**划分（同一头的 5 个剂量档必须放同一侧，否则同一解剖结构会泄漏到验证集）：

| 集合 | 头 | 卷数（×5 剂量） | 样本数（×100 视角） |
|---|---|---|---|
| train | `01`~`16` | 80 | **8000** |
| valid | `17`~`18` | 10 | **1000** |
| test | — | — | —（按要求不设测试集） |

- 头 `19`/`20` 数据不完整，**不纳入**
- 训练样本 8000，约为 ximenzi（15000）的一半
- 验证 1000 个全尺寸样本；配合 `val_sample_ratio: 0.2` → 200 个/次，`val_freq: 5000` → 18 次，开销可忽略
- 5 个剂量档全部使用（GT 相同、输入不同，相当于 5 倍的轻度数据增强）

---

## 五、运行步骤（titan 服务器）

### 5.1 ⚠️ 环境还没就绪

实测当前状态：

| | 现状 |
|---|---|
| GPU | 4 × **NVIDIA TITAN Xp**（Pascal，**12 GB**，无 Tensor Core） |
| 驱动 | 580.178.04 |
| conda 环境 | 只有 `MCGPU`（Python 3.9） |
| `MCGPU` 里的包 | numpy、cv2、skimage、tmux、tree ✓；**torch 缺失** ❌；yaml/einops/lmdb/tensorboardX 也缺 |
| 数据盘 | `/mnt/no3` 3.6 TB，可用 2.5 TB ✓ |

**必须先装 PyTorch，且要选仍支持 Pascal（compute capability 6.1 / sm_61）的版本。**
装完必须验证——不能只看 `import torch` 成功：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
# 期望 capability 为 (6, 1)

# 真正的验证：一次 conv 前向+反向
python -c "
import torch, torch.nn as nn
m = nn.Conv2d(1,48,3,padding=1).cuda()
x = torch.randn(4,1,96,96, device='cuda', requires_grad=True)
m(x).sum().backward(); torch.cuda.synchronize(); print('conv fwd+bwd OK')
"
```

> TITAN Xp 是 2017 年的 Pascal 架构，单卡 fp32 算力约为 3090 的 1/3。好消息是有 **4 张卡**，可用多卡训练补偿：yml 里 `num_gpu: 4` + `--launcher pytorch`。

> 同样要警惕 [7-TROUBLESHOOTING.md](7-TROUBLESHOOTING.md) 案例 2 的 cuDNN 冲突——新机器先跑一遍 `torch.backends.cudnn.version()`。

### 5.2 准备数据与索引

```bash
cd /mnt/no3/<你的工作目录>/restormer

# 1) 生成索引
python prepare_sim_proj_data.py --data-root /mnt/no3/tengmeng/simulation_change_params_20heads --output ./

# 2) 复核值域与裁剪边界（注意 --shape 要传本数据的形状）
python analyze_proj_range.py \
    --data-root /mnt/no3/tengmeng/simulation_change_params_20heads \
    --shape 100,500,500 \
    --stride 1 \
    --crop-rows 122 --crop-cols 2
```

> `analyze_proj_range.py` 的「病人」是单层目录，这份数据是两层——需要先按 §3.1 改造成支持 `<头>/<剂量>`，或者直接用它输出的 ①（值域）部分手工核对。**最简单**：跑一次确认全局 max ≈ 0.162、包围盒并集 ≈ 行[91,408]，与本文件 §一 的实测值对齐即可。

### 5.3 训练 / 推理

```bash
# 训练
python basicsr/train.py -opt Options/CT_ProjectionDomain_Sim_Restormer.yml --launcher none

# 推理（--txt 模式）
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Sim_Restormer/models/net_g_latest.pth \
    --config Options/CT_ProjectionDomain_Sim_Restormer.yml \
    --txt ./test_sim_proj.txt
# 产出 results/ct_proj/<头>_<剂量>_pred/input/gt.raw，形状 (100, 256, 496)
```

---

## 六、已确定的配置

| 项 | 决定 |
|---|---|
| `crop_rows` | **122**（仿真数据同样受边界 FFT 影响，主动裁掉；接受 11~15% 能量损失） |
| 剂量档 | **5 档全用** |
| 划分 | 头 **`01`~`16` 训练 / `17`~`18` 验证**，**不设测试集**；`19`/`20` 不纳入 |
| 训练规模 | **单卡**（`num_gpu: 1`） |

> TITAN Xp 是 Pascal 架构，单卡 fp32 算力约为 3090 的 1/3。若后续觉得慢，可改成 `num_gpu: 4` + `--launcher pytorch` 用 4 卡并行（需先验证显存与多卡配置）。

---

## 七、与 ximenzi 数据的对照

| | ximenzi（上一套） | simulation（本套） |
|---|---|---|
| 目录层级 | `<病人>/` | `<头>/<剂量>/` |
| 病人数 | 20 | 20（可用 18） |
| 每病人卷数 | 1 | **5**（剂量档） |
| 平面形状 | `(1000, 500, 500)` | **`(100, 500, 500)`** |
| 视角数 | 1000 | **100** |
| 全局 max | 0.115 | **0.162** |
| 裁剪后平面 | 256 × 496 | 256 × 496 |
| `clip_max` | 0.12 | **0.17** |
| `crop_rows / cols` | 122 / 2 | **122 / 2**（相同） |
| 训练样本 | 15000 | 7000 |
| 代码改动 | 有（参数化那次） | **无** |

---

## 附：一句话总结

上一轮把 `Dataset_CT` 参数化的收益在这里体现了——**换数据集只需要两个新文件（prepare 脚本 + yml）**。
真正需要动脑的不是代码，而是 **`clip_max` 和 `crop_rows` 必须按新数据重新标定**，照抄上一套的值会分别造成 3~4% 的像素截断和 11~15% 的目标能量损失。
