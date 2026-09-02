# 训练配置参数说明（图像域 + 投影域）

> 本文档逐字段解释两个训练配置：
> - `Options/CT_ImageDomain_Restormer.yml`（图像域）
> - `Options/CT_ProjectionDomain_Restormer.yml`（投影域）
>
> 两者共用同一个数据集类 `Dataset_CT`（`basicsr/data/ct_image_dataset.py`），靠 yml 里 `datasets.train/val` 的 **`domain`** 字段切换；网络、损失、优化器、调度器完全一致。
> 配套说明：运行流程见 [2-EXECUTION.md](2-EXECUTION.md)，完整背景见 [1-RUN_GUIDE.md](1-RUN_GUIDE.md)，投影域方案见 [5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md)。

---

## 0. 参数从哪来、改了什么

- **模板**：结构与绝大多数参数照搬官方 Restormer 的 `Deraining/Options/Deraining_Restormer.yml` / `Motion_Deblurring/Options/Deblurring_Restormer.yml`（同构的"图像恢复"任务模板）。
- **单通道灰度**：`inp/out_channels: 1` 的写法来自官方 `GaussianGrayDenoising_Restormer.yml`。
- **图像域归一化** `[-1000, 1000]`：来自你的 `AICTVer2/experiments/denseunet_phasect.yaml`，保证与 DenseUNet 基线可比。
- **投影域归一化** `clip(x, 0, 0.28)/0.28` 与黑边裁剪 `crop_rows: 16`：来自 AICT-code `proj_reader`，保证与 `DenseUNet_ProjCT` 基线可比。
- **缩小规模**：总迭代 90k（官方 300k）、patch 最大 256（官方 384）、batch 按单卡缩小——因为你的数据只有 13 病人、且是单卡训练。

### 两个 yml 的差异总览

| 配置 | `domain` | 索引 txt | 归一化 | 黑边裁剪 | 几何增强 | `gt_size` | `mixup` |
|------|----------|---------|--------|---------|---------|-----------|---------|
| 图像域 | `image` | `train_img.txt`（第三列=切片号 z） | HU `(x+1000)/2000` | — | `true` | 256 | `true` |
| 投影域 | `proj` | `train_proj_img.txt`（第三列=视角号 v） | `clip(x,0,0.28)/0.28` | `proj_crop_rows: 16`（→96 行） | `false` | 96 | `false` |

---

## 1. general settings（实验标识，两 yml 同）

| 字段 | 含义 |
|------|------|
| `name: CT_ImageDomain_Restormer` / `CT_ProjectionDomain_Restormer` | 实验名 → 决定 `experiments/<name>/` 与 `tb_logger/<name>/` 目录名 |
| `model_type: ImageCleanModel` | 使用的模型类（`basicsr/models/image_restoration_model.py`），负责网络定义、L1 loss、训练/验证循环 |
| `scale: 1` | 分辨率比例，1 = 不放大（不超分）。固定 |
| `num_gpu: 1` | 显卡数；多卡时 `--nproc_per_node` 必须与它一致；0 = 纯 CPU（极慢） |
| `manual_seed: 100` | 随机种子，保证可复现 |

---

## 2. datasets（数据入口）

### 2.1 公共字段

| 字段 | 含义 |
|------|------|
| `type: Dataset_CT` | 统一数据集类（`basicsr/data/ct_image_dataset.py`）。**旧名 `Dataset_CTImage` 保留为别名**（等价 `domain: image`），老 yml 不换也能跑 |
| **`domain: image / proj`** | **域开关**：`image` 读 `rec_*.raw` 切片 `(512,512)`，`proj` 读 `proj_*.raw` 视角 `(128,512)`。同一索引格式 `<输入raw> <目标raw> <索引>`，索引第三列图像域=切片号、投影域=视角号 |
| `txt: ./train_img.txt` / `./train_proj_img.txt` | 索引文件（分别由 `prepare_ct_data.py` / `prepare_ct_proj_data.py` 生成） |

### 2.2 域专属归一化 / 裁剪 / 增强

| 字段 | 生效域 | 含义 |
|------|--------|------|
| `hu_min / hu_max` | `image` | HU 归一化窗口：`(HU - (-1000)) / 2000 → [0,1]`。**训练/验证/推理三处必须一致** |
| `proj_norm_clip_max: 0.28` | `proj` | 投影域归一化：`clip(x, 0, 0.28)/0.28 → [0,1]`（同 AICT-code `proj_reader`，全数据集统计范围约 `[-0.066, 0.276]`）。**必须全局固定，勿逐样本 min-max**——会破坏 Restormer 残差结构 `out=net(x)+x` 的可加性 |
| `proj_crop_rows: 16` | `proj` | 裁剪探测器上下黑边：`128 - 2×16 = 96` 行。96 是 8 的倍数，Restormer 3 次下采样无奇偶问题 |
| `geometric_augs: true/false` | 两域 | 训练时随机翻转/旋转增强。**投影域必须 `false`**：90° 旋转会交换视角轴/通道轴，物理无意义 |

### 2.3 DataLoader

| 字段 | 含义 |
|------|------|
| `use_shuffle: true` | 每个 epoch 打乱数据 |
| `num_worker_per_gpu: 8` | 每 GPU 的数据加载子进程数（读 .raw 较慢，8 个合理） |
| `batch_size_per_gpu: 4`（两域同） | 每 GPU 每步喂的样本数。**必须 = `mini_batch_sizes[0]`**（渐进式各阶段从这里随机子采样，设大了只会多加载再丢掉） |

### 2.4 渐进式训练（progressive training）四件套

思想：**先小 patch 大 batch 快速收敛，再逐步放大 patch 同时减小 batch 以省显存**。这是 Restormer 官方训练策略。

| 字段 | 图像域 | 投影域 |
|------|--------|--------|
| `gt_size` | `256`（最终最大 patch） | **`96`（≤ 裁剪后高度）** |
| `gt_sizes` | `[128, 160, 192, 256]` | `[48, 64, 80, 96]` |
| `mini_batch_sizes` | `[4, 3, 2, 1]` | `[4, 3, 2, 1]` |
| `iters` | `[30000, 24000, 18000, 18000]` | `[30000, 24000, 18000, 18000]` |

> ⚠️ **投影域最重要的一条约束**：`gt_sizes` 最大值必须 ≤ 裁剪后高度 **96**。框架的 `padding()` 会把样本 reflect 补到 `gt_size` 再裁剪，若沿用 256，96 行会被补到 256，模型就训练在大部分人造反射边上。48/64/80/96 都是 8 的倍数（Restormer 需 2³ 整除）。
>
> `iters` 各阶段之和**必须 = `train.total_iter` = 90000**（框架硬性要求），改 `iters` 时同步改 `total_iter`。

### 2.5 其他

| 字段 | 含义 |
|------|------|
| `dataset_enlarge_ratio: 1` | 数据集重复放大倍数（1 = 不放大） |
| `prefetch_mode: ~` | 数据预取方式（None = 不预取） |

---

## 3. network_g（Restormer 网络结构，两 yml 同）

| 字段 | 含义 |
|------|------|
| `type: Restormer` | 网络类（`basicsr/models/archs/restormer_arch.py`） |
| `inp_channels / out_channels: 1` | 输入/输出通道数。CT 灰度/投影 = 1；**Restormer 是残差结构（`out = net(x) + x`），两者必须相等**；双能投影可设 2 |
| `dim: 48` | 第一层特征通道数。越大越强但越吃显存（官方默认 48） |
| `num_blocks: [4, 6, 6, 8]` | 编码器 4 级每级的 Transformer block 数量 |
| `heads: [1, 2, 4, 8]` | 4 级各自的自注意力头数 |
| `num_refinement_blocks: 4` | 解码后精修（refinement）block 数 |
| `ffn_expansion_factor: 2.66` | Feed-Forward 隐藏层放大倍数 |
| `bias: False` | 卷积不加偏置（官方默认） |
| `LayerNorm_type: WithBias` | LayerNorm 是否带偏置（`WithBias` / `BiasFree`） |
| `dual_pixel_task: False` | 双像素（Dual-Pixel）去模糊专用开关，CT 任务固定 False |

---

## 4. path（权重/断点路径，两 yml 同）

| 字段 | 含义 |
|------|------|
| `pretrain_network_g: ~` | 预训练权重路径（None = 从零训练） |
| `strict_load_g: true` | 严格加载权重（参数名必须完全匹配） |
| `resume_state: ~` | 手动指定断点。留空 = **自动**从 `experiments/<name>/training_states/` 最新 `.state` 恢复 |

---

## 5. train（训练超参，两 yml 基本同）

| 字段 | 含义 |
|------|------|
| `total_iter: 90000` | 总迭代数（必须 = `datasets.train.iters` 之和） |
| `warmup_iter: -1` | 学习率热身迭代（-1 = 不用） |
| `use_grad_clip: true` | 梯度裁剪（norm 裁剪到 0.01），防梯度爆炸 |
| `scheduler: CosineAnnealingRestartCyclicLR` | 余弦退火重启调度器，三个子参数见下 |
| ├ `periods: [30000, 60000]` | 两个周期的迭代长度 |
| ├ `restart_weights: [1, 1]` | 每周期重启后学习率恢复比例 |
| └ `eta_mins: [0.0003, 0.000001]` | 每周期学习率最低值。效果：**第一周期恒 3e-4（常数），第二周期 3e-4 余弦退火到 1e-6**（模仿官方 92k/208k 结构） |
| `mixing_augs.mixup` | 数据混合增强：两个样本按 Beta 分布线性混合。**图像域 `true`**（数据量小，正则化有效）；**投影域 `false`**（样本 7200 个，AICT 投影域配置未开，想开设 `true`，混合两个视角的 sinogram 物理无害） |
| `optim_g: AdamW` | 优化器类型 |
| ├ `lr: 3e-4` | 初始学习率 |
| ├ `weight_decay: 1e-4` | L2 正则强度 |
| └ `betas: [0.9, 0.999]` | Adam 的一阶/二阶动量系数 |
| `pixel_opt: L1Loss` | 像素级 L1（MAE）损失，权重 1。post-log 噪声近似高斯，L1 是合理起点；想对齐 AICTVer2（图像域 MSE / 投影域 MSE）可改为 `MSELoss` |

---

## 6. val（验证，两 yml 同）

| 字段 | 含义 |
|------|------|
| `window_size: 8` | 推理前把图补到 8 的倍数（Restormer 有 3 次下采样 → 2³=8），再裁回原尺寸。96/512 已是 8 的倍数，投影域实际不补 |
| `val_freq: 1000` | 每 1000 迭代验证一次（只算 PSNR，不占磁盘） |
| `save_img: false` | 验证时不保存图像 |
| `rgb2bgr: true` | 存图时做 BGR 转换（单通道无影响，保持框架默认） |
| `use_image: false` | 指标**直接在张量上算**（[0,1] float），不转 uint8，避免量化丢精度 |
| `max_minibatch: 8` | 验证 batch 上限 |
| `metrics.psnr.type: calculate_psnr` | PSNR 指标 |
| ├ `crop_border: 0` | 计算前不裁边 |
| └ `test_y_channel: false` | 不转 YCbCr Y 通道（CT 不需要） |

> 投影域注意：训练日志里的 val PSNR 是 **sinogram 域** 的，只用于监控收敛；**最终验收需把 `<病人>_pred.raw` FBP 重建到图像域**与 `rec_no_pbi.raw` 比（`test_ct_proj.py` 本身不含重建，精确几何用 AICT-code 重建代码）。

---

## 7. logger（日志）与 dist_params（分布式，两 yml 同）

| 字段 | 含义 |
|------|------|
| `print_freq: 200` | 每 200 迭代打印一次日志 |
| `save_checkpoint_freq: 5000` | **每 5000 迭代保存一次权重+断点**（关于频率选择的磁盘考量，见下） |
| `use_tb_logger: true` | 开启 tensorboard（`tensorboard --logdir tb_logger`） |
| `wandb.project: ~` | 不启用 wandb |
| `dist_params.backend: nccl` | 多卡分布式后端（单卡忽略） |
| `dist_params.port: 29500` | 分布式通信端口 |

### 为什么 `save_checkpoint_freq: 5000`

每次保存 ≈ **300MB**（`net_g_<iter>.pth` 模型权重 ~104MB + `<iter>.state` 优化器状态 ~200MB）。90k 迭代：

| 频率 | 保存次数 | 占用 |
|------|---------|------|
| 1000 | 90 | ≈27GB（**会爆 30G 系统盘**） |
| **5000（当前）** | 18 | ≈5.4GB |
| 10000 | 9 | ≈2.7GB |

崩溃时最多重跑"频率值"那么多迭代（5000 ≈ 13 轮），优化器状态完整恢复，不从头学。**清理旧快照时必须按迭代成对删**（`net_g_<iter>.pth` + `<iter>.state`），续训需要两者配套。

---

## 8. 最值得调整的三个旋钮

| 旋钮 | 作用 | 调整方向 |
|------|------|---------|
| `network_g.dim` | 模型容量/显存 | 显存不足降到 32；效果不够升到 64 |
| `datasets.train.gt_size`（+`gt_sizes`） | 最终 patch 大小 | **图像域**显存不足降到 192（同步改 `gt_sizes`/`iters`）；**投影域**受 `≤ 96` 约束，显存不足继续往下压（如 `[48,64,80]`） |
| `train.total_iter`（+`scheduler.periods`、`datasets.train.iters`） | 训练时长 | 过拟合就减小；欠拟合就增大（注意 `sum(iters)=total_iter` 的约束） |
