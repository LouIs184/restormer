# 训练配置参数说明（图像域 + 投影域 × 联影 / 西门子）

> 本文档逐字段解释三个训练配置：
> - `Options/CT_ImageDomain_Restormer.yml`（图像域，联影双能相位）
> - `Options/CT_ProjectionDomain_Restormer.yml`（投影域，联影双能相位）
> - `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml`（投影域，西门子 PBI）
>
> 三者共用同一个数据集类 `Dataset_CT`（`basicsr/data/ct_image_dataset.py`），靠 yml 里 `datasets.train/val` 的 **`domain`** 字段切换；网络、损失、优化器、调度器完全一致。
> 西门子 PBI 数据集的实测差异、裁剪/归一化定标过程与运行步骤见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md)。
> 配套说明：运行流程见 [2-EXECUTION.md](2-EXECUTION.md)，完整背景见 [1-RUN_GUIDE.md](1-RUN_GUIDE.md)，投影域方案见 [5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md)。
>
> 注意：西门子 PBI 数据**只有投影数据、没有重建数据**（无 `rec_*.raw`），故**图像域配置 `CT_ImageDomain_Restormer.yml` 在该数据集上不可用**，只能训投影域。

---

## 0. 参数从哪来、改了什么

- **模板**：结构与绝大多数参数照搬官方 Restormer 的 `Deraining/Options/Deraining_Restormer.yml` / `Motion_Deblurring/Options/Deblurring_Restormer.yml`（同构的"图像恢复"任务模板）。
- **单通道灰度**：`inp/out_channels: 1` 的写法来自官方 `GaussianGrayDenoising_Restormer.yml`。
- **图像域归一化** `[-1000, 1000]`：来自你的 `AICTVer2/experiments/denseunet_phasect.yaml`，保证与 DenseUNet 基线可比。
- **投影域归一化** `clip(x, 0, 0.28)/0.28` 与黑边裁剪 `crop_rows: 16`：来自 AICT-code `proj_reader`，保证与 `DenseUNet_ProjCT` 基线可比。
- **缩小规模**：总迭代 90k（官方 300k）、patch 最大 256（官方 384）、batch 按单卡缩小——因为联影数据只有 13 病人、且是单卡训练。（西门子投影域沿用同一套迭代/网络规模，但其数据更多：15 个训练病人 × 1000 视角 = 15000 样本。）

### 三个 yml 的差异总览

| 配置 | `domain` | 索引 txt | 归一化 | 裁剪 | 几何增强 | `gt_size` | `mixup` |
|------|----------|---------|--------|---------|---------|-----------|---------|
| 图像域（联影） | `image` | `train_img.txt`（第三列=切片号 z） | HU `(x+1000)/2000` | — | `true` | 256 | `true` |
| 投影域（联影） | `proj` | `train_proj_img.txt`（第三列=视角号 v） | `clip(x,0,0.28)/0.28` | `proj_crop_rows: 16`（→96 行） | `false` | 96 | `false` |
| 投影域（西门子） | `proj` | `train_ximenzi_proj.txt`（第三列=视角号 v） | `clip(x,0,clip_max)/clip_max`（**`clip_max` 待标定**，见 §2.2） | `proj_crop_rows: 122` + `proj_crop_cols: 2`（→256×496） | `false` | 192 | `false` |

> `mixup` 一列的判断依据**不是同一套**：联影图像域开 `mixup` 是因为**样本量小**（正则化有效）；联影投影域样本 7200 个、从未开过；**西门子投影域样本有 15000 个（15 病人 × 1000 视角），数据量更多，故同样 `mixup: false`**。

---

## 1. general settings（实验标识，三 yml 同）

| 字段 | 含义 |
|------|------|
| `name: CT_ImageDomain_Restormer` / `CT_ProjectionDomain_Restormer` / `CT_ProjectionDomain_Ximenzi_Restormer` | 实验名 → 决定 `experiments/<name>/` 与 `tb_logger/<name>/` 目录名（三个实验目录相互隔离） |
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
| **`domain: image / proj`** | **域开关**：`image` 读 `rec_*.raw` 切片 `(512,512)`，`proj` 读 `proj_*.raw` 视角（联影 `(128,512)`、西门子 `(500,500)`，实际尺寸由 `proj_shape` 决定）。同一索引格式 `<输入raw> <目标raw> <索引>`，索引第三列图像域=切片号、投影域=视角号 |
| `txt: ./train_img.txt` / `./train_proj_img.txt` / `./train_ximenzi_proj.txt` | 索引文件（分别由 `prepare_ct_data.py` / `prepare_ct_proj_data.py` / `prepare_ximenzi_proj_data.py` 生成） |

### 2.2 域专属归一化 / 裁剪 / 增强

| 字段 | 生效域 | 含义 |
|------|--------|------|
| `hu_min / hu_max` | `image` | HU 归一化窗口：`(HU - (-1000)) / 2000 → [0,1]`。**训练/验证/推理三处必须一致** |
| **`proj_shape: [720,128,512]`** | `proj` | 投影 `.raw` 的原始形状 `(视角数, H, W)`，**默认 `[720,128,512]`（联影）**，西门子填 `[1000,500,500]`。**为什么需要**：形状原先硬编码在 `ct_image_dataset.py` 里，yml 无法覆盖，换数据集就必须改代码；参数化之后换数据集只改 yml。**不传此字段则回退默认值，老 yml 行为不变** |
| **`image_shape: [160,512,512]`** | `image` | 重建 `.raw` 的原始形状 `(切片数, H, W)`，**默认 `[160,512,512]`**。与 `proj_shape` 同因（把硬编码形状参数化）；西门子数据集没有重建数据，用不到该字段 |
| `proj_norm_clip_max` | `proj` | 投影域归一化：`clip(x, 0, clip_max)/clip_max → [0,1]`，**只取上界做单侧截断**。**投影域下此项为必填：留空（`~`）或缺失会直接 `raise ValueError`，不再静默回退 0.28**（该校验只在 `domain: proj` 触发；图像域走 `hu_min`/`hu_max`，不受影响）。取值：**联影 `0.28`**（同 AICT-code `proj_reader`，全数据集统计范围约 `[-0.066, 0.276]`）；**西门子待标定**（实测值域 −0.0385 ~ 0.0939，病人 01 隔 50 张抽样得目标 max ≈ 0.0946、99.99% 分位 ≈ 0.0886，**全 20 个病人的真值须在服务器上跑 `analyze_proj_range.py` 后定稿**，见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) §四）。**train/val 两处必须填同一个值**；**必须全局固定，勿逐样本 min-max**——会破坏 Restormer 残差结构 `out=net(x)+x` 的可加性 |
| `proj_crop_rows` | `proj` | 裁剪探测器**上下**（行方向）黑边，最终行数 = `H − 2×rows`。**联影 `16`**（`128 − 2×16 = 96` 行；96 是 8 的倍数，Restormer 3 次下采样无奇偶问题）；**西门子 `122`**（`500 − 2×122 = 256` 行；目标非零包围盒行 `[118,381]` 恒成立、边界外能量 0.000000%，裁到 122 仅损失 1.74% 目标能量，122 是从 128 向下取的第一个满足 16 整除的合格值）。注：现在**行、列都可裁**，列方向见 `proj_crop_cols` |
| **`proj_crop_cols: 0`** | `proj` | 裁剪探测器**左右**（列方向）黑边，最终列数 = `W − 2×cols`，**默认 `0`（不裁）**。**为什么需要**：西门子数据列方向各视角的投影足迹宽度不同（旋转展开），全 1000 视角并集为列 `[0,499]`，**不存在对全部视角都无损的固定列裁剪**，不能按内容边界裁；裁 2 列纯粹是为了把宽度凑成 16 整除——**西门子 `2`**（`500 − 4 = 496` → 平面 256×496，代价仅 0.018% 能量，边缘列能量只有中间列的约 1%）。联影不填（默认 0），行为不变 |
| `geometric_augs: true/false` | 三 yml | 训练时随机翻转/旋转增强。**投影域必须 `false`**（联影、西门子投影域都是 `false`）：90° 旋转会交换视角轴/通道轴，物理无意义 |

### 2.3 DataLoader

| 字段 | 含义 |
|------|------|
| `use_shuffle: true` | 每个 epoch 打乱数据 |
| `num_worker_per_gpu: 8` | 每 GPU 的数据加载子进程数（读 .raw 较慢，8 个合理） |
| `batch_size_per_gpu: 4`（三 yml 同） | 每 GPU 每步喂的样本数。**必须 = `mini_batch_sizes[0]`**（渐进式各阶段从这里随机子采样，设大了只会多加载再丢掉） |

### 2.4 渐进式训练（progressive training）四件套

思想：**先小 patch 大 batch 快速收敛，再逐步放大 patch 同时减小 batch 以省显存**。这是 Restormer 官方训练策略。

| 字段 | 图像域（联影） | 投影域（联影） | 投影域（西门子） |
|------|--------|--------|--------|
| `gt_size` | `256`（最终最大 patch） | **`96`（≤ 裁剪后高度）** | **`192`（≤ 256，且 8 整除）** |
| `gt_sizes` | `[128, 160, 192, 256]` | `[48, 64, 80, 96]` | `[96, 128, 160, 192]` |
| `mini_batch_sizes` | `[4, 3, 2, 1]` | `[4, 3, 2, 1]` | `[4, 3, 2, 1]` |
| `iters` | `[30000, 24000, 18000, 18000]` | `[30000, 24000, 18000, 18000]` | `[30000, 24000, 18000, 18000]` |

> ⚠️ **投影域最重要的一条约束**：`gt_sizes` 最大值必须 ≤ 裁剪后**平面边长**（上限 = `min(裁剪后行数, 裁剪后列数)`，联影为 96，西门子为 256）。框架的 `padding()` 会把样本 reflect 补到 `gt_size` 再裁剪，若沿用更大的值，96 行会被补到该值，模型就训练在大部分人造反射边上。各 `gt_size` 都必须是 8 的倍数（Restormer 需 2³ 整除）；**西门子取 192 而非上限 256，是为了保留纵向随机裁剪的余地**（平面 256×496，等比放大后 `gt_sizes: [96,128,160,192]`）。
>
> `iters` 各阶段之和**必须 = `train.total_iter` = 90000**（框架硬性要求），改 `iters` 时同步改 `total_iter`。

### 2.5 其他

| 字段 | 含义 |
|------|------|
| `dataset_enlarge_ratio: 1` | 数据集重复放大倍数（1 = 不放大） |
| `prefetch_mode: ~` | 数据预取方式（None = 不预取） |

---

## 3. network_g（Restormer 网络结构，三 yml 同）

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

## 4. path（权重/断点路径，三 yml 同）

| 字段 | 含义 |
|------|------|
| `pretrain_network_g: ~` | 预训练权重路径（None = 从零训练） |
| `strict_load_g: true` | 严格加载权重（参数名必须完全匹配） |
| `resume_state: ~` | 手动指定断点。留空 = **自动**从 `experiments/<name>/training_states/` 最新 `.state` 恢复 |

---

## 5. train（训练超参，三 yml 基本同）

| 字段 | 含义 |
|------|------|
| `total_iter: 90000` | 总迭代数（必须 = `datasets.train.iters` 之和）。三个 yml 都是 90000：西门子 15000 训练样本 / batch 4 = 3750 iter/epoch → 约 24 epoch（联影约 50 epoch），若欠拟合可酌情调大 |
| `warmup_iter: -1` | 学习率热身迭代（-1 = 不用） |
| `use_grad_clip: true` | 梯度裁剪（norm 裁剪到 0.01），防梯度爆炸 |
| `scheduler: CosineAnnealingRestartCyclicLR` | 余弦退火重启调度器，三个子参数见下 |
| ├ `periods: [30000, 60000]` | 两个周期的迭代长度 |
| ├ `restart_weights: [1, 1]` | 每周期重启后学习率恢复比例 |
| └ `eta_mins: [0.0003, 0.000001]` | 每周期学习率最低值。效果：**第一周期恒 3e-4（常数），第二周期 3e-4 余弦退火到 1e-6**（模仿官方 92k/208k 结构） |
| `mixing_augs.mixup` | 数据混合增强：两个样本按 Beta 分布线性混合。**联影图像域 `true`**（数据量小，正则化有效）；**投影域 `false`**——联影投影域样本 7200 个、AICT 投影域配置未开（想开设 `true`）；**西门子投影域样本 15000 个，比联影更多，故同样 `false`**。混合两个视角的 sinogram 物理无害，需要时可开 |
| `optim_g: AdamW` | 优化器类型 |
| ├ `lr: 3e-4` | 初始学习率 |
| ├ `weight_decay: 1e-4` | L2 正则强度 |
| └ `betas: [0.9, 0.999]` | Adam 的一阶/二阶动量系数 |
| `pixel_opt: L1Loss` | 像素级 L1（MAE）损失，权重 1。post-log 噪声近似高斯，L1 是合理起点；想对齐 AICTVer2（图像域 MSE / 投影域 MSE）可改为 `MSELoss` |

---

## 6. val（验证）

| 字段 | 含义 |
|------|------|
| `window_size: 8` | 推理前把图补到 8 的倍数（Restormer 有 3 次下采样 → 2³=8），再裁回原尺寸。联影 96/512 已是 8 的倍数、实际不补；西门子裁剪后平面 256×496，两个边长都是 8 的倍数，同样实际不补 |
| `val_freq` | 每多少迭代验证一次（只算 PSNR，不占磁盘）。联影两个 yml 都是 `1000`；**西门子 `5000`**，理由见 §6.1 |
| **`val_sample_ratio`** | **验证集抽样比例**，只在 `datasets.val` 段生效（实现在 `basicsr/data/ct_image_dataset.py`）。不写 / 写 `1.0` = 全用（与改动前行为一致）；`0.1` = 每 10 个样本取 1。**固定步长抽样、非随机**，保证每次验证用同一批样本、val PSNR 可比。抽样结果会打进日志 |
| `save_img: false` | 验证时不保存图像 |
| `rgb2bgr: true` | 存图时做 BGR 转换（单通道无影响，保持框架默认） |
| `use_image: false` | 指标**直接在张量上算**（[0,1] float），不转 uint8，避免量化丢精度 |
| `max_minibatch: 8` | 验证 batch 上限 |
| `metrics.psnr.type: calculate_psnr` | PSNR 指标 |
| ├ `crop_border: 0` | 计算前不裁边 |
| └ `test_y_channel: false` | 不转 YCbCr Y 通道（CT 不需要） |

### 6.1 为什么西门子必须把验证调稀（有实测数据）

验证跑的是**全尺寸前向**（不是 patch），而且 `validation()` 会跑完**整个** val 集（没有提前退出），
并且 `batch_size=1`、`num_workers=0`（`basicsr/data/__init__.py:101`）。
所以它的开销由「**验证样本数 × 单样本尺寸**」决定，和训练 patch 完全不是一个量级。

联影投影域那次跑完 90000 iter 的实测训练日志：

| | |
|---|---|
| 总时长 | 9.53 h |
| ├ 验证 | **5.00 h**（90 次 × 3.4 min） |
| └ 训练 | 4.53 h |

**验证本来就占了总时长的一半以上**，只是旧配置的量级还能忍（过一夜能跑完），所以没被注意。
西门子的单次验证像素量是联影的 **3.6 倍**：

| | 联影投影域 | 西门子投影域 | 倍数 |
|---|---|---|---|
| 验证样本数 | 3 病人 × 720 视角 = 2160 | 3 × 1000 = 3000 | 1.39× |
| 单样本尺寸 | `96×512` = 49,152 px | `256×496` = 126,976 px | 2.58× |
| **单次验证像素量** | **1.06 亿** | **3.81 亿** | **3.6×** |

按原样跑，验证总耗时会到 **~18 h**。调两个参数后：

```
val_freq: 5000         → 验证次数 90 → 18
val_sample_ratio: 0.1  → 单次 12 min → 1.2 min
合计 ≈ 18 × 1.2 min ≈ 22 min
```

> ⚠️ **训练日志里的 `eta` 会严重低估总时长**——它是「从开始到现在的平均速度 × 剩余迭代数」，而 `iter 1000` 那行打印时**第一次验证还没跑**。等验证累积几次之后 ETA 才会爬到接近真实值。详见 [7-TROUBLESHOOTING.md](7-TROUBLESHOOTING.md) 案例 7。

> 投影域注意：训练日志里的 val PSNR 是 **sinogram 域** 的，只用于监控收敛；**最终验收需把 `<病人>_pred.raw` FBP 重建到图像域**与 `rec_no_pbi.raw` 比（`test_ct_proj.py` 本身不含重建，精确几何用 AICT-code 重建代码）。
>
> 另注意：`clip_max` 改变归一化尺度（`use_image: false`，PSNR 直接在归一化域算），**联影与西门子的验证 PSNR 数值不可横向比较**。西门子数据**没有重建数据**，无法做上述 FBP 验收，只能看 `test_ct_proj.py` 输出的原始值域指标。详见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) §四。

---

## 7. logger（日志）与 dist_params（分布式，三 yml 同）

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
| `datasets.train.gt_size`（+`gt_sizes`） | 最终 patch 大小 | **联影图像域**显存不足降到 192（同步改 `gt_sizes`/`iters`）；**联影投影域**受 `≤ 96` 约束，显存不足继续往下压（如 `[48,64,80]`）；**西门子投影域** `192`（上限 256），显存不足可降到 160/128 并同步改 `gt_sizes`/`iters` |
| `train.total_iter`（+`scheduler.periods`、`datasets.train.iters`） | 训练时长 | 过拟合就减小；欠拟合就增大（注意 `sum(iters)=total_iter` 的约束） |
