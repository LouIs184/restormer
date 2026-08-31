# 训练后勤手册（产物位置 / 监控 / 断点 / TensorBoard）

> 训练跑起来之后，"东西在哪、怎么盯、坏了怎么办"这类问题都在这。
> 配套：执行流程见 [2-EXECUTION.md](2-EXECUTION.md)，配置参数见 [4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md)。

服务器上仓库位置：`/root/imageReconstruction/restormer`（当前 AutoDL 实例）。

---

## 1. 训练产物保存在哪

yml 里 `name: CT_ImageDomain_Restormer`，所有产物都在仓库内的 `experiments/` 下：

```
/root/imageReconstruction/restormer/
├── experiments/CT_ImageDomain_Restormer/
│   ├── models/                       # ★ 模型权重
│   │   ├── net_g_latest.pth          #   最新权重（推理/评估用这个）
│   │   ├── net_g_5000.pth            #   每 5000 迭代一个快照（iter 编号）
│   │   ├── net_g_10000.pth
│   │   └── ...
│   ├── training_states/              # ★ 断点状态（优化器+调度器）
│   │   ├── 5000.state
│   │   ├── 10000.state
│   │   └── ...
│   ├── train_CT_ImageDomain_Restormer_2026-08-25-*.log   # 完整训练日志
│   └── visualization/                # 验证图（yml save_img:false 时为空）
└── tb_logger/
    └── CT_ImageDomain_Restormer/     # TensorBoard 数据
```

| 文件 | 内容 | 大小 |
|------|------|------|
| `net_g_<iter>.pth` | 模型权重（仅 `params`） | ~104MB |
| `net_g_latest.pth` | 最新权重，每次保存时覆盖写 | ~104MB（常驻） |
| `<iter>.state` | 优化器+调度器状态（**不含权重**） | ~200MB |
| `train_*.log` | 完整训练日志（含每步 loss、验证 PSNR） | 小 |
| `tb_logger/` | TensorBoard 事件文件 | 小~几十 MB |

> `experiments/` 和 `tb_logger/` 都在 SFTP 的 `ignore` 里，**不会**同步回本地电脑，避免几十 GB 的网络传输。

---

## 2. 断点续训（训练中断了怎么办）

**自动恢复，不用做任何事**：直接重跑训练命令，脚本会从 `training_states/` 里最新的 `.state` 自动续上：

```bash
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
```

日志出现 `Resuming training from epoch: ..., iter: ...` 即成功恢复。

**续训需要成对的 `.pth` + `.state`**（`check_resume` 会用 `net_g_<iter>.pth` 恢复权重、`.state` 恢复优化器）。**清理旧快照必须按迭代成对删**，单删一个会导致该点无法续训：

```bash
# 例如删掉 iter 20000 的旧快照（权重和状态一起删）
rm experiments/CT_ImageDomain_Restormer/models/net_g_20000.pth
rm experiments/CT_ImageDomain_Restormer/training_states/20000.state
```

---

## 3. 训练中怎么实时监控

### 3.1 看日志

```bash
cd /root/imageReconstruction/restormer
tail -f train_ct.log          # 后台跑时用的 nohup 日志
# 或
tail -f experiments/CT_ImageDomain_Restormer/train_*.log
```

关注点：`l_pix` 应逐步下降；每 1000 迭代出现一行 `Validation ValSet, # psnr: xx.xx`，PSNR 应逐步上升。

### 3.2 TensorBoard（推荐）

**启动（服务器上）**：

```bash
conda activate /root/autodl-tmp/envs/restormer
tensorboard --logdir tb_logger --port 6006 --host 0.0.0.0
```

**本地浏览器访问**（二选一）：

- **AutoDL 自定义服务**（最简单）：AutoDL 控制台 → 实例 → 更多 → **自定义服务** → 添加端口 `6006` → 复制生成的 URL 在浏览器打开。
- **SSH 隧道**：
  ```bash
  # 本地电脑终端执行（host/port 是实例当前的 SSH 信息）
  ssh -p <SSH端口> root@<SSH地址> -L 6006:localhost:6006
  # 然后浏览器打开 http://localhost:6006
  ```

TensorBoard 里看：`loss`（l_pix）、`lr`、`metrics/psnr` 三条曲线。

### 3.3 显卡占用

```bash
nvidia-smi
# 看显存使用、GPU 利用率、有没有其他进程抢卡
```

### 3.4 看懂 ETA

日志 `eta: 7:52:34` 是剩余时间估计（按当前迭代速度外推）。注意渐进式训练后期 patch 变大（256）batch 变小（1），每迭代会变慢，实际总时长可能比早期 ETA 更长。

---

## 4. 权重保存策略与磁盘

- 频率：`save_checkpoint_freq: 5000`（yml 已设）→ 90k 迭代共 18 个快照，约 5.4GB。
- **系统盘只有 30G**（仓库 + experiments 都在系统盘），建议隔一段时间检查：
  ```bash
  du -sh experiments/CT_ImageDomain_Restormer
  df -h /        # 看系统盘剩余
  ```
- 训练完，需要腾空间就按第 2 节的"成对删"清掉中间快照，只留 `net_g_latest.pth` 和最后几个。

---

## 5. 训练跑完/中途评估

用最新权重推理验证集（文档 `docs/2-EXECUTION.md` 第 7 节）：

```bash
python test_ct.py --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

输出到 `results/ct/`：校正后 `.raw`（写回 HU）、可视化 PNG、终端打印 PSNR/SSIM。

---

## 6. 训练异常处理

| 现象 | 原因 | 处理 |
|------|------|------|
| `Killed` | 内存不足被 OOM（2GB 无卡实例常见） | 训练请务必在 GPU 实例上跑 |
| `CUDA out of memory` | 显存不够 | 减小 `batch_size_per_gpu`/`mini_batch_sizes`，或 `gt_size` 降到 192 |
| `libgomp: Invalid value for OMP_NUM_THREADS` | OpenMP 环境变量值非法 | 无害警告，可 `export OMP_NUM_THREADS=8` 消除 |
| `l_pix` 不降或爆炸 | 学习率/数据问题 | 检查归一化窗口是否一致、`hu_min/hu_max` 是否改过 |
| val PSNR 持续下降 | 过拟合 | 看 mixup 是否开着、减小 `total_iter`、尽早用中间快照 |
| 日志中断、进程退出 | SSH 断开/实例重启 | 用 nohup 起训练，`Resuming` 自动续训 |

---

## 7. 实例重建/迁移时的注意事项

| 内容 | 位置 | 保存镜像后？ |
|------|------|-------------|
| conda 环境 | 数据盘 `/root/autodl-tmp/envs/` | ❌ 不保存，需重建 |
| 数据 | 数据盘 `/root/autodl-tmp/联影双能相位数据2080` | ❌ 不保存 |
| 仓库+训练产物 | 系统盘 `/root/imageReconstruction/restormer` | ✅ 随镜像保存 |

- 换新实例后，`conda env list` 确认环境还在（数据盘跟着新实例走的话），SSH 的 host/port 会变，TensorBoard 隧道和 SFTP 配置要更新。
- 如果保存镜像后重开导致数据盘为空，需要重新上传数据、重装环境，但**系统盘上的仓库和已训练权重还在**。

---

## 8. 一句话速查

```bash
# 看训练是否在跑 / 进度
tail -f train_ct.log
nvidia-smi

# 启动 TensorBoard（然后本地开自定义服务端口 6006 或 SSH 隧道）
tensorboard --logdir tb_logger --port 6006 --host 0.0.0.0

# 中断后续训
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none

# 推理验证集
python test_ct.py --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth --patient ...

# 看磁盘余量 / 产物大小
df -h /
du -sh experiments/CT_ImageDomain_Restormer
```
