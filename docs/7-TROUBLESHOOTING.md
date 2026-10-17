# 排错手册（实战记录）

> 本文档记录**实际踩过的坑**及其排查过程，供换机器 / 换环境 / 换数据集时对照。
> 每条都写明：**现象 → 排查 → 根因 → 修复**。
> 相关：[2-EXECUTION.md](2-EXECUTION.md)（执行流程）、[6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md)（西门子数据集）。

---

## 〇、通用排查方法（先看这一节）

出问题时**不要一上来就改代码**，按下面的顺序分层缩小范围。本项目绝大多数"看起来像代码 bug"的问题，最后都是环境问题。

### 0.1 先判断「环境问题」还是「代码问题」

CUDA / 深度学习框架的报错，**栈顶往往不是真凶**。典型误导：

```
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

报错自己都说了栈可能不对——真正出错的 kernel 可能在前面几十行。判断办法：

```bash
CUDA_LAUNCH_BLOCKING=1 python basicsr/train.py -opt Options/<你的yml>
```

每个 kernel 同步执行，栈就会**指向真正出错的位置**。排查 CUDA 问题第一步永远先加这个环境变量。

### 0.2 用最小复现剥离变量

不要在全流程里猜。写个 10 行脚本，只保留可疑操作：

```bash
# 测试 1：纯 PyTorch 能不能跑（排除本项目的所有代码）
python -c "
import torch, torch.nn as nn
m = nn.Sequential(nn.Conv2d(1,48,3,padding=1), nn.ReLU()).cuda()
x = torch.randn(4,1,96,96, device='cuda', requires_grad=True)
m(x).sum().backward(); torch.cuda.synchronize()
print('conv fwd+bwd OK')
"

# 测试 2：只跑网络，不碰数据加载（排除 dataloader / 索引 / .raw 读取）
python -c "
import torch
from basicsr.models.archs.restormer_arch import Restormer
net = Restormer(inp_channels=1, out_channels=1, dim=48, num_blocks=[4,6,6,8],
                num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
                bias=False, LayerNorm_type='WithBias', dual_pixel_task=False).cuda()
x = torch.randn(4,1,96,96, device='cuda')
net(x).sum().backward(); torch.cuda.synchronize()
print('restormer fwd+bwd OK')
"
```

- **测试 1 就崩** → 环境/驱动/硬件问题，跟本项目无关，去查 ①
- **测试 1 过、测试 2 崩** → 是 Restormer 在这套 CUDA/PyTorch 组合上的问题
- **两个都过** → 问题在训练循环 / 数据加载 / 配置，看后面各案例

### 0.3 先确认环境版本三件套

```bash
nvidia-smi                                    # 看 Driver Version 和右上角 CUDA Version
python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda)"
python -c "import torch;print('cudnn',torch.backends.cudnn.version())"
```

**`torch.version.cuda` 必须 ≤ `nvidia-smi` 显示的 CUDA Version**（驱动向后兼容，不能向前）。
`cudnn` 必须能打印出版本号——**打印此处就报错，直接跳到案例 2**。

---

## 一、环境类问题

### 案例 1：`ModuleNotFoundError: No module named 'basicsr'`

**现象**

```bash
$ python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml
Traceback (most recent call last):
  File "/mnt/zhujingping/restormer/basicsr/train.py", line 10, in <module>
    from basicsr.data import create_dataloader, create_dataset
ModuleNotFoundError: No module named 'basicsr'
```

明明人在仓库根目录，`basicsr/` 就在眼前，却说不存在。

**原因**

两个因素叠加：

1. `python basicsr/train.py` 会把 **`basicsr/` 这个目录本身**放进 `sys.path[0]`，**而不是当前工作目录**。所以仓库根目录不在导入路径上，顶层包 `basicsr` 自然找不到。
2. `basicsr/` 里**没有 `__init__.py`**（里面的文件是 `data/ metrics/ models/ utils/ test.py train.py version.py`）。它是一个 **PEP 420 命名空间包**，必须靠外部把**仓库根目录**加进 `sys.path` 才能被解析。

`INSTALL.md` 第 4 步的 `python setup.py develop --no_cuda_ext` 做的就是这个"把仓库根目录注册进环境"的事。**做过这步的机器上能跑，新机器上没做过就会报这个错。**

**排查**

```bash
python -c "import basicsr; print(basicsr.__path__)"
```

- 正常（已注册）：打印 `_NamespacePath(['/mnt/zhujingping/restormer/basicsr'])`
- 未注册：`ModuleNotFoundError`

> ⚠️ **不要用 `python -c "import basicsr; print(basicsr.__file__)"` 来验证**——命名空间包没有 `__init__.py`，`__file__` 规定就是 `None`。打印 `None` **不代表失败**，那恰恰说明导入成功了。用 `__path__` 判断。

**修复（推荐：写 `.pth` 文件）**

`.pth` 就是 `setup.py develop` 底层实际做的事（往 site-packages 写路径），跳过 setuptools 直接做：

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/mnt/zhujingping/restormer" > "$SP/restormer.pth"
```

（把路径换成你服务器上仓库的实际位置。）永久生效，重启终端也在。验证：

```bash
python -c "from basicsr.data import create_dataset; print('import ok')"
```

**修复（临时：不装，跑一次）**

```bash
PYTHONPATH=$PWD python basicsr/train.py -opt Options/<你的yml>

# 或者（-m 会把当前目录加进 sys.path）
python -m basicsr.train -opt Options/<你的yml>
```

`basicsr/train.py` 结尾有 `if __name__ == '__main__': main()`，所以两种都能用。

**❌ 不要用 `pip install -e .` 代替**

`setup.py` 的 `--no_cuda_ext` 是自己实现的自定义 flag（`setup.py:125`），pip 传不进去，会走默认分支去编译 `deform_conv_ext` / `fused_act_ext` / `upfirdn2d_ext` 三个 CUDA 算子（`setup.py:129-148`）。**Restormer 根本用不到这三个**，白编译几分钟还可能失败。

**附：`python setup.py develop --no_cuda_ext` 本身也可能失败**

新版 setuptools 里 `develop` 命令已废弃，且 `setup.py:172` 的 `setup_requires=['cython', 'numpy']` 会触发已废弃的 egg 抓取流程：

```
_DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.
```

**这是警告不是报错**，但它可能卡在下载/编译 numpy、cython 上很久。遇到卡住就 Ctrl-C，直接用上面的 `.pth` 方案。

---

### 案例 2：`CUDA error: an illegal memory access`（cuDNN 版本冲突）

**现象**

训练能启动，前向和 loss 都算过去了，**在 `backward()` 崩溃**：

```
File "basicsr/models/image_restoration_model.py", line 165, in optimize_parameters
    l_pix.backward()
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

容易被误导为"数据形状不对"或"模型有 bug"，因为崩在 backward。但输入只是 `(4,1,96,96)` 的小 patch，这个规模下出非法访存**几乎必然是环境问题**。

**排查**

第一步（最关键）——直接问 PyTorch 的 cuDNN 能不能初始化：

```bash
python -c "import torch; print(torch.backends.cudnn.version())"
```

本例输出：

```
RuntimeError: cuDNN version incompatibility: PyTorch was compiled against (8, 9, 2)
but found runtime version (8, 5, 0). PyTorch already comes bundled with cuDNN.
One option to resolving this error is to ensure PyTorch can find the bundled cuDNN.
Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn.
Please either remove it from the path or install cudnn (8, 9, 2)
```

**根因**

`LD_LIBRARY_PATH` 里有一个**系统安装的 cuDNN 8.5.0**，它的搜索优先级高于 PyTorch 自带的 cuDNN 8.9.2，于是被动态链接器选中了。cuDNN 大版本之间 **ABI 不兼容**，卷积的 backward 实现按 8.9 的符号布局去读 8.5 的库，读到错位的显存 → `illegal memory access`。

这也解释了为什么**前向能过、backward 才炸**——cuDNN 的卷积反向实现正是 ABI 不匹配的重灾区。

> 注意 `torch 2.3.0+cu121` 配驱动 535（CUDA 12.2）本身是兼容的，**问题只在 cuDNN 被顶掉**，不需要换 PyTorch。

**定位污染源**

```bash
echo $LD_LIBRARY_PATH
# 逐个目录看谁带了 libcudnn
for d in $(echo $LD_LIBRARY_PATH | tr ':' ' '); do
    ls "$d"/libcudnn* 2>/dev/null && echo "   ^^^ 来自 $d"
done
```

常见来源：`/usr/local/cuda/lib64`、`/usr/local/cuda-11.x/lib64`、某个系统级 cuDNN 安装目录。

**修复（临时验证）**

```bash
env -u LD_LIBRARY_PATH python -c "import torch; print('cudnn', torch.backends.cudnn.version())"
# 期望: cudnn 8902  ← 8.9.2，PyTorch 自带的那个
```

打印出 `8902` 就坐实了。然后**每一条 Python 命令前都加上这个前缀**：

```bash
# 训练
env -u LD_LIBRARY_PATH python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml

# 推理（最容易忘：训练加了前缀，推理忘了加，会以同样的 illegal memory access 崩掉）
env -u LD_LIBRARY_PATH python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Ximenzi_Restormer/models/net_g_latest.pth \
    --config Options/CT_ProjectionDomain_Ximenzi_Restormer.yml \
    --txt ./test_ximenzi_proj.txt

# 其它任何入口同理
env -u LD_LIBRARY_PATH python analyze_proj_range.py --data-root /mnt/tm/ximenzi_pbi_data
```

> ⚠️ **`env -u LD_LIBRARY_PATH` 只对当前这一条命令生效**——换终端、换命令、写进 shell 脚本忘了加，都会重新加载那个错的 cuDNN。
> 实测踩过这个坑：训练时加了前缀能正常跑，之后推理忘了加，直接在同一个地方崩。
> **所以不要依赖每次手加前缀，直接用下面的「永久修复」。**

**修复（永久）**

从 `~/.bashrc` / `~/.profile` 里把那条包含 libcudnn 的 `export LD_LIBRARY_PATH=...` 去掉或注释掉，重开终端。

> ❌ 不建议"往系统里装一个 cuDNN 8.9.2"。PyTorch 轮子已经自带正确版本，**让系统路径别去干扰它**才是正解。

**同类的其它表现**

cuDNN 冲突不一定报成 `illegal memory access`，也可能是：

| 表现 | 说明 |
|------|------|
| `illegal memory access` | 本例 |
| 训练 loss 突然变 NaN / 随机崩溃 | 同样是 ABI 错位读到垃圾数据 |
| `CUDNN_STATUS_...` 系列错误 | 更直接的版本不匹配报错 |
| 同样的代码在不同机器上行为不一致 | 两台机器的 `LD_LIBRARY_PATH` 不同 |

**先做的检查**：任何时候怀疑 CUDA 环境，先跑一遍 0.3 的三件套命令。

---

### 案例 3：GPU 被其它进程占用

**现象**：训练能跑但异常慢，或 OOM。

**排查**

```bash
nvidia-smi
```

本例的输出里注意到：

```
|   1  NVIDIA GeForce RTX 3090 ... |  13296MiB / 24576MiB |     94%      Default |
| Processes:  1  ...  3277559  C  python  13290MiB |
```

GPU 1 已被另一个 python 进程占了 13GB。PyTorch 默认用 device 0，所以本例没冲突，**但多卡机器上要显式确认**。

**修复**

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Options/<你的yml>
```

`CUDA_VISIBLE_DEVICES` 会重新编号——设成 `0` 后，程序里看到的"第 0 号卡"就是物理 0 号。

---

## 二、配置/数据类问题（多为刻意设计，不是 bug）

### 案例 4：`ValueError: 投影域缺少 proj_norm_clip_max（或留空为 ~）`

**现象**：一启动训练就报错，提示去跑 `analyze_proj_range.py`。

**这不是 bug，是刻意设计。** 投影域归一化是 `clip(x, 0, clip_max)/clip_max`，`clip_max` 直接决定数据的动态范围利用率。旧代码用 `opt.get('proj_norm_clip_max', 0.28)` 静默回退到联影数据集标定的 0.28——换到西门子数据集（实测 max 仅 0.115）会让归一化值全挤在 `[0, 0.41]`，**白白浪费 59% 动态范围，而且不会有任何报错**。这类错误最难发现，所以改成显式报错。

**修复**

```bash
python analyze_proj_range.py --data-root <数据根目录>
```

看输出 ② 的「建议 proj_norm_clip_max」，填进 yml 的 **`train` 和 `val` 两处**（两处必须一致）。

> 取值时注意：脚本的 `max`/`min` 是在**完整切片**上算的（准确），但 `--stride` 只覆盖部分平面。默认 `--stride 10` 只扫 10% 的平面，所以"100% 分位"不等于全集真值，建议留一点余量。想要精确值就 `--stride 1`。

### 案例 5：`ValueError: LQ (...) is smaller than patch size`

**现象**

```
ValueError: LQ (256, 496) is smaller than patch size (320, 320).
Please remove <数据路径>
```

**原因**：`paired_random_crop`（`basicsr/data/transforms.py:58`）有个硬约束——**裁剪后平面的边长必须 ≥ `gt_size`**。换数据集时平面的尺寸变了，但 `gt_size` 忘了改，就会撞上。

**修复**

`gt_size` 必须满足两个条件：

1. ≤ `min(H − 2×proj_crop_rows, W − 2×proj_crop_cols)`
2. **8 的倍数**（Restormer 有 3 次 `PixelUnshuffle(2)`）

各数据集的合法值：

| 数据集 | 裁剪后平面 | `gt_size` 上限 | 配置取值 |
|--------|-----------|---------------|---------|
| 联影投影域 | 96 × 512 | 96 | `96` |
| 西门子投影域 | 256 × 496 | 256 | `192` |
| 联影图像域 | 512 × 512 | 512 | `256` |

> 顺便：**整个平面本身不需要是 8 的倍数**。训练时网络只见 `gt_size×gt_size` 的 patch；验证走 `pad_test` 会自动补齐到 8 的倍数再裁回；推理脚本同样会补齐。只有 `gt_size` 有整除要求。

### 案例 6：索引 txt 不存在 / 路径是 Windows 路径

**现象**：`FileNotFoundError` 打开 `train_ximenzi_proj.txt`，或读到的路径是 `D:/ImageReconstruction/...`。

**原因**：索引 txt 被 `.gitignore` 忽略（`*_img.txt` / `*_proj.txt`），**clone 下来是没有的**，必须在服务器上跑 prepare 脚本生成。索引里存的是**绝对路径**，在本地生成再传上去是无效的。

**修复**

```bash
# 联影
python prepare_ct_data.py     --data-root <联影数据根目录> --output ./
python prepare_ct_proj_data.py --data-root <联影数据根目录> --output ./
# 西门子
python prepare_ximenzi_proj_data.py --data-root /mnt/tm/ximenzi_pbi_data --output ./
```

用 `head -2 train_ximenzi_proj.txt` 确认路径指向服务器上的真实位置。

---

### 案例 7：训练时间远超 `eta` 预估（验证开销被低估）

**现象**

训练日志里 `eta` 显示 5:48，但实际要跑 20 小时以上：

```
14:44:00 INFO: [CT_Pr..][epoch:  0, iter:     200, ...] [eta: 6:16:40, time (data): 0.234 (0.002)] l_pix: 1.0139e-02
14:47:05 INFO: [CT_Pr..][epoch:  0, iter:   1,000, ...] [eta: 5:48:45, time (data): 0.208 (0.002)] l_pix: 3.4131e-03
```

看起来像"6 小时就能跑完"。

**原因（两个叠加）**

**① `eta` 没算验证。** 它的算法是「从训练开始到现在的平均速度 × 剩余迭代数」：

```python
# basicsr/utils/logger.py
time_sec_avg = total_time / (current_iter - self.start_iter + 1)
eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
```

`iter 1000` 那行是**打印时**算的，而第一次验证紧接着那行才开始跑。所以那一刻 ETA 里**一次验证都没算进去**。

**② 验证本身就是大头，而且比训练还贵。** `validation()` 会跑完**整个** val 集（没有提前退出），每个样本是**全尺寸平面**（不是 patch），且 `batch_size=1`、`num_workers=0`。

联影投影域那次跑完 90000 iter 的实测日志：

| | |
|---|---|
| 总时长 | 9.53 h |
| ├ **验证** | **5.00 h**（90 次 × 3.4 min） |
| └ 训练 | 4.53 h |

**验证占了总时长一半以上**——一直是这个项目的时间大头，只是旧配置总量还忍得住，所以没被察觉。

**排查：直接从日志量出验证耗时**

不用估，日志里就能拆出来。验证会让某两条相邻日志之间的间隔突然变大：

```bash
# 看 iter 1000 那行到 iter 1200 那行的时间差
grep -E "iter:\s+(1,000|1,200)" train_CT_ProjectionDomain_Ximenzi_Restormer_*.log
```

正常间隔约为 `print_freq × 单iter耗时`（本例 200 × 0.23 s ≈ 46 s）；**跨验证的那段会变成十几分钟**，差值就是单次验证耗时。

**修复**

调两个参数（`Options/CT_ProjectionDomain_Ximenzi_Restormer.yml`）：

```yaml
val:
  val_freq: !!float 5000        # 验证次数 90 → 18
  val_sample_ratio: 0.1         # 单次验证样本 3000 → 300
```

- `val_sample_ratio` 是 `Dataset_CT` 读的字段（只对 `datasets.val` 生效），**固定步长抽样、非随机**——随机抽样会让 val PSNR 曲线抖动，无法判断是模型在变还是样本在变。
- 效果：单次验证 12 min → 1.2 min，验证总耗时 18 h → **~22 min**。
- 抽样结果会打进日志：`[Dataset_CT] val_sample_ratio=0.1 → 每 10 个取 1，验证样本 3000 → 300`。

> 详细的验证开销分析见 [4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md) §6.1。

**教训**

`eta` 只在**验证跑过几次、且各阶段 patch 尺寸稳定**之后才可信。早期看到的小数字不要当真。


| 报错关键词 | 最可能的原因 | 去哪一节 |
|-----------|-------------|---------|
| `No module named 'basicsr'` | 包没注册进环境 | 案例 1 |
| `illegal memory access` / 随机崩溃 | cuDNN / CUDA 库冲突 | 案例 2 |
| `cuDNN version incompatibility` | `LD_LIBRARY_PATH` 污染 | 案例 2 |
| 训练极慢 / OOM | GPU 被占 | 案例 3 |
| `proj_norm_clip_max` 相关 | 没标定归一化尺度（**刻意的**） | 案例 4 |
| `LQ (...) is smaller than patch size` | `gt_size` > 裁剪后平面边长 | 案例 5 |
| `FileNotFoundError` 打开 `*.txt` | 索引没生成 | 案例 6 |
| **训练时间远超 `eta` 预估** | **验证开销（`eta` 不含验证）** | **案例 7** |
| `iters` 之和 != `total_iter` | yml 两处没同步改 | 见 [4-CONFIG_PARAMS.md](4-CONFIG_PARAMS.md) §2.4 |
| 日志样本数与预期不符 | 索引生成时视角/切片范围不对 | 案例 6 |

---

## 四、通用原则

1. **加了新机器 / 新环境，先跑 0.2 的两个最小复现**，再跑本项目。能把"环境问题"和"代码问题"一刀切开。
2. **CUDA 报错先加 `CUDA_LAUNCH_BLOCKING=1`**，不要相信默认栈。
3. **怀疑环境时先查版本三件套**（0.3），尤其是 `torch.backends.cudnn.version()` 能不能打印出来。
4. **报错信息要读全**。案例 2 的答案就写在报错正文里（`Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn`），只是被 `illegal memory access` 这个更吓人的标题盖住了。
5. **本项目的 `ValueError`/报错多数是刻意设计的护栏**（把静默错误变成显式报错），先读报错文案，它通常直接告诉你怎么修。
