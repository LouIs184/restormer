# Restormer 训练 · 执行文档（图像域 + 投影域）

## ⚠️ 本仓库支持两套数据集，命令不要混用

| 数据集 | 服务器路径 | yml | 数据准备脚本 | 详细文档 |
|--------|-----------|-----|-------------|---------|
| **联影双能相位 CT**（原有） | `/root/autodl-tmp/联影双能相位数据2080`（13 个病人） | `Options/CT_ImageDomain_Restormer.yml`、`Options/CT_ProjectionDomain_Restormer.yml` | `prepare_ct_data.py`、`prepare_ct_proj_data.py` | [../DATASET_FORMAT.md](../DATASET_FORMAT.md) |
| **西门子 PBI**（新增，**只有投影域**，无重建数据） | `/mnt/tm/ximenzi_pbi_data`（本项目用 `01`~`20`） | `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml` | `prepare_ximenzi_proj_data.py` + 标定脚本 `analyze_proj_range.py`（**必须先跑，否则 dataset 直接报错**） | [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) |

> 两套数据集**产物目录互相隔离**（联影 `experiments/CT_*_Restormer/`，西门子 `experiments/CT_ProjectionDomain_Ximenzi_Restormer/`）。
> 下文凡是给命令的地方都会**并列给出两套数据集的命令**；文中出现的联影专属数值（形状 `720×128×512`、96 行、13 个病人、7200 样本、`_100000` 后缀等）在西门子上都不同，**西门子的对应值一律以 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) 为准**。

> 命令优先，照抄即可。配套代码都已在本仓库就位：
>
> - **图像域**：`prepare_ct_data.py`、`Options/CT_ImageDomain_Restormer.yml`、`test_ct.py`
> - **投影域**：`prepare_ct_proj_data.py`、`Options/CT_ProjectionDomain_Restormer.yml`、`test_ct_proj.py`
> - **统一数据集**：`basicsr/data/ct_image_dataset.py`（`Dataset_CT`，靠 yml 的 `domain` 一键切换两个域）
>
> 背景与原理见 [1-RUN_GUIDE.md](1-RUN_GUIDE.md)，投影域方案见 [5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md)。
>
> 前提：数据已放在服务器（两套数据集的路径见上表，西门子为 `/mnt/tm/ximenzi_pbi_data`），有 ≥1 张 NVIDIA GPU。

---

## 0. 一分钟速览

```bash
cd /path/to/Restormer

# ① 装环境（只做一次）
conda create -n restormer python=3.10 -y && conda activate restormer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tensorboard yapf lpips
python setup.py develop --no_cuda_ext

# ② 生成数据索引（联影）
python prepare_ct_data.py          # 图像域（train_img.txt / valid_img.txt）
python prepare_ct_proj_data.py     # 投影域（train_proj_img.txt / valid_proj_img.txt）

# ②' 西门子 PBI（只有投影域）：先标定归一化尺度，再生成索引
python analyze_proj_range.py --data-root /mnt/tm/ximenzi_pbi_data
#   -> 把推荐的 proj_norm_clip_max 填进 Ximenzi yml 的 train 和 val 两处
python prepare_ximenzi_proj_data.py --data-root /mnt/tm/ximenzi_pbi_data --output ./
#   -> 产出 train_ximenzi_proj.txt / valid_ximenzi_proj.txt / test_ximenzi_proj.txt

# ③ 开训（单卡）—— 选一个 yml 即可
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none              # 联影·图像域
python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml --launcher none         # 联影·投影域
python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml --launcher none # 西门子·投影域
```

**几个 yml 的 `datasets` 段结构相同**：`type` 都是 `Dataset_CT`，靠 `domain: image / domain: proj` 区分；联影两个域与西门子投影域的网络、损失、优化器、调度器完全一致，差异只在数据形状、裁剪量与 `proj_norm_clip_max`（西门子的取值与理由见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md)）。下面每步展开说明。

---

## 1. 装环境（一次性）

```bash
cd /path/to/Restormer

conda create -n restormer python=3.10 -y
conda activate restormer

# 按服务器 CUDA 版本选 index-url：
#   CUDA 11.8 -> cu118，CUDA 12.1 -> cu121，CPU 调试 -> cpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tensorboard yapf lpips

# 必须带 --no_cuda_ext：跳过编译 dcn/fused_act/upfirdn2d 三个 CUDA op（Restormer 用不到）
# 若新版 setuptools 报错：先 pip install "setuptools<60"，再重跑本条命令
python setup.py develop --no_cuda_ext
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from basicsr.models.archs.restormer_arch import Restormer; print('restormer ok')"
python -c "from basicsr.data.ct_image_dataset import Dataset_CT; print('dataset ok')"
```

> 若 `import basicsr` 报 numpy 相关错：`pip install "numpy<2"`。
> 若 PyTorch ≥2.6 加载权重报 `weights_only` 错：装 `torch<2.6`（`test_ct_proj.py` 已加 `weights_only=False` 兼容；`test_ct.py` 若报错同样加一行，或直接降 torch）。

---

## 2. 生成数据索引（每次换数据/划分后重新跑）

在 **Restormer 仓库根目录**执行。**联影**的两个脚本共用同一份病人划分（train 10 次 / valid 3 次），只是样本粒度不同；**西门子**由 `prepare_ximenzi_proj_data.py` 按病人划分（`01`~`15` / `16`~`18` / `19`~`20`）：

| 数据集 | 脚本 | 生成文件 | 样本数 | 索引第三列含义 |
|--------|------|---------|--------|---------------|
| 联影 | `prepare_ct_data.py` | `train_img.txt` / `valid_img.txt` | 1580 / 474 | 切片号 z（1..158） |
| 联影 | `prepare_ct_proj_data.py` | `train_proj_img.txt` / `valid_proj_img.txt` | 7200 / 2160 | 视角号 v（0..719） |
| 西门子 | `prepare_ximenzi_proj_data.py` | `train_ximenzi_proj.txt` / `valid_ximenzi_proj.txt` / `test_ximenzi_proj.txt` | 15000 / 3000 / 2000 | 视角号（每人 1000 视角） |

```bash
# —— 联影 ——
python prepare_ct_data.py
python prepare_ct_proj_data.py
# 默认数据根目录: /root/autodl-tmp/联影双能相位数据2080
# 指定路径: python prepare_ct_proj_data.py --data-root /你的/数据目录 --output ./

# —— 西门子 PBI（投影域）——
python prepare_ximenzi_proj_data.py --data-root /mnt/tm/ximenzi_pbi_data --output ./
# 脚本会校验每个 .raw 恰为 1,000,000,000 字节，形状不符直接报错
```

> 索引 txt 被 `.gitignore` 忽略（`*_img.txt` / `*_proj.txt`），**不在 git 里**——换机器/换划分后必须在服务器上重跑对应 prepare 脚本生成。

确认生成成功：

```bash
# —— 联影 ——
wc -l train_img.txt valid_img.txt train_proj_img.txt valid_proj_img.txt
# 应为 1580  474  7200  2160
head -2 train_proj_img.txt   # 路径应指向 /root/autodl-tmp/联影双能相位数据2080/.../proj_*.raw ... 0

# —— 西门子 PBI ——
wc -l train_ximenzi_proj.txt valid_ximenzi_proj.txt test_ximenzi_proj.txt
# 应为 15000  3000  2000
```

---

## 3. 训练前快速校验（可选但推荐，30 秒）

确认各数据集都能读取、形状正确（西门子的 yml 需先按 `analyze_proj_range.py` 填好 `proj_norm_clip_max`，否则会直接报错）：

```bash
python - <<'EOF'
import yaml
from basicsr.data import create_dataset
for yml in ['Options/CT_ImageDomain_Restormer.yml',
            'Options/CT_ProjectionDomain_Restormer.yml',
            'Options/CT_ProjectionDomain_Ximenzi_Restormer.yml']:
    opt = yaml.safe_load(open(yml))
    tr = opt['datasets']['train']; tr['phase'] = 'train'; tr['scale'] = opt['scale']
    ds = create_dataset(tr)
    x = ds[0]
    print(f"{yml} | domain={tr.get('domain')} | samples={len(ds)} "
          f"| lq={tuple(x['lq'].shape)} {x['lq'].dtype}")
    va = opt['datasets']['val'];  va['phase'] = 'val';  va['scale'] = opt['scale']
    print(f"   val samples={len(create_dataset(va))}")
EOF
```

期望输出：

```
Options/CT_ImageDomain_Restormer.yml | domain=image | samples=1580 | lq=(1, 256, 256) torch.float32
   val samples=474
Options/CT_ProjectionDomain_Restormer.yml | domain=proj | samples=7200 | lq=(1, 96, 96) torch.float32
   val samples=2160
```

> 西门子 PBI（`Options/CT_ProjectionDomain_Ximenzi_Restormer.yml`）的对应样例：索引行数为 `15000 3000 2000`（train/valid/test），**验证集**样本张量形状为 `lq=(1, 256, 496)`（平面 256×496）。该 yml 的 `proj_norm_clip_max` 标定前是 `~`，**不先跑 `analyze_proj_range.py` 填值就会直接报错**；形状/裁剪量取值理由见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md)。

> 注意：yml 里 `txt: ./train_*.txt` 是相对路径，**必须从仓库根目录启动训练**，否则请改成绝对路径。

---

## 4. 开始训练

### 单卡（最常用）

```bash
conda activate restormer
cd /path/to/Restormer
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
# 投影域换成:
python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml --launcher none
# 西门子 PBI 投影域（产物落在 experiments/CT_ProjectionDomain_Ximenzi_Restormer/，与联影隔离）:
python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml --launcher none
```

### 多卡（例如 4 卡）

先把对应 yml 的 `num_gpu: 1` 改成 `4`，然后：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
    basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher pytorch
```

### 后台跑 + 留存日志

```bash
nohup python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml \
    --launcher none > train_ct_proj.log 2>&1 &
echo $!   # 记下 PID，方便 kill

# 西门子 PBI 同理，换成 Ximenzi yml 即可：
nohup python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml \
    --launcher none > train_ct_proj_ximenzi.log 2>&1 &
echo $!
```

**正常启动的迹象**（前几秒日志里应看到）：
- 打印网络结构、`Dataset Dataset_CT - TrainSet is created.`（图像域 1580 条 / 联影投影域 7200 条 / 西门子投影域 15000 条）
- 迭代号 + `# l_pix: 0.x` 的 loss 稳步下降
- 每 1000 迭代出现 `Validation ValSet, # psnr: xx.xx`

### 显存不够？

| 域 | 做法 |
|----|------|
| 图像域 | `batch_size_per_gpu` 和 `mini_batch_sizes` 一起减半，或 `gt_size/gt_sizes` 从 256 降到 192（对应 `[128,160,192]`、`iters` 改 3 段、`total_iter` 同步改） |
| 投影域（联影） | **`gt_sizes` 最大值不能超过 96**（裁剪后高度，见 5 文档 §4.1）。显存不够就把 `gt_sizes` 往下压（如 `[48,64,80]`），`mini_batch_sizes` 对应减，`total_iter` 保持 `sum(iters)` |
| 投影域（西门子） | 裁剪后平面为 **256×496**，`gt_size` / `gt_sizes` 上限随之不同（yml 里为 `gt_size: 192`、`gt_sizes: [96,128,160,192]`）；显存不够同样把 `gt_sizes` 往下压并对应减 `mini_batch_sizes`，理由见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) |

---

## 5. 监控

```bash
tail -f train_ct_proj.log                        # 实时日志（西门子后台跑时是 train_ct_proj_ximenzi.log）
tensorboard --logdir tb_logger --port 6006       # 浏览器开 http://服务器IP:6006
nvidia-smi                                       # 显存/占用
```

产物位置（实验目录按 yml 的 `name` 区分，两套数据集/两个域互不覆盖）：

```
experiments/CT_ImageDomain_Restormer/
experiments/CT_ProjectionDomain_Restormer/
experiments/CT_ProjectionDomain_Ximenzi_Restormer/   # 西门子 PBI 投影域（与联影隔离）
├── models/            # 权重 net_g_latest.pth + 每5000步快照 net_g_<iter>.pth
├── training_states/   # 断点 *.state（自动续训用）
└── ...日志
```

---

## 6. 断点续训（自动）

**训练中断后，直接用同一条命令重跑即可**，脚本会自动从对应实验目录的 `training_states/` 里最新的 `.state` 恢复：

```bash
python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml --launcher none         # 联影
python basicsr/train.py -opt Options/CT_ProjectionDomain_Ximenzi_Restormer.yml --launcher none # 西门子
```

日志出现 `Resuming training from epoch: ..., iter: ...` 即恢复成功。（续训从各自实验目录的 `training_states/` 恢复，两套数据集互不影响。）

---

## 7. 推理 / 评估

> **`--data_root` 与 `--txt` 的用法**
>
> - **`--patient` 模式（联影常用）**：`test_ct.py` / `test_ct_proj.py` 的 `--data_root` 默认都是服务器路径 `/root/autodl-tmp/联影双能相位数据2080`。**在本地（WSL/Windows）跑推理必须显式传 `--data_root`** 指向本地数据目录（如 `/mnt/d/.../联影双能相位数据2080`），否则读不到数据、报 PermissionError。
> - **`--txt` 模式（`test_ct_proj.py` 新增，西门子用）**：从索引 txt 逐行读路径，**路径来自索引文件本身，不需要再传 `--data_root`，也不需要 `--patient`**，直接 `--txt ./test_ximenzi_proj.txt` 即可。

### 图像域

> 仅**联影**数据集可用。西门子 PBI **没有重建数据**，图像域整条线不可用（见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) §1.2）。

```bash
python test_ct.py \
    --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
    --data_root /root/autodl-tmp/联影双能相位数据2080 \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

**`test_ct.py` 参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 训练权重 `net_g_latest.pth` |
| `--data_root` | `/root/autodl-tmp/联影双能相位数据2080` | 数据根目录；本地推理必须显式传 |
| `--patient` | 必填，可多个 | 要推理的病人文件夹（空格分隔） |
| `--result_dir` | `./results/ct` | 输出目录 |
| `--config` | `Options/CT_ImageDomain_Restormer.yml` | 训练 yml：取网络结构 + `hu_min/hu_max` |
| `--hu_min` | 读 yml（默认 -1000） | 覆盖归一化下限 |
| `--hu_max` | 读 yml（默认 1000） | 覆盖归一化上限 |

输出到 `results/ct/`：

| 文件 | 内容 |
|------|------|
| `<病人>_pred.raw` | 校正后体积（float32 HU，`(158,512,512)`） |
| `<病人>_input.raw` | 输入切片（已剔首尾，与 pred 同形状） |
| `<病人>_gt.raw` | 目标切片（同上，与 pred 同形状） |
| 终端打印 | input vs gt / pred vs gt 的 MSE·RMSE·MAE·PSNR·SSIM（HU 值域，同 AICT-code infer.py） |

### 投影域

> **⚠️ 环境注意**：若你的机器存在 cuDNN 版本冲突（报 `CUDNN_STATUS_NOT_SUPPORTED` 或 `CUDA error: an illegal memory access`），**所有 python 命令前都要加 `env -u LD_LIBRARY_PATH`**——训练、推理、脚本一视同仁，而且它只对当前这一条命令生效。
> 判断与永久修复见 [7-TROUBLESHOOTING.md](7-TROUBLESHOOTING.md) 案例 2。例如西门子推理要写成：
> ```bash
> env -u LD_LIBRARY_PATH python test_ct_proj.py \
>     --weights experiments/CT_ProjectionDomain_Ximenzi_Restormer/models/net_g_latest.pth \
>     --config Options/CT_ProjectionDomain_Ximenzi_Restormer.yml \
>     --txt ./test_ximenzi_proj.txt
> ```

**联影（`--patient` 模式）**：

```bash
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Restormer/models/net_g_latest.pth \
    --data_root /root/autodl-tmp/联影双能相位数据2080 \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

**西门子 PBI（`--txt` 模式，路径来自索引文件，不需要 `--data_root`）**：

```bash
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Ximenzi_Restormer/models/net_g_latest.pth \
    --config Options/CT_ProjectionDomain_Ximenzi_Restormer.yml \
    --txt ./test_ximenzi_proj.txt
# 产出 results/ct_proj/<病人>_{pred,input,gt}.raw（裁剪后平面 256×496，见下表）
```

**`test_ct_proj.py` 参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 训练权重 `net_g_latest.pth` |
| `--data_root` | `/root/autodl-tmp/联影双能相位数据2080` | 数据根目录；`--patient` 模式下本地推理必须显式传。**用 `--txt` 时不需要** |
| `--patient` | 必填，可多个 | 要推理的病人文件夹（空格分隔）。**用 `--txt` 时不需要** |
| `--txt` | 无 | 从索引 txt 逐行读路径（如 `./test_ximenzi_proj.txt`，西门子用）；路径写在索引文件里，**用它时不需要传 `--data_root` / `--patient`** |
| `--result_dir` | `./results/ct_proj` | 输出目录 |
| `--config` | `Options/CT_ProjectionDomain_Restormer.yml` | 训练 yml：取网络结构 + `proj_shape` / `proj_crop_rows` / `proj_crop_cols` / `proj_norm_clip_max`（西门子请填 `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml`） |
| `--crop_rows` | 读 yml（联影默认 16，西门子为 122） | 覆盖黑边裁剪行数（input/gt 按它裁，保证三卷同形状） |
| `--proj_clip_max` | 读 yml（联影默认 0.28，西门子为标定值） | 覆盖投影域归一化上限 |

输出到 `results/ct_proj/`：

| 文件 | 内容 |
|------|------|
| `<病人>_pred.raw` | 校正后 sinogram 体积（float32，联影 `(720,96,512)`；西门子 `(1000,256,496)`，可直接接你的重建流程） |
| `<病人>_input.raw` | 输入投影（已按 `proj_crop_rows`/`proj_crop_cols` 裁黑边，与 pred 同形状） |
| `<病人>_gt.raw` | 目标投影（同上裁剪，与 pred 同形状） |
| 终端打印 | input vs gt / pred vs gt 的 MSE·RMSE·MAE·PSNR·SSIM（原始值域，同 AICT-code infer.py） |

> 投影域 `.raw` 是「校正后的 sinogram」，**要看图像效果必须重建**——把 `<病人>_pred.raw` 接你的重建流程（几何参数用扫描协议 / AICT-code 的），脚本本身不含重建。
> 对比 AICTVer2 基线时，用同一批验证病人、同一归一化跑它那边的评估再比（联影为 `clip(x,0,0.28)/0.28`；西门子的 `clip_max` 是重新标定的值，**两套数据集的验证 PSNR 不能横向比较**，见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) §四）。

---

## 8. 常见问题对照

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: numpy` 等 | 先 `conda activate restormer` 再跑 |
| `basicsr` 报 numpy 2.x 兼容错 | `pip install "numpy<2"` |
| 权重/续训报 `weights_only` | `pip install "torch<2.6"`，或在对应 `torch.load` 加 `weights_only=False`（`test_ct_proj.py` 已加） |
| `Dataset ... is not found` | `ct_image_dataset.py` 文件名必须以 `_dataset.py` 结尾且在 `basicsr/data/` 下 |
| 投影域训练样本大部分是人造反射边（联影） | **`gt_sizes` 最大值超了裁剪后高度 96**：改回 `[48,64,80,96]` |
| 投影域训练样本大部分是人造反射边（西门子） | `gt_sizes` 最大值超了裁剪后平面边长（256×496）：改回 yml 默认的 `[96,128,160,192]`，见 [6-XIMENZI_ADAPTATION.md](6-XIMENZI_ADAPTATION.md) |
| 投影域启动即报 `proj_norm_clip_max` 相关的 `None`/缺失错误 | 投影域下该字段留空/缺失会**直接报错**（不再静默回退 0.28）。先跑 `python analyze_proj_range.py --data-root /mnt/tm/ximenzi_pbi_data`，把推荐值填进 yml 的 `train` 和 `val` 两处 |
| 投影域开了 `geometric_augs: true` 效果差 | 90° 旋转会交换视角轴/通道轴，物理无意义，改 `false` |
| `xxx is not in ... paths` / 样本数不对 | 重跑对应的 prepare 脚本（`prepare_ct_*.py` / `prepare_ximenzi_proj_data.py`），检查 `txt` 路径是否是服务器路径 |
| 启动即 `CUDA out of memory` | 见第 4 节「显存不够」 |
| 训练几轮 val PSNR 不升反降 | 过拟合：图像域开 `mixup: true`（已默认开）；投影域可尝试开 `mixup`、减小 `total_iter`、早停看快照 |
| 多卡只在 rank0 打日志/验证 | 正常现象 |

---

## 附：本次执行需要的文件清单

| 文件 | 状态 |
|------|------|
| `prepare_ct_data.py` | ✅ 已创建 |
| `prepare_ct_proj_data.py` | ✅ 已创建 |
| `basicsr/data/ct_image_dataset.py`（`Dataset_CT`，含 `domain` 开关） | ✅ 已创建 |
| `Options/CT_ImageDomain_Restormer.yml`（`domain: image`） | ✅ 已创建 |
| `Options/CT_ProjectionDomain_Restormer.yml`（`domain: proj`） | ✅ 已创建 |
| `test_ct.py` | ✅ 已创建 |
| `test_ct_proj.py`（含 `--txt` 模式） | ✅ 已创建 |
| `train_img.txt` / `valid_img.txt` | ⏳ 第 2 步生成 |
| `train_proj_img.txt` / `valid_proj_img.txt` | ⏳ 第 2 步生成 |
| `Options/CT_ProjectionDomain_Ximenzi_Restormer.yml`（`domain: proj`，西门子） | ✅ 已创建（`proj_norm_clip_max` 待标定） |
| `prepare_ximenzi_proj_data.py`（西门子，含 test 划分） | ✅ 已创建 |
| `analyze_proj_range.py`（西门子 `clip_max` 标定） | ✅ 已创建 |
| `train_ximenzi_proj.txt` / `valid_ximenzi_proj.txt` / `test_ximenzi_proj.txt` | ⏳ 第 2 步生成 |
