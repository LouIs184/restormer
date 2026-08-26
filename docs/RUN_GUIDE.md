# Restormer 运行指南（双能相位 CT 束硬化 / 相位模糊伪影校正）

> 适用数据：`DATASET_FORMAT.md` 描述的 13 个病人 `.raw` 数据（服务器上位于 `/root/autodl-tmp/联影双能相位数据2080`）。
> 目标任务：以「模糊相位校正后的重建/投影」（含伪影）为输入，恢复「无相位束硬化校正的干净重建/投影」。
> 本文档已与你的历史训练代码 `C:\imageReconstruction\AICTVer2` 对齐（病人划分来自 `patients.yaml`，归一化、投影域做法与其一致）。
> 本文档回答两个问题：**怎么跑这个项目**、**用图像域还是投影域训练**。

---

## 0. 结论速览（TL;DR）

- **推荐用「图像域」训练**：直接对 `512×512` 的 CT 切片对（输入 `rec_pbi_fs_dec_blur_phase_100000.raw`，目标 `rec_no_pbi.raw`）做 2D 去伪影 + 去模糊。这与你在 AICTVer2 里的 `DenseUNet_PhaseCT` 是同一域、同一任务，可直接做模型对比（DenseUNet2d vs Restormer）。
- **三个决定性理由**：
  1. 这个仓库本质是 **2D 图像恢复框架**（Restormer 是 2D Transformer），图像域直接产出最终要交付的 CT 图像；
  2. **Restormer 仓库里没有任何 CT 重建（FBP）代码** —— 投影域训练得到的是「校正后的 sinogram」，无法在这个仓库里直接变回图像来验证/交付（你 AICTVer2 里有 ProjCT 管线，但本项目没有重建环节）；
  3. 图像域的 PSNR/SSIM 直接反映临床影像质量，评估闭环。
- **运行三步**：① 生成数据索引 `train_img.txt` / `valid_img.txt`（第 4 节，按你 `patients.yaml` 的划分）；② 新增一个自定义数据集文件 + 一个 yml 配置（第 5、6 节给了完整代码）；③ 一条命令开始训练。
- 环境：Python 3.8~3.10 + PyTorch 2.x + 常用 pip 包 + `python setup.py develop --no_cuda_ext`（第 2 节给完整命令）。

---

## 1. 项目运行机制（先读懂再跑）

### 1.1 这个仓库是什么

- **Restormer**（CVPR 2022）是一个 **2D 图像到图像的恢复网络**，`forward` 最后一行是 `out = self.output(...) + inp_img`，即**残差学习**。
- 输入输出通道数由配置控制：`inp_channels` / `out_channels`，灰度图用 `1`（两者必须相等，因为是残差结构）。
- 这个仓库不是「一个针对你的数据写好的代码」，而是一个 **BasicSR 框架 + Restormer 模型** 的通用架子。仓库里已有的 Deraining / Denoising / Deblurring 只是四个「示例任务」，各自有 yml 配置 + dataset 类。**你的 CT 任务需要仿照它们，自己加一个 yml 配置和一个小 dataset 类**（第 5、6 节给了完整可直接用的代码）。

### 1.2 训练流程（BasicSR 约定）

```
yml 配置
  ├─ datasets.train / datasets.val  数据源（dataset 类的参数）
  ├─ network_g                      Restormer 网络结构
  ├─ train                          优化器 / 学习率 / loss / 迭代数
  └─ val / logger                   验证频率 / PSNR 指标 / 日志

basicsr/train.py -opt xxx.yml       启动训练（命令行入口）
  ├─ create_dataset()   按 datasets.* 的 type 动态 import 对应 Dataset 类
  │                      （规则：basicsr/data/ 下所有 *s_dataset.py 会被自动扫描）
  ├─ create_dataloader() 组 DataLoader，支持 progressive training
  ├─ ImageCleanModel     定义网络 + L1 loss + AdamW + 余弦退火重启调度器
  └─ 循环迭代：每 val_freq 次在 val 上算 PSNR
```

关键点：
- **框架默认只读 PNG/JPG 图像文件**（`cv2.imdecode`，HWC 布局，`[0,1]` float32）。你的数据是 `.raw` 体积文件，所以**需要一个小型自定义 Dataset 直接读 .raw**（保留 float32 HU 精度，不丢精度）。这就是第 5 节那个文件。
- 训练产物统一落在 `experiments/<name>/`：模型权重在 `models/`，断点状态在 `training_states/`，日志在根目录。

---

## 2. 环境要求

> 你在云服务器上跑，下面命令按 **Linux + NVIDIA GPU** 写。仓库官方说 PyTorch 1.8.1，但实际上 **PyTorch 1.8+ 都能跑，推荐直接上 2.x**（你 AICTVer2 用的是 py3.10/3.12，Python 3.8~3.10 都兼容本仓库）。

### 2.1 硬件

- **至少 1 张 ≥ 12GB 显存的 GPU** 即可开始；24GB 更舒服。
- 数据总共约 8.8GB、13 个病人，**单卡完全够**。多卡只是更快，不是必须。

### 2.2 软件依赖

```bash
# 1) 创建 conda 环境
conda create -n restormer python=3.10 -y
conda activate restormer

# 2) 安装 PyTorch（按你服务器 CUDA 版本选 index-url，下面示例为 CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   CUDA 12.1 用:  https://download.pytorch.org/whl/cu121
#   CPU 调试用:    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   ⚠️ 建议用 torch>=1.8,<2.6；若用 2.6+，torch.load 默认 weights_only 可能影响断点续训（见第 10 节）

# 3) 其余依赖
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tensorboard yapf lpips

# 4) 安装本仓库的 basicsr（重点：必须加 --no_cuda_ext，跳过 dcn/fused_act/upfirdn2d 三个 CUDA op，
#    它们 Restormer 用不到，不跳过会尝试用 nvcc 编译、耗时且易失败）
#    若新版 setuptools 报错：先 pip install "setuptools<60"，再重跑本条命令
python setup.py develop --no_cuda_ext
#    实在不行可用免安装方案：export PYTHONPATH=$PWD（使 import basicsr 可用）
```

### 2.3 环境验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "from basicsr.models.archs.restormer_arch import Restormer; print('restormer ok')"
```

> 若 `import basicsr` 报 numpy 相关错误（numpy 2.x 兼容问题），`pip install "numpy<2"` 后重试。

---

## 3. 图像域 vs 投影域：怎么选

你的数据天然支持两个域。**两个域你在 AICTVer2 里都训练过**（PhaseCT / ProjCT），所以这里是「选哪个域先跑 Restormer」。逐项对比：

| 维度 | 图像域（推荐） | 投影域 |
|------|--------------|--------|
| 训练样本 | 每病人 158 张切片 → 训练 **1580 对**（10 病人） | 每病人 720 个视角 → 训练 **7200 对**（10 病人） |
| 单样本尺寸 | 512×512，灰度（整张 CT 切片） | 单视角探测器平面 128×512（crop 黑边后 96×512），灰度 |
| 输出形式 | **直接得到校正后的 CT 图像**（交付物） | 校正后的视角（sinogram 域），**还需重建**才有图像 |
| 在本仓库能否评估 | ✅ 直接算图像 PSNR/SSIM | ⚠️ 仓库无 FBP 重建代码，转不回图像；除非另接重建 |
| 物理合理性 | 后验（learned image-domain correction） | 更本质（束硬化是 sinogram 域现象，post-log 伪影可加） |
| 与 Restormer 匹配度 | 高（本就是 2D 图像恢复模型） | 中（视角尺寸 128×512 非方形、各向异性，但可 patch 化） |
| 与你的基线可比性 | 直接对标 `DenseUNet_PhaseCT` | 对标 `DenseUNet_ProjCT` |
| 实现成本 | 低（本文档给现成代码） | 中（移植你 AICTVer2 的 `proj_reader` + `prepare_proj_data.py`，见第 9 节） |

### 3.1 为什么推荐图像域

1. **闭环交付**：Restormer 输出直接是图像，PSNR/SSIM 直接衡量你真正关心的东西。投影域训练完你连「效果好不好」都看不到——除非在 Restormer 仓库外再套一层重建（ASTRA / cuCIM 等），那属于另一个工程。
2. **模型匹配**：输入是 `rec_pbi_fs_dec_blur_phase`（既模糊又有束硬化），本质是 **去模糊 + 去束硬化** 的组合。Restormer 在去模糊上是 SOTA，其长程自注意力对「束硬化 cupping 这类全局低频伪影」很合适（CNN 感受野不够，Transformer 正好）。
3. **精度无损**：图像域直接读 float32 HU 值，归一化到 `[0,1]` 后训练，没有中间量化损失。
4. **公平对比**：你在 AICTVer2 里已经有 `DenseUNet_PhaseCT` 的基线（同一图像域、同一 `[-1000,1000]` 归一化），Restormer 跑图像域可以直接在同一测试集上对比两者提升。

### 3.2 什么时候才值得先跑投影域

- 你**已有重建管线**（或愿意在 Restormer 之外接上 AICTVer2 里的重建/评估环节）；
- 想直接对标自己的 `DenseUNet_ProjCT` 基线，看 Transformer 在 sinogram 域能不能赢过 DenseUNet2d；
- 数据量上投影域更大（7200 vs 1580），且 post-log 伪影可加、残差学习物理意义成立（`DATASET_FORMAT.md` 第 118 行也确认了）。

如果你走投影域，第 9 节给了按你 AICTVer2 做法移植的适配。

---

## 4. 数据准备（图像域）

### 4.1 数据位置与目录结构

数据在服务器 `/root/autodl-tmp/联影双能相位数据2080`，13 个病人文件夹（每个含 4 个 .raw）。**路径含中文，Linux 下 UTF-8 没问题**，但索引文件编码要处理（第 5 节数据集代码已做兼容）。

### 4.2 训练 / 验证集划分（单点事实：`patients.yaml`，与 AICTVer2 完全一致）

| 集 | 病人文件夹 | 扫描 | 切片对数 |
|----|-----------|------|---------|
| 训练 | 11034、11849、15962(×2)、17238、18537、18631、38423、51741、51789 | 10 | 1580 |
| 验证 | 72278、91963(×2) | 3 | 474 |

> 与 `DATASET_FORMAT.md` 第 210 行的建议不同：你的 `patients.yaml` 把 17238 和平扫的 **18537 放进了训练集**，验证集用 72278 和 91963。**以 `patients.yaml` 为准**（划分跨域一致，ProjCT 也用同一份）。

### 4.3 索引文件（数据加载入口，对齐你的 AICTVer2 工作流）

你 AICTVer2 的做法是 `prepare_data.py` 生成 `train_img.txt` / `valid_img.txt`，每行：

```
<输入.raw绝对路径> <目标.raw绝对路径> <切片号>
```

切片号取 `1..158`（剔除首尾切片，160-2=158）。**⚠️ 你 AICTVer2 目录里现存的 `train_img.txt`/`valid_img.txt` 是旧划分、且写的是 Windows 路径，不能直接拿来用，必须在服务器上重新生成。**

新建 `prepare_ct_data.py`（放 Restormer 仓库根目录，内容即你 AICTVer2 的 `prepare_data.py` 简化版，划分写死为 `patients.yaml`）：

```python
## 生成 train_img.txt / valid_img.txt（图像域，切片 1..158）
import os

DATA_ROOT = "/root/autodl-tmp/联影双能相位数据2080"
TRAIN = [
    "11034_307811_960+_AXIAL_CE1_M067Y_20211215_Thick1_Incre1",
    "11849_308361_960+_AXIAL_CE1_M044Y_20211216_Thick1_Incre1",
    "15962_15633_960+_AXIAL_CE1_M034Y_20200601_Thick1_Incre1",
    "15962_35868_960+_AXIAL_CE1_M034Y_20200601_Thick1_Incre1",
    "17238_33314_960+_AXIAL_CE1_F063Y_20220106_Thick1_Incre1",
    "18537_93073_960+_AXIAL_CE0_M067Y_20210430_Thick1_Incre1",
    "18631_303226_960+_AXIAL_CE1_F058Y_20211216_Thick1_Incre1",
    "38423_35948_960+_AXIAL_CE1_M049Y_20200601_Thick1_Incre1",
    "51741_308269_960+_AXIAL_CE1_M048Y_20211215_Thick1_Incre1",
    "51789_393117_960+_AXIAL_CE1_M074Y_20211215_Thick1_Incre1",
]
VALID = [
    "72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1",
    "91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1",
    "91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1",
]
INP = "rec_pbi_fs_dec_blur_phase_100000.raw"
GT = "rec_no_pbi.raw"

def write_list(fname, patients):
    with open(fname, "w", encoding="utf-8") as f:
        for p in patients:
            for z in range(1, 159):
                f.write(f"{os.path.join(DATA_ROOT, p, INP)} "
                        f"{os.path.join(DATA_ROOT, p, GT)} {z}\n")

write_list("train_img.txt", TRAIN)
write_list("valid_img.txt", VALID)
print("done: train", 158 * len(TRAIN), "valid", 158 * len(VALID))
```

> 若你改 `patients.yaml`，改上面两个列表即可，两边保持一致。

### 4.4 归一化窗口（与 AICTVer2 对齐）

数据是 **HU 值**。AICTVer2 的 `raw_reader.py` 用的是：

```
img_norm = clip((HU - hu_min) / (hu_max - hu_min), 0, 1)，其中 hu_min=-1000, hu_max=1000
```

**本项目沿用 `hu_min=-1000, hu_max=1000`**（第 5、6、8 节均为此默认值），这样与你的 DenseUNet 基线可比。输入和目标用同一组窗口。该窗口已由 `analyze_hu_range.py` 的数据分析确认：与数据驱动窗口 `[-1050, 1050]` 对解剖组织的映射几乎等价，窗外只是背景空气与少量金属（金属两侧同裁、loss 为 0，无影响），无需更换。

---

## 5. 代码：新增自定义数据集（读索引文件 + memmap，不丢精度）

新建文件 `basicsr/data/ct_image_dataset.py`（文件名必须以 `_dataset.py` 结尾，框架自动扫描注册，见 `basicsr/data/__init__.py`）：

```python
## 双能相位 CT 图像域去伪影数据集：按 train_img.txt/valid_img.txt 读取
## 每行: <输入raw路径> <目标raw路径> <切片号1..158>
## 归一化: (HU - (-1000)) / 2000 -> [0,1]（与 AICTVer2 raw_reader 一致）
import locale
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding


class Dataset_CTImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_CTImage, self).__init__()
        self.opt = opt
        self.phase = opt['phase']          # 'train' / 'val'，由框架自动注入
        self.scale = opt['scale']          # 固定为 1
        # 索引文件路径含中文，做编码回退（同 AICTVer2 raw_reader）
        txt = opt['txt']
        try:
            with open(txt, encoding='utf-8') as f:
                self.paired = f.readlines()
        except UnicodeDecodeError:
            with open(txt, encoding=locale.getpreferredencoding()) as f:
                self.paired = f.readlines()
        self.hu_min = float(opt.get('hu_min', -1000.0))
        self.hu_range = float(opt.get('hu_max', 1000.0)) - self.hu_min
        self.geometric_augs = opt.get('geometric_augs', True)
        # np.memmap 惰性映射体积文件，不显式占 RAM（OS 页缓存负责）
        self._vol_cache = {}

    def _volume(self, path):
        if path not in self._vol_cache:
            self._vol_cache[path] = np.memmap(
                path, dtype=np.float32, mode='r', shape=(160, 512, 512))
        return self._vol_cache[path]

    def _norm(self, x):
        return np.clip((x - self.hu_min) / self.hu_range, 0, 1).astype(np.float32)

    def __getitem__(self, index):
        line = self.paired[index].strip()
        inp_path, gt_path, z = line.split()
        z = int(z)

        inp_vol = self._volume(inp_path)
        gt_vol = self._volume(gt_path)

        img_lq = np.expand_dims(self._norm(inp_vol[z]), axis=2)  # (512,512,1)
        img_gt = np.expand_dims(self._norm(gt_vol[z]), axis=2)

        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_lq, img_gt = padding(img_lq, img_gt, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                self.scale, None)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # HWC -> CHW, numpy -> tensor（单通道，无需 BGR->RGB）
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': inp_path,
            'gt_path': gt_path,
        }

    def __len__(self):
        return len(self.paired)
```

> 性能说明：AICTVer2 的 `raw_reader` 每次 `__getitem__` 都 `np.fromfile` 读整个 160MB 体积再取一张切片，慢；这里改用 `np.memmap`，只按需读被访问的页，首轮磁盘 IO 后由页缓存命中。

---

## 6. 代码：新增训练配置

新建文件 `Options/CT_ImageDomain_Restormer.yml`：

```yaml
# general settings
name: CT_ImageDomain_Restormer
model_type: ImageCleanModel
scale: 1
num_gpu: 1            # 改成你的 GPU 数；0 表示纯 CPU（极慢，仅调试）
manual_seed: 100

# dataset and data loader settings
datasets:
  train:
    name: TrainSet
    type: Dataset_CTImage
    txt: ./train_img.txt
    hu_min: -1000     # 与 AICTVer2 PhaseCT 一致
    hu_max: 1000
    geometric_augs: true

    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 4

    ### ------- 渐进式训练（小 patch 先训，再逐步放大）--------
    mini_batch_sizes: [4, 3, 2, 1]       # 每 GPU 批大小
    iters: [30000, 24000, 18000, 18000]  # 每个阶段的迭代数
    gt_size: 256                         # 最大 patch 尺寸
    gt_sizes: [128, 160, 192, 256]       # 对应各阶段 patch 尺寸
    ### -------------------------------------------------------

    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: ValSet
    type: Dataset_CTImage
    txt: ./valid_img.txt
    hu_min: -1000
    hu_max: 1000
    io_backend:
      type: disk

# network structures
network_g:
  type: Restormer
  inp_channels: 1     # 灰度 CT 切片
  out_channels: 1     # 残差结构，必须与 inp_channels 相同
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: WithBias
  dual_pixel_task: False

# path
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

# training settings
train:
  total_iter: 90000
  warmup_iter: -1
  use_grad_clip: true

  # 前 3 万轮固定 3e-4，之后余弦退火到 1e-6
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods: [30000, 60000]
    restart_weights: [1, 1]
    eta_mins: [0.0003, 0.000001]

  mixing_augs:
    mixup: true        # 数据量小，建议开 mixup 缓解过拟合
    mixup_beta: 1.2
    use_identity: true

  optim_g:
    type: AdamW
    lr: !!float 3e-4
    weight_decay: !!float 1e-4
    betas: [0.9, 0.999]

  # losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1
    reduction: mean

# validation settings
val:
  window_size: 8
  val_freq: !!float 1000
  save_img: false
  rgb2bgr: true
  use_image: false        # 直接在张量上算 PSNR（归一化域），避免 uint8 量化
  max_minibatch: 8

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false

# logging settings
logger:
  print_freq: 200
  save_checkpoint_freq: !!float 5000
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500
```

**yml 里需要你改的东西（其余照抄）**：
- `num_gpu`：你的 GPU 张数；
- `txt`：指向第 4.3 节生成的 `train_img.txt` / `valid_img.txt` 的**绝对路径**（或相对仓库根目录的路径）；
- 想对齐 AICTVer2 的 MSE 损失：把 `pixel_opt.type` 改成 `L1Loss`→`L1Loss` 不用改，换 MSE 就把 `type` 改为 `L1Loss` 对应库里的 `MSELoss`（`basicsr/models/losses/losses.py` 里有）；
- 显存紧张（<16GB）时把 `batch_size_per_gpu` 和 `mini_batch_sizes` 各减半，或把 `gt_size` 降到 192。

---

## 7. 开始训练

### 7.1 单卡（最常用）

```bash
conda activate restormer
cd /path/to/Restormer
# 先确保 train_img.txt / valid_img.txt 已生成（第 4.3 节）
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
```

### 7.2 多卡（如 4 卡）

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
    basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher pytorch
```

> `train.sh` 里写死 `--nproc_per_node=8`，按需修改。

### 7.3 后台跑 + 看日志

```bash
nohup python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml \
    --launcher none > train_ct.log 2>&1 &
tail -f train_ct.log

# 另开终端看 tensorboard
tensorboard --logdir tb_logger --port 6006
```

### 7.4 产物位置与断点续训

- 权重：`experiments/CT_ImageDomain_Restormer/models/`（`net_g_latest.pth` + 每 1000 步一个快照）
- 断点：`experiments/CT_ImageDomain_Restormer/training_states/`
- 续训：把 yml 里 `path.resume_state` 指向最近的 `.state` 文件再启动即可。

---

## 8. 推理与评估

新建 `test_ct.py`（放仓库根目录）：读 `.raw` → 逐切片推理 → 同时输出**可视化 PNG**、**写回 HU 的 .raw**、**PSNR/SSIM**。

```python
## 图像域推理：Restormer 权重 -> 校正后的 CT 切片
import argparse
import os
import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.models.archs.restormer_arch import Restormer

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='训练得到的 net_g_latest.pth')
parser.add_argument('--data_root', default='/root/autodl-tmp/联影双能相位数据2080')
parser.add_argument('--patient', nargs='+', required=True, help='要推理的病人文件夹')
parser.add_argument('--result_dir', default='./results/ct')
parser.add_argument('--hu_min', type=float, default=-1000.0)
parser.add_argument('--hu_max', type=float, default=1000.0)
parser.add_argument('--config', default='Options/CT_ImageDomain_Restormer.yml')
args = parser.parse_args()
os.makedirs(args.result_dir, exist_ok=True)

# 从训练配置取网络结构
cfg = yaml.safe_load(open(args.config, 'r'))
net_cfg = dict(cfg['network_g']); net_cfg.pop('type')
model = Restormer(**net_cfg)
ckpt = torch.load(args.weights, map_location='cpu')
model.load_state_dict(ckpt['params'])
model = model.cuda().eval()
print('loaded', args.weights)

INP = 'rec_pbi_fs_dec_blur_phase_100000.raw'
GT = 'rec_no_pbi.raw'
R = args.hu_max - args.hu_min

def norm(x): return np.clip((x - args.hu_min) / R, 0, 1)
def denorm(x): return x * R + args.hu_min

def to_png(h, path):
    cv2.imwrite(path, (norm(h) * 255).astype(np.uint8))

for patient in args.patient:
    inp = np.fromfile(os.path.join(args.data_root, patient, INP),
                      dtype=np.float32).reshape(160, 512, 512)
    gt = np.fromfile(os.path.join(args.data_root, patient, GT),
                     dtype=np.float32).reshape(160, 512, 512)
    out = np.zeros_like(inp)
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for z in range(1, 159):            # 跳过首尾
            x = torch.from_numpy(norm(inp[z])).unsqueeze(0).unsqueeze(0).cuda()
            out[z] = denorm(model(x)[0, 0].cpu().numpy())
            psnr_list.append(peak_signal_noise_ratio(norm(gt[z]), norm(out[z]), data_range=1.0))
            ssim_list.append(structural_similarity(norm(gt[z]), norm(out[z]), data_range=1.0))
    out.astype(np.float32).tofile(os.path.join(args.result_dir, f'{patient}_rec_corrected.raw'))
    np.save(os.path.join(args.result_dir, f'{patient}_out.npy'), out)
    to_png(inp[80], os.path.join(args.result_dir, f'{patient}_z80_input.png'))
    to_png(out[80], os.path.join(args.result_dir, f'{patient}_z80_output.png'))
    to_png(gt[80],  os.path.join(args.result_dir, f'{patient}_z80_gt.png'))
    print(f'{patient}: PSNR={np.mean(psnr_list):.3f} dB, SSIM={np.mean(ssim_list):.4f}')
```

运行（默认推理验证集 3 个病人）：

```bash
python test_ct.py --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
                  --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
                           91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
                           91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

> PSNR/SSIM 在**归一化域**（`[0,1]`，`data_range=1.0`）计算，和 AICTVer2 `metrics.compute_psnr(mse, max_val=1.0)` 口径一致，用于相对比较；不等于 HU 域的绝对值。

---

## 9. （可选）投影域方案（对齐你的 ProjCT 做法）

你 AICTVer2 的 ProjCT 是**按视角**处理的：每个视角是一个 `128×512` 的探测器平面（1 通道），归一化 `clip [0, 0.28] → [0,1]`，并 `crop_rows=16` 裁掉上下黑边。移植到 Restormer 的要点：

### 9.1 数据索引（`prepare_ct_proj_data.py`）

每行 `<输入proj_raw> <目标proj_raw> <视角号0..719>`，视角取全 720 个（首尾视角如确认有运动伪影，可改成 `1..719`）：

```python
## 生成 train_proj_img.txt / valid_proj_img.txt（投影域，视角 0..719）
import os

DATA_ROOT = "/root/autodl-tmp/联影双能相位数据2080"
# TRAIN / VALID 与第 4.3 节完全一致
TRAIN = [ ...同第 4.3 节... ]
VALID = [ ...同第 4.3 节... ]
INP = "proj_pbi_fs_dec_blur_phase_100000.raw"
GT = "proj_no_pbi.raw"

def write_list(fname, patients):
    with open(fname, "w", encoding="utf-8") as f:
        for p in patients:
            for v in range(720):
                f.write(f"{os.path.join(DATA_ROOT, p, INP)} "
                        f"{os.path.join(DATA_ROOT, p, GT)} {v}\n")

write_list("train_proj_img.txt", TRAIN)
write_list("valid_proj_img.txt", VALID)
print("done: train", 720 * len(TRAIN), "valid", 720 * len(VALID))
```

### 9.2 数据集（`basicsr/data/ct_proj_dataset.py`）

直接移植你 AICTVer2 `proj_reader.py` 的高效读取（**按字节偏移只读单个视角**，不必 memmap 整个体积）：

```python
## 投影域数据集：每行 <输入raw> <目标raw> <视角号>，读取单个视角 (128,512)
import locale
import numpy as np
from torch.utils import data as data
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding

class Dataset_CTProj(data.Dataset):
    def __init__(self, opt):
        super(Dataset_CTProj, self).__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.scale = opt['scale']
        txt = opt['txt']
        try:
            with open(txt, encoding='utf-8') as f:
                self.paired = f.readlines()
        except UnicodeDecodeError:
            with open(txt, encoding=locale.getpreferredencoding()) as f:
                self.paired = f.readlines()
        self.norm_clip_max = float(opt.get('norm_clip_max', 0.28))
        self.crop_rows = int(opt.get('crop_rows', 16))

    def _read_view(self, path, v):
        view_size = 128 * 512
        off = v * view_size * 4
        return np.fromfile(path, dtype=np.float32,
                           count=view_size, offset=off).reshape(128, 512)

    def __getitem__(self, index):
        line = self.paired[index].strip()
        inp_path, gt_path, v = line.split()
        v = int(v)
        img_lq = self._read_view(inp_path, v)
        img_gt = self._read_view(gt_path, v)
        if self.crop_rows > 0:                      # 裁上下黑边
            img_lq = img_lq[self.crop_rows:-self.crop_rows, :]
            img_gt = img_gt[self.crop_rows:-self.crop_rows, :]
        img_lq = np.clip(img_lq, 0, self.norm_clip_max).astype(np.float32) / self.norm_clip_max
        img_gt = np.clip(img_gt, 0, self.norm_clip_max).astype(np.float32) / self.norm_clip_max
        img_lq = np.expand_dims(img_lq, axis=2)     # (H,512,1)
        img_gt = np.expand_dims(img_gt, axis=2)
        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_lq, img_gt = padding(img_lq, img_gt, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                self.scale, None)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        return {'lq': img_lq, 'gt': img_gt,
                'lq_path': inp_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paired)
```

### 9.3 配置

复制第 6 节的 yml，改三处即可：`name: CT_ProjDomain_Restormer`、两个 `txt` 指向 `train_proj_img.txt`/`valid_proj_img.txt`、`datasets.train.type: Dataset_CTProj`（val 同理），并在 train/val 各加 `norm_clip_max: 0.28`、`crop_rows: 16`。网络结构 `inp_channels: 1` 不变。

### 9.4 评估/交付提醒

- 投影域输出是「校正后的视角」，**要看效果必须重建**。Restormer 仓库内没有 FBP；你 AICTVer2 里若有重建/评估环节，把校正后的 720 个视角写回 `(720,128,512)` 体积后接你的重建流程。
- 也可顺带用你的 `DenseUNet_ProjCT` 基线在**同一批视角**上做对比。

---

## 10. 常见问题与注意事项

1. **首尾切片无信号**：图像域索引文件取 `z=1..158`（第 4.3 节已处理）；投影域首尾视角（0/719）可能有运动伪影，如确认可改成 `1..719`。
2. **别用 8-bit PNG 存切片当训练数据**：会把 HU 精度量化到 ~4HU/级。要么直接读 .raw（推荐，第 5 节 memmap 方案），要么存 16-bit/float `.npy`。
3. **数据量小（13 病人）容易过拟合**：建议 `mixup: true`、开几何增强、监控 val PSNR 是否随训练掉头；可加 5 折交叉验证（每次留不同病人做 val）看稳定性。
4. **划分已考虑域差异**：你的 `patients.yaml` 把唯一的平扫病人 18537 放进了训练集，验证集全为增强（CE1），训练/验证域分布接近，这点比 `DATASET_FORMAT.md` 的原始建议更稳。
5. **PSNR 口径**：默认 `use_image: false` 在归一化张量上算，与 AICTVer2 `compute_psnr(max_val=1.0)` 一致；不要和 HU 域的论文 PSNR 直接比。
6. **旧索引文件作废**：AICTVer2 目录里的 `train_img.txt`/`valid_img.txt` 是旧划分 + Windows 路径，务必在服务器上用第 4.3 节脚本重新生成（生成后用 `head` 确认路径是 `/root/autodl-tmp/联影双能相位数据2080/...`）。
7. **PyTorch 2.6+**：`torch.load` 默认 `weights_only=True`，若断点续训/加载报 `weights_only` 相关错误，装 `torch<2.6`，或在 `basicsr/models/base_model.py` 的 `torch.load` 加 `weights_only=False`。
8. **多卡**：`dist_params.backend: nccl`，确保 `--nproc_per_node` 与本机 GPU 数一致；日志里 `LOCAL_RANK` 相关的验证只在 rank 0 执行，属正常。
9. **内存**：图像域数据集用 `np.memmap`（不显式占 RAM，比 AICTVer2 的整卷 `fromfile` 快且省）；投影域用单视角 seek 读取（同 `proj_reader.py`）。
10. **progressive training**：`iters` 各阶段之和必须等于 `train.total_iter`，`mini_batch_sizes` 与 `gt_sizes` 长度一致；改 batch 时两者同步改。
11. **对比基线时统一测试集**：跟 `DenseUNet_PhaseCT` 对比时，用同一验证集（72278、91963×2）同一归一化窗口跑完 AICTVer2 的 `test.py` 再比 PSNR。

---

## 附：文件改动清单

| 操作 | 路径 | 内容 |
|------|------|------|
| 新增 | `prepare_ct_data.py` | 第 4.3 节，生成 `train_img.txt` / `valid_img.txt` |
| 新增 | `basicsr/data/ct_image_dataset.py` | 第 5 节代码 |
| 新增 | `Options/CT_ImageDomain_Restormer.yml` | 第 6 节代码 |
| 新增（可选） | `prepare_ct_proj_data.py`、`basicsr/data/ct_proj_dataset.py` | 第 9 节投影域 |
| 新增（可选） | `test_ct.py` | 第 8 节代码 |
| 环境 | conda + pip | 第 2 节命令 |
