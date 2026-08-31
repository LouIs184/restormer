# Restormer 训练 · 执行文档（图像域 + 投影域）

> 命令优先，照抄即可。配套代码都已在本仓库就位：
>
> - **图像域**：`prepare_ct_data.py`、`Options/CT_ImageDomain_Restormer.yml`、`test_ct.py`
> - **投影域**：`prepare_ct_proj_data.py`、`Options/CT_ProjectionDomain_Restormer.yml`、`test_ct_proj.py`
> - **统一数据集**：`basicsr/data/ct_image_dataset.py`（`Dataset_CT`，靠 yml 的 `domain` 一键切换两个域）
>
> 背景与原理见 [1-RUN_GUIDE.md](1-RUN_GUIDE.md)，投影域方案见 [5-PROJECTION_DOMAIN.md](5-PROJECTION_DOMAIN.md)。
>
> 前提：数据在服务器 `/root/autodl-tmp/联影双能相位数据2080`（13 个病人），有 ≥1 张 NVIDIA GPU。

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

# ② 生成数据索引
python prepare_ct_data.py          # 图像域（train_img.txt / valid_img.txt）
python prepare_ct_proj_data.py     # 投影域（train_proj_img.txt / valid_proj_img.txt）

# ③ 开训（单卡）—— 选一个 yml 即可
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml --launcher none
```

**两个 yml 只差 `datasets` 段**：`type` 都是 `Dataset_CT`，靠 `domain: image / domain: proj` 区分，网络、损失、优化器、调度器完全一致。下面每步展开说明。

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

在 **Restormer 仓库根目录**执行。两个脚本共用同一份病人划分（train 10 次 / valid 3 次），只是样本粒度不同：

| 脚本 | 生成文件 | 样本数 | 索引第三列含义 |
|------|---------|--------|---------------|
| `prepare_ct_data.py` | `train_img.txt` / `valid_img.txt` | 1580 / 474 | 切片号 z（1..158） |
| `prepare_ct_proj_data.py` | `train_proj_img.txt` / `valid_proj_img.txt` | 7200 / 2160 | 视角号 v（0..719） |

```bash
python prepare_ct_data.py
python prepare_ct_proj_data.py
# 默认数据根目录: /root/autodl-tmp/联影双能相位数据2080
# 指定路径: python prepare_ct_proj_data.py --data-root /你的/数据目录 --output ./
```

确认生成成功：

```bash
wc -l train_img.txt valid_img.txt train_proj_img.txt valid_proj_img.txt
# 应为 1580  474  7200  2160
head -2 train_proj_img.txt   # 路径应指向 /root/autodl-tmp/联影双能相位数据2080/.../proj_*.raw ... 0
```

---

## 3. 训练前快速校验（可选但推荐，30 秒）

确认两个域的数据集都能读取、形状正确：

```bash
python - <<'EOF'
import yaml
from basicsr.data import create_dataset
for yml in ['Options/CT_ImageDomain_Restormer.yml',
            'Options/CT_ProjectionDomain_Restormer.yml']:
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
```

**正常启动的迹象**（前几秒日志里应看到）：
- 打印网络结构、`Dataset Dataset_CT - TrainSet is created.`（图像域 1580 条 / 投影域 7200 条）
- 迭代号 + `# l_pix: 0.x` 的 loss 稳步下降
- 每 1000 迭代出现 `Validation ValSet, # psnr: xx.xx`

### 显存不够？

| 域 | 做法 |
|----|------|
| 图像域 | `batch_size_per_gpu` 和 `mini_batch_sizes` 一起减半，或 `gt_size/gt_sizes` 从 256 降到 192（对应 `[128,160,192]`、`iters` 改 3 段、`total_iter` 同步改） |
| 投影域 | **`gt_sizes` 最大值不能超过 96**（裁剪后高度，见 5 文档 §4.1）。显存不够就把 `gt_sizes` 往下压（如 `[48,64,80]`），`mini_batch_sizes` 对应减，`total_iter` 保持 `sum(iters)` |

---

## 5. 监控

```bash
tail -f train_ct_proj.log                        # 实时日志
tensorboard --logdir tb_logger --port 6006       # 浏览器开 http://服务器IP:6006
nvidia-smi                                       # 显存/占用
```

产物位置（实验目录按 yml 的 `name` 区分，两个域互不覆盖）：

```
experiments/CT_ImageDomain_Restormer/
experiments/CT_ProjectionDomain_Restormer/
├── models/            # 权重 net_g_latest.pth + 每5000步快照 net_g_<iter>.pth
├── training_states/   # 断点 *.state（自动续训用）
└── ...日志
```

---

## 6. 断点续训（自动）

**训练中断后，直接用同一条命令重跑即可**，脚本会自动从对应实验目录的 `training_states/` 里最新的 `.state` 恢复：

```bash
python basicsr/train.py -opt Options/CT_ProjectionDomain_Restormer.yml --launcher none
```

日志出现 `Resuming training from epoch: ..., iter: ...` 即恢复成功。

---

## 7. 推理 / 评估

### 图像域

```bash
python test_ct.py \
    --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

输出到 `results/ct/`：`*_rec_corrected.raw` / `*_out.npy` / `*_z80_*.png` + 终端 PSNR/SSIM。

### 投影域

```bash
python test_ct_proj.py \
    --weights experiments/CT_ProjectionDomain_Restormer/models/net_g_latest.pth \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

输出到 `results/ct_proj/`：

| 文件 | 内容 |
|------|------|
| `*_proj_corrected.raw` | 校正后 sinogram 体积（float32，`(720,96,512)`，可直接接你的重建流程） |
| `*_proj_out.npy` | 同上，numpy 格式 |
| `*_v360_input/output/gt.png` | 第 360 视角可视化对比 |
| `*_fbp_row64.png` / `*_fbp_row64_gt.png` | FBP 重建（pred vs GT，平行束近似） |
| 终端打印 | sinogram 域 PSNR/SSIM + FBP 重建域 PSNR（定性） |

> 投影域 sinogram 上的 PSNR 只用于训练监控；**最终验收看 FBP 到图像域后的质量**。`skimage` 的 `iradon` 是平行束近似（数据是扇/锥束），精确几何请用 AICT-code 的重建代码。
> 对比 AICTVer2 基线时，用同一批验证病人、同一 `clip(x,0,0.28)/0.28` 归一化跑它那边的评估再比。

---

## 8. 常见问题对照

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: numpy` 等 | 先 `conda activate restormer` 再跑 |
| `basicsr` 报 numpy 2.x 兼容错 | `pip install "numpy<2"` |
| 权重/续训报 `weights_only` | `pip install "torch<2.6"`，或在对应 `torch.load` 加 `weights_only=False`（`test_ct_proj.py` 已加） |
| `Dataset ... is not found` | `ct_image_dataset.py` 文件名必须以 `_dataset.py` 结尾且在 `basicsr/data/` 下 |
| 投影域训练样本大部分是人造反射边 | **`gt_sizes` 最大值超了裁剪后高度 96**：改回 `[48,64,80,96]` |
| 投影域开了 `geometric_augs: true` 效果差 | 90° 旋转会交换视角轴/通道轴，物理无意义，改 `false` |
| `xxx is not in ... paths` / 样本数不对 | 重跑对应的 `prepare_ct_*.py`，检查 `txt` 路径是否是服务器路径 |
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
| `test_ct_proj.py` | ✅ 已创建 |
| `train_img.txt` / `valid_img.txt` | ⏳ 第 2 步生成 |
| `train_proj_img.txt` / `valid_proj_img.txt` | ⏳ 第 2 步生成 |
