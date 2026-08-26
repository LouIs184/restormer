# Restormer 图像域训练 · 执行文档

> 命令优先，照抄即可。配套代码都已在本仓库就位：
> `prepare_ct_data.py`、`basicsr/data/ct_image_dataset.py`、`Options/CT_ImageDomain_Restormer.yml`、`test_ct.py`。
> 背景与原理见 `RUN_GUIDE.md`。
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
python prepare_ct_data.py

# ③ 开训（单卡）
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
```

下面每步展开说明。

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
```

> 若 `import basicsr` 报 numpy 相关错：`pip install "numpy<2"`。
> 若 PyTorch ≥2.6 加载权重报 `weights_only` 错：装 `torch<2.6`。

---

## 2. 生成数据索引（每次换数据/划分后重新跑）

在 **Restormer 仓库根目录**执行（生成 `train_img.txt` / `valid_img.txt`，共 2054 行，路径写绝对路径）：

```bash
python prepare_ct_data.py
# 默认数据根目录: /root/autodl-tmp/联影双能相位数据2080
# 指定路径: python prepare_ct_data.py --data-root /你的/数据目录 --output ./
```

确认生成成功：

```bash
wc -l train_img.txt valid_img.txt      # 应为 1580 / 474
head -2 train_img.txt                   # 路径应指向 /root/autodl-tmp/联影双能相位数据2080/...
```

---

## 3. 训练前快速校验（可选但推荐，30 秒）

确认数据集能正常读取、形状正确：

```bash
python - <<'EOF'
import yaml
from basicsr.data import create_dataset
opt = yaml.safe_load(open('Options/CT_ImageDomain_Restormer.yml'))
tr = opt['datasets']['train']; tr['phase'] = 'train'; tr['scale'] = opt['scale']
ds = create_dataset(tr)
x = ds[0]
print('train samples:', len(ds))           # 1580
print('lq:', x['lq'].shape, x['lq'].dtype) # torch.Size([1, patch, patch]) float32
va = opt['datasets']['val'];  va['phase'] = 'val';  va['scale'] = opt['scale']
print('val  samples:', len(create_dataset(va)))    # 474
EOF
```

> 注意：yml 里 `txt: ./train_img.txt` 是相对路径，**必须从仓库根目录启动训练**，否则请改成绝对路径。

---

## 4. 开始训练

### 单卡（最常用）

```bash
conda activate restormer
cd /path/to/Restormer
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
```

### 多卡（例如 4 卡）

先把 yml 的 `num_gpu: 1` 改成 `4`，然后：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
    basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher pytorch
```

### 后台跑 + 留存日志

```bash
nohup python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml \
    --launcher none > train_ct.log 2>&1 &
echo $!   # 记下 PID，方便 kill
```

**正常启动的迹象**（前几秒日志里应看到）：
- 打印网络结构、`Dataset Dataset_CTImage - TrainSet is created.`（train 1580 条）
- 迭代号 + `# l_pix: 0.x` 的 loss 稳步下降
- 每 1000 迭代出现 `Validation ValSet, # psnr: xx.xx`

### 显存不够？

把 yml 里 `batch_size_per_gpu` 和 `mini_batch_sizes` 一起减半，或把 `gt_size/gt_sizes` 从 256 降到 192（对应 `[128,160,192]`、`iters` 改成 3 段、`total_iter` 同步改）。

---

## 5. 监控

```bash
tail -f train_ct.log                                # 实时日志
tensorboard --logdir tb_logger --port 6006          # 浏览器开 http://服务器IP:6006
nvidia-smi                                          # 显存/占用
```

产物位置：

```
experiments/CT_ImageDomain_Restormer/
├── models/            # 权重 net_g_latest.pth + 每5000步快照 net_g_<iter>.pth
├── training_states/   # 断点 *.state（自动续训用）
└── ...日志
```

---

## 6. 断点续训（自动）

**训练中断后，直接用同一条命令重跑即可**，脚本会自动从 `experiments/CT_ImageDomain_Restormer/training_states/` 里最新的 `.state` 恢复：

```bash
python basicsr/train.py -opt Options/CT_ImageDomain_Restormer.yml --launcher none
```

日志出现 `Resuming training from epoch: ..., iter: ...` 即恢复成功。

---

## 7. 推理 / 评估

训练到足够迭代后（或直接用 `net_g_latest.pth`）：

```bash
python test_ct.py \
    --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
    --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
              91963_301643_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1 \
              91963_53624_960+_AXIAL_CE1_M070Y_20211216_Thick1_Incre1
```

默认推理验证集 3 个病人，输出到 `results/ct/`：

| 文件 | 内容 |
|------|------|
| `*_rec_corrected.raw` | 校正后体积（float32 HU，`(160,512,512)`，可直接给重建/对比用） |
| `*_out.npy` | 同上，numpy 格式 |
| `*_z80_input/output/gt.png` | 第 80 层切片可视化对比 |
| 终端打印 | 全病人平均 PSNR / SSIM（归一化域） |

> 对比 AICTVer2 的 `DenseUNet_PhaseCT` 时，用同一批验证病人、同一 `[-1000,1000]` 归一化跑它那边的 `test.py` 再比。

---

## 8. 常见问题对照

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: numpy` 等 | 先 `conda activate restormer` 再跑 |
| `basicsr` 报 numpy 2.x 兼容错 | `pip install "numpy<2"` |
| 权重/续训报 `weights_only` | `pip install "torch<2.6"` |
| `Dataset ... is not found` | `ct_image_dataset.py` 文件名必须以 `_dataset.py` 结尾且在 `basicsr/data/` 下 |
| `xxx is not in ... paths` / 样本数不对 | 重跑 `prepare_ct_data.py`，检查 `train_img.txt` 路径是否是服务器路径 |
| 启动即 `CUDA out of memory` | 见第 4 节「显存不够」 |
| 训练几轮 val PSNR 不升反降 | 过拟合：开 `mixup: true`（yml 已默认开）、减小 `total_iter`、早停看快照 |
| 多卡只在 rank0 打日志/验证 | 正常现象 |

---

## 附：本次执行需要的文件清单

| 文件 | 状态 |
|------|------|
| `prepare_ct_data.py` | ✅ 已创建 |
| `basicsr/data/ct_image_dataset.py` | ✅ 已创建 |
| `Options/CT_ImageDomain_Restormer.yml` | ✅ 已创建（改 `num_gpu`、`txt` 路径） |
| `test_ct.py` | ✅ 已创建 |
| `train_img.txt` / `valid_img.txt` | ⏳ 第 2 步生成 |
