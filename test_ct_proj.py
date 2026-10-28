## 投影域推理：Restormer 权重 -> 逐视角去伪影 -> 保存 {病人}_pred/input/gt.raw + 指标
## 产出只有 .raw 和终端指标（同 AICT-code infer_proj 的做法，不存图）。
## 形状/裁剪/归一化尺度全部从训练 yml 读取，脚本内不出现数据集相关的魔数。
## 输入/目标的裁剪与训练 yml 的 proj_crop_rows/cols 一致，pred/input/gt 三卷同形状。
##
## 用法（推荐：直接吃 prepare 脚本生成的测试集索引，无需关心文件名）:
##   python test_ct_proj.py \
##       --weights experiments/CT_ProjectionDomain_Ximenzi_Restormer/models/net_g_latest.pth \
##       --txt ./test_ximenzi_proj.txt
##
## 用法（旧的按病人名指定，用 --inp_name/--gt_name 指定文件名）:
##   python test_ct_proj.py \
##       --weights experiments/CT_ProjectionDomain_Restormer/models/net_g_latest.pth \
##       --data_root /root/autodl-tmp/联影双能相位数据2080 \
##       --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1 \
##       --inp_name proj_pbi_fs_dec_blur_phase_100000.raw
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.models.archs.restormer_arch import Restormer

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='训练得到的 net_g_latest.pth')
parser.add_argument('--txt', default=None,
                    help='索引 txt（如 prepare_ximenzi_proj_data.py 生成的 '
                         'test_ximenzi_proj.txt）；与 --patient 二选一，推荐')
parser.add_argument('--patient', nargs='+', default=None,
                    help='要推理的病人文件夹（需配合 --data_root / --inp_name）')
parser.add_argument('--data_root', default='/root/autodl-tmp/联影双能相位数据2080',
                    help='--patient 模式下的数据根目录；--txt 模式下不使用（路径已在 txt 里）')
parser.add_argument('--inp_name', default='proj_pbi_fs_dec_blur_phase_100000.raw',
                    help='--patient 模式下的输入文件名')
parser.add_argument('--gt_name', default='proj_no_pbi.raw',
                    help='--patient 模式下的目标文件名')
parser.add_argument('--result_dir', default='./results/ct_proj',
                    help='输出目录（写 <病人>_pred/input/gt.raw）')
parser.add_argument('--config', default='Options/CT_ProjectionDomain_Restormer.yml',
                    help='训练 yml（取网络结构 + proj_shape / 裁剪量 / 归一化尺度）')
parser.add_argument('--crop_rows', type=int, default=None,
                    help='覆盖 yml 里的 proj_crop_rows（默认读 yml，保持与训练一致）')
parser.add_argument('--crop_cols', type=int, default=None,
                    help='覆盖 yml 里的 proj_crop_cols（默认读 yml）')
parser.add_argument('--proj_clip_max', type=float, default=None,
                    help='覆盖 yml 里的 proj_norm_clip_max（默认读 yml，保持与训练一致）')
parser.add_argument('--proj_shape', default=None,
                    help='覆盖 yml 里的 proj_shape，格式 n_view,H,W（如 720,500,500）。'
                         '测试数据的视角数与训练不同时必须指定，否则会按训练的视角数解析')
parser.add_argument('--batch_size', type=int, default=4,
                    help='推理批大小。各视角形状相同，可批处理加速；'
                         'Restormer 只有 LayerNorm、无 BatchNorm，批处理与逐张结果等价。'
                         '显存不够时降到 1。默认 4')
parser.add_argument('--progress', type=int, default=100,
                    help='每处理多少个视角打印一次进度（0=不打印）。默认 100')
args = parser.parse_args()

# ---- 网络结构 + 投影域参数从训练 yml 读取 ----
cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
net_cfg = dict(cfg['network_g'])
net_cfg.pop('type')
data_cfg = cfg['datasets'].get('val', cfg['datasets'].get('train', {}))

shape_cfg = args.proj_shape.split(',') if args.proj_shape \
    else data_cfg.get('proj_shape', (720, 128, 512))
n_view, H0, W0 = (int(x) for x in shape_cfg)
crop_rows = args.crop_rows if args.crop_rows is not None else int(
    data_cfg.get('proj_crop_rows', 0))
crop_cols = args.crop_cols if args.crop_cols is not None else int(
    data_cfg.get('proj_crop_cols', 0))

clip_raw = args.proj_clip_max if args.proj_clip_max is not None else \
    data_cfg.get('proj_norm_clip_max', None)
if clip_raw is None:
    sys.exit('错误：yml 里的 proj_norm_clip_max 为空。\n'
             '      该值决定归一化尺度，必须与训练时用的一致。\n'
             '      先跑 analyze_proj_range.py 标定并填进 yml，'
             '或用 --proj_clip_max 显式指定。')
clip_max = float(clip_raw)

H, W = H0 - 2 * crop_rows, W0 - 2 * crop_cols
# Restormer 要求边长 8 整除（3 次 PixelUnshuffle）。这里与验证集 pad_test 保持一致地
# 补齐再推理、输出裁回；平面本身整除时 pad 为 0，等于不做任何事。
pad_h, pad_w = (-H) % 8, (-W) % 8
print(f'yml: {args.config}')
print(f'平面形状 {n_view}x{H0}x{W0}  裁剪 rows={crop_rows} cols={crop_cols} '
      f'→ {H}x{W}  clip_max={clip_max}')
if pad_h or pad_w:
    print(f'推理时 reflect 补齐 +{pad_h}/+{pad_w} → '
          f'{H + pad_h}x{W + pad_w}（输出裁回 {H}x{W}）')

# ---- 组装待推理清单: [(标签, 输入路径, 目标路径)] ----
if args.txt:
    pairs, seen = [], set()
    with open(args.txt, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            if (parts[0], parts[1]) in seen:
                continue
            seen.add((parts[0], parts[1]))
            pairs.append((parts[0], parts[1]))

    # 标签 = 输入文件所在目录相对于「所有输入目录的公共祖先」的路径：
    #   单层目录（ximenzi 的 01/）        -> "01"
    #   两层目录（仿真数据的 01/dose_1e4/）-> "01_dose_1e4"
    # 只取 basename(dirname(...)) 在两层级数据上会重名（都叫 dose_1e4），
    # 不同头的输出会互相覆盖。
    dirs = [os.path.dirname(p[0]) for p in pairs]
    try:
        prefix = os.path.commonpath(dirs) if dirs else ''
    except ValueError:          # 跨盘符等极端情况
        prefix = ''
    jobs = []
    for inp_path, gt_path in pairs:
        rel = os.path.relpath(os.path.dirname(inp_path), prefix) if prefix else ''
        label = rel.replace(os.sep, '_') if rel not in ('', '.') \
            else os.path.basename(os.path.dirname(inp_path))
        jobs.append((label, inp_path, gt_path))
elif args.patient:
    jobs = [(p,
             os.path.join(args.data_root, p, args.inp_name),
             os.path.join(args.data_root, p, args.gt_name)) for p in args.patient]
else:
    parser.error('必须提供 --txt 或 --patient 之一')

model = Restormer(**net_cfg)
try:  # PyTorch 2.6+ 默认 weights_only=True，会拒收含非张量的 ckpt
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
except TypeError:  # torch < 2.6 无 weights_only 参数
    ckpt = torch.load(args.weights, map_location='cpu')
model.load_state_dict(ckpt['params'])
model = model.cuda().eval()
print('loaded', args.weights)


def norm(x):
    return np.clip(x, 0, clip_max) / clip_max


def denorm(x):
    return x * clip_max


def compute_metrics(pred, gt, tag=''):
    """同 AICT-code infer.py：在原始值域上算 MSE/RMSE/MAE/PSNR/SSIM。"""
    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    dr = max(gt.max() - gt.min(), 1e-8)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=dr)
    ssim = np.mean([
        structural_similarity(gt[i], pred[i],
                              data_range=max(gt[i].max() - gt[i].min(), 1e-8))
        for i in range(gt.shape[0])
    ])
    print(f'  {tag}  MSE: {mse:.6f}  RMSE: {rmse:.6f}  MAE: {mae:.6f}  '
          f'PSNR: {psnr:.2f}  SSIM: {ssim:.4f}')
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}


for label, inp_path, gt_path in jobs:
    if not (os.path.exists(inp_path) and os.path.exists(gt_path)):
        print(f'跳过 {label}：文件缺失\n  {inp_path}\n  {gt_path}')
        continue

    inp = np.fromfile(inp_path, dtype=np.float32).reshape(n_view, H0, W0)
    gt = np.fromfile(gt_path, dtype=np.float32).reshape(n_view, H0, W0)

    # 裁剪（与训练一致），保证 pred/input/gt 同形状 (n_view, H, W)
    if crop_rows or crop_cols:
        input_cropped = inp[:, crop_rows:H0 - crop_rows, crop_cols:W0 - crop_cols]
        gt_cropped = gt[:, crop_rows:H0 - crop_rows, crop_cols:W0 - crop_cols]
    else:
        input_cropped, gt_cropped = inp, gt

    pred = np.zeros((n_view, H, W), dtype=np.float32)
    bs = max(1, int(args.batch_size))
    t0 = time.time()
    print(f'  {label}: {n_view} 视角, batch={bs}', flush=True)
    with torch.no_grad():
        for start in range(0, n_view, bs):
            end = min(start + bs, n_view)
            batch = np.stack([norm(input_cropped[v]) for v in range(start, end)])
            x = torch.from_numpy(batch).unsqueeze(1)     # (B,1,H,W)
            if pad_h or pad_w:               # 与验证集 pad_test 一致：补齐到 8 的倍数
                x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
            y = model(x.cuda())                          # (B,1,H',W')
            if pad_h or pad_w:
                y = y[:, :, :H, :W]                      # 输出裁回原尺寸
            pred[start:end] = denorm(y[:, 0].cpu().numpy())
            if args.progress > 0 and (end % args.progress == 0 or end == n_view):
                el = time.time() - t0
                eta = el / end * (n_view - end)
                print(f'    {end}/{n_view} 视角  已用 {el:.0f}s  剩余约 {eta:.0f}s',
                      flush=True)

    os.makedirs(args.result_dir, exist_ok=True)
    base = os.path.join(args.result_dir, label)
    pred.astype(np.float32).tofile(base + '_pred.raw')
    input_cropped.astype(np.float32).tofile(base + '_input.raw')
    gt_cropped.astype(np.float32).tofile(base + '_gt.raw')
    print(f'Saved {label}_pred/input/gt.raw  shape={pred.shape}')

    print(f'  -- Metrics for {label} --')
    compute_metrics(input_cropped, gt_cropped, tag='input vs gt :')
    compute_metrics(pred, gt_cropped, tag='pred  vs gt :')
