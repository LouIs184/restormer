## 图像域推理：Restormer 权重 -> 逐切片去伪影 -> 保存 {病人}_pred/input/gt.raw + 指标
## 产出只有 .raw 和终端指标（同 AICT-code infer_phase 的做法，不存图）。
## 输入/目标剔除首尾切片（z=1..158，与训练一致），pred/input/gt 三卷同形状 (158,512,512)。
## 用法:
##   python test_ct.py \
##       --weights experiments/CT_ImageDomain_Restormer/models/net_g_latest.pth \
##       --data_root /root/autodl-tmp/联影双能相位数据2080 \
##       --patient 72278_406010_960+_AXIAL_CE1_F071Y_20211216_Thick1_Incre1
import argparse
import os
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.models.archs.restormer_arch import Restormer

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='训练得到的 net_g_latest.pth')
parser.add_argument('--data_root', default='/root/autodl-tmp/联影双能相位数据2080',
                    help='数据根目录（含 13 个病人文件夹）；本地推理务必改成本地路径')
parser.add_argument('--patient', nargs='+', required=True, help='要推理的病人文件夹')
parser.add_argument('--result_dir', default='./results/ct',
                    help='输出目录（写 <病人>_pred/input/gt.raw）')
parser.add_argument('--config', default='Options/CT_ImageDomain_Restormer.yml',
                    help='训练 yml（取网络结构 + hu_min / hu_max）')
parser.add_argument('--hu_min', type=float, default=None,
                    help='覆盖 yml 里的 hu_min（默认读 yml，保持与训练一致）')
parser.add_argument('--hu_max', type=float, default=None,
                    help='覆盖 yml 里的 hu_max（默认读 yml，保持与训练一致）')
args = parser.parse_args()
os.makedirs(args.result_dir, exist_ok=True)

# ---- 网络结构 + HU 归一化窗口从训练 yml 读取 ----
cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
net_cfg = dict(cfg['network_g'])
net_cfg.pop('type')
data_cfg = cfg['datasets'].get('val', cfg['datasets'].get('train', {}))
hu_min = args.hu_min if args.hu_min is not None else float(data_cfg.get('hu_min', -1000.0))
hu_max = args.hu_max if args.hu_max is not None else float(data_cfg.get('hu_max', 1000.0))
hu_range = hu_max - hu_min

model = Restormer(**net_cfg)
try:  # PyTorch 2.6+ 默认 weights_only=True，会拒收含非张量的 ckpt
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
except TypeError:  # torch < 2.6 无 weights_only 参数
    ckpt = torch.load(args.weights, map_location='cpu')
model.load_state_dict(ckpt['params'])
model = model.cuda().eval()
print('loaded', args.weights)

INP = 'rec_pbi_fs_dec_blur_phase_100000.raw'   # 输入（含伪影）
GT = 'rec_no_pbi.raw'                          # 目标（干净）


def norm(x):
    return np.clip((x - hu_min) / hu_range, 0, 1).astype(np.float32)


def denorm(x):
    return x * hu_range + hu_min


def compute_metrics(pred, gt, tag=''):
    """同 AICT-code infer.py：在原始值域（HU）上算 MSE/RMSE/MAE/PSNR/SSIM。"""
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


for patient in args.patient:
    inp = np.fromfile(os.path.join(args.data_root, patient, INP),
                      dtype=np.float32).reshape(160, 512, 512)
    gt = np.fromfile(os.path.join(args.data_root, patient, GT),
                     dtype=np.float32).reshape(160, 512, 512)

    # 剔除首尾切片（z=1..158，与训练一致），pred/input/gt 同形状 (158, 512, 512)
    input_slices = inp[1:-1]
    gt_slices = gt[1:-1]

    pred = np.zeros_like(input_slices)
    with torch.no_grad():
        for i in range(158):                # 逐切片推理
            x = torch.from_numpy(norm(input_slices[i])).unsqueeze(0).unsqueeze(0).cuda()
            pred[i] = denorm(model(x)[0, 0].cpu().numpy())

    base = os.path.join(args.result_dir, patient)
    pred.astype(np.float32).tofile(base + '_pred.raw')
    input_slices.astype(np.float32).tofile(base + '_input.raw')
    gt_slices.astype(np.float32).tofile(base + '_gt.raw')
    print(f'Saved {patient}_pred/input/gt.raw  shape={pred.shape}')

    print(f'  -- Metrics for {patient} --')
    compute_metrics(input_slices, gt_slices, tag='input vs gt :')
    compute_metrics(pred, gt_slices, tag='pred  vs gt :')
