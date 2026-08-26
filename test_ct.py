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
cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
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
