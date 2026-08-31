## 投影域推理：Restormer 权重 -> 逐视角去伪影 -> 保存 sinogram + FBP 图像域验证
## 用法:
##   python test_ct_proj.py \
##       --weights ./experiments/CT_ProjectionDomain_Restormer/models/net_g_latest.pth \
##       --patient 11034_307811_960+_AXIAL_CE1_M067Y_20211215_Thick1_Incre1
import argparse
import os
import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import iradon
from basicsr.models.archs.restormer_arch import Restormer

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='训练得到的 net_g_latest.pth')
parser.add_argument('--data_root', default='/root/autodl-tmp/联影双能相位数据2080')
parser.add_argument('--patient', nargs='+', required=True, help='要推理的病人文件夹')
parser.add_argument('--result_dir', default='./results/ct_proj')
parser.add_argument('--config', default='Options/CT_ProjectionDomain_Restormer.yml')
parser.add_argument('--proj_clip_max', type=float, default=0.28)
parser.add_argument('--crop_rows', type=int, default=16)
parser.add_argument('--view_save', type=int, default=360, help='存哪张视角的 PNG')
parser.add_argument('--fbp_row', type=int, default=64,
                    help='FBP 用的探测器行（原始行坐标，默认中间行）')
parser.add_argument('--fbp_views', type=int, default=360,
                    help='FBP 用多少视角（720 视角取前 360 覆盖 180°，平行束近似）')
args = parser.parse_args()
os.makedirs(args.result_dir, exist_ok=True)

# 从训练配置取网络结构
cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
net_cfg = dict(cfg['network_g'])
net_cfg.pop('type')
model = Restormer(**net_cfg)
try:  # PyTorch 2.6+ 默认 weights_only=True，会拒收含非张量的 ckpt
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
except TypeError:  # torch < 2.6 无 weights_only 参数
    ckpt = torch.load(args.weights, map_location='cpu')
model.load_state_dict(ckpt['params'])
model = model.cuda().eval()
print('loaded', args.weights)

INP = 'proj_pbi_fs_dec_blur_phase_100000.raw'
GT = 'proj_no_pbi.raw'


def norm(x):
    return np.clip(x, 0, args.proj_clip_max) / args.proj_clip_max


def denorm(x):
    return x * args.proj_clip_max


def rescale(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


for patient in args.patient:
    inp = np.fromfile(os.path.join(args.data_root, patient, INP),
                      dtype=np.float32).reshape(720, 128, 512)
    gt = np.fromfile(os.path.join(args.data_root, patient, GT),
                     dtype=np.float32).reshape(720, 128, 512)
    H = 128 - 2 * args.crop_rows
    pred = np.zeros((720, H, 512), dtype=np.float32)

    psnr_in, psnr_out, ssim_out = [], [], []
    with torch.no_grad():
        for v in range(720):            # 逐视角推理
            x = norm(inp[v, args.crop_rows:-args.crop_rows, :])
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda()
            out = model(x)[0, 0].cpu().numpy()          # (H,512) 归一化域
            pred[v] = denorm(out)                        # 转回投影域物理值
            g = norm(gt[v, args.crop_rows:-args.crop_rows, :])
            psnr_in.append(peak_signal_noise_ratio(
                norm(inp[v, args.crop_rows:-args.crop_rows, :]), g, data_range=1.0))
            psnr_out.append(peak_signal_noise_ratio(out, g, data_range=1.0))
            ssim_out.append(structural_similarity(out, g, data_range=1.0))

    # 保存校正后的 sinogram 体积（供你的重建/对比流程使用）
    pred.astype(np.float32).tofile(
        os.path.join(args.result_dir, f'{patient}_proj_corrected.raw'))
    np.save(os.path.join(args.result_dir, f'{patient}_proj_out.npy'), pred)

    # 单视角可视化（input / output / gt）
    v = args.view_save

    def to_png(h, path):
        cv2.imwrite(path, (norm(h) * 255).astype(np.uint8))

    to_png(inp[v, args.crop_rows:-args.crop_rows, :],
           os.path.join(args.result_dir, f'{patient}_v{v}_input.png'))
    to_png(pred[v], os.path.join(args.result_dir, f'{patient}_v{v}_output.png'))
    to_png(gt[v, args.crop_rows:-args.crop_rows, :],
           os.path.join(args.result_dir, f'{patient}_v{v}_gt.png'))

    print(f'{patient}: sinogram PSNR in={np.mean(psnr_in):.3f} dB -> '
          f'out={np.mean(psnr_out):.3f} dB, SSIM out={np.mean(ssim_out):.4f}')

    # FBP 图像域验证（平行束近似，定性）。同一 FBP 管线下比较
    # recon(pred) vs recon(gt)，避免 HU↔投影单位换算偏差。
    rc = args.fbp_row - args.crop_rows     # 转成裁剪后行坐标
    if 0 <= rc < H:
        theta = np.linspace(0, 180, args.fbp_views, endpoint=False)
        sino_pred = pred[:args.fbp_views, rc, :].T     # (512, 360)，每列一个视角
        sino_gt = gt[:args.fbp_views, rc, :].T
        recon = iradon(sino_pred, theta=theta)
        recon_gt = iradon(sino_gt, theta=theta)

        cv2.imwrite(os.path.join(args.result_dir, f'{patient}_fbp_row{args.fbp_row}.png'),
                    (rescale(recon) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.result_dir, f'{patient}_fbp_row{args.fbp_row}_gt.png'),
                    (rescale(recon_gt) * 255).astype(np.uint8))

        psnr_img = peak_signal_noise_ratio(rescale(recon), rescale(recon_gt),
                                           data_range=1.0)
        print(f'  FBP(平行束近似) 行{args.fbp_row}: 重建域 PSNR='
              f'{psnr_img:.3f} dB（对比 recon(pred) vs recon(gt)，见 *_fbp_row*.png）')
        print('  提示: 数据是扇/锥束，skimage iradon 仅定性；精确几何请用 AICT-code 的重建代码')
    else:
        print(f'  --fbp_row {args.fbp_row} 超出有效行范围，跳过 FBP')
