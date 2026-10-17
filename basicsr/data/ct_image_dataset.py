## CT 数据集：通过 yml 的 domain 一键切换 图像域/投影域
## domain = 'image' -> 读 rec_*.raw 切片 (512,512)，HU 归一化
## domain = 'proj'  -> 读 proj_*.raw 视角平面 (H,W)，clip 归一化（同 AICT proj_reader）
## 索引文件每行: <输入raw> <目标raw> <索引>（image=切片号z，proj=视角号v）
##
## 体积形状 / 裁剪量 / 归一化尺度全部由 yml 提供，换数据集无需改代码：
##   image_shape       [160, 512, 512]      图像域体积形状
##   proj_shape        [720, 128, 512]      投影域平面形状 (n_view, H, W)
##   proj_norm_clip_max  0.28               投影域归一化上界（必填，见下）
##   proj_crop_rows/cols 0                  投影域上下/左右裁剪量
import locale
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import get_root_logger, img2tensor, padding


class Dataset_CT(data.Dataset):
    def __init__(self, opt):
        super(Dataset_CT, self).__init__()
        self.opt = opt
        self.phase = opt['phase']          # 'train' / 'val'，框架自动注入
        self.scale = opt['scale']          # 固定为 1
        txt = opt['txt']
        try:
            with open(txt, encoding='utf-8') as f:
                self.paired = f.readlines()
        except UnicodeDecodeError:
            with open(txt, encoding=locale.getpreferredencoding()) as f:
                self.paired = f.readlines()

        # ---- 域开关（yml 里切换）----
        self.domain = opt.get('domain', 'image')      # 'image' | 'proj'

        # ---- 验证集抽样（yml: val_sample_ratio，只对 val 生效）----
        # 验证集样本数远多于训练时，每次验证的全尺寸前向是主要耗时来源。
        # 用固定步长抽样（非随机）：保证每次验证用同一批样本，val PSNR 才可比。
        # 1.0 = 全用（默认，不写此字段时行为与改动前完全一致）。
        if self.phase == 'val':
            ratio = float(opt.get('val_sample_ratio', 1.0))
            if ratio < 1.0:
                stride = max(1, int(round(1.0 / max(ratio, 1e-9))))
                n_all = len(self.paired)
                self.paired = self.paired[::stride]
                get_root_logger().info(
                    f'[Dataset_CT] val_sample_ratio={ratio} → 每 {stride} 个取 1，'
                    f'验证样本 {n_all} → {len(self.paired)}')

        # 图像域参数
        self.hu_min = float(opt.get('hu_min', -1000.0))
        self.hu_range = float(opt.get('hu_max', 1000.0)) - self.hu_min
        # 投影域参数（同 AICT-code proj_reader）
        # clip_max 决定归一化尺度，必须按当前数据集实测标定。
        # 投影域下留空(None)/缺失直接报错，避免静默套用别的数据集的旧尺度；
        # 图像域不需要它（用 hu_min/hu_max），故不做校验。
        clip_max = opt.get('proj_norm_clip_max', None)
        if self.domain == 'proj' and clip_max is None:
            raise ValueError(
                "投影域缺少 proj_norm_clip_max（或留空为 ~）。\n"
                "该值决定归一化尺度，必须按当前数据集实测标定，不能沿用旧值。\n"
                "请先运行：\n"
                "    python analyze_proj_range.py --data-root <数据根目录>\n"
                "把输出中的『建议 proj_norm_clip_max』填进 yml 后再训练/推理。")
        self.proj_norm_clip_max = float(clip_max) if clip_max is not None else None
        self.proj_crop_rows = int(opt.get('proj_crop_rows', 0))
        self.proj_crop_cols = int(opt.get('proj_crop_cols', 0))

        # 体积形状（缺省值 = 旧数据集，保证旧 yml 不加字段行为不变）
        self.image_shape = tuple(opt.get('image_shape', (160, 512, 512)))
        self.proj_shape = tuple(opt.get('proj_shape', (720, 128, 512)))

        self.geometric_augs = opt.get('geometric_augs', True)
        self._vol_cache = {}

    def _volume(self, path):
        if path not in self._vol_cache:
            shape = self.image_shape if self.domain == 'image' else self.proj_shape
            self._vol_cache[path] = np.memmap(
                path, dtype=np.float32, mode='r', shape=shape)
        return self._vol_cache[path]

    def _norm(self, x):
        if self.domain == 'image':
            # 图像域：HU 线性归一化 (x+1000)/2000
            return np.clip((x - self.hu_min) / self.hu_range, 0, 1).astype(np.float32)
        else:
            # 投影域：AICT proj_reader 的 clip 归一化
            return np.clip(x, 0, self.proj_norm_clip_max) / self.proj_norm_clip_max

    def __getitem__(self, index):
        line = self.paired[index].strip()
        inp_path, gt_path, idx = line.split()
        idx = int(idx)

        inp_vol = self._volume(inp_path)
        gt_vol = self._volume(gt_path)

        if self.domain == 'image':
            img_lq, img_gt = inp_vol[idx], gt_vol[idx]        # (512,512) 切片
        else:
            img_lq, img_gt = inp_vol[idx], gt_vol[idx]        # (H,W) 视角平面
            r, c = self.proj_crop_rows, self.proj_crop_cols
            if r or c:                                         # 裁上下/左右黑边
                # 用显式上下界而非 [r:-r]：后者在 r=0 时会得到空数组
                h, w = img_lq.shape
                img_lq = img_lq[r:h - r, c:w - c]
                img_gt = img_gt[r:h - r, c:w - c]

        img_lq = np.expand_dims(self._norm(img_lq), axis=2)   # (H,W,1)
        img_gt = np.expand_dims(self._norm(img_gt), axis=2)

        if self.phase == 'train':
            gt_size = self.opt['gt_size']
            img_lq, img_gt = padding(img_lq, img_gt, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                self.scale, None)
            if self.geometric_augs and self.domain == 'image':
                # 投影域禁 90° 旋转（交换视角轴/通道轴，物理无意义）
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': inp_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paired)


# 兼容别名：旧的 type: Dataset_CTImage 仍可用（等价 domain: image）
Dataset_CTImage = Dataset_CT
