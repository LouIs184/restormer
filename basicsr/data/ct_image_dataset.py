## 双能相位 CT 数据集：通过 yml 的 domain 一键切换 图像域/投影域
## domain = 'image' -> 读 rec_*.raw 切片 (512,512)，HU 归一化
## domain = 'proj'  -> 读 proj_*.raw 视角 (128,512)，clip 归一化（同 AICT proj_reader）
## 索引文件每行: <输入raw> <目标raw> <索引>（image=切片号z，proj=视角号v）
import locale
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding


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

        # 图像域参数
        self.hu_min = float(opt.get('hu_min', -1000.0))
        self.hu_range = float(opt.get('hu_max', 1000.0)) - self.hu_min
        # 投影域参数（同 AICT-code proj_reader）
        self.proj_norm_clip_max = float(opt.get('proj_norm_clip_max', 0.28))
        self.proj_crop_rows = int(opt.get('proj_crop_rows', 0))

        self.geometric_augs = opt.get('geometric_augs', True)
        self._vol_cache = {}

    def _volume(self, path):
        if path not in self._vol_cache:
            shape = (160, 512, 512) if self.domain == 'image' else (720, 128, 512)
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
            img_lq, img_gt = inp_vol[idx], gt_vol[idx]        # (128,512) 视角
            if self.proj_crop_rows > 0:                        # 裁上下黑边
                img_lq = img_lq[self.proj_crop_rows:-self.proj_crop_rows, :]
                img_gt = img_gt[self.proj_crop_rows:-self.proj_crop_rows, :]

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
