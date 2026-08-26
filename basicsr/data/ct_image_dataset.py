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
