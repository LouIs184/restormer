#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图像域数据索引 train_img.txt / valid_img.txt（切片 z=1..158，剔除首尾）。
病人划分与 AICTVer2 的 patients.yaml 一致：train 10 次扫描 / valid 3 次扫描。

每行格式：<输入.raw绝对路径> <目标.raw绝对路径> <切片号>

用法:
    python prepare_ct_data.py
    python prepare_ct_data.py --data-root /root/autodl-tmp/联影双能相位数据2080 --output ./
"""

import argparse
import os

INP = "rec_pbi_fs_dec_blur_phase_100000.raw"
GT = "rec_no_pbi.raw"

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/root/autodl-tmp/联影双能相位数据2080",
                   help="图像域数据根目录（含 13 个病人子文件夹）")
    p.add_argument("--output", "-o", default="./",
                   help="输出目录（写 train_img.txt / valid_img.txt）")
    args = p.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"错误：数据根目录不存在: {args.data_root}")

    os.makedirs(args.output, exist_ok=True)
    for fname, patients in [("train_img.txt", TRAIN), ("valid_img.txt", VALID)]:
        with open(os.path.join(args.output, fname), "w", encoding="utf-8") as f:
            for pat in patients:
                inp = os.path.join(args.data_root, pat, INP)
                gt = os.path.join(args.data_root, pat, GT)
                if not (os.path.exists(inp) and os.path.exists(gt)):
                    raise SystemExit(f"错误：缺少文件 {pat}: {inp} / {gt}")
                for z in range(1, 159):
                    f.write(f"{inp} {gt} {z}\n")
        print(f"已生成 {fname}: {len(patients)} 病人 x 158 切片 "
              f"= {len(patients) * 158} 行")


if __name__ == "__main__":
    main()
