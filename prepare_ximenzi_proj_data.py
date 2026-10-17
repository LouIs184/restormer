#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成西门子 PBI 投影域数据索引：train/valid/test 三个 txt。

数据来源: <data-root>/<病人>/ 下的两个大 raw
    输入(含伪影): proj_pbi_fs_dec_blur_phase.raw   ← 注意没有 _100000 后缀
    目标(干净)  : proj_no_pbi.raw
    平面形状    : (1000, 500, 500) = (视角数, H, W)，float32 小端
    （已实测：raw[k] 与切片目录 gt/gt_k.raw 逐字节相同，最外层即视角号）

每行格式: <输入.raw绝对路径> <目标.raw绝对路径> <视角号 v>

划分（共 20 个病人，编号 01~20）:
    train 01~15 (15人) / valid 16~18 (3人) / test 19~20 (2人)
    —— 按病人划分，同一病人的所有视角只出现在一个集合里

用法:
    python prepare_ximenzi_proj_data.py
    python prepare_ximenzi_proj_data.py --data-root /mnt/tm/ximenzi_pbi_data --output ./
    python prepare_ximenzi_proj_data.py --train 01 02 --valid 03 --test 04   # 自定义划分

注意: 生成的 *_proj.txt 已被 .gitignore 忽略，需在服务器上本地生成。
"""

import argparse
import os
import sys

INP = "proj_pbi_fs_dec_blur_phase.raw"   # 输入（含伪影）
GT = "proj_no_pbi.raw"                   # 目标（干净）
NUM_VIEWS = 1000                         # 全 1000 视角，不需剔除首尾
SHAPE = (1000, 500, 500)                 # 用于校验文件大小

SPLIT = {
    "train_ximenzi_proj.txt": [f"{i:02d}" for i in range(1, 16)],    # 01~15
    "valid_ximenzi_proj.txt": [f"{i:02d}" for i in range(16, 19)],   # 16~18
    "test_ximenzi_proj.txt": [f"{i:02d}" for i in range(19, 21)],    # 19~20
}
EXPECT_BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2] * 4   # 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/mnt/tm/ximenzi_pbi_data",
                   help="数据根目录（含 01..30 病人子文件夹）")
    p.add_argument("--output", "-o", default="./",
                   help="输出目录（写 train/valid/test 三个 txt）")
    p.add_argument("--train", nargs="+", default=None, help="覆盖训练集病人列表")
    p.add_argument("--valid", nargs="+", default=None, help="覆盖验证集病人列表")
    p.add_argument("--test", nargs="+", default=None, help="覆盖测试集病人列表")
    p.add_argument("--num-views", type=int, default=NUM_VIEWS,
                   help=f"视角数（默认 {NUM_VIEWS}）")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        sys.exit(f"错误：数据根目录不存在: {args.data_root}")

    split = dict(SPLIT)
    overrides = {
        "train_ximenzi_proj.txt": args.train,
        "valid_ximenzi_proj.txt": args.valid,
        "test_ximenzi_proj.txt": args.test,
    }
    for name, pats in overrides.items():
        if pats:
            split[name] = pats

    # 划分重叠检查：同一病人不得跨集合
    seen = {}
    for name, pats in split.items():
        for pat in pats:
            if pat in seen:
                sys.exit(f"错误：病人 {pat} 同时出现在 {seen[pat]} 和 {name}")
            seen[pat] = name

    os.makedirs(args.output, exist_ok=True)
    for fname, patients in split.items():
        out_path = os.path.join(args.output, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for pat in patients:
                inp = os.path.join(args.data_root, pat, INP)
                gt = os.path.join(args.data_root, pat, GT)
                for p in (inp, gt):
                    if not os.path.exists(p):
                        sys.exit(f"错误：缺少文件 {p}")
                    size = os.path.getsize(p)
                    if size != EXPECT_BYTES:
                        sys.exit(
                            f"错误：文件大小不符，形状可能不是 {SHAPE}\n"
                            f"  {p}\n  实际 {size} 字节，期望 {EXPECT_BYTES} 字节")
                for v in range(args.num_views):
                    f.write(f"{inp} {gt} {v}\n")
        print(f"已生成 {out_path}: {len(patients)} 病人 "
              f"({', '.join(patients)}) x {args.num_views} 视角 "
              f"= {len(patients) * args.num_views} 行")

    print(f"\n总计 {len(seen)} 个病人："
          f"train {len(split['train_ximenzi_proj.txt'])} / "
          f"valid {len(split['valid_ximenzi_proj.txt'])} / "
          f"test {len(split['test_ximenzi_proj.txt'])}")
    print("下一步：先跑 analyze_proj_range.py 标定 proj_norm_clip_max 并填进 yml。")


if __name__ == "__main__":
    main()
