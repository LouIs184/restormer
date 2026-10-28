#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成仿真数据（simulation_change_params_20heads）的投影域索引。

数据来源: <data-root>/<头>/<剂量>/ 下的两个 raw
    输入(含伪影): proj_pbi_fs_dec_blur_phase.raw
    目标(干净)  : proj_no_pbi.raw
    平面形状    : (100, 500, 500) = (视角数, H, W)，float32 小端，post_log
    （实测确认：按 (100,500,500) 解析时 100 个平面均值几乎相同，
      按字面的 (500,500,100) 解析则完全错乱；最外层是视角号）

⚠️ 本数据集是**两级目录**（<头>/<剂量>），与 ximenzi 的 <病人>/ 不同。
   每个 (头, 剂量) 组合算一条"卷"，每卷 100 个视角。

每行格式: <输入.raw绝对路径> <目标.raw绝对路径> <视角号 v>

划分（按**头**划分，同一头的 5 个剂量档必须落在同一集合，避免泄漏）:
    train 01~16 (16 头) / valid 17~18 (2 头) / 无 test
    —— 头 19/20 数据不完整，不纳入

用法:
    python prepare_sim_proj_data.py --output ./
    python prepare_sim_proj_data.py --train-heads 01 02 --valid-heads 03
    python prepare_sim_proj_data.py --doses dose_1e4          # 只用单一剂量档

注意: 生成的 *_proj.txt 已被 .gitignore 忽略，需在服务器上本地生成。
"""

import argparse
import os
import sys

INP = "proj_pbi_fs_dec_blur_phase.raw"   # 输入（含伪影）
GT = "proj_no_pbi.raw"                   # 目标（干净）
NUM_VIEWS = 100
SHAPE = (100, 500, 500)                  # (视角数, H, W)
EXPECT_BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2] * 4    # 100,000,000

ALL_DOSES = ["dose_1e4", "dose_1e5", "dose_1e6", "dose_1e7", "dose_1e8"]

SPLIT = {
    "train_sim_proj.txt": [f"{i:02d}" for i in range(1, 17)],   # 01~16
    "valid_sim_proj.txt": ["17", "18"],                        # 17~18
    # 无 test：19/20 数据不完整，按要求不做测试集
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root",
                   default="/mnt/no3/tengmeng/simulation_change_params_20heads",
                   help="数据根目录（含 01..20 头目录）")
    p.add_argument("--output", "-o", default="./",
                   help="输出目录（写 train/valid 两个 txt）")
    p.add_argument("--train-heads", nargs="+", default=None, help="覆盖训练集头列表")
    p.add_argument("--valid-heads", nargs="+", default=None, help="覆盖验证集头列表")
    p.add_argument("--doses", nargs="+", default=None,
                   help=f"使用哪些剂量档（默认全部 5 档: {' '.join(ALL_DOSES)}）")
    p.add_argument("--num-views", type=int, default=NUM_VIEWS,
                   help=f"每卷使用的视角数（默认 {NUM_VIEWS}）")
    p.add_argument("--skip-size-check", action="store_true",
                   help="跳过文件大小校验（不建议）")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        sys.exit(f"错误：数据根目录不存在: {args.data_root}")

    doses = args.doses or ALL_DOSES
    for d in doses:
        if d not in ALL_DOSES:
            sys.exit(f"错误：未知剂量档 {d}，可选: {ALL_DOSES}")

    split = dict(SPLIT)
    if args.train_heads:
        split["train_sim_proj.txt"] = args.train_heads
    if args.valid_heads:
        split["valid_sim_proj.txt"] = args.valid_heads

    # 划分重叠检查：同一个头不得跨集合
    seen = {}
    for name, heads in split.items():
        for h in heads:
            if h in seen:
                sys.exit(f"错误：头 {h} 同时出现在 {seen[h]} 和 {name}")
            seen[h] = name

    os.makedirs(args.output, exist_ok=True)
    total_vol = 0
    for fname, heads in split.items():
        out_path = os.path.join(args.output, fname)
        n_line = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for h in heads:
                for d in doses:
                    inp = os.path.join(args.data_root, h, d, INP)
                    gt = os.path.join(args.data_root, h, d, GT)
                    for p in (inp, gt):
                        if not os.path.exists(p):
                            sys.exit(f"错误：缺少文件 {p}\n"
                                     f"      （头 {h} / {d} 的数据可能未生成完整）")
                        if not args.skip_size_check:
                            size = os.path.getsize(p)
                            if size != EXPECT_BYTES:
                                sys.exit(
                                    f"错误：文件大小不符，形状可能不是 {SHAPE}\n"
                                    f"  {p}\n  实际 {size} 字节，期望 {EXPECT_BYTES} 字节")
                    for v in range(args.num_views):
                        f.write(f"{inp} {gt} {v}\n")
                    n_line += args.num_views
        n_vol = len(heads) * len(doses)
        total_vol += n_vol
        print(f"已生成 {out_path}: {len(heads)} 头 x {len(doses)} 剂量 = {n_vol} 卷 "
              f"x {args.num_views} 视角 = {n_line} 行")

    print(f"\n总计 {len(seen)} 个头、{total_vol} 卷："
          f"train {len(split['train_sim_proj.txt'])} 头 / "
          f"valid {len(split['valid_sim_proj.txt'])} 头")
    print("下一步：确认 yml 的 proj_shape=[100,500,500]、clip_max=0.17、crop 122/2")


if __name__ == "__main__":
    main()
