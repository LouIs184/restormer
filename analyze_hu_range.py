#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计图像域 .raw 数据的 HU 值范围，并给出推荐的归一化窗口。

数据约定（见 DATASET_FORMAT.md）：
  - 输入: rec_pbi_fs_dec_blur_phase_100000.raw（含伪影）
  - 目标: rec_no_pbi.raw（干净）
  - 体积形状 (160, 512, 512)，float32 小端，HU 值
  - 只统计有效切片 z=1..158（剔除首尾）

用法:
    python analyze_hu_range.py                                    # 全量扫描 13 个病人
    python analyze_hu_range.py --stride 2                         # 隔一张切片扫描（更快）
    python analyze_hu_range.py --patients 18537_93073_...         # 只看指定病人
    python analyze_hu_range.py --low-pct 0.1 --high-pct 99.99     # 更保守/更贴极值的窗口
    python analyze_hu_range.py --exclude-air -500                 # 只用解剖组织(HU>-500)算窗口

输出:
    - 输入 / 目标 各自的 min / max / 各分位数 / mean / std
    - 推荐窗口 A: 全量像素按 [low_pct, high_pct] 分位（圆整到 round_step）
    - 推荐窗口 B: 仅解剖组织(HU > exclude_air)按同分位，避免空气峰主导
    - 各窗口的"窗外像素占比"按真实像素统计（非线性外推）
"""

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_ROOT = r"C:/imageReconstruction/联影双能相位数据2080"
FILES = {
    "input": "rec_pbi_fs_dec_blur_phase_100000.raw",
    "target": "rec_no_pbi.raw",
}
VOL_SHAPE = (160, 512, 512)
Z_START, Z_END = 1, 158          # 有效切片区间（不含首尾）
DISPLAY_PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_ROOT,
                   help="图像域数据根目录，含各病人子文件夹")
    p.add_argument("--stride", type=int, default=1,
                   help="隔几张切片抽样（1=全部，2=隔一张……）")
    p.add_argument("--patients", nargs="+", default=None,
                   help="只统计指定病人（默认：根目录下全部病人）")
    p.add_argument("--low-pct", type=float, default=1.0,
                   help="推荐窗口下界分位数（默认 1%）")
    p.add_argument("--high-pct", type=float, default=99.9,
                   help="推荐窗口上界分位数（默认 99.9%）")
    p.add_argument("--round-step", type=float, default=50.0,
                   help="推荐窗口圆整步长（默认 50 HU）")
    p.add_argument("--exclude-air", type=float, default=None,
                   help="额外给出'仅解剖组织'窗口：只统计 HU > 该值的像素"
                        "（默认 None=不额外算；建议 -500 试试）")
    p.add_argument("--max-pool-pixels", type=int, default=20_000_000,
                   help="用于分位数/均值估计的像素采样上限")
    return p.parse_args()


def get_patients(root, only=None):
    if only:
        missing = [p for p in only if not os.path.isdir(os.path.join(root, p))]
        if missing:
            sys.exit(f"错误：以下病人目录不存在: {missing}")
        return list(only)
    dirs = sorted(os.listdir(root))
    patients = [d for d in dirs
                if os.path.isdir(os.path.join(root, d)) and
                os.path.exists(os.path.join(root, d, FILES["input"])) and
                os.path.exists(os.path.join(root, d, FILES["target"]))]
    if not patients:
        sys.exit(f"错误：{root} 下没有找到含 .raw 的图像域病人目录。")
    return patients


def scan(root, patients, stride, max_pool_pixels):
    """返回 {key: {mn, mx, mean, std, pcts, pool}}。pool 用于后续精确占比统计。"""
    n_slices = len(range(Z_START, Z_END, stride)) * len(patients)
    per_slice_target = max(1, max_pool_pixels // n_slices)

    results = {}
    for key, fname in FILES.items():
        t0 = time.time()
        mn, mx = float("inf"), float("-inf")
        pool = []
        for pat in patients:
            path = os.path.join(root, pat, fname)
            vol = np.memmap(path, dtype=np.float32, mode="r", shape=VOL_SHAPE)
            for z in range(Z_START, Z_END, stride):
                sl = vol[z]
                mn = min(mn, float(sl.min()))
                mx = max(mx, float(sl.max()))
                step = max(1, int(np.sqrt(sl.size / per_slice_target)))
                pool.append(sl[::step, ::step].ravel())
        pool = np.concatenate(pool)
        results[key] = {
            "mn": mn,
            "mx": mx,
            "mean": float(pool.mean()),
            "std": float(pool.std()),
            "pcts": np.percentile(pool, DISPLAY_PCTS),
            "pool": pool,
        }
        print(f"  已扫描 {key:6s} {len(patients)} 病人 "
              f"({n_slices} 张切片, {pool.size:,} 采样像素), "
              f"耗时 {time.time()-t0:.1f}s")
    return results


def window_from(pools, low, high, round_step):
    """按 low/high 分位给出一个窗口，圆整到 round_step。pools: list[ndarray]"""
    los = [np.percentile(pool, low) for pool in pools]
    his = [np.percentile(pool, high) for pool in pools]
    hu_min = float(np.floor(min(los) / round_step) * round_step)
    hu_max = float(np.ceil(max(his) / round_step) * round_step)
    return hu_min, hu_max


def coverage(pool, hu_min, hu_max):
    """窗口外像素的真实占比（精确统计，非外推）。"""
    frac_lo = float((pool < hu_min).mean()) * 100
    frac_hi = float((pool > hu_max).mean()) * 100
    return frac_lo, frac_hi


def fmt_pcts(p):
    labels = ["0.1%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "99.9%"]
    return "  ".join(f"{l}:{v:9.1f}" for l, v in zip(labels, p))


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        sys.exit(f"错误：数据根目录不存在: {args.data_root}")

    patients = get_patients(args.data_root, args.patients)
    print(f"数据根目录 : {args.data_root}")
    print(f"统计病人   : {len(patients)} 个")
    print(f"切片抽样   : z={Z_START}..{Z_END}, stride={args.stride}")
    print()

    res = scan(args.data_root, patients, args.stride, args.max_pool_pixels)

    print("\n========== 各集合 HU 统计（mean/std 来自采样池） ==========")
    for key in ("input", "target"):
        r = res[key]
        print(f"\n[{key}]  min={r['mn']:.1f}  max={r['mx']:.1f}  "
              f"mean={r['mean']:.2f}  std={r['std']:.2f}")
        print(f"  分位数: {fmt_pcts(r['pcts'])}")

    pools = [res["input"]["pool"], res["target"]["pool"]]

    print("\n========== 推荐归一化窗口 ==========")
    # 窗口 A: 全量像素（含空气）
    lo, hi = window_from(pools, args.low_pct, args.high_pct, args.round_step)
    print(f"\n[窗口 A: 全量像素 {args.low_pct}% ~ {args.high_pct}% 分位] "
          f"hu_min={lo:.0f}  hu_max={hi:.0f}")
    for key in ("input", "target"):
        fl, fh = coverage(res[key]["pool"], lo, hi)
        print(f"    {key:6s}: 窗外占比 低于下界 {fl:.3f}% / 高于上界 {fh:.3f}%")

    # 窗口 B: 仅解剖组织（排除空气）
    if args.exclude_air is not None:
        th = args.exclude_air
        anat_pools = [p[p > th] for p in pools]
        n_keep = sum(p.size for p in anat_pools)
        n_all = sum(p.size for p in pools)
        lo2, hi2 = window_from(anat_pools, args.low_pct, args.high_pct,
                               args.round_step)
        print(f"\n[窗口 B: 仅解剖组织(HU>{th:.0f})，占全量 "
              f"{100.0*n_keep/n_all:.1f}%] hu_min={lo2:.0f}  hu_max={hi2:.0f}")
        for key in ("input", "target"):
            fl, fh = coverage(res[key]["pool"], lo2, hi2)
            print(f"    {key:6s}: 窗外占比 低于下界 {fl:.3f}% / 高于上界 {fh:.3f}%")

    # 对照: AICTVer2 默认窗口
    print("\n[对照: AICTVer2 默认 [-1000, 1000]]")
    for key in ("input", "target"):
        fl, fh = coverage(res[key]["pool"], -1000.0, 1000.0)
        print(f"    {key:6s}: 窗外占比 低于下界 {fl:.3f}% / 高于上界 {fh:.3f}%")


if __name__ == "__main__":
    main()
