#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计投影域 .raw 数据的值域，并给出推荐的 proj_norm_clip_max；
同时复核裁剪边界（proj_crop_rows / proj_crop_cols）与首尾切片是否退化。

数据约定（见 ximenzi_pbi_data/README.md，形状已实测校正）：
  - 输入: proj_pbi_fs_dec_blur_phase.raw（含伪影）
  - 目标: proj_no_pbi.raw（干净）
  - 平面形状 (n_view, H, W) = (1000, 500, 500)，float32 小端，postlog 值
  - [!] README 写的 (500,500,1000) 是轴向描述，实际存储为连续 1000 个 500x500 平面
       （实测 raw[k] 与切片目录 gt/gt_k.raw 逐字节相同）

投影域归一化为单侧截断：norm(x) = clip(x, 0, clip_max) / clip_max
  → 关键量是**上界**；下界恒为 0（负值一律被截断）

用法:
    python analyze_proj_range.py                              # 默认扫 01~20，隔 10 张
    python analyze_proj_range.py --patients 01 02 03          # 只看指定病人
    python analyze_proj_range.py --patients all               # 扫全部 30 个病人
    python analyze_proj_range.py --stride 1                   # 全部切片（慢，但最准）
    python analyze_proj_range.py --crop-rows 128 --crop-cols 45   # 复核你打算用的裁剪量

输出:
    ① 输入 / 目标 各自的 min/max/mean/std/分位数
    ② 推荐 clip_max 候选（100% / 99.99% / 99.9% / 99% 分位）及各自的截断像素占比
    ③ 每个病人的目标非零包围盒 → 用来确认 proj_crop_rows / proj_crop_cols 是否对全体成立
    ④ 候选裁剪量下的目标能量保留率
    ⑤ 首尾切片退化检查（是否出现全零/常量片，决定要不要剔除）
"""

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_ROOT = r"/mnt/tm/ximenzi_pbi_data"
DEFAULT_SHAPE = (1000, 500, 500)          # (n_view, H, W)
FILES = {
    "input": "proj_pbi_fs_dec_blur_phase.raw",
    "target": "proj_no_pbi.raw",
}
DISPLAY_PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99, 100.0]
CAND_PCTS = [100.0, 99.99, 99.9, 99.0]    # clip_max 候选分位
DEFAULT_PATIENTS = [f"{i:02d}" for i in range(1, 21)]   # 01~20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_ROOT,
                   help="投影域数据根目录，含 01..30 病人子文件夹")
    p.add_argument("--patients", nargs="+", default=None,
                   help="只统计指定病人（如 01 02 ...），或 all=全部（默认 01~20）")
    p.add_argument("--stride", type=int, default=10,
                   help="隔几张平面抽样（默认 10；1=全部 1000 张，最准但最慢）")
    p.add_argument("--shape", default=",".join(map(str, DEFAULT_SHAPE)),
                   help="平面形状 n_view,H,W（默认 1000,500,500）")
    p.add_argument("--crop-rows", type=int, default=None,
                   help="复核：按此上下裁剪量算目标能量保留率")
    p.add_argument("--crop-cols", type=int, default=None,
                   help="复核：按此左右裁剪量算目标能量保留率")
    p.add_argument("--round-step", type=float, default=0.005,
                   help="推荐 clip_max 的圆整步长（默认 0.005）")
    p.add_argument("--max-pool-pixels", type=int, default=20_000_000,
                   help="用于分位数/均值估计的像素采样上限")
    return p.parse_args()


def get_patients(root, only):
    if only and len(only) == 1 and only[0].lower() == "all":
        only = None
    if only:
        missing = [d for d in only if not os.path.isdir(os.path.join(root, d))]
        if missing:
            sys.exit(f"错误：以下病人目录不存在: {missing}")
        return list(only)
    dirs = sorted(os.listdir(root))
    pats = [d for d in dirs
            if os.path.isdir(os.path.join(root, d)) and
            os.path.exists(os.path.join(root, d, FILES["input"])) and
            os.path.exists(os.path.join(root, d, FILES["target"]))]
    if not pats:
        sys.exit(f"错误：{root} 下没有找到含投影 .raw 的病人目录。")
    return pats


def scan(root, patients, shape, stride, max_pool_pixels):
    """返回 {key: {...}}。顺带记录每个病人的目标非零包围盒与首尾切片统计。"""
    n_view = shape[0]
    idx = list(range(0, n_view, stride))
    per_slice_target = max(1, max_pool_pixels // max(1, len(idx) * len(patients)))

    res = {}
    bboxes = {}       # {patient: (r0,r1,c0,c1)} 目标非零包围盒（抽样取并集）
    edge_stat = {}    # {patient: {0: (...), last: (...)}} 首尾切片

    for key, fname in FILES.items():
        t0 = time.time()
        mn, mx = float("inf"), float("-inf")
        pool = []
        for pat in patients:
            path = os.path.join(root, pat, fname)
            vol = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
            for z in idx:
                sl = vol[z]
                mn = min(mn, float(sl.min()))
                mx = max(mx, float(sl.max()))
                step = max(1, int(np.sqrt(sl.size / per_slice_target)))
                pool.append(sl[::step, ::step].ravel())

                if key == "target":
                    # 非零包围盒（用抽样网格足够，边界是整行/整列的结构性零）
                    nz = np.argwhere(sl != 0)
                    if nz.size:
                        r0, c0 = nz.min(0)
                        r1, c1 = nz.max(0)
                        old = bboxes.get(pat)
                        bboxes[pat] = (min(old[0], r0), max(old[1], r1),
                                       min(old[2], c0), max(old[3], c1)) if old \
                            else (r0, r1, c0, c1)
                    if z in (0, n_view - 1):
                        edge_stat.setdefault(pat, {})[z] = (
                            float(sl.min()), float(sl.max()),
                            float(np.abs(sl).sum()), int(np.count_nonzero(sl)))
            del vol
        pool = np.concatenate(pool)
        res[key] = {
            "mn": mn, "mx": mx,
            "mean": float(pool.mean()), "std": float(pool.std()),
            "pcts": np.percentile(pool, DISPLAY_PCTS),
            "neg_frac": float((pool < 0).mean()) * 100,
            "pool": pool,
        }
        print(f"  已扫描 {key:6s}  {len(patients)} 病人 x {len(idx)} 张平面 "
              f"({pool.size:,} 采样像素), 耗时 {time.time() - t0:.1f}s")
    return res, bboxes, edge_stat


def energy_retention(root, patients, shape, stride, crop_rows, crop_cols):
    """目标在给定裁剪量下的能量保留率，逐病人 + 总体。"""
    n_view, H, W = shape
    idx = list(range(0, n_view, stride))
    keep_e, tot_e = 0.0, 0.0
    per_pat = {}
    for pat in patients:
        path = os.path.join(root, pat, FILES["target"])
        vol = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        k, t = 0.0, 0.0
        for z in idx:
            sl = vol[z]
            r0, r1 = crop_rows, H - crop_rows
            c0, c1 = crop_cols, W - crop_cols
            t += float(np.abs(sl).sum())
            k += float(np.abs(sl[r0:r1, c0:c1]).sum())
        del vol
        per_pat[pat] = 100.0 * k / t if t else float("nan")
        keep_e += k
        tot_e += t
    return (100.0 * keep_e / tot_e if tot_e else float("nan")), per_pat


def fmt_pcts(p):
    return "  ".join(f"{l:g}%:{v:9.5f}" for l, v in zip(DISPLAY_PCTS, p))


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        sys.exit(f"错误：数据根目录不存在: {args.data_root}")

    shape = tuple(int(x) for x in args.shape.split(","))
    if len(shape) != 3:
        sys.exit("错误：--shape 需要 3 个值，如 1000,500,500")

    only = args.patients if args.patients else DEFAULT_PATIENTS
    patients = get_patients(args.data_root, only)

    print(f"数据根目录 : {args.data_root}")
    print(f"统计病人   : {len(patients)} 个  ({', '.join(patients)})")
    print(f"平面形状   : {shape}   stride={args.stride}")
    print()

    res, bboxes, edge_stat = scan(args.data_root, patients, shape,
                                  args.stride, args.max_pool_pixels)

    # ---------- ① 值域统计 ----------
    print("\n========== ① 值域统计（mean/std 来自采样池） ==========")
    for key in ("input", "target"):
        r = res[key]
        print(f"\n[{key}]  min={r['mn']:.6f}  max={r['mx']:.6f}  "
              f"mean={r['mean']:.6f}  std={r['std']:.6f}")
        print(f"  负值占比(会被 clip 到 0): {r['neg_frac']:.3f}%")
        print(f"  分位数: {fmt_pcts(r['pcts'])}")

    # ---------- ② 推荐 clip_max ----------
    print("\n========== ② 推荐 proj_norm_clip_max ==========")
    print("  投影域 norm(x)=clip(x,0,c)/c，只用上界；下表给出各候选 c 的截断情况：")
    print(f"  {'候选 c':>10}  {'来源分位':>10}  "
          f"{'目标截断%':>10}  {'输入截断%':>10}  {'归一化后均值':>12}")
    tgt_pool, inp_pool = res["target"]["pool"], res["input"]["pool"]
    best = None
    for pct in CAND_PCTS:
        c_raw = float(np.percentile(tgt_pool, pct))
        c = float(np.ceil(c_raw / args.round_step) * args.round_step)
        t_clip = float((tgt_pool > c).mean()) * 100
        i_clip = float((inp_pool > c).mean()) * 100
        nm = float(np.clip(tgt_pool, 0, c).mean()) / c
        print(f"  {c:>10.4f}  {pct:>9g}%  {t_clip:>10.4f}  {i_clip:>10.4f}  {nm:>12.5f}")
        if best is None or (t_clip <= 0.001 and c < best[0]):
            best = (c, pct, t_clip)
    if best:
        print(f"\n  → 建议 proj_norm_clip_max: {best[0]:.4f}"
              f"（取自 {best[1]:g}% 分位，目标截断 {best[2]:.4f}%）")
    print(f"  → 当前 yml 若为 0.28，则归一化值仅落在 [0, "
          f"{min(1.0, res['target']['mx'] / 0.28):.3f}]，动态范围严重浪费")

    # ---------- ③ 裁剪边界复核 ----------
    print("\n========== ③ 目标非零包围盒（确认 crop 对全体成立） ==========")
    rs, cs = [], []
    for pat in patients:
        if pat not in bboxes:
            print(f"  {pat}: 抽样到的平面全是零（可疑，请用 --stride 1 复查）")
            continue
        r0, r1, c0, c1 = bboxes[pat]
        rs += [r0, r1]
        cs += [c0, c1]
        print(f"  {pat}: 行[{r0},{r1}] 列[{c0},{c1}]  "
              f"→ 无损裁剪 rows/cols = {r0}/{c0}")
    if rs and cs:
        print(f"\n  全体并集: 行[{min(rs)},{max(rs)}] 列[{min(cs)},{max(cs)}]")
        print(f"  → 无损 proj_crop_rows >= {min(rs)},  proj_crop_cols >= {min(cs)}")
        print(f"  → 裁剪后尺寸: "
              f"{shape[1] - 2 * min(rs)} x {shape[2] - 2 * min(cs)}")
        gt_max = min(shape[1] - 2 * min(rs), shape[2] - 2 * min(cs))
        print(f"  → gt_size 上限 = {gt_max // 8 * 8}（需 8 整除，Restormer 要求）")
        print("  [!] 若某病人边界明显不同 → 探测器/准直有差异，不要用全局固定裁剪量")

    # ---------- ④ 候选裁剪量的能量保留率 ----------
    if args.crop_rows is not None or args.crop_cols is not None:
        cr = args.crop_rows or 0
        cc = args.crop_cols or 0
        tot, per_pat = energy_retention(args.data_root, patients, shape,
                                        args.stride, cr, cc)
        print(f"\n========== ④ 裁剪 rows={cr} cols={cc} 的目标能量保留率 ==========")
        print(f"  总体: {tot:.4f}%   （损失 {100 - tot:.4f}%）")
        worst = sorted(per_pat.items(), key=lambda kv: kv[1])[:5]
        print("  损失最大的 5 个病人: "
              + "  ".join(f"{p}:{v:.2f}%" for p, v in worst))
        if tot < 99.9:
            print("  [!] 损失 >0.1%，说明该裁剪量切到了真实信号，考虑改为无损边界")

    # ---------- ⑤ 首尾切片退化检查 ----------
    print("\n========== ⑤ 首尾切片退化检查 ==========")
    n_view = shape[0]
    degen = []
    for pat in patients:
        e = edge_stat.get(pat, {})
        parts = []
        for z in (0, n_view - 1):
            if z in e:
                mn, mx, s, nz = e[z]
                flat = (nz == 0)
                parts.append(f"z={z}: |sum|={s:.2f} 非零={nz} "
                             f"{'[!]全零' if flat else 'ok'}")
                if flat:
                    degen.append((pat, z))
        print(f"  {pat}: " + "   ".join(parts))
    if degen:
        print(f"\n  [!] {len(degen)} 个退化首尾平面，建议剔除: {degen}")
    else:
        print(f"\n  → 首尾切片未见退化，无需剔除（全 {n_view} 张可用）")


if __name__ == "__main__":
    main()
