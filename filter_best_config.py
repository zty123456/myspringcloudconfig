"""
按 hw + seq_len 分组，每组取 step_time_ms 最小的配置行，输出到新 CSV。

用法:
  python scripts/filter_best_config.py <csv_path>
  python scripts/filter_best_config.py <csv_path> --hw-groups '[["ascend_910c","ascend_910b"],["nvidia_h100_sxm","nvidia_h800"]]'
  python scripts/filter_best_config.py <csv_path> --seq-lens 4096,8192
  python scripts/filter_best_config.py <csv_path> --hw-groups '[["1","2"],["3","4"]]' --seq-lens 4096,8192 --output best.csv

说明:
  --hw-groups: 筛选 hw 列的值，仅保留出现在分组中的 hw。每个 hw 独立按 seq_len 取最优配置。
               例如 [["ascend_910c","ascend_910b"],["nvidia_h100_sxm","nvidia_h800"]]
               会分别取 ascend_910c、ascend_910b、nvidia_h100_sxm、nvidia_h800 各自的最优行。
               不指定时，保留所有 hw。
  --seq-lens:  逗号分隔的 seq_len 列表，只保留这些 seq_len 的行。
               不指定时，保留所有 seq_len。
  --output:    输出文件名（默认 best_per_hw_seq_len.csv，写入 CSV 所在目录）。
"""
import sys
import json
import pandas as pd
from pathlib import Path


def parse_hw_groups(raw: str):
    normalized = raw.replace("'", '"')
    groups = json.loads(normalized)
    if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
        raise ValueError("--hw-groups must be a JSON array of arrays")
    return [[str(item) for item in g] for g in groups]


def main(csv_path: str, hw_groups=None, seq_lens=None, output_name=None) -> None:
    df = pd.read_csv(csv_path)
    original_columns = list(df.columns)

    if df.empty:
        print("CSV is empty.")
        return

    if "step_time_ms" not in df.columns:
        print("ERROR: 'step_time_ms' column not found.")
        return

    required = ["hw", "seq_len"]
    missing = [k for k in required if k not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        return

    df["hw"] = df["hw"].astype(str)

    df = df.sort_values("step_time_ms", ascending=True)

    if seq_lens:
        seq_set = set(seq_lens)
        df = df[df["seq_len"].isin(seq_set)]
        if df.empty:
            print(f"ERROR: no rows match seq_len in {sorted(seq_set)}")
            return

    if hw_groups:
        allowed_hw = {hw for group in hw_groups for hw in group}
        df = df[df["hw"].isin(allowed_hw)]
        # 构建 hw → (group_idx, hw_in_group_idx, label) 映射
        hw_info = {}
        for gi, group in enumerate(hw_groups):
            label = "+".join(group)
            for hi, hw_val in enumerate(group):
                hw_info[hw_val] = (gi, hi, label)
        df["_group_order"] = df["hw"].map(lambda h: hw_info[h][0] if h in hw_info else 999)
        df["_hw_order"] = df["hw"].map(lambda h: hw_info[h][1] if h in hw_info else 999)

    # 每个 hw 独立按 seq_len 取 step_time_ms 最小的配置
    result = df.groupby(["hw", "seq_len"], as_index=False, sort=False).first()

    # 按 hw_group 顺序 → seq_len → hw 在组内顺序 排序
    if hw_groups:
        result = result.sort_values(["_group_order", "seq_len", "_hw_order"], ascending=[True, True, True])
        result.drop(columns=["_group_order", "_hw_order"], inplace=True)
        result = result[[c for c in original_columns if c in result.columns]]
    else:
        result = result[[c for c in original_columns if c in result.columns]]

    out_dir = Path(csv_path).parent
    out_name = output_name or "best_per_hw_seq_len.csv"
    out_path = out_dir / out_name
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} rows to {out_path}")
    print(result.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter best config per HW group and seq_len")
    parser.add_argument("csv_path", help="Path to results_summary.csv")
    parser.add_argument("--hw-groups", default=None,
                        help='HW grouping, e.g. \'[["1","2"],["3","4"]]\'')
    parser.add_argument("--seq-lens", default=None,
                        help='Comma-separated seq_len values, e.g. 4096,8192')
    parser.add_argument("--output", default=None,
                        help='Output CSV filename (default: best_per_hw_seq_len.csv)')
    args = parser.parse_args()

    hw_groups = parse_hw_groups(args.hw_groups) if args.hw_groups else None
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")] if args.seq_lens else None

    main(args.csv_path, hw_groups=hw_groups, seq_lens=seq_lens, output_name=args.output)
