"""
按 hw 分组 + seq_len 分组，每组取 step_time_ms 最小的配置行，输出到新 CSV。

用法:
  python scripts/filter_best_config.py <csv_path>
  python scripts/filter_best_config.py <csv_path> --hw-groups '[["1","2"],["3","4"]]'
  python scripts/filter_best_config.py <csv_path> --seq-lens 4096,8192
  python scripts/filter_best_config.py <csv_path> --hw-groups '[["1","2"],["3","4"]]' --seq-lens 4096,8192 --output best.csv

说明:
  --hw-groups: 将 hw 列的值分组，同组内竞争取 step_time_ms 最小的配置。
               例如 [["1","2"],["3","4"]] 表示节点数 1 和 2 作为一组，3 和 4 作为另一组。
               不指定时，每个 hw 值独立作为一组。
  --seq-lens:  逗号分隔的 seq_len 列表，只保留这些 seq_len 的行。
               不指定时，保留所有 seq_len。
  --output:    输出文件名（默认 best_per_hw_seq_len.csv，写入 CSV 所在目录）。
"""
import sys
import json
import pandas as pd
from pathlib import Path


def parse_hw_groups(raw: str):
    """解析 [[1,2],[3,4]] 或 [['1','2'],['3','4']] 格式的 JSON 数组。"""
    # 允许用户使用单引号或中文字符等
    normalized = raw.replace("'", '"')
    groups = json.loads(normalized)
    if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
        raise ValueError("--hw-groups must be a JSON array of arrays")
    return [[str(item) for item in g] for g in groups]


def main(csv_path: str, hw_groups=None, seq_lens=None, output_name=None) -> None:
    df = pd.read_csv(csv_path)

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

    # 确保 hw 列是字符串，以便与分组值匹配
    df["hw"] = df["hw"].astype(str)

    # 按 step_time_ms 升序排列
    df = df.sort_values("step_time_ms", ascending=True)

    # 按 seq_len 筛选
    if seq_lens:
        seq_set = set(seq_lens)
        df = df[df["seq_len"].isin(seq_set)]
        if df.empty:
            print(f"ERROR: no rows match seq_len in {sorted(seq_set)}")
            return

    # 按 hw 分组筛选
    if hw_groups:
        allowed_hw = {hw for group in hw_groups for hw in group}
        df = df[df["hw"].isin(allowed_hw)]
        # 将 hw 映射到组标签上
        hw_to_group = {}
        for i, group in enumerate(hw_groups):
            label = "+".join(group)
            for hw_val in group:
                hw_to_group[hw_val] = label
        df["hw_group"] = df["hw"].map(hw_to_group)
        group_keys = ["hw_group", "seq_len"]
    else:
        group_keys = ["hw", "seq_len"]

    # 每组取 step_time_ms 最小的一行
    result = df.groupby(group_keys, as_index=False, sort=False).first()

    # 如果使用了 hw_group，移除临时列，保留原始 hw（即最小 step_time 对应的 hw 值）
    if hw_groups:
        result.drop(columns=["hw_group"], inplace=True)

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
