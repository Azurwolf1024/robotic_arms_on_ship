"""
把 experiments/run_nn_study.py 的结果 (output/nn_study.pkl) 整理成 Markdown 表格。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.common import load  # noqa: E402

CONFIGS = ["TSID", "TSID+NN", "TSID+LADRC", "TSID+NN+LADRC"]
TASKS = ["fixed_point", "circle"]
SCALES = [0.5, 1.0, 2.0]


def main():
    try:
        res = load("nn_study")
    except FileNotFoundError:
        print("[skip] output/nn_study.pkl 尚未生成; 请先运行 run_nn_study.py")
        return

    for tk in TASKS:
        print(f"\n### {tk}\n")
        hdr = "| 船体幅度 scale | " + " | ".join(CONFIGS) + " |"
        sep = "|" + "---|" * (len(CONFIGS) + 1)
        print(hdr)
        print(sep)
        for sc in SCALES:
            row = f"| {sc:>13} |"
            for c in CONFIGS:
                m = res.get((tk, c, sc))
                if m is None:
                    row += "  —  |"
                else:
                    row += f"  {m['pos_mean']:.2f} mm (max {m['pos_max']:.1f})  |"
            print(row)
        # 相对 TSID 的改善
        print("\n相对 TSID 的均方根位置误差改善 (越小越好):")
        for sc in SCALES:
            base = res.get((tk, "TSID", sc), {}).get("pos_mean")
            line = f"  scale={sc}:"
            for c in CONFIGS[1:]:
                m = res.get((tk, c, sc))
                if base and m:
                    imp = (1 - m["pos_mean"] / base) * 100
                    line += f"  {c}: {imp:+.1f}%"
            print(line)


if __name__ == "__main__":
    main()
