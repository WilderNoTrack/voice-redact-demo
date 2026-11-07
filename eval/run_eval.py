"""跑评测，输出一份可以直接贴进 README 或邮件的对照表。

评测口径只有两条，但都要说清楚，否则数字好看却没用：

· 漏检（FN）：标注里该替换的，没被找到。这是脱敏产品的致命伤——
  漏一个身份证号，整份材料就不能外发。
· 误替换（FP）：标注里没有，但被替换了。这是可用性问题——
  把"第八十七条"当成金额替换掉，律师会立刻弃用。

两者不能只看一个。只追求召回的系统会把整篇文章涂黑；只追求精确的系统会漏掉该拦的。
所以这张表两个都报。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.redact import redact  # noqa: E402

CASES = Path(__file__).resolve().parent / "cases.json"
OUT = ROOT / "eval" / "REPORT.md"


def covered(entity: str, spans) -> bool:
    """标注实体是否被某个 span 覆盖（允许 span 比实体长）。"""
    for sp in spans:
        if entity in sp.text or sp.text in entity:
            return True
    return False


def run() -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rows, total_exp, total_hit, total_fp, total_ms = [], 0, 0, 0, 0.0

    for case in cases:
        persons = [p.strip() for p in (case.get("persons") or "").split(",") if p.strip()]
        t0 = time.perf_counter()
        res = redact(case["text"], persons=persons)
        ms = (time.perf_counter() - t0) * 1000

        expect = case.get("expect", [])
        hit = [e for e in expect if covered(e["text"], res.spans)]
        miss = [e for e in expect if not covered(e["text"], res.spans)]

        # 误替换：spans 里没有任何一个标注实体与之重叠
        expect_texts = [e["text"] for e in expect]
        fp = [s for s in res.spans
              if not any(t in s.text or s.text in t for t in expect_texts)]

        total_exp += len(expect)
        total_hit += len(hit)
        total_fp += len(fp)
        total_ms += ms

        rows.append({
            "id": case["id"], "note": case["note"],
            "expect": len(expect), "hit": len(hit),
            "miss": [f'{e["text"]}({e["kind"]})' for e in miss],
            "fp": [f'{s.text}→{s.placeholder}' for s in fp],
            "spans": len(res.spans), "ms": ms,
            "redacted": res.redacted,
        })

    recall = total_hit / total_exp if total_exp else 1.0
    # 误替换率的分母用"实际替换的 span 总数"更直观：替换出去的东西里有多少是错的
    total_spans = sum(r["spans"] for r in rows)
    fp_rate = total_fp / total_spans if total_spans else 0.0

    return {"rows": rows, "recall": recall, "fp_rate": fp_rate,
            "total_exp": total_exp, "total_hit": total_hit,
            "total_fp": total_fp, "avg_ms": total_ms / len(cases)}


def to_markdown(r: dict) -> str:
    lines = [
        "# 脱敏评测报告",
        "",
        f"- 用例 {len(r['rows'])} 条，标注实体 {r['total_exp']} 个",
        f"- **召回率 {r['recall']:.1%}**（{r['total_hit']}/{r['total_exp']}），漏检 "
        f"{r['total_exp'] - r['total_hit']} 个",
        f"- **误替换率 {r['fp_rate']:.1%}**（{r['total_fp']} 个替换不在标注内）",
        f"- 单条平均耗时 {r['avg_ms']:.1f} ms（纯规则，无模型调用）",
        "",
        "| 用例 | 说明 | 标注 | 命中 | 漏检 | 误替换 | 耗时 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['id']} | {row['note']} | {row['expect']} | {row['hit']} | "
            f"{'、'.join(row['miss']) or '—'} | {'、'.join(row['fp']) or '—'} | {row['ms']:.1f} ms |"
        )
    lines += ["", "## 逐条输出", ""]
    for row in r["rows"]:
        lines += [f"### {row['id']}", "", f"> {row['note']}", "",
                  "```", row["redacted"], "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run()
    md = to_markdown(result)
    print(md)
    OUT.write_text(md, encoding="utf-8")
    print(f"\n已写入 {OUT}")
