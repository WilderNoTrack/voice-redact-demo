"""敏感信息识别与替换。

设计取舍（写在这里是因为这几条决定了整个模块的样子）：

1. 规则优先，模型靠后。身份证、手机号、银行卡这类"有格式、有校验位"的东西，
   正则 + 校验位能做到零误判，比任何模型都可靠，而且能解释"为什么匹配"。
   模型只用来兜规则兜不住的部分（人名）。

2. 每个命中都带置信度和命中理由。用户的界面需要知道"哪条可以自动替换、
   哪条必须人来确认"，所以 Span 里必须有 confidence 和 method，不能只给结果。

3. 替换用占位符 + 内存映射表，不回填原文。占位符按类型编号（[当事人1]、
   [身份证1]），下游模型看到的是占位符；映射表只在本次请求的内存里存在，
   不落盘、不进日志。这条是产品边界，不是技术选择。

4. 姓名的启发式召回是有意做成"低置信度"的。中文人名没有边界、没有校验位，
   靠姓氏表硬猜必然误杀。所以它只输出 0.6 置信度，界面上标黄，等人点确认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Literal

Kind = Literal[
    "id_card", "phone", "bank_card", "case_no", "email",
    "amount", "date", "plate", "address", "person", "org",
]

# 占位符前缀。中文场景下用中文标签更易读，也避免和正文里的英文方括号冲突。
LABELS: dict[str, str] = {
    "id_card": "身份证",
    "phone": "手机号",
    "bank_card": "银行卡",
    "case_no": "案号",
    "email": "邮箱",
    "amount": "金额",
    "date": "日期",
    "plate": "车牌",
    "address": "地址",
    "person": "当事人",
    "org": "机构",
}


@dataclass
class Span:
    start: int
    end: int
    kind: str
    text: str
    confidence: float
    method: str          # 命中理由，界面上直接展示
    placeholder: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RedactResult:
    original: str
    redacted: str
    spans: list[Span] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)  # 占位符 -> 原文，仅内存

    def to_dict(self, with_mapping: bool = False) -> dict:
        d = {
            "original": self.original,
            "redacted": self.redacted,
            "spans": [s.to_dict() for s in self.spans],
            "stats": self.stats(),
        }
        if with_mapping:
            d["mapping"] = self.mapping
        return d

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        auto = pending = 0
        for s in self.spans:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
            if s.confidence >= AUTO_THRESHOLD:
                auto += 1
            else:
                pending += 1
        return {
            "total": len(self.spans),
            "by_kind": by_kind,
            "auto_replaced": auto,
            "need_confirm": pending,
        }


AUTO_THRESHOLD = 0.85   # 到这个分数才自动替换，其余交给人


# ── 校验位 ────────────────────────────────────────────────────────────────

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


def id_card_valid(s: str) -> bool:
    """GB 11643 校验位。18 位身份证的最后一位不是随便写的。"""
    if len(s) != 18:
        return False
    body = s[:17]
    if not body.isdigit():
        return False
    total = sum(int(c) * w for c, w in zip(body, _ID_WEIGHTS))
    return _ID_CHECK[total % 11] == s[17].upper()


def luhn_valid(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 16 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ── 规则表 ────────────────────────────────────────────────────────────────
# 顺序有意义：先长后短、先强后弱。身份证必须排在"银行卡"之前，
# 否则 18 位身份证会被当成卡号吃掉。

_RULES: list[tuple[str, re.Pattern, str, float]] = [
    ("id_card", re.compile(r"(?<![0-9A-Za-z])([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?![0-9A-Za-z])"), "18 位格式 + 校验位通过", 1.0),
    ("case_no", re.compile(r"[（(]\s*\d{4}\s*[)）]\s*[\u4e00-\u9fa5]{0,6}\d{0,6}\s*[\u4e00-\u9fa5]{0,4}\s*\d+\s*号"), "案号格式", 0.95),
    ("bank_card", re.compile(r"(?<![0-9])(?:\d[ -]?){15,18}\d(?![0-9])"), "16–19 位 + Luhn 通过", 0.9),
    ("phone", re.compile(r"(?<![0-9])(?:1[3-9]\d{9})(?![0-9])"), "手机号段格式", 0.95),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "邮箱格式", 0.95),
    ("plate", re.compile(r"(?<![A-Za-z0-9\u4e00-\u9fa5])[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳](?![A-Za-z0-9])"), "车牌格式", 0.9),
    ("amount", re.compile(r"(?<![0-9.])\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:元|万元|亿元)|(?<![0-9.])\d+(?:\.\d+)?\s*(?:万元|亿元|元)(?![0-9])"), "金额格式", 0.85),
    ("date", re.compile(r"(?:19|20)\d{2}\s*[年\-/.]\s*(?:0?[1-9]|1[0-2])\s*[月\-/.]\s*(?:3[01]|[12]\d|0?[1-9])\s*日?"), "日期格式", 0.8),
]

# 地址用关键词触发，避免"北京市"这种单独出现也被吃掉——
# 只替换到门牌号级别的连续片段。
_ADDRESS = re.compile(
    r"[\u4e00-\u9fa5]{2,8}(?:省|自治区|市)[\u4e00-\u9fa5]{2,10}?(?:区|县|市)"
    r"[\u4e00-\u9fa5]{2,12}?(?:路|街|道|巷|胡同|里|村)"
    r"[\u4e00-\u9fa5\d]{0,10}?(?:\d+号)?"
)

# 姓名线索词：只在这些人出现的位置附近认人名，避免满篇误杀。
_NAME_CUES = ("被告人", "原告", "被告", "证人", "被害人", "申请人", "被执行人",
              "委托人", "当事人", "姓名", "联系人", "法定代表人", "负责人")

# 单姓（覆盖约 95% 的中文姓氏，够 demo 用；生产环境应换完整姓氏表 + 人名库）
_SURNAMES = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘"
    "于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江"
    "尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
)


def _iter_matches(text: str, span_taken: list[tuple[int, int]]):
    """按规则表跑一遍，跳过已被更靠前规则占用的区间。"""
    for kind, pattern, reason, conf in _RULES:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if any(not (end <= a or start >= b) for a, b in span_taken):
                continue
            # 身份证要过校验位，银行卡要过 Luhn，否则不认——
            # method 上写着"校验位通过"，就必须真的校验过，不能只是格式像
            raw = m.group(0)
            if kind == "id_card" and not id_card_valid(raw):
                continue
            if kind == "bank_card":
                if not luhn_valid(raw):
                    continue
            yield kind, start, end, raw.strip(), conf, reason
            span_taken.append((start, end))


def redact(
    text: str,
    persons: list[str] | None = None,
    orgs: list[str] | None = None,
    use_name_heuristic: bool = True,
) -> RedactResult:
    """识别并替换敏感信息。

    persons / orgs 是外部词表（当事人名单、机构名单）。词表命中的置信度高于
    姓氏启发式——这是有意为之：能确定的走词表，不确定的才让规则猜。
    """
    text = text or ""
    spans: list[Span] = []
    taken: list[tuple[int, int]] = []

    for kind, start, end, raw, conf, reason in _iter_matches(text, taken):
        spans.append(Span(start, end, kind, raw, conf, reason))

    # 词表：姓名、机构。长词优先，避免"张三丰"被"张三"截断。
    for words, kind, conf in ((persons or [], "person", 0.98), (orgs or [], "org", 0.95)):
        for w in sorted((w for w in words if w), key=len, reverse=True):
            for m in re.finditer(re.escape(w), text):
                s, e = m.start(), m.end()
                if any(not (e <= a or s >= b) for a, b in taken):
                    continue
                spans.append(Span(s, e, kind, w, conf, "词表命中"))
                taken.append((s, e))

    # 地址
    for m in _ADDRESS.finditer(text):
        s, e = m.start(), m.end()
        if any(not (e <= a or s >= b) for a, b in taken):
            continue
        spans.append(Span(s, e, "address", m.group(0), 0.75, "省市区 + 路街门牌结构"))
        taken.append((s, e))

    # 姓名启发式：只在线索词后 6 个字内认 2–3 字人名，且必须落在姓氏表里
    if use_name_heuristic:
        for cue in _NAME_CUES:
            for m in re.finditer(re.escape(cue), text):
                tail_start = m.end()
                tail = text[tail_start:tail_start + 4]
                mm = re.match(r"[：:，,\s]{0,2}([\u4e00-\u9fa5]{2,3})", tail)
                if not mm:
                    continue
                name = mm.group(1)
                if len(name) < 2 or name[0] not in _SURNAMES:
                    continue
                s = tail_start + mm.start(1)
                e = tail_start + mm.end(1)
                if any(not (e <= a or s >= b) for a, b in taken):
                    continue
                spans.append(Span(s, e, "person", name, 0.6, f"「{cue}」后的姓氏命中（启发式，需确认）"))
                taken.append((s, e))

    # 排序、编号、生成占位符
    spans.sort(key=lambda x: (x.start, -x.end))
    counters: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for sp in spans:
        n = counters.get(sp.kind, 0) + 1
        counters[sp.kind] = n
        label = LABELS.get(sp.kind, sp.kind)
        ph = f"[{label}{n}]"
        # 同一段原文只保留一个映射，避免 [当事人1] 和 [当事人2] 指向同一人时混乱
        if sp.text in mapping.values():
            for k, v in mapping.items():
                if v == sp.text:
                    ph = k
                    break
        else:
            mapping[ph] = sp.text
        sp.placeholder = ph

    # 拼装脱敏文本
    out, cursor = [], 0
    for sp in spans:
        if sp.start < cursor:      # 区间重叠时以先出现的为准
            continue
        out.append(text[cursor:sp.start])
        out.append(sp.placeholder)
        cursor = sp.end
    out.append(text[cursor:])

    return RedactResult(original=text, redacted="".join(out), spans=spans, mapping=mapping)


def apply_manual(text: str, result: RedactResult, keep_spans: set[int]) -> str:
    """人工确认后重建文本：keep_spans 是用户选择"不要替换"的 span 索引。"""
    out, cursor = [], 0
    for i, sp in enumerate(result.spans):
        if sp.start < cursor:
            continue
        out.append(text[cursor:sp.start])
        out.append(sp.text if i in keep_spans else sp.placeholder)
        cursor = sp.end
    out.append(text[cursor:])
    return "".join(out)


if __name__ == "__main__":
    demo = (
        "被告人张建军，身份证号110105199003078211，手机13812345678，"
        "住北京市朝阳区建国路88号。2024年3月15日通过尾号6222 0202 0000 1235的"
        "银行卡转账123,000元。案号（2024）京0105刑初123号。"
    )
    r = redact(demo, persons=["张建军"], orgs=["北京市朝阳区人民检察院"])
    print("【有词表】", r.redacted)
    for s in r.spans:
        print(f"  {s.placeholder:12} {s.kind:9} {s.confidence:.2f}  {s.method}")
    print("stats:", r.stats())

    # 没有词表时，姓名只能靠线索词 + 姓氏表猜，置信度会降到 0.6
    r2 = redact(demo)
    print("\n【无词表】", r2.redacted)
    for s in r2.spans:
        if s.kind == "person":
            print(f"  {s.placeholder:12} {s.confidence:.2f}  {s.method}")
