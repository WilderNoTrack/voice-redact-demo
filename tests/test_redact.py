"""单元测试，用标准库 unittest，不引第三方。

跑：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.redact import (  # noqa: E402
    apply_manual, id_card_valid, luhn_valid, redact,
)


class TestChecksums(unittest.TestCase):
    def test_id_card(self):
        self.assertTrue(id_card_valid("110105199003078211"))
        self.assertTrue(id_card_valid("110105198505120016"))
        # 改最后一位应失败
        self.assertFalse(id_card_valid("110105199003078218"))

    def test_luhn(self):
        self.assertTrue(luhn_valid("6222020200001235"))
        self.assertFalse(luhn_valid("6222020200001234"))


class TestRedact(unittest.TestCase):
    def test_phonenum(self):
        r = redact("联系13812345678。")
        self.assertEqual(r.redacted, "联系[手机号1]。")
        self.assertEqual(len(r.spans), 1)

    def test_bad_luhn_not_replaced(self):
        # 校验位不通过就放过，宁可漏不可错
        r = redact("账户6222 0202 0000 1234。")
        self.assertEqual(r.spans, [])
        self.assertIn("6222 0202 0000 1234", r.redacted)

    def test_legal_text_untouched(self):
        text = ("《劳动合同法》第八十七条规定，用人单位违反本法规定解除或者终止劳动合同的，"
                "应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。")
        r = redact(text)
        self.assertEqual(r.spans, [])
        self.assertEqual(r.redacted, text)

    def test_placeholder_numbering(self):
        r = redact("甲110105199003078211，乙110105198505120016。")
        phs = [s.placeholder for s in r.spans]
        self.assertEqual(phs, ["[身份证1]", "[身份证2]"])

    def test_person_lexicon_beats_heuristic(self):
        r = redact("被告人张建军到案。", persons=["张建军"])
        self.assertEqual(len(r.spans), 1)
        self.assertGreaterEqual(r.spans[0].confidence, 0.95)

    def test_person_heuristic_low_confidence(self):
        r = redact("被告人张建军到案。")
        self.assertEqual(len(r.spans), 1)
        self.assertLess(r.spans[0].confidence, 0.85)   # 必须落在待确认区间

    def test_overlap_not_double_replaced(self):
        # 身份证里包含数字，不能被银行卡规则再吃一遍
        r = redact("证件110105199003078211。")
        self.assertEqual([s.kind for s in r.spans], ["id_card"])

    def test_same_value_same_placeholder(self):
        r = redact("张建军称，张建军当日不在场。", persons=["张建军"])
        self.assertEqual(len({s.placeholder for s in r.spans}), 1)

    def test_apply_manual_keeps_selected(self):
        r = redact("被告人张建军到案。", persons=["张建军"])
        kept = apply_manual(r.original, r, {0})
        self.assertEqual(kept, "被告人张建军到案。")
        restored = apply_manual(r.original, r, set())
        self.assertEqual(restored, "被告人[当事人1]到案。")


class TestProperties(unittest.TestCase):
    """不管怎么改规则，这几条必须一直成立。"""

    def test_no_original_leak_in_redacted(self):
        text = "被告人张建军，手机13812345678，身份证号110105199003078211。"
        r = redact(text, persons=["张建军"])
        for s in r.spans:
            if s.confidence >= 0.85:
                self.assertNotIn(s.text, r.redacted,
                                 f"自动替换的 {s.kind} 仍出现在输出里")

    def test_spans_sorted_and_non_overlapping(self):
        text = "张建军13812345678北京市朝阳区建国路88号110105199003078211"
        r = redact(text, persons=["张建军"])
        last = -1
        for s in r.spans:
            self.assertGreaterEqual(s.start, last)
            last = s.end


if __name__ == "__main__":
    unittest.main(verbosity=2)
