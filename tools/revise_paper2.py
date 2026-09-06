# -*- coding: utf-8 -*-
"""추가 지적 3건을 논문 docx 에 반영한다.

1 Ⅱ장 제목 "5. AMI" → "6. AMI"   (앞 절이 이미 5 다)
2 각주 인용을 IEEE 인라인 [n] 으로 통일하고 각주 정의를 제거한다.
  각주 7(Koo 재인용)은 각주 6과 같은 문헌이므로 둘 다 [6] 이 된다.
3 Bonferroni 문장을 방어적이지 않은 표현으로 교체한다.

각주 id → 참고문헌 번호
  1 Kim [1] · 2 Wu [2] · 3 Vinh [3] · 4 Engel [4] · 5 Sobot [5] · 6·7 Koo [6]
"""
import re
from pathlib import Path

W = "unpacked"

FN_TO_REF = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6}

# 각주 참조 런 — add_captions_footnotes.py 가 넣은 형태 그대로
FN_RUN = re.compile(
    r'<w:r><w:rPr><w:rFonts w:hint="eastAsia"/><w:vertAlign w:val="superscript"/>'
    r'<w:color w:val="000000"/></w:rPr><w:footnoteReference w:id="(\d+)"/></w:r>')

# IEEE 인라인 인용은 위첨자가 아니라 본문 기준선에 온다
CITE_RUN = '<w:r><w:rPr><w:color w:val="000000"/></w:rPr><w:t>[{}]</w:t></w:r>'

HEADING = ('<w:t xml:space="preserve">5. AMI</w:t>',
           '<w:t xml:space="preserve">6. AMI</w:t>')

BONFERRONI = (
    "각 질문은 여러 축과 구간을 동시에 보므로 다중 비교가 발생한다. 원 분석은 "
    "조합별 신뢰구간만 냈기에, 질문 단위 Bonferroni 보정을 사후에 계산해 병기한다. ",
    "각 연구 질문 내 다중 비교를 고려하여, percentile bootstrap 표본으로 "
    "Bonferroni 보정 95% 신뢰구간을 추가 산출하였다. ")


def main():
    doc_p = Path(W) / "word" / "document.xml"
    fn_p = Path(W) / "word" / "footnotes.xml"
    x = doc_p.read_text(encoding="utf-8")
    f = fn_p.read_text(encoding="utf-8")

    # ── 1. 제목 번호 ──────────────────────────────────────────
    assert x.count(HEADING[0]) == 1, "AMI 제목을 찾지 못했다"
    x = x.replace(*HEADING)

    # ── 3. Bonferroni 문장 ───────────────────────────────────
    assert x.count(BONFERRONI[0]) == 1, "Bonferroni 문장을 찾지 못했다"
    x = x.replace(*BONFERRONI)

    # ── 2. 각주 참조 → 인라인 [n] ─────────────────────────────
    seen = []

    def to_cite(m):
        fid = int(m.group(1))
        seen.append(fid)
        return CITE_RUN.format(FN_TO_REF[fid])

    x, n = FN_RUN.subn(to_cite, x)
    assert sorted(seen) == sorted(FN_TO_REF), f"각주 참조 {sorted(seen)}"
    assert '<w:footnoteReference' not in x, "치환되지 않은 각주 참조가 남았다"

    # 각주 정의 제거 — 구분선/이어짐(id -1, 0)은 남겨야 Word 가 문서를 연다
    for fid in FN_TO_REF:
        pat = re.compile(rf'<w:footnote w:id="{fid}"[^>]*>.*?</w:footnote>', re.S)
        f, k = pat.subn("", f)
        assert k == 1, f"각주 정의 {fid} 제거 실패"
    left = re.findall(r'<w:footnote w:id="(-?\d+)"', f)
    assert set(left) <= {"-1", "0"}, f"남은 각주 {left}"

    doc_p.write_text(x, encoding="utf-8")
    fn_p.write_text(f, encoding="utf-8")
    print(f"  제목 5.→6. · Bonferroni 문장 교체")
    print(f"  각주 {n}개 → 인라인 [n] · 남은 각주 정의 {left}")


if __name__ == "__main__":
    main()
