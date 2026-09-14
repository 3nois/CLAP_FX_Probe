# -*- coding: utf-8 -*-
"""논문 docx 를 IEEE 식 2단 조판으로 바꾼다.

입력  docs/개인_포트폴리오_논문_최종.docx
출력  같은 파일 (unpacked/ 를 덮어쓴 뒤 다시 압축)

배치
  전폭(1단)  제목·작성자표·초록 / 그림 7개 / 표 2개 / 단 폭을 넘는 수식 2개
  2단        나머지 본문

폭 계산 (US Letter 12240 twip)
  좌우 여백 900 씩  →  본문 10440
  단 5040 × 2 + 단 간격 360 = 10440          (5040 twip = 3.5 in, IEEE 규격)

★ 중간에서 단 수를 바꾸려면 '연속' 구역 나누기를 쓴다. 문단의 pPr 안에 들어간
  sectPr 은 '그 문단까지의 앞 내용'에 적용되고, 본문 맨 끝의 sectPr 이 마지막
  구역을 맡는다.
"""
import re
import shutil
import subprocess
from pathlib import Path

W = "unpacked"

PG_MAR = ('<w:pgMar w:top="1080" w:right="900" w:bottom="1152" w:left="900" '
          'w:header="504" w:footer="504" w:gutter="0"/>')
COLS_1 = '<w:cols w:space="360"/>'
COLS_2 = '<w:cols w:num="2" w:space="360" w:equalWidth="1"/>'

FULL_W, COL_W = 10440, 5040

# 전폭으로 뺄 구간 (블록 번호, 양끝 포함)
FULL_SPANS = [
    (0, 8),      # 제목 · 작성자 표 · 초록
    (12, 13),    # 그림 1
    (48, 49),    # 표 1
    (61, 61),    # 수식 (6)   — 번호까지 넣으면 단 폭을 넘는다 (렌더 실측)
    (79, 79),    # 수식 (12)  — 단 폭의 1.33배
    (82, 82),    # 수식 (13)  — 단 폭의 1.14배
    (95, 96),    # 표 2
    (101, 101),  # 수식 (17)  — 같은 이유 (렌더 실측)
    (122, 123),  # 그림 2
    (125, 126),  # 그림 3
    (130, 131),  # 그림 4
    (135, 136),  # 그림 5
    (138, 139),  # 그림 6
    (144, 145),  # 그림 7
]

# 원 양식이 제목 아래에 넣어둔 빈 문단 — 2단 지면에서는 첫 쪽을 통째로 비운다
DROP = {1, 2, 3, 4}

EQ_MARK = '<w:tab w:val="center" w:pos="4752"/>'

# 그림 치수 상한 (EMU). 균일 축소는 답이 아니다 — 세로로 긴 그림은 쪽을 다 먹고,
# 가로로 납작한 그림은 줄이면 글씨가 안 보인다. 그래서 폭·높이 각각에 상한을 두고
# 둘 중 빡빡한 쪽에 맞춘다. 세로 상한은 본문 높이의 약 45%.
EMU_PT = 12700
MAX_W_EMU = 522 * EMU_PT        # 전폭 10440 twip = 522 pt
MAX_H_EMU = 306 * EMU_PT


def scale_figures(x):
    """<wp:extent> 와 짝이 되는 <a:ext> 를 같은 배율로 줄인다.

    ★ 둘이 어긋나면 Word 가 그림을 잘라낸다. 반드시 쌍으로 고쳐야 한다.
    """
    scales, n = [], 0

    def measure(m):
        cx, cy = int(m.group(2)), int(m.group(3))
        scales.append(min(MAX_W_EMU / cx, MAX_H_EMU / cy, 2.0))
        return m.group(0)

    re.sub(r'(<wp:extent )cx="(\d+)" cy="(\d+)"', measure, x)

    it = iter(scales)
    cur = [None]

    def apply(m):
        nonlocal n
        tag = m.group(1)
        if tag.startswith("<wp:extent"):
            cur[0] = next(it)
        s = cur[0]
        if s is None:
            return m.group(0)
        n += 1
        return f'{tag}cx="{round(int(m.group(2))*s)}" cy="{round(int(m.group(3))*s)}"'

    x = re.sub(r'(<wp:extent |<a:ext )cx="(\d+)" cy="(\d+)"', apply, x)
    return x, len(scales), scales


def sectpr(base, two_col, final):
    """기존 sectPr 을 본떠 새 sectPr 을 만든다. 머리말·꼬리말 참조는 유지한다."""
    s = base
    s = re.sub(r"<w:pgMar[^/]*/>", PG_MAR, s)
    s = re.sub(r"<w:cols[^/]*/>", COLS_2 if two_col else COLS_1, s)
    if not final:  # 연속 구역 — type 은 스키마상 pgSz 앞
        s = s.replace("<w:pgSz", '<w:type w:val="continuous"/><w:pgSz', 1)
    return s


def eq_tabs(blk, width):
    """수식 문단의 가운데·오른쪽 탭을 주어진 폭에 맞춘다."""
    return (blk.replace('<w:tab w:val="center" w:pos="4752"/>',
                        f'<w:tab w:val="center" w:pos="{width // 2}"/>')
               .replace('<w:tab w:val="right" w:pos="9504"/>',
                        f'<w:tab w:val="right" w:pos="{width}"/>'))


def keep_next(blk):
    """그림과 캡션이 쪽을 넘어 갈라지지 않게 붙인다.

    ★ CT_PPr 순서상 keepNext 는 pStyle 바로 뒤, 맨 앞쪽이다.
    """
    if "<w:keepNext/>" in blk:
        return blk
    kn = "<w:keepNext/>"
    if "<w:pPr>" in blk:
        head, rest = blk.split("<w:pPr>", 1)
        ppr, tail = rest.split("</w:pPr>", 1)
        m = re.match(r"(<w:pStyle[^/]*/>)", ppr)
        ppr = (m.group(1) + kn + ppr[m.end():]) if m else kn + ppr
        return head + "<w:pPr>" + ppr + "</w:pPr>" + tail
    return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + kn + "</w:pPr>", blk, count=1)


def justify(blk):
    """2단 본문은 양끝 맞춤이 보기 좋다. 제목·캡션·수식·그림은 건드리지 않는다."""
    if ("<w:pStyle" in blk or "<w:drawing>" in blk or EQ_MARK in blk
            or "<w:jc " in blk or blk.startswith("<w:tbl")):
        return blk
    txt = re.sub(r"<[^>]+>", "", blk).strip()
    if len(txt) < 40 or txt.startswith(("그림", "표 ")):
        return blk
    jc = '<w:jc w:val="both"/>'
    if "<w:pPr>" in blk:
        head, rest = blk.split("<w:pPr>", 1)
        ppr, tail = rest.split("</w:pPr>", 1)
        # CT_PPr 순서상 jc 는 rPr 앞
        ppr = ppr.replace("<w:rPr>", jc + "<w:rPr>", 1) if "<w:rPr>" in ppr else ppr + jc
        return head + "<w:pPr>" + ppr + "</w:pPr>" + tail
    return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + jc + "</w:pPr>", blk, count=1)


# (틀림, 맞음, 예상 건수) — 2단 조판에 맞춘 스타일 축소
STYLE_FIX = [
    # docDefaults
    ('<w:sz w:val="22"/><w:szCs w:val="22"/>',
     '<w:sz w:val="19"/><w:szCs w:val="19"/>', 1),
    ('<w:spacing w:after="200" w:line="276" w:lineRule="auto"/>',
     '<w:spacing w:after="0" w:line="252" w:lineRule="auto"/>', 1),
    # Normal — 본문 줄간격 1.45 → 1.05, 10.5pt → 9.5pt
    ('<w:spacing w:after="140" w:line="348" w:lineRule="auto"/>',
     '<w:spacing w:after="0" w:line="252" w:lineRule="auto"/>', 1),
    ('<w:rFonts w:ascii="맑은 고딕" w:eastAsia="맑은 고딕" w:hAnsi="맑은 고딕"/>'
     '<w:sz w:val="21"/>',
     '<w:rFonts w:ascii="맑은 고딕" w:eastAsia="맑은 고딕" w:hAnsi="맑은 고딕"/>'
     '<w:sz w:val="19"/>', 1),
    # heading 1 (Ⅰ. 서론 …) 15pt → 11.5pt, 앞뒤 여백 축소
    ('<w:spacing w:before="400" w:after="160"/>',
     '<w:spacing w:before="200" w:after="80"/>', 1),
    ('<w:sz w:val="30"/><w:szCs w:val="28"/>',
     '<w:sz w:val="23"/><w:szCs w:val="23"/>', 1),
    # heading 2 (1. 연구의 배경 …) 12pt → 10pt
    ('<w:spacing w:before="280" w:after="100"/>',
     '<w:spacing w:before="150" w:after="60"/>', 1),
    ('<w:sz w:val="24"/><w:szCs w:val="26"/>',
     '<w:sz w:val="20"/><w:szCs w:val="20"/>', 1),
]


def restyle(root):
    """2단 지면에 맞게 본문·제목 크기와 줄간격을 줄인다."""
    p = root / "word" / "styles.xml"
    s = p.read_text(encoding="utf-8")
    for old, new, n_exp in STYLE_FIX:
        n = s.count(old)
        assert n == n_exp, f"스타일 {old[:40]!r} {n}건 (예상 {n_exp})"
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print(f"  스타일 {len(STYLE_FIX)}건 조정")


def main():
    root = Path(W)
    restyle(root)

    doc = root / "word" / "document.xml"
    x = doc.read_text(encoding="utf-8")

    m = re.search(r"^(.*?<w:body>)(.*?)(<w:sectPr\b.*?</w:sectPr>)(\s*</w:body>.*)$",
                  x, re.S)
    head, body, base_sect, tail = m.groups()
    assert head.startswith("<?xml") and "xmlns:w=" in head, "네임스페이스 선언 유실"

    body, n_fig, scales = scale_figures(body)
    print("  그림 배율 " + " · ".join(f"{s:.2f}" for s in scales))

    blocks = re.findall(r"<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>", body, re.S)
    assert "".join(blocks) == body, "문단 분해 손실"

    full = set()
    for a, b in FULL_SPANS:
        full.update(range(a, b + 1))

    # 표지용으로 잡혀 있던 제목 위 여백 1600 twip 을 줄인다
    assert blocks[0].count('<w:spacing w:before="1600"/>') == 1, "제목 문단 여백"
    blocks[0] = blocks[0].replace('<w:spacing w:before="1600"/>',
                                  '<w:spacing w:before="0" w:after="160"/>')

    # 블록별 손질. 버릴 블록은 빼되 단 배치 판정은 원래 번호로 한다.
    kept = []
    for i, b in enumerate(blocks):
        if i in DROP:
            assert not re.sub(r"<[^>]+>", "", b).strip(), f"블록 {i} 는 비어 있지 않다"
            continue
        if EQ_MARK in b:
            b = eq_tabs(b, FULL_W if i in full else COL_W)
        elif i not in full:
            b = justify(b)
        if "<w:drawing>" in b:           # 그림 → 바로 아래 캡션과 한 쪽에
            b = keep_next(b)
        kept.append((i in full, b))

    # 같은 모드끼리 묶어 구역으로 나눈다
    runs, cur, mode = [], [], kept[0][0]
    for is_full, b in kept:
        if is_full != mode:
            runs.append((mode, cur))
            cur, mode = [], is_full
        cur.append(b)
    runs.append((mode, cur))

    pieces = []
    for k, (is_full, blks) in enumerate(runs):
        pieces.extend(blks)
        if k < len(runs) - 1:
            sp = sectpr(base_sect, two_col=not is_full, final=False)
            pieces.append(f"<w:p><w:pPr>{sp}</w:pPr></w:p>")
    final_sect = sectpr(base_sect, two_col=not runs[-1][0], final=True)

    doc.write_text(head + "".join(pieces) + final_sect + tail, encoding="utf-8")
    print(f"  구역 {len(runs)}개 (전폭 {sum(1 for f, _ in runs if f)} · "
          f"2단 {sum(1 for f, _ in runs if not f)})")
    print(f"  전폭 블록 {len(full)}개 · 빈 문단 {len(DROP)}개 삭제 · "
          f"본문 블록 {len(blocks)}개")


if __name__ == "__main__":
    main()
