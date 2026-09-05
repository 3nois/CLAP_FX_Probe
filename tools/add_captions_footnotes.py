# -*- coding: utf-8 -*-
"""논문 docx 에 인용 각주·캡션·수식번호를 추가하고 오타를 고친다.

입력  개인_포트폴리오_논문_양식.docx (사용자 작성 본문)
출력  개인_포트폴리오_논문_최종.docx

각주는 **참고문헌을 어디에 썼는지 밝히는 인용 각주**다. 설명 각주가 아니다.
각주 번호는 본문 등장 순서를 따라야 하므로 앵커 위치로 정렬한 뒤 id 를 매긴다.
"""
import re
from pathlib import Path

W = "unpacked"

# ── 인용 각주: (본문 앵커, 각주 내용) — 순서는 아래에서 위치로 재정렬 ──────
CITES = [
    ("신디사이저이다",
     "K. Kim, J. Koo, S. Lee, H. Joung, and K. Lee, “TokenSynth: A token-based neural "
     "synthesizer for instrument cloning and text-to-instrument,” in Proc. ICASSP 2025, "
     "Hyderabad, India, Apr. 2025, pp. 1–5."),
    ("Contrastive Language-Audio Pretraining",
     "Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale "
     "contrastive language-audio pretraining with feature fusion and keyword-to-caption "
     "augmentation,” in Proc. ICASSP 2023, Rhodes Island, Greece, Jun. 2023, pp. 1–5."),
    ("조정 상호정보량",
     "N. X. Vinh, J. Epps, and J. Bailey, “Information theoretic measures for clusterings "
     "comparison: Variants, properties, normalization and correction for chance,” Journal "
     "of Machine Learning Research, vol. 11, pp. 2837–2854, 2010."),
    ("Koo et al. 2023",
     "J. Koo, M. A. Martínez-Ramírez, W.-H. Liao, S. Uhlich, K. Lee, and Y. Mitsufuji, "
     "“Music mixing style transfer: A contrastive learning approach to disentangle audio "
     "effects,” in Proc. ICASSP 2023, Rhodes Island, Greece, Jun. 2023, pp. 1–5."),
    ("test split 1200개를 활용한다",
     "J. Engel, C. Resnick, A. Roberts, S. Dieleman, M. Norouzi, D. Eck, and K. Simonyan, "
     "“Neural audio synthesis of musical notes with WaveNet autoencoders,” in Proc. 34th "
     "Int. Conf. Machine Learning (ICML), Sydney, Australia, Aug. 2017, pp. 1068–1077."),
    ("Pedalboard를 사용한다",
     "P. Sobot, “Pedalboard,” Zenodo, 2021, doi: 10.5281/zenodo.7817838."),
    ("FXencoder를 추가하여",
     "J. Koo, M. A. Martínez-Ramírez, W.-H. Liao, S. Uhlich, K. Lee, and Y. Mitsufuji, "
     "앞의 글."),
]

# ── 오타 수정: (틀림, 맞음, 예상 건수) ─────────────────────────────
TYPOS = [("3계역", "3계열", 1), ("유효개수", "유효계수", 2)]

# ── 그림 캡션 (그림 아래) ────────────────────────────────────────
FIG_CAPTIONS = [
    "TokenSynth 구조. 점선은 학습 중 동결되는 모듈이다. 음색은 맨 앞 프리픽스 토큰 하나로만 전달된다.",
    "축별 held-out R² — 좁은 창(하위 1/3)에서 전 범위까지. 붉은 띠는 널 바닥이며, "
    "널 축 두 개는 두 점 모두 띠 안에 있다.",
    "축별 JND(분해능), 실무 범위 대비 백분율. 로그축. 채운 점은 정밀 측정, "
    "빈 점과 화살표는 상한만 확정된 축이다.",
    "Q3 — within·between 코사인과 gap의 95% 신뢰구간. 15축 × 4구간 = 60조합 전부에서 "
    "gap의 하한이 0을 배제한다.",
    "Q4 — B2(파라미터 미지) 조건의 방향 예측과 between 기준선. 5축 × 4구간 = 20조합 전부가 "
    "기준선을 넘는다. 위 축은 각도 환산.",
    "Q5 — 생성 경로의 directional_agreement(왼쪽)와 읽기·쓰기 각도 대조(오른쪽). "
    "붉은 띠는 무작위 널의 95% 범위다.",
    "Q6 — 검색 경로. M1 자기 dry 회수율, M2 실사용 조건의 dry 비율, M3 패밀리 보존율. "
    "세 조건이 모두 충족되어야 성공으로 판정한다.",
]

# ── 표 캡션 (표 위). 머리말 표는 제외 ─────────────────────────────
TBL_CAPTIONS = [
    "오디오 이펙트 세 계열의 수학적 성격과 멜 스펙트로그램에서의 가시성.",
    "축 구성 — 계열별 파라미터 범위와 고정 조건.",
]

# ── 본문에서 실제로 인용되는 참고문헌(원래 번호)만 남긴다 ──────────────
#    [4] DAC · [6] Lakh · [7] MT3 · [8] MusicGen · [9] 악기검색 ·
#    [11] modality gap · [13] CLIP 은 본문에 근거가 없어 삭제한다.
KEEP_REFS = [1, 2, 3, 5, 10, 12]

FN_RPR = ('<w:rPr><w:rFonts w:hint="eastAsia"/><w:vertAlign w:val="superscript"/>'
          '<w:color w:val="000000"/></w:rPr>')
# ★ CT_PPr 는 자식 순서가 스키마로 강제된다 — spacing 이 jc 보다 앞, rPr 이 맨 뒤.
CAP_PPR_FIG = ('<w:pPr><w:spacing w:before="60" w:after="180"/><w:jc w:val="center"/>'
               '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>')
CAP_PPR_TBL = ('<w:pPr><w:spacing w:before="180" w:after="60"/>'
               '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>')
CAP_RPR = '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/><w:color w:val="000000"/></w:rPr>'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def center_para(blk):
    """문단에 w:jc=center 를 넣는다. CT_PPr 순서상 jc 는 rPr 앞에 와야 한다."""
    if '<w:jc w:val="center"/>' in blk:
        return blk
    jc = '<w:jc w:val="center"/>'
    if "<w:pPr>" in blk:
        head, rest = blk.split("<w:pPr>", 1)
        ppr, tail = rest.split("</w:pPr>", 1)
        ppr = re.sub(r'<w:jc w:val="[^"]*"/>', "", ppr)          # 다른 정렬은 제거
        ppr = (ppr.replace("<w:rPr>", jc + "<w:rPr>", 1)
               if "<w:rPr>" in ppr else ppr + jc)
        return head + "<w:pPr>" + ppr + "</w:pPr>" + tail
    return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + jc + "</w:pPr>", blk, count=1)


def center_table(blk):
    """표 자체를 가운데 정렬. CT_TblPr 순서상 jc 는 tblW 바로 뒤."""
    if re.search(r"<w:tblPr>.*?<w:jc ", blk, re.S):
        return blk
    return re.sub(r"(<w:tblPr>.*?<w:tblW[^/]*/>)",
                  r'\1<w:jc w:val="center"/>', blk, count=1, flags=re.S)


# 본문 폭 = pgSz 12240 − 좌우 여백 1368×2 = 9504 twip
TEXT_W = 9504
EQ_TABS = (f'<w:tabs><w:tab w:val="center" w:pos="{TEXT_W//2}"/>'
           f'<w:tab w:val="right" w:pos="{TEXT_W}"/></w:tabs>')


def number_equation(blk, n):
    """별행 수식을 가운데 두고 번호를 오른쪽 끝에 붙인다.

    ★ <m:oMathPara> 뒤에 런을 덧붙이면 Word 는 그 문단을 더 이상 '수식 문단'으로
      보지 않아 centerGroup 정렬이 풀린다(LibreOffice 는 유지해서 렌더로는 안 잡힘).
      그래서 래퍼를 벗기고 가운데 탭 + 오른쪽 탭으로 배치한다. 정렬 속성에 의존하지
      않으므로 Word·LibreOffice 양쪽에서 같은 결과가 나온다.
    """
    inner = re.search(r"<m:oMathPara>(?:<m:oMathParaPr>.*?</m:oMathParaPr>)?(.*)</m:oMathPara>",
                      blk, re.S)
    if not inner:
        return blk
    math = inner.group(1)
    tab = '<w:r><w:tab/></w:r>'
    num = (f'<w:r><w:rPr><w:color w:val="000000"/></w:rPr>'
           f'<w:tab/><w:t>({n})</w:t></w:r>')
    blk = blk[:inner.start()] + tab + math + num + blk[inner.end():]
    # 문단 정렬은 제거하고(탭이 위치를 정한다) 탭 스톱을 심는다
    if "<w:pPr>" in blk:
        head, rest = blk.split("<w:pPr>", 1)
        ppr, tail = rest.split("</w:pPr>", 1)
        ppr = re.sub(r'<w:jc w:val="[^"]*"/>', "", ppr)
        blk = head + "<w:pPr>" + EQ_TABS + ppr + "</w:pPr>" + tail
    else:
        blk = re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + EQ_TABS + "</w:pPr>", blk, count=1)
    return blk


def caption_para(label, text, ppr):
    return (f'<w:p>{ppr}'
            f'<w:r><w:rPr><w:b/><w:sz w:val="18"/><w:szCs w:val="18"/>'
            f'<w:color w:val="000000"/></w:rPr><w:t xml:space="preserve">{esc(label)} </w:t></w:r>'
            f'<w:r>{CAP_RPR}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def main():
    doc_p = Path(W) / "word" / "document.xml"
    fn_p = Path(W) / "word" / "footnotes.xml"
    x = doc_p.read_text(encoding="utf-8")
    f = fn_p.read_text(encoding="utf-8")

    # ── 0. 오타 ───────────────────────────────────────────────
    for wrong, right, n_exp in TYPOS:
        n = x.count(wrong)
        assert n == n_exp, f"오타 {wrong!r} 건수 {n} (예상 {n_exp})"
        x = x.replace(wrong, right)
        print(f"  오타 {wrong} → {right} ({n}건)")

    # ── 1. 인용 각주 — 본문 등장 순서로 번호 부여 ──────────────────
    for a, _ in CITES:
        assert x.count(a) == 1, f"앵커 {a!r} 가 {x.count(a)}건 (1이어야 함)"
    ordered = sorted(CITES, key=lambda c: x.index(c[0]))

    defs = []
    for i, (anchor, body) in enumerate(ordered, start=1):
        ref = (f'</w:t></w:r><w:r>{FN_RPR}<w:footnoteReference w:id="{i}"/></w:r>'
               f'<w:r><w:rPr><w:color w:val="000000"/></w:rPr><w:t xml:space="preserve">')
        x = x.replace(anchor, anchor + ref, 1)
        defs.append(
            f'<w:footnote w:id="{i}"><w:p><w:pPr><w:spacing w:after="0" w:line="240" '
            f'w:lineRule="auto"/><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:vertAlign w:val="superscript"/><w:sz w:val="16"/></w:rPr>'
            f'<w:footnoteRef/></w:r>'
            f'<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
            f'<w:t xml:space="preserve"> {esc(body)}</w:t></w:r></w:p></w:footnote>')
        print(f"  각주 {i}: {anchor[:28]}")
    f = f.replace("</w:footnotes>", "".join(defs) + "</w:footnotes>")

    # ── 2. 그림·표 캡션 ────────────────────────────────────────
    # ★ head 는 XML 선언과 <w:document> 네임스페이스 선언까지 전부 포함해야 한다.
    m = re.search(r"^(.*?<w:body>)(.*)(<w:sectPr\b.*?</w:sectPr>\s*</w:body>.*)$", x, re.S)
    assert m, "body 구간을 찾지 못했다"
    head, body, tail = m.group(1), m.group(2), m.group(3)
    assert head.startswith("<?xml") and "xmlns:w=" in head, "네임스페이스 선언 유실"

    blocks = re.findall(r"<w:p\b[^>]*>.*?</w:p>|<w:p/>|<w:tbl>.*?</w:tbl>", body, re.S)
    assert "".join(blocks) == body, "블록 분해 불일치 — 중단"

    # ── 2-a. 미인용 참고문헌 삭제 + 재번호 ────────────────────────
    def ref_no(blk):
        t = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", blk))
        m = re.match(r"\s*\[(\d+)\]\s", t)
        return int(m.group(1)) if m else None

    found = [n for n in (ref_no(b) for b in blocks) if n]
    assert found == sorted(found) and len(found) == len(set(found)), f"참고문헌 번호 이상: {found}"
    renum = {old: i + 1 for i, old in enumerate(sorted(KEEP_REFS))}
    dropped = [n for n in found if n not in renum]

    kept_blocks = []
    for b in blocks:
        n = ref_no(b)
        if n is None:
            kept_blocks.append(b)
            continue
        if n not in renum:
            continue  # 미인용 — 삭제
        # 첫 <w:t> 의 "[old]" 를 "[new]" 로
        kept_blocks.append(re.sub(r"(<w:t[^>]*>\s*)\[%d\]" % n,
                                  r"\g<1>[%d]" % renum[n], b, count=1))
    blocks = kept_blocks
    print(f"  참고문헌 {len(found)} → {len(renum)}개 "
          f"(삭제 {dropped}) · 재번호 {[f'{o}→{n}' for o, n in renum.items() if o != n]}")

    # ── 2-b. 중앙정렬 — 표 · 그림 · 별행 수식 ────────────────────
    n_c_tbl = n_c_fig = n_c_eq = 0
    for i, b in enumerate(blocks):
        if b.startswith("<w:tbl>"):
            nb = center_table(b)
            n_c_tbl += nb != b
        elif "<w:drawing>" in b:
            nb = center_para(b)
            n_c_fig += nb != b
        elif "<m:oMathPara>" in b:
            continue  # 수식은 번호 붙이는 단계에서 탭으로 배치한다
        else:
            continue
        blocks[i] = nb
    print(f"  중앙정렬 — 표 {n_c_tbl} · 그림 {n_c_fig}")

    out, fig_n, tbl_n = [], 0, 0
    for b in blocks:
        if b.startswith("<w:tbl>"):
            tbl_n += 1
            k = tbl_n - 2  # 1번은 작성자/학번 머리말 표
            if 0 <= k < len(TBL_CAPTIONS):
                out.append(caption_para(f"표 {k+1}.", TBL_CAPTIONS[k], CAP_PPR_TBL))
            out.append(b)
            continue
        out.append(b)
        if "<w:drawing>" in b:
            if fig_n < len(FIG_CAPTIONS):
                out.append(caption_para(f"그림 {fig_n+1}.", FIG_CAPTIONS[fig_n], CAP_PPR_FIG))
            fig_n += 1

    # ── 3. 별행 수식 — 가운데 배치 + 오른쪽 번호 ────────────────
    eq_n = 0
    for i, b in enumerate(out):
        if "<m:oMathPara>" not in b:
            continue
        eq_n += 1
        out[i] = number_equation(b, eq_n)

    doc_p.write_text(head + "".join(out) + tail, encoding="utf-8")
    fn_p.write_text(f, encoding="utf-8")
    print(f"\n인용 각주 {len(ordered)} · 그림 캡션 {fig_n} · 표 캡션 "
          f"{min(max(tbl_n-1,0),len(TBL_CAPTIONS))} · 수식 번호 {eq_n}")


if __name__ == "__main__":
    main()
