# -*- coding: utf-8 -*-
"""이력서(CV) 1쪽을 docx 로 만든다. PDF 는 soffice 로 변환한다.

출력  out/apply/김성현_CV.docx
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

OUT = Path("out/apply")
FONT = "Noto Sans CJK KR"          # 샌드박스에서 PDF 를 뽑을 때 쓰는 한글 폰트
GRAY = RGBColor(0x55, 0x55, 0x55)
LINE = RGBColor(0xAA, 0xAA, 0xAA)
TEXT_W = Cm(17.0)                   # A4 폭 21 - 좌우 여백 2 씩


def setup(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(2.0)
    s.top_margin = Cm(1.5)
    s.bottom_margin = Cm(1.2)
    n = doc.styles["Normal"]
    n.font.name = FONT
    n.font.size = Pt(9)
    # ★ 한글은 eastAsia 속성을 따로 지정해야 폰트가 적용된다.
    n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = n.paragraph_format
    pf.space_before, pf.space_after = Pt(0), Pt(0)
    pf.line_spacing = 1.14


def para(doc, space_before=0, space_after=0, align=None, indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    if indent is not None:
        p.paragraph_format.left_indent = Cm(indent)
    return p


def run(p, text, size=9, bold=False, color=None, space=0):
    r = p.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(size)
    r.bold = bold
    if color is not None:
        r.font.color.rgb = color
    if space:
        r.font.element.get_or_add_rPr().append(
            _spacing(space))
    r.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    return r


def _spacing(twips):
    from docx.oxml import OxmlElement
    e = OxmlElement("w:spacing")
    e.set(qn("w:val"), str(twips))
    return e


def rule(doc, before=4, after=2):
    """제목 아래 가로선. 문단 하단 테두리로 그린다."""
    from docx.oxml import OxmlElement
    p = para(doc, space_before=before, space_after=after)
    pPr = p._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "AAAAAA")
    bd.append(bottom)
    pPr.append(bd)
    return p


def section(doc, title):
    p = para(doc, space_before=6, space_after=0)
    run(p, title, size=10, bold=True, space=24)
    rule(doc, before=0, after=3)


def entry(doc, period, title, bold_title=True):
    """왼쪽 기간 + 오른쪽 내용. 탭 정렬로 열을 맞춘다."""
    p = para(doc, space_before=2.5)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(3.6), WD_TAB_ALIGNMENT.LEFT)
    p.paragraph_format.left_indent = Cm(3.6)
    p.paragraph_format.first_line_indent = Cm(-3.6)
    run(p, period, size=8.5, color=GRAY)
    p.add_run("\t")
    run(p, title, bold=bold_title)
    return p


def bullet(doc, text):
    p = para(doc, space_before=1, indent=3.6)
    p.paragraph_format.first_line_indent = Cm(-0.38)
    run(p, "· ", size=9, color=GRAY)
    run(p, text, size=8.5)
    return p


def main():
    doc = Document()
    setup(doc)

    # ── 머리말 ───────────────────────────────────────────────
    p = para(doc)
    run(p, "김 성 현", size=17, bold=True, space=26)
    p = para(doc, space_before=4)
    run(p, "서울대학교 첨단융합학부 융합데이터과학전공  ·  학번 2025-16318", size=9)
    p = para(doc, space_before=2)
    run(p, "ryan0823@snu.ac.kr  ·  010-8888-1183  ·  github.com/3nois",
        size=9, color=GRAY)
    p = para(doc, space_before=5)
    run(p, "관심 분야  ", size=9, bold=True)
    run(p, "오디오 신호처리와 기계학습이 만나는 지점 — 음색 표현 학습, "
           "제어 가능한 음악·음향 생성, 오디오 이펙트 모델링", size=9)
    rule(doc, before=5, after=0)

    # ── 학력 ────────────────────────────────────────────────
    section(doc, "학력")
    entry(doc, "2025.03 ~ 현재",
          "서울대학교 첨단융합학부 융합데이터과학전공  (학사과정 2학년 재학)")
    entry(doc, "~ 2025.02", "광주과학고등학교 졸업")

    # ── 연구 경험 ────────────────────────────────────────────
    section(doc, "연구 경험")
    entry(doc, "2026.07 ~ 2026.09",
          "CLAP 음색 임베딩의 오디오 이펙트 정보 분석  (개인 연구)")
    bullet(doc, "TokenSynth(ICASSP 2025)가 음색 지표 저하의 원인으로 추정했으나 "
                "측정하지는 않은 진술을 대상으로, CLAP 임베딩이 이펙트 정보를 담는지와 "
                "그 정보로 이펙트를 제어할 수 있는지를 여섯 질문으로 나누어 측정")
    bullet(doc, "NSynth 1,200개 소스 × 이펙트 3계열 23축 × 25레벨, 570,000회 렌더링. "
                "릿지 프로브·조정 상호정보량·방향 예측·오디오 재생성·임베딩 검색")
    bullet(doc, "이펙트 정보는 존재하나 악기 정체성보다 4~5배 약함. 조작 방향은 소스마다 "
                "고유하며 임베딩만으로 예측되나(cos 0.71~0.82), 생성 경로를 지나면 방향 "
                "일치도가 84.5°~88.1°로 무너짐(무작위 널 89.97°)")
    bullet(doc, "손실 지점을 projection 층의 국소 유효계수 95.2/512, 프리픽스 토큰 "
                "하나를 공유하는 조건 채널 경쟁, 자기회귀 샘플링 잔차 57.4%로 특정. "
                "재학습 없는 개입으로 1.77배 회복")
    bullet(doc, "세 장애가 모두 생성기 내부에 있다는 점에 착안해 생성기를 우회하는 "
                "검색 과제로 같은 방향 정보를 보냈고, 검증한 6조건 전부에서 회수율이 "
                "개선됨(R@10 0.96~1.00, 악기 패밀리 보존율 0.93~0.98)")
    bullet(doc, "MacBook M5 CPU 단독 수행 · github.com/3nois/CLAP_FX_Probe")

    # ── 논문 ────────────────────────────────────────────────
    section(doc, "논문")
    p = entry(doc, "2023.12",
              "스마트 홈 환경을 위한 고해상도 Tactile 센서와 딥러닝을 사용한 "
              "사람/신체 인식")
    bullet(doc, "김성현, 설채환. 한국정보과학회 2023 한국소프트웨어종합학술대회"
                "(KSC 2023) 논문집, pp. 1987–1989.  제1저자")
    bullet(doc, "촉각 센서 배열과 딥러닝으로 카메라 없이 사람·자세를 인식하는 "
                "Ambient AI 구성을 제안")

    # ── 기술 ────────────────────────────────────────────────
    section(doc, "기술")
    entry(doc, "언어 · 도구",
          "Python, PyTorch, scikit-learn, NumPy/SciPy, librosa, pedalboard, Git",
          bold_title=False)
    entry(doc, "방법",
          "표현 프로빙, 대조학습 임베딩 분석, 생성 모델 조건화, 부트스트랩 "
          "신뢰구간·다중비교 보정, 사전등록 기반 실험 설계",
          bold_title=False)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "김성현_CV.docx"
    doc.save(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
