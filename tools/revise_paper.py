# -*- coding: utf-8 -*-
"""검토 지적 9건을 논문 docx 본문에 반영한다.

입력  개인_포트폴리오_논문_최종.docx  (add_captions_footnotes.py 의 산출물)
출력  같은 파일을 덮어쓴다

지적 내용
  1 렌더링 횟수    575000 → 570000  (캐시 실측: 17축×1200 + 6축×400, 각 25레벨)
  2 Q5 서술        "유의하지 않다" → "방향성은 관찰되나 효과가 작다"
  3 통계 정보      표본 단위·부트스트랩·CI·다중비교 보정을 실험 설계에 명시
  4 누수 여부      Q1·Q2 가 source-grouped split 임을 크게 적음
  5 결론 강도      Q6 "모든 조건" → "검증한 조건"
  6 인용 체계      참고문헌 번호를 각주 등장 순서에 맞게 재정렬
  7 순환성 한계    같은 CLAP 으로 채점하는 구조를 한계에 추가
  8 제목          "정보 분석:" · "TokenSynth 성능 향상"
  9 멜 설명        저역 강화 → 주파수 축의 비선형 재배치

★ 본문 치환은 전부 단일 <w:t> 안에서 일어나는 것만 골랐다(사전 확인 완료).
  런 경계를 넘는 문자열은 건드리지 않는다.
"""
import re
from pathlib import Path

W = "unpacked"

# ── 단순 치환: (틀림, 맞음, 예상 건수) ────────────────────────────
REPLACES = [
    # 8 제목
    ("CLAP 음색 임베딩의 오디오 이펙트 정보: ",
     "CLAP 음색 임베딩의 오디오 이펙트 정보 분석: ", 1),
    ("의 성능 향상을 중심으로", " 성능 향상을 중심으로", 1),

    # 1 렌더링 횟수 — 초록과 실험 설계 두 곳
    ("575000회 렌더링", "570000회 렌더링", 2),

    # 9 멜 설명
    ("즉 사람 귀에 맞도록 저역을 더 강화시킨 것이 멜 스펙트로그램이다.",
     "이를 사람의 주파수 지각에 맞게 주파수 축을 비선형적으로 재배치한 것이 "
     "멜(Mel) 스펙트로그램이다.", 1),

    # 2 초록의 Q5 서술
    ("그러나 그 방향으로 조건 벡터를 이동시켜 오디오를 생성하면 유의하지 않은 결과가 "
     "나온다.",
     "그러나 그 방향으로 조건 벡터를 이동시켜 오디오를 생성하면, 방향성은 통계적으로 "
     "관찰되나(20조합 중 19개가 신뢰구간으로 0을 배제, Bonferroni 보정 후 16개) "
     "효과 크기가 작아 실용 수준의 제어에는 이르지 못한다.", 1),

    # 5 결론 강도 — 초록
    ("회수율이 6가지 조건에서 유의하게 개선되었으며 악기 패밀리 보존율은 하락하지 "
     "않았다.",
     "천장 효과가 없는 2축 3레벨을 검증했고 그 6조건 전부에서 회수율이 개선되었으며"
     "(Bonferroni 보정 후 5조건), 악기 패밀리 보존율은 하락하지 않았다.", 1),

    # 5 결론 강도 — 결론
    ("즉 검색으로 같은 방향 정보를 흘렸더니 모든 조건에서 유의하게 문제가 개선되었다.",
     "즉 검색으로 같은 방향 정보를 흘렸더니 검증한 조건 전부에서 문제가 개선되었다. "
     "다만 이는 천장 효과가 없는 2축 3레벨에 한정된 결과이며, 실사용 조건(M2)에서는 "
     "distortion 이 물리 지표 4개 중 3개, reverb 가 최대 강도 구간에서만 4개 전부의 "
     "지지를 받는다.", 1),

    # 2 Q5 결과 본문
    (" 20개의 조합 중 19개가 ",
     " 표본 단위는 소스이고 신뢰구간은 소스 부트스트랩 2000회의 백분위 95% 구간이다. "
     "20개의 조합 중 19개가 ", 1),
]

# ── 새 문단 삽입: (이 문장이 든 문단 뒤에, 문단 목록) ──────────────
PARA = ('<w:p><w:pPr><w:ind w:firstLineChars="50" w:firstLine="105"/>'
        '<w:rPr><w:rFonts w:hint="eastAsia"/><w:lang w:eastAsia="ko-KR"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:hint="eastAsia"/><w:lang w:eastAsia="ko-KR"/></w:rPr>'
        '<w:t xml:space="preserve"> {}</w:t></w:r></w:p>')

INSERTS = [
    # 3·4 통계 규약과 누수 차단 — 실험 설계 마지막(Ror 문단) 뒤
    # ★ "Ror. 실제 보정 방향" 은 런 경계로 쪼개져 있어 뒷부분만 앵커로 쓴다.
    ("실제 보정 방향", [
        "모든 질문에서 표본 단위는 레벨이 아닌 소스다. 한 소스의 25레벨은 같은 원음에서 "
        "나온 관측이라 독립 표본이 아니기 때문이다. 따라서 Q1의 릿지 프로브와 Q2의 분류·"
        "회귀는 sklearn 의 GroupShuffleSplit 에 groups = src_id 를 넘겨 소스 단위로 "
        "분할했다. 한 소스의 25레벨은 전부 학습이거나 전부 시험이며 양쪽에 걸치지 "
        "않는다. 이 처리가 없으면 프로브가 이펙트가 아니라 음색을 외워 맞힐 수 있어 "
        "성능이 과대평가된다. Q4~Q6의 패밀리 층화 분할도 같은 src_id 목록 위에서 "
        "이루어지며, Q6는 라이브러리의 바이트 중복 20곡을 묶어 group-aware 로 "
        "회수·제외한다. 다만 이 중복 처리는 Q1·Q2의 프로브 분할에는 적용하지 않았다.",

        "불확실도는 전부 소스 단위 부트스트랩의 백분위 95% 신뢰구간이다. 반복 수는 "
        "Q3이 300회, Q5가 2000회, Q6가 1000회이며 시드는 0으로 고정했다. Q1은 "
        "GroupShuffleSplit 3겹(시험 20%), Q2는 5겹(시험 30%), Q4는 패밀리 층화 "
        "80/10/10, Q6는 60/20/20이다. 초매개변수는 검증 분할에서만 고르고 시험 분할은 "
        "1회만 평가한다.",

        "각 질문은 여러 축과 구간을 동시에 보므로 다중 비교가 발생한다. 원 분석은 "
        "조합별 신뢰구간만 냈기에, 질문 단위 Bonferroni 보정을 사후에 계산해 병기한다. "
        "Q3은 60조합 전부, Q4는 20조합 전부가 보정 후에도 유지되나, Q5는 19개에서 "
        "16개로, Q6의 회수율은 6조건에서 5조건으로 줄어든다. 판정이 바뀌는 두 곳은 "
        "결과에 함께 적었다.",
    ]),

    # 7 순환성 한계 — 한계 마지막 문단 뒤
    ("더 잘게 재지 못했다", [
        "Q5의 평가 구조에도 한계가 있다. 생성한 오디오를 다시 같은 CLAP 으로 인코딩해 "
        "방향 일치도를 재기 때문에, 방향을 검출한 모델과 그 반영 여부를 채점하는 모델이 "
        "동일하다. 완전히 독립적인 검증이 아니며 양방향으로 치우칠 수 있다 — CLAP 이 "
        "보는 축에서만 점수가 나므로 과대평가될 수도, CLAP 이 압축한 축이라 바닥이 낮아 "
        "과소평가될 수도 있다. 각도의 절대 수준을 논할 때 특히 문제가 된다. Q6의 M2 "
        "에서 CLAP 밖의 물리 지표 8종을 쓴 것이 부분적 대응이나 Q5에는 같은 대응을 "
        "적용하지 않았으므로, 외부 음향 지표와 블라인드 청취 평가가 추가로 필요하다.",
    ]),
]

# ── 6 참고문헌 재정렬: 각주 등장 순서(Kim·Wu·Vinh·Engel·Sobot·Koo) ──
REF_ORDER = ["Kim", "Wu", "Vinh", "Engel", "Sobot", "Koo"]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def reorder_refs(blocks):
    """참고문헌 문단들을 각주 순서로 재배열하고 [n] 을 다시 매긴다."""
    idx = [i for i, b in enumerate(blocks)
           if re.match(r"^\[\d+\]", re.sub(r"<[^>]+>", "", b).strip())]
    assert idx, "참고문헌 문단을 찾지 못했다"
    assert idx == list(range(idx[0], idx[0] + len(idx))), "참고문헌 문단이 연속이 아니다"
    assert len(idx) == len(REF_ORDER), f"참고문헌 {len(idx)}개 (예상 {len(REF_ORDER)})"

    def author(b):
        txt = re.sub(r"<[^>]+>", "", b)
        for a in REF_ORDER:
            if f"{a}," in txt or f"{a} " in txt:
                return a
        raise AssertionError(f"저자 판별 실패: {txt[:60]}")

    cur = {author(blocks[i]): blocks[i] for i in idx}
    assert set(cur) == set(REF_ORDER), sorted(cur)

    out = []
    for new_n, a in enumerate(REF_ORDER, 1):
        b = cur[a]
        # 첫 <w:t> 의 선두 [n] 만 새 번호로 바꾼다
        b = re.sub(r"(<w:t[^>]*>)\s*\[\d+\]", rf"\g<1>[{new_n}]", b, count=1)
        out.append(b)
    blocks[idx[0]:idx[0] + len(idx)] = out
    return blocks, [author(b) for b in out]


def main():
    p = Path(W) / "word" / "document.xml"
    x = p.read_text(encoding="utf-8")

    for old, new, n_exp in REPLACES:
        n = x.count(old)
        assert n == n_exp, f"{old[:40]!r} {n}건 (예상 {n_exp})"
        x = x.replace(old, esc(new) if "<" in new else new)
    print(f"  본문 치환 {len(REPLACES)}건")

    # 앞뒤 공백이 생긴 <w:t> 는 xml:space="preserve" 가 없으면 Word 가 공백을 버린다.
    def fix_space(m):
        attrs, text = m.group(1) or "", m.group(2)
        if text != text.strip() and "xml:space" not in attrs:
            attrs += ' xml:space="preserve"'
        return f"<w:t{attrs}>{text}</w:t>"

    # ★ <w:t(\s…)?> 로 묶어야 한다. [^>]* 로 두면 <w:tc>·<w:tbl> 까지 잡힌다.
    x, n_sp = re.subn(r"<w:t(\s[^>]*)?>((?:(?!</w:t>).)*)</w:t>", fix_space, x, flags=re.S)
    print(f"  xml:space 점검 {n_sp}개 run")

    # ★ head 는 XML 선언과 <w:document> 네임스페이스 선언까지 전부 포함해야 한다.
    m = re.search(r"^(.*?<w:body>)(.*)(<w:sectPr\b.*?</w:sectPr>\s*</w:body>.*)$", x, re.S)
    head, body, tail = m.groups()
    assert head.startswith("<?xml") and "xmlns:w=" in head, "네임스페이스 선언 유실"

    blocks = re.findall(r"<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>", body, re.S)
    assert "".join(blocks) == body, "문단 분해 손실"

    n_ins = 0
    for anchor, paras in INSERTS:
        hit = [i for i, b in enumerate(blocks) if anchor in b]
        assert len(hit) == 1, f"앵커 {anchor!r} {len(hit)}건"
        i = hit[0]
        blocks[i + 1:i + 1] = [PARA.format(esc(t)) for t in paras]
        n_ins += len(paras)
    print(f"  문단 삽입 {n_ins}개")

    blocks, order = reorder_refs(blocks)
    print(f"  참고문헌 재정렬 → {' · '.join(order)}")

    p.write_text(head + "".join(blocks) + tail, encoding="utf-8")


if __name__ == "__main__":
    main()
