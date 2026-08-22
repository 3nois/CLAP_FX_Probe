"""9차 F-2 — 악기-이펙트 조합 필터.

실무에서 흔치 않은 조합(베이스+리버브 등)은 CLAP 학습 데이터에도 드물 것이므로
제외한다.

근거 갱신(11차 Phase 6, 2026-08-22): 9차의 원래 근거("블라인드 청취자의 판단
오염 방지")는 Phase 6가 청취 실험이 아니므로 더 이상 성립하지 않는다. 이 필터를
유지하는 진짜 근거는 주장의 생태적 타당성(ecological validity)이다 — 실무에서
쓰이지 않는 조합(베이스+리버브 등)에서 나온 directional_agreement 수치는 "이
도구가 실무에서 쓸 만한가"라는 질문에 답하지 못한다. 드문 조합은 CLAP 학습
분포에서도 벗어나 있어 결과가 이 도구의 성능이 아니라 분포 밖 거동을 반영할
위험도 있다.

highshelf/lowshelf/peak(벨) EQ는 셋 다 스펙트럼 조작이고 실무에서 트랙(악기
패밀리)을 가리지 않고 범용으로 쓰이므로 전 패밀리를 허용한다.
"""

ALL_FAMILIES = {"bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal"}

EFFECT_ALLOWED_FAMILIES = {
    "reverb": {"brass", "flute", "keyboard", "mallet", "organ", "reed", "string", "vocal"},
    "distortion": {"bass", "guitar", "keyboard", "organ", "reed"},
    "highshelf": set(ALL_FAMILIES),
    "lowshelf": set(ALL_FAMILIES),  # 11차 Phase 6 추가 — 벨/저역 셸프는 실무에서 트랙을 가리지 않음
    "peak": set(ALL_FAMILIES),      # 11차 Phase 6 추가 — 위와 동일 근거
}

EXCLUSION_REASON = {
    ("reverb", "bass"): "저역이 뭉개져 실무에서 잘 안 씀",
    ("reverb", "guitar"): "저역이 뭉개져 실무에서 잘 안 씀",
    ("distortion", "flute"): "관악기 왜곡은 비주류 조합",
    ("distortion", "vocal"): "보컬 왜곡은 특수효과이며 표준 조합 아님",
    ("distortion", "mallet"): "타악기 왜곡은 비주류 조합",
    ("distortion", "string"): "현악기 왜곡(바이올린 등)은 비주류 조합",
    ("distortion", "brass"): "금관 왜곡은 비주류 조합",
}


def is_allowed(effect: str, family: str) -> bool:
    return family in EFFECT_ALLOWED_FAMILIES[effect]


def exclusion_reason(effect: str, family: str) -> str:
    return EXCLUSION_REASON.get((effect, family), "제외 목록에 없음(허용)")
