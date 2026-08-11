"""9차 F-2 — 악기-이펙트 조합 필터.

실무에서 흔치 않은 조합(베이스+리버브 등)은 지각적으로 어색하고 CLAP 학습
데이터에도 드물 것이므로 제외한다. highshelf는 스펙트럼 조작이라 악기를
가리지 않으므로 전 패밀리를 그대로 둔다.
"""

ALL_FAMILIES = {"bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal"}

EFFECT_ALLOWED_FAMILIES = {
    "reverb": {"brass", "flute", "keyboard", "mallet", "organ", "reed", "string", "vocal"},
    "distortion": {"bass", "guitar", "keyboard", "organ", "reed"},
    "highshelf": set(ALL_FAMILIES),
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
