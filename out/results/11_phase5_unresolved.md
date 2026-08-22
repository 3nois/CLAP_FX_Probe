# Phase 5-F — 미해결분 정리

지시서의 두 항목은 이번 라운드 초반(Phase 2 무결성 검증 후속 조치)에서 이미 처리됐다.
재실행 없음 — 아래는 링크와 요약이다.

## 1. eq_cascade_intensity 타깃 교체

`out/results/11_phase2_cascade_relabel.md` 참고.

- 기존 타깃(s, 소스 무관) vs 신규 타깃(s·‖g_source‖₂, 실효 EQ 강도) 둘 다 산출·병기함
- **결과: 가설과 달리 R²가 개선되지 않았다**(100% 폭 기준 0.0716 → 0.0679, 좁은 윈도우에서는 오히려 음수)
- 원인은 `windowed_r2_general`의 방법론적 특성(소스 간 스케일 분산이 타깃에 추가되며 분모만 커짐)으로 설명, 변위-R² 격차 자체는 미해결로 남김
- 결함 15로 문서화 완료(레이블 식별가능성 결함)

## 2. 부호 축(*_gain) JND 방향 분리

`out/results/11_phase2_bend_signedjnd.md` §(b), `out/results/11_phase2jnd_final.md` 참고.

- 0에서 boost(+)/cut(−) 양방향 분리 재측정 완료(300소스, 로그 20단계 정밀 렌더링)
- 세 EQ 타입(highshelf/lowshelf/peak) 모두 boost(+) 0.6036dB, cut(−) −0.6036dB에서 검출(실무범위 15dB 대비 4.02%)
- 세 축이 정확히 같은 값에서 검출된 것은 로그 격자의 해당 구간이 넓어 우연히 같은 이산 구간에서 검출된 것 — 버그 아님, 근거는 해당 문서에 상세 기술
