# 곡률(bend 각도) + 부호축 방향분리 JND — 사용자 지시 §2

널 바닥(재사용, `11_phase2_doseresponse.md`와 동일): JND p95=0.0026

## (a) 곡률 bend 각도 (도, deg)

| 축 | 중앙값 | 최댓값 | 최대 위치 | 참고: 원값 κ(구 곡률) |
|---|---|---|---|---|
| distortion_drive_db | 26.0° | 28.9° | 상위1/3 | mean=0.0186 |
| reverb_wet_level | 19.4° | 62.2° | 하위1/3 | mean=0.0161 |
| reverb_room_size | 27.4° | 31.7° | 상위1/3 | mean=0.0137 |
| reverb_damping | 26.0° | 28.1° | 상위1/3 | mean=0.0037 |
| reverb_width | 8.8° | 10.4° | 하위1/3 | mean=0.0009 |
| highshelf_gain | 11.0° | 18.1° | 하위1/3 | mean=0.0053 |
| lowshelf_gain | 21.4° | 22.7° | 하위1/3 | mean=0.0048 |
| peak_gain | 10.4° | 10.9° | 하위1/3 | mean=0.0042 |
| highshelf_cutoff_gp6 | 12.6° | 14.5° | 하위1/3 | mean=0.0020 |
| lowshelf_cutoff_gp6 | 33.5° | 67.3° | 하위1/3 | mean=0.0046 |
| peak_cutoff_gp6 | 23.8° | 28.2° | 상위1/3 | mean=0.0097 |
| highshelf_q_gp6 | 8.9° | 11.6° | 상위1/3 | mean=0.0018 |
| lowshelf_q_gp6 | 26.1° | 31.0° | 상위1/3 | mean=0.0041 |
| peak_q_gp6 | 10.4° | 12.0° | 상위1/3 | mean=0.0014 |
| eq_cascade_intensity | 19.4° | 20.0° | 상위1/3 | mean=0.0045 |
| null_12k_gain | 9.5° | 12.1° | 하위1/3 | mean=0.0005 |
| null_15k_gain | 11.6° | 14.7° | 하위1/3 | mean=0.0003 |

★ bend가 크면(180°에 가까우면 완전 반전, 90°면 직각으로 꺾임) 손잡이 방향이 구간마다 다르다는 뜻 — 8차의 방향 예측(cos 0.71~0.82)이 전역이 아니라 국소적으로만 유효할 수 있음을 뜻한다. 위 표는 축별 대표 각도(중앙값)와 최악(최댓값) 구간을 함께 보여준다.

## (b) 부호축(*_gain) 방향분리 JND — boost(+)/cut(-) 각각

★ 아래는 원 25레벨 격자(1.25dB 간격)로만 검사한 잠정치였다. §3 정밀 렌더링
완료 후 진짜 값은 [`11_phase2jnd_final.md`](11_phase2jnd_final.md)에 있다 —
**세 축 모두 boost(+) 0.6036dB, cut(-) −0.6036dB**로 동일(실무범위 15dB 대비
4.02%, 우연히 세 축이 같은 로그 이산 구간에서 검출된 것이며 버그 아님,
근거는 해당 문서 참고).

| 축 | boost(+) 첫 JND(원 격자, 잠정) | cut(-) 첫 JND(원 격자, 잠정) |
|---|---|---|
| highshelf_gain | 1.25dB (8.3%) | -1.25dB (8.3%) |
| lowshelf_gain | 1.25dB (8.3%) | -1.25dB (8.3%) |
| peak_gain | 1.25dB (8.3%) | -1.25dB (8.3%) |