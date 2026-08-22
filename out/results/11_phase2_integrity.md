# Phase 2 무결성 검증

검증 스크립트: `11_phase2_integrity.py`

## 1~2. 축 파일 존재 + shape

| 축 | 존재 | shape | 판정 |
|---|---|---|---|
| distortion_drive_db | O | (400, 25, 512) | OK |
| reverb_wet_level | O | (400, 25, 512) | OK |
| reverb_room_size | O | (400, 25, 512) | OK |
| reverb_damping | O | (400, 25, 512) | OK |
| reverb_width | O | (400, 25, 512) | OK |
| highshelf_gain | O | (400, 25, 512) | OK |
| highshelf_cutoff_gp6 | O | (400, 25, 512) | OK |
| highshelf_q_gp6 | O | (400, 25, 512) | OK |
| highshelf_cutoff_gn6 | O | (400, 25, 512) | OK |
| highshelf_q_gn6 | O | (400, 25, 512) | OK |
| lowshelf_gain | O | (400, 25, 512) | OK |
| lowshelf_cutoff_gp6 | O | (400, 25, 512) | OK |
| lowshelf_q_gp6 | O | (400, 25, 512) | OK |
| lowshelf_cutoff_gn6 | O | (400, 25, 512) | OK |
| lowshelf_q_gn6 | O | (400, 25, 512) | OK |
| peak_gain | O | (400, 25, 512) | OK |
| peak_cutoff_gp6 | O | (400, 25, 512) | OK |
| peak_q_gp6 | O | (400, 25, 512) | OK |
| peak_cutoff_gn6 | O | (400, 25, 512) | OK |
| peak_q_gn6 | O | (400, 25, 512) | OK |
| eq_cascade_intensity | O | (400, 25, 512) | OK |
| null_12k_gain | O | (400, 25, 512) | OK |
| null_15k_gain | O | (400, 25, 512) | OK |

총 23/23개 축 확인.

## 3. bypass 앵커

행 수 = 400 — OK

## 4. neutral check — cos(e_bypass, e(theta_min))

| 축 | theta_min | min cos | mean cos | 기대 | 판정 |
|---|---|---|---|---|---|
| highshelf_gain | 0 | 0.999998 | 1.000000 | >0.9999 (진짜 dry) | OK |
| lowshelf_gain | 0 | 0.999198 | 0.999986 | >0.9999 (진짜 dry) | **★ FAIL** |
| peak_gain | 0 | 0.999870 | 1.000000 | >0.9999 (진짜 dry) | **★ FAIL** |
| eq_cascade_intensity | 0 | 0.999137 | 0.999985 | >0.9999 (진짜 dry) | **★ FAIL** |
| distortion_drive_db | 0 | 0.811765 | 0.964188 | 0.90~0.98 (insertion cost, 실패 아님) | OK |
| reverb_wet_level | 0 | 0.962922 | 0.986046 | 0.90~0.98 (insertion cost, 실패 아님) | OK |
| reverb_room_size | 0.05 | 0.755271 | 0.955546 | 0.90~0.98 (insertion cost, 실패 아님) | OK |
| reverb_damping | 0 | 0.636340 | 0.921126 | 0.90~0.98 (insertion cost, 실패 아님) | OK |
| reverb_width | 0 | 0.728480 | 0.940294 | 0.90~0.98 (insertion cost, 실패 아님) | OK |
| null_12k_gain | 0 | 1.000000 | 1.000000 | >0.9999 (진짜 dry) | OK |
| null_15k_gain | 0 | 1.000000 | 1.000000 | >0.9999 (진짜 dry) | OK |

**★★★ EQ gain 축이 0.9999를 넘지 못했다 — 즉시 중단하고 보고할 것 (지시 §A.4).**

## 5. 게이트 축 vs 널 축 — 끝점 간 변위 비교

게이트 축(gp6, gain=+6dB 고정에서 cutoff/q 스윕)의 끝점 변위가 널 축(초음파, 무효과 기대)의 끝점 변위와 겹치면 게이트 고정이 실패한 것이다.

| 축 | mean displacement(끝점) | 95% CI |
|---|---|---|
| **null(12k+15k 통합, 바닥선)** | 0.000188 | [0.000174, 0.000203] |
| highshelf_cutoff_gp6 | 0.008371 | [0.007985, 0.008794]  |
| highshelf_q_gp6 | 0.019059 | [0.017577, 0.020634]  |
| lowshelf_cutoff_gp6 | 0.004866 | [0.004390, 0.005368]  |
| lowshelf_q_gp6 | 0.008549 | [0.007800, 0.009353]  |
| peak_cutoff_gp6 | 0.009803 | [0.009365, 0.010240]  |
| peak_q_gp6 | 0.009010 | [0.008662, 0.009356]  |


## 종합 판정: **★ FAIL — 사람 보고 필요**


---

# Phase 2 무결성 검증 (v2 — §4 기준 정정)

v1(`11_phase2_integrity.md`)에서 쓴 `cos > 0.9999` 절대 기준은 IIR biquad에 부적절하다는 지적을 반영해 판정 기준을 교체했다. 원본 v1 기록은 보존하고 본 문서가 §4 판정을 대체한다.

## 널 바닥 (기준)

null_12k_gain + null_15k_gain, 25레벨 전체 풀링 (N=20000): **95백분위 = 1.003e-04**

## §4 정정 — 축별 insertion_cost / neutral_offset

| 축 | theta_min | insertion_cost(min cos) | insertion_cost(mean cos) | neutral_offset(median d) | neutral_offset(p95 d) | 널 기준 판정 |
|---|---|---|---|---|---|---|
| distortion_drive_db | 0 | 0.811765 | 0.964188 | 1.667e-02 | 1.343e-01 | 해당없음(실효과, insertion cost) |
| reverb_wet_level | 0 | 0.962922 | 0.986046 | 1.318e-02 | 2.381e-02 | 해당없음(실효과, insertion cost) |
| reverb_room_size | 0.05 | 0.755271 | 0.955546 | 3.891e-02 | 1.060e-01 | 해당없음(실효과, insertion cost) |
| reverb_damping | 0 | 0.636340 | 0.921126 | 6.432e-02 | 1.930e-01 | 해당없음(실효과, insertion cost) |
| reverb_width | 0 | 0.728480 | 0.940294 | 5.002e-02 | 1.486e-01 | 해당없음(실효과, insertion cost) |
| highshelf_gain | 0 | 0.999998 | 1.000000 | 0.000e+00 | 5.960e-08 | PASS(널 이하) |
| highshelf_cutoff_gp6 | 500 | 0.960636 | 0.989165 | 9.848e-03 | 1.983e-02 | 해당없음(실효과, insertion cost) |
| highshelf_q_gp6 | 0.1 | 0.991420 | 0.997016 | 2.793e-03 | 5.143e-03 | 해당없음(실효과, insertion cost) |
| highshelf_cutoff_gn6 | 500 | 0.968724 | 0.989674 | 9.705e-03 | 1.886e-02 | 해당없음(실효과, insertion cost) |
| highshelf_q_gn6 | 0.1 | 0.992352 | 0.997078 | 2.714e-03 | 5.214e-03 | 해당없음(실효과, insertion cost) |
| lowshelf_gain | 0 | 0.999198 | 0.999986 | 0.000e+00 | 7.116e-05 | PASS(널 이하) |
| lowshelf_cutoff_gp6 | 30 | 0.988302 | 0.999412 | 5.034e-05 | 3.349e-03 | 해당없음(실효과, insertion cost) |
| lowshelf_q_gp6 | 0.1 | 0.989114 | 0.997560 | 2.082e-03 | 5.736e-03 | 해당없음(실효과, insertion cost) |
| lowshelf_cutoff_gn6 | 30 | 0.990758 | 0.999587 | 3.082e-05 | 2.555e-03 | 해당없음(실효과, insertion cost) |
| lowshelf_q_gn6 | 0.1 | 0.991869 | 0.997742 | 1.875e-03 | 5.412e-03 | 해당없음(실효과, insertion cost) |
| peak_gain | 0 | 0.999870 | 1.000000 | 0.000e+00 | 1.192e-07 | PASS(널 이하) |
| peak_cutoff_gp6 | 200 | 0.977361 | 0.994912 | 4.225e-03 | 1.244e-02 | 해당없음(실효과, insertion cost) |
| peak_q_gp6 | 0.1 | 0.969670 | 0.988976 | 1.015e-02 | 1.897e-02 | 해당없음(실효과, insertion cost) |
| peak_cutoff_gn6 | 200 | 0.979918 | 0.995795 | 3.382e-03 | 1.113e-02 | 해당없음(실효과, insertion cost) |
| peak_q_gn6 | 0.1 | 0.970279 | 0.989467 | 9.778e-03 | 1.859e-02 | 해당없음(실효과, insertion cost) |
| eq_cascade_intensity | 0 | 0.999137 | 0.999985 | 0.000e+00 | 7.954e-05 | PASS(널 이하) |
| null_12k_gain | 0 | 1.000000 | 1.000000 | 0.000e+00 | 5.960e-08 | PASS(널 이하) |
| null_15k_gain | 0 | 1.000000 | 1.000000 | 0.000e+00 | 5.960e-08 | PASS(널 이하) |

**전 축 새 기준 PASS.**

## 확인 1 — theta≈0 근방 오프셋이 평탄한가, |gain| 따라 벌어지는가

- **lowshelf_gain**: theta=0에서 mean_d=1.419e-05 (곡선 전체 최솟값 근방), 인접 3점 [0.000113, 1.4e-05, 0.000131], 먼 지점들 범위 [1.428e-03, 1.568e-02] — theta=0을 중심으로 양방향 모두 |theta| 증가에 따라 **단조·대칭적으로 증가**(발산 아닌 매끄러운 용량-반응 곡선의 최솟값 = 진짜 dry 지점). 오프셋이 theta=0 근방에서만 평평하게 떠 있다가 갑자기 벌어지는 패턴은 없음.
- **peak_gain**: theta=0에서 mean_d=3.576e-07 (곡선 전체 최솟값 근방), 인접 3점 [0.000297, 0.0, 0.000303], 먼 지점들 범위 [4.481e-03, 3.831e-02] — theta=0을 중심으로 양방향 모두 |theta| 증가에 따라 **단조·대칭적으로 증가**(발산 아닌 매끄러운 용량-반응 곡선의 최솟값 = 진짜 dry 지점). 오프셋이 theta=0 근방에서만 평평하게 떠 있다가 갑자기 벌어지는 패턴은 없음.
- **eq_cascade_intensity**: theta=0에서 mean_d=1.508e-05 (곡선 전체 최솟값 근방), 인접 3점 [1.5e-05, 1.5e-05, 9.8e-05], 먼 지점들 범위 [1.348e-03, 3.896e-02] — theta=0을 중심으로 양방향 모두 |theta| 증가에 따라 **단조·대칭적으로 증가**(발산 아닌 매끄러운 용량-반응 곡선의 최솟값 = 진짜 dry 지점). 오프셋이 theta=0 근방에서만 평평하게 떠 있다가 갑자기 벌어지는 패턴은 없음.

**판정: 평탄한 상수 오프셋 — theta 의존적으로 벌어지는 패턴 아님. 진행 조건 충족.**

## 확인 2 — lowshelf_gain 문제 소스의 저역(<=200Hz) 에너지 상관

문제 소스(cos<0.9999) N=17/400.

- 문제군 저역 RMS: median=0.00778, mean=0.00946
- 나머지 383개 저역 RMS: median=0.05199, mean=0.07364
- Mann-Whitney U, 단측(문제군 < 나머지): **p=3.46e-09**
- 문제 소스 패밀리 분포: {'bass': 1, 'guitar': 6, 'keyboard': 2, 'mallet': 6, 'organ': 1, 'reed': 1}

**결과: 가설(저역 에너지가 높은 소스일수록 영향받는다)과 반대 방향의 유의한 상관이 나왔다.** 문제 소스는 저역 에너지가 오히려 하위 20백분위 이내로 낮다(주로 guitar/mallet — 발현 후 급격히 감쇠하는 발현형 악기, 지속되는 저음이 아니라 짧은 트랜지언트). 해석: 저역 셸프 필터의 고정된 절대 크기 수치 잔차가, 저역 에너지 자체가 원래 작은 소스에서는 상대적으로 더 큰 스펙트럴 왜곡으로 작용해 CLAP이 감지했을 가능성 — 그러나 이는 사후 추정이며 기전이 완전히 확인된 것은 아니다. 원 가설(저역 에너지 高 → 영향 大)은 **기각**하고, 실제로는 반대 방향의 상관관계를 관측 사실로 남긴다.

## 결함 14 (신규)

> IIR biquad는 gain=0에서도 구현체가 항등이 아니다 — 저주파 셸프에서 CLAP 코사인 0.9992까지 벗어난다. Phase 0.5의 6소스 스모크 테스트로는 잡히지 않았다(표본 부족). 400소스 규모에서 처음 드러났다. 널 바닥(1.0e-4) 대비로는 무해한 수준(§4 정정 기준 전 축 PASS)이며, 발현형/트랜지언트 위주 악기에서 상대적으로 더 크게 나타나는 경향이 있다(확인 2, 방향은 가설과 반대).

## 종합 판정

**PASS (정정된 기준) — B → D → C 진행.**
