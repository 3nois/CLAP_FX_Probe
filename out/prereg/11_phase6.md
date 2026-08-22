# Phase 6 사전 등록 — 오디오 검증 (Q5)

날짜: 2026-08-19.

## 6-A. 재사용 확인

`tokensynth_bridge/inject.py`(`synthesize_from_embedding`), `midi_gen.py`
(`generate_midi_variants_for_source`), `phase_f2_filter.py`(`is_allowed`),
`phase_f4_full.py`의 소스 선정·분산분해(`two_way_anova_ss`)·부트스트랩
(`bootstrap_ci_by_source`) 로직을 import로 재사용한다. 재구현 없음. `TokenSynth`,
`CLAP`, `DACDecoder`는 `tokensynth_bridge/tokensynth` 패키지에서 그대로 로드 확인함
(import 테스트 통과).

Phase 2 임베딩(`out/caches/11_phase2_<axis>.npz`)은 `inject.py`의
`our_load_and_preprocess`(48kHz, 4.0초, 피크 0.7 정규화)와 **동일 전처리**로 만들어져
있어 별도 변환 없이 `synthesize_from_embedding`에 직접 주입 가능함을 확인했다.

## 6-B. 무엇을 넘기는가 — 정확한 범위 확정

5-C에서 B2가 between 기준선을 넘은 (축,구간) 조합: **20개 전부**(5축×4구간
[전범위/하/중/상] — 인접구간 3개는 5-C에서 B2 자체를 계산 안 했으므로 애초에 대상
아님). Branch B이므로 5-D에서 context를 확인한 4개 축쌍(`{highshelf,lowshelf,
peak}_gain`×cutoff, `reverb_room_size`×wet_level)은 전범위에 한해 context 조건도
추가한다.

## 6-C. 측정 (지시 그대로)

```
v_generated = e_regen(예측 dry 주입) − e_regen(e_wet 주입)
v_original  = e_dry_true − e_wet
directional_agreement = cos(v_generated, v_original)
```

"예측 dry 주입"은 **오라클(e_dry_true)이 아니라 5-C의 학습된 B2 모델(MLPDualHead)의
실제 예측**을 쓴다 — `e_wet + direction_pred × magnitude_pred`. 이 모델은 Phase 2의
1,200소스로 학습됐으므로, Phase 6 오디오 생성 소스(아래 §소스)와 **파일명 중복이
없는지 확인하고, 겹치면 그 소스를 B2 학습 데이터에서 제외**해 진짜 held-out 예측이
되게 한다.

## 6-D. 규모 확정 — 배수의 정확한 의미

`phase_f4_full.py`의 소스 선정(`select_sources`, N_PER_FAMILY_TARGET=120/MIN=60/
seed=0 → 10패밀리×5=50소스, `SELECT_SEED=1`)을 그대로 재사용해 9차와 동일한 50소스
풀을 재현했다. `phase_f2_filter`를 축에 매핑(`distortion_drive_db→distortion`,
`reverb_room_size→reverb`, `highshelf_gain→highshelf`)하고, `lowshelf_gain`·
`peak_gain`은 `highshelf`와 같은 근거(악기 무관 스펙트럼 조작)로 전 패밀리 허용을
**확장 적용**한다(원 필터에 없던 두 항목을 highshelf와 동일 규칙으로 추가 — 근거
명시, 표결 아님).

**실측 결과**(같은 50소스 풀, 필터 적용 후 허용 소스 수):

| 축 | 허용 소스 수 |
|---|---|
| distortion_drive_db | 25 |
| reverb_room_size | 40 |
| highshelf_gain | 50 |
| lowshelf_gain | 50 |
| peak_gain | 50 |
| **합(1구간 기준)** | **215** |

9차는 3이펙트로 정확히 115였다(25+40+50). 5축이 되며 이미 215로 늘었다 — "9차와
동일 규모에서 시작"은 **한 구간 기준으로 대응 이펙트 수만큼 자연 증가한 값(215)을
출발점으로 삼는다**로 해석한다. 이후 "조합 수만큼 배수"는 **구간 수(4)를 곱한다**로
해석한다: **215 × 4 = 860 조합**. MIDI는 1변형만(지시대로, 3변형 대비 결과 기여
0.2%였음 — `results_9_phase_f4.json`의 `pct_midi_variant` 참고), 조건은 c(e_wet
주입)/d(예측 dry 주입) 2가지 → **860 × 2 = 1,720회 생성**.

context 조건(4축쌍×전범위, 대표 context 3레벨 — 하/중/상): (50+50+50+40) × 3 × 2 =
**1,140회 생성 추가**.

**총 생성 횟수: 2,860회.**

### 시간 추정

9차 F-4 실측: 690회 생성에 4,216초(= 6.11초/회, 모델 로딩 제외). 같은 속도 가정 시
2,860 × 6.11초 ≈ 17,475초 ≈ **약 4.9시간**.

★ 위 해석(215 출발점, 구간수 배수, context 대표 3레벨)은 지시문이 명시하지 않은
부분에 대한 제 판단이다 — 특히 "n≈115에서 시작, 조합 수만큼 배수"를 "215×4"로
읽은 것과, context 레벨을 13개 전부가 아니라 대표 3개로 축소한 것 두 가지는 규모에
크게 영향을 준다(다르게 읽으면 최대 13/3 ≈ 4.3배까지 커질 수 있음). **실행 전 이
해석을 확인받는다.**

## 6-E. 한계 (사전 확정, 실행 후에도 그대로 유지)

- 재구성 충실도가 낮다(9차 관측 cos 0.25~0.65) — NSynth 단음 4초가 TokenSynth 학습
  분포(폴리포닉 5초) 밖이기 때문. directional_agreement는 이 낮은 절대 충실도
  안에서의 **방향** 지표이지 오디오 품질 지표가 아니다.
- 블라인드 청취를 시도하지 않는다 — 9차에서 대부분 "자연스러움 하"로 평가돼 판단
  근거가 오염될 위험이 확인됐다(`results_9_blind.json` 참고). 정량 지표로만 판정.

## 예측 (실행 전 확정)

**"임베딩에서 잘 되던 구간이 오디오에서도 잘 될 것인가"** — 아니오라고 예측한다.
9차 원 실험(B2 cos 0.71~0.82, 임베딩 단계)이 실제 생성에서 directional_agreement
0.03~0.06(거의 무작위)까지 무너진 전례가 있다. 이번 라운드의 B2 cos가 5-C에서
더 높게 나온 구간(예: distortion 전범위 0.80)이라도, 임베딩 단계 성능과 생성 단계
directional_agreement 사이에 **강한 양의 상관이 없을 것**으로 예측한다 — 즉 병목은
"손잡이를 못 찾아서"가 아니라 "TokenSynth의 projection layer/생성 경로 자체가
이펙트 정보를 소거하는 것"(10차 진단)이라는 가설을 재확인할 것으로 예측한다.

---

**위 6-D의 해석(2,860회, ~4.9시간)을 확인받은 뒤 실행한다.**
