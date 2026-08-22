# Phase 1 사전 등록 — EQ 계열 설계 + 사용자 검토 4건 반영

날짜: 2026-08-13
본 문서는 `out/results/11_phase0_ranges.md`(승인됨)의 범위 표를 **아래 4건 반영분으로
대체(supersede)**한다. Phase 0 감사 방식·결론, tanh/wet_level 무효과 실측, 게이트
스캔, ±6/±9/±12 구간별 사후 질의 방침은 그대로 유지한다(변경 없음).

검증 스크립트: `11_phase1_prereg_checks.py` · 원시 로그: `out/results/11_phase1_prereg_checks.json`,
`out/logs/11_phase1_prereg_checks.log`

---

## 1. OAT 기준점 표 (지시 1건 — 최우선)

pedalboard 기본값(`gain_db=0`)으로 "나머지는 기본값 고정"하고 cutoff·q를 스윕하면
EQ 세 타입 × 2축 = 6축이 전부 null이 된다는 지적은 §3(Phase 0.5 게이트 스캔)의
연장선이며 정확하다. 아래를 축별 고정 기준점으로 사전 등록한다.

| 스윕 대상 축 | 고정 기준점 | 비고 |
|---|---|---|
| EQ(3타입 공통) cutoff, q | `gain_db = +6dB` 조건과 `gain_db = −6dB` 조건 **둘 다** 렌더링 | 2차에서 확립된 부호 분리 관행 유지 |
| EQ(3타입 공통) gain | cutoff = 타입별 대표값(HighShelf 2000Hz / LowShelf 100Hz / PeakFilter 1000Hz), q = 0.7071(Koo 기본) | gain=0이 진짜 dry이므로(§Phase0.5) 게이팅 문제 없음, 대표 cutoff는 §4 갱신 범위의 로그 중앙값 |
| Reverb room_size, damping, width | `wet_level = 0.3`, `dry_level = 0.7` 고정 | wet_level=0이면 세 축 전부 무효화(Phase 0.5 §3) |
| Reverb wet_level | 나머지 Koo 기본값(room_size=0.5, damping=0.1, width=0.7) | 이 축 자체는 게이트 없음 |
| Distortion drive_db | 해당 없음(단일 축) | — |
| eq_cascade_intensity | 소스별 고정 5밴드 gain 패턴을 스칼라 s로 동시 스케일(§3) | s=0이 전 밴드 gain=0이므로 자체로 진짜 dry |

### 1.1 실측 검증 (`11_phase1_prereg_checks.py` §1, §1b)

6소스 감사셋으로 위 기준점에서 스윕 대상이 실제로 움직이는지 확인했다. **18개 조건
전부 비퇴화(spread > 1e-4) 확인**, 새로운 숨은 무효화 관계는 발견되지 않았다.

| 축 | 기준점 | cos spread(4점 스윕) | 판정 |
|---|---|---|---|
| HighShelf.cutoff | gain=+6/−6dB | 8.04e-03 / 6.86e-03 | OK |
| HighShelf.q | gain=+6/−6dB | 3.00e-02 / 1.68e-02 | OK |
| LowShelf.cutoff | gain=+6/−6dB | 1.24e-02 / 7.44e-03 | OK |
| LowShelf.q | gain=+6/−6dB | 1.86e-02 / 1.53e-03 | OK |
| PeakFilter.cutoff | gain=+6/−6dB | 1.10e-02 / 7.16e-03 | OK |
| PeakFilter.q | gain=+6/−6dB | 3.39e-03 / 2.87e-03 | OK |
| Reverb.room_size | wet_level=0.3 | 1.86e-01 | OK |
| Reverb.damping | wet_level=0.3 | 3.40e-03 | OK |
| Reverb.width | wet_level=0.3 | 4.84e-02 | OK |

`LowShelf.q`의 spread(1.53e-03)가 다른 축보다 한 자릿수 작다 — 퇴화는 아니지만
(1e-4 기준 통과) Phase 2에서 이 축의 JND가 다른 축보다 클(둔감할) 가능성을 미리
예상해둔다.

---

## 2. bypass 앵커와 θ_min 분리 (지시 2건)

### 2.1 정의

- **`e_bypass`**: pedalboard를 아예 통과시키지 않은 진짜 dry. 용량-반응 곡선의
  "θ=0" 표기점이자 도구가 최종적으로 도달을 확인하려는 목표점.
- **`e(θ_min)`**: 프로세서를 통과하되 그 축의 실무 범위 최솟값을 준 지점. JND의
  첫 구간, 변위 벡터의 원점은 **이 지점**으로 잡는다(`e_bypass`가 아니라).
- **`insertion_cost(축) = cos(e_bypass, e(θ_min))`**: 축마다 별도 수치로 보고.
  범위 스윕과 무관하게 "프로세서를 켜는 것 자체"의 대가를 분리해서 드러낸다.

이 분리는 EQ 3타입에는 적용하지 않는다 — `gain_db=0`이 실측으로 진짜 dry임이
확인됐으므로(Phase 0.5 §2, cos>0.9999) EQ의 θ=0은 `e_bypass`와 동일하고
insertion_cost는 자명하게 0이다. eq_cascade_intensity도 동일(§3, s=0이 전 밴드
gain=0).

### 2.2 insertion_cost 표

| 축 | θ_min | insertion_cost (min cos, 6소스) | insertion_cost (mean cos) | 근거 |
|---|---|---|---|---|
| Distortion.drive_db | 0 dB (Koo 하한, §4) | 0.900335 | 0.978586 | Phase 0.5 §2 재사용 — tanh(x)≠x, 수학적으로 무효과 값 없음 |
| Reverb.wet_level(자체 축) | 0.0 | 0.984033 | 0.985676 | Phase 0.5 §2 재사용(`dry_level=1.0` 조건) |
| Reverb.room_size | 0.05 (wet_level=0.3 기준점) | 0.861674 | 0.946790 | 신규 측정, §2.2.1 주의사항 참고 |
| Reverb.damping | 0.0 (wet_level=0.3 기준점) | 0.813468 | 0.927844 | 〃 |
| Reverb.width | 0.0 (wet_level=0.3 기준점) | 0.848173 | 0.941226 | 〃 |
| EQ 3타입 전체 | gain_db=0 | 1.000000(자명) | 1.000000(자명) | Phase 0.5 §2, 무효과 값 실측 PASS |
| eq_cascade_intensity | s=0 | 1.000000(자명) | 1.000000(자명) | §3.2, 6소스 전부 cos@s=0=1.000000 |

#### 2.2.1 해석 주의 — room_size/damping/width의 insertion_cost는 순수 "삽입 잡음"이 아니다

wet_level 자체 축(θ_min=0)의 insertion_cost(0.984)와 room_size/damping/width
축들의 insertion_cost(0.81\~0.86)는 **성격이 다르다.** 후자 세 축은 OAT 기준점이
`wet_level=0.3`이라, θ_min 지점이라도 이미 30% 웻 믹스라는 **실재하는 강한 리버브
효과**가 걸려 있다 — 이 수치는 "프로세서 삽입 아티팩트"가 아니라 "wet_level=0.3
자체가 진짜 효과다"를 대부분 반영한다. 세 축의 용량-반응 곡선을 해석할 때
`e(θ_min)`을 원점으로 삼는 것은 여전히 옳지만(그 축 자체의 스윕은 θ_min에서
시작해야 하므로), 이 insertion_cost 수치 자체를 "reverb 삽입은 항상 이만큼
아티팩트를 유발한다"로 일반화해서 읽으면 안 된다. 순수 삽입 아티팩트 추정치는
wet_level 자체 축의 0.984 쪽이 더 가깝다.

---

## 3. eq_cascade_intensity 축 (지시 3건)

### 3.1 설계

Koo et al. 원본이 실제로 거는 5밴드 동시 적용(Phase 0 §2.1)을 재현하는 축이다.
단일 밴드 3타입 축은 그대로 유지하고, 이 축을 **15번째 축으로 추가**한다.

- 밴드 5개 고정 주파수: `low_shelf=100Hz, first_band=400Hz, second_band=2000Hz,
  third_band=4000Hz, high_shelf=3500Hz`
  - first_band·second_band·third_band는 Koo 기본값(400/2000/4000Hz) 그대로.
  - low_shelf는 Koo 기본 80Hz 대신 우리 LowShelf 축의 갱신 범위(§4, 30\~200Hz)
    중앙 근처인 100Hz로 대체 — Koo 범위 안이므로 이탈 아님.
  - **high_shelf는 Koo 기본값 8000Hz를 쓰지 않는다.** 8000Hz는 NSynth 16kHz
    소스의 Nyquist와 정확히 같아 측정 불가 지점이다(§4.1 한계 참고). 대신
    단일 HighShelf 축의 안전 범위(500\~4000Hz) 밖이지만 여전히 Nyquist에서
    충분히 떨어진 3500Hz를 대표값으로 채택했다 — Koo 충실도보다 측정 가능성을
    우선한 명시적 이탈이다.
- 소스마다 5밴드 gain을 Koo 범위(−15\~+15dB)에서 **소스별 고정 시드**로 1회
  추출한다. 시드 규칙: `seed = 42 + idx`, `idx`는 Phase 2 확정 400소스 목록에서
  해당 소스의 순번(0-based, 목록은 렌더링 시작 시 고정·로그 기록). 이 문서의
  검증에서는 감사용 6소스에 `idx=0..5`를 그대로 적용했다(아래 §3.2 표).
- 스칼라 `s ∈ [0, 1]`로 5밴드 gain을 동시 배율(`gain_i(s) = s · gain_i_fixed`).
  **선형** 25등분(0, 1/24, ..., 1) — s=0이 진짜 dry(전 밴드 gain=0)이므로 로그
  스케일을 쓸 이유가 없다(로그 스케일은 0을 포함할 수 없다).

### 3.2 퇴화 검증 결과

6소스 각각 독립 시드로 뽑은 패턴에 대해 s=[0, 0.25, 0.5, 0.75, 1.0] 스윕:

| 소스 | seed | cos@s=0 | cos@s=1.0 | spread |
|---|---|---|---|---|
| bass_electronic | 42 | 1.000000 | 0.902563 | 9.74e-02 |
| brass_acoustic | 43 | 1.000000 | 0.890466 | 1.10e-01 |
| flute_acoustic | 44 | 1.000000 | 0.970590 | 2.94e-02 |
| guitar_acoustic | 45 | 1.000000 | 0.965820 | 3.42e-02 |
| keyboard_acoustic | 46 | 1.000000 | 0.950624 | 4.94e-02 |
| reed_acoustic | 47 | 1.000000 | 0.967666 | 3.23e-02 |

전 소스 s=0에서 cos=1.000000(진짜 무효과 값 확인), s 증가에 따라 단조 감소,
퇴화(패턴이 서로 상쇄돼 효과가 사라지는 경우) 없음. 설계 그대로 Phase 2에 채택한다.

---

## 4. 범위 출처 정책 고정 (지시 4건)

**정책(고정)**: *"Koo et al. 구현 범위를 1순위로 채택한다. NSynth 16kHz 제약 또는
실무 근거와 충돌할 때만 좁히고 그 사유를 적는다."*

### 4.1 갱신된 범위 표 (본 문서가 `11_phase0_ranges.md`의 표를 대체)

| 파라미터 | 하한 | 상한 | 근거/정책 적용 | θ_min 취급 |
|---|---|---|---|---|
| EQ gain_db(공통) | **−15 dB** | **+15 dB** | Koo 그대로 복원. 이전 초안(±12dB)은 "출처 간 절충"이라는 임의적 사유로 좁혔던 것 — 정책상 근거 미달이라 폐기. 실무 사용자가 흔히 쓰는 ±6dB는 렌더링 후 사후 구간 질의로 별도 보고 | 0 dB(진짜 dry, 게이팅 없음) |
| EQ q(공통) | 0.1 | 2.0 | Koo 범위와 일치, 변경 없음 | 게이트 아님 |
| HighShelf cutoff_frequency_hz | 500 Hz | 4000 Hz | **Koo(5000\~10000Hz)에서 좁힘 — 사유: NSynth 16kHz 소스 Nyquist(8000Hz) 제약.** Koo 범위의 절반(8000\~10000Hz)은 물리적으로 넘을 수 없고, 8000Hz 근방도 실측상 대역 소실로 측정 불가(3\~4차 전례, R²=0.004) | — |
| LowShelf cutoff_frequency_hz | **30 Hz** | **200 Hz** | **Koo로 원복.** 이전 초안의 60\~500Hz는 "독립 출처 없음"을 자인하며 임의로 넓힌 것 — Nyquist 충돌도 실무 근거 충돌도 없으므로 정책상 Koo를 그대로 써야 함 | — |
| PeakFilter cutoff_frequency_hz | 200 Hz | **6000 Hz** | 하한은 Koo(first_band) 그대로. **상한을 8000Hz→6000Hz로 하향 — 사유: 8000Hz=Nyquist라 그 지점의 스윕 격자점이 원리적으로 무효(HighShelf와 동일한 문제).** 6000Hz는 Nyquist와 2kHz 여유 확보 | — |
| Distortion drive_db | **0 dB** | 20 dB | **Koo로 원복(이전 초안 0.5dB 하한 폐기).** §2 분리 도입으로 "θ=0이 진짜 dry가 아님"을 insertion_cost로 명시 보고하는 방식으로 해결했으므로, 인위적으로 하한을 밀어 올릴 이유가 없어짐 — Koo 범위를 그대로 쓰고 θ_min=0dB의 삽입 비용을 별도 수치로 낸다(§2.2) | 0 dB, insertion_cost=0.900(§2.2) |
| Reverb room_size | 0.05 | 0.85 | Koo 그대로, 변경 없음 | wet_level=0.3 기준점에서 0.05, insertion_cost=0.862(§2.2) |
| Reverb damping | 0.0 | 1.0 | Koo 그대로, 변경 없음 | 〃 0.0, insertion_cost=0.813 |
| Reverb width | 0.0 | 1.0 | Koo 그대로, 변경 없음 | 〃 0.0, insertion_cost=0.848 |
| Reverb wet_level | 0.0 | **0.5** | **Koo(0\~1.0)에서 의도적으로 좁힘 — 사유: 실무 근거.** "인서트 리버브 실무 wet 비율은 12\~40%대"라는 소싱된 믹싱 가이드 기준, wet_level=1.0(100% 웻)은 실사용 맥락에서 사실상 나타나지 않는 극단값이라 판단. 정책의 "실무 근거와 충돌 시 좁힘" 예외 조항 적용 사례로 명시 | 0.0, insertion_cost=0.984(자체 축, §2.2) |
| eq_cascade_intensity(s) | 0.0 | 1.0 | 신규 축(§3). 밴드별 gain은 Koo 범위(±15dB)에서 추출, s는 그 배율 | 0.0(진짜 dry) |
| Reverb freeze_mode | — | — | 변경 없음, 0 고정(스윕 대상 아님) | 0(무효과 값) |

### 4.2 한계 절 추가 (신규)

**Koo et al. 원본 EQ의 high_shelf 밴드 기본 주파수 범위(5000\~10000Hz)는 절반
이상이 NSynth 16kHz 소스의 Nyquist 주파수(8000Hz)를 넘는다.** 즉 이 프로젝트가
쓰는 데이터(16kHz 소스를 48kHz로 업샘플)로는 **원본 EQ의 high_shelf 밴드를 있는
그대로 재현하는 것 자체가 물리적으로 불가능하다** — 우리가 렌더링 파이프라인을
잘못 짜서가 아니라 소스 데이터의 대역 자체가 없기 때문이다. 단일축 HighShelf
(500\~4000Hz)와 cascade 축의 high_shelf 대역(3500Hz)은 모두 이 제약을 피하기 위해
Koo 원 범위에서 하향 이탈한 근사치이며, **"논문이 실제로 걸었을 EQ의 상단 밴드
거동"을 이 프로젝트가 완전히 재현할 수는 없다**는 점을 결과 해석의 명시적 한계로
못박는다. Phase 5 재산출 표(과거 highshelf 수치 재검토) 작성 시 이 한계를 함께
기술한다.

---

## 5. Phase 1 최종 축 구성 (15축)

| 그룹 | 축 | 레벨 수 |
|---|---|---|
| Distortion | drive_db | 25 |
| Reverb | wet_level, room_size, damping, width | 25 × 4 |
| EQ 단일밴드 | HighShelf/LowShelf/PeakFilter × {gain, cutoff, q} | 25 × 9 |
| EQ 캐스케이드(신규) | eq_cascade_intensity(s) | 25 |
| 널 축(변화 없음) | ultrasonic shelf 12kHz, 15kHz | 25 × 2(기존 계획 유지) |

주 실험축 합계: 1+4+9+1 = **15축 × 25레벨 = 375조건 + dry 앵커 1 = 376조건/소스**
(널 축 2개 별도). 이전 보고(14축)보다 cascade 1축이 늘어 조건 수가 350→375로
증가했다.

## 6. Phase 1 예측·판정 기준 사전 등록 (원칙 3)

**예측 1 — EQ 3타입 측정 가능성 순서**: `LowShelf ≳ PeakFilter > HighShelf`.
근거: NSynth 악기 하모닉 에너지는 저\~중역에 집중되고, HighShelf는 Nyquist
제약으로 유효 스윕 대역이 가장 좁게 깎였다(§4.1 한계). "CLAP이 EQ를 못 읽는다"는
기존 결론이 highshelf 고유의 문제인지 EQ 전반의 문제인지를 이 순서로 가른다.
- **판정: 확정(band-specific)** — LowShelf의 windowed R² 피크가 HighShelf의
  windowed R² 피크보다 절대값 0.10 이상 높으면.
- **판정: 기각(EQ 전반 문제)** — 세 타입의 windowed R² 피크가 서로 0.05 이내로
  수렴하면.
- 그 외는 **혼재**로 보고.

**예측 2 — 캐스케이드 vs 단일밴드**: eq_cascade_intensity의 windowed R² 피크가
단일밴드 3타입 중 최댓값보다 높다(5밴드 동시 적용이 논문의 실제 조건에 더
가깝고, 여러 밴드의 스펙트럴 변화가 누적되므로).
- **판정: 확정** — cascade 피크가 단일밴드 최댓값보다 0.05 이상 높으면.
- **판정: 기각(중복/비가산적)** — cascade 피크가 단일밴드 최댓값과 0.05 이내로
  같거나 낮으면(밴드 간 상쇄·중복으로 해석).

**예측 3 — insertion_cost 순서**: `HighShelf/LowShelf/PeakFilter(≈0) <
Reverb.wet_level(0.984) < Distortion.drive_db(0.900) < Reverb.room/damping/width
(0.81~0.86, 단 §2.2.1 주의사항대로 wet=0.3 자체 효과 포함)`.
이미 실측(§2.2)으로 확인됐으므로 이 예측은 Phase 2 렌더링 없이 이 문서 시점에
**확정**으로 기록한다.

## 7. Phase 2 소요 시간·조건 수 추정

벤치마크(`.venv`, CPU, M-시리즈): render+embed 1회 = **159.8ms**(N=40 반복 평균).

- 총 렌더링 수 = 376조건 × 400소스 = **150,400회** (널 축 2×25×400=20,000 별도,
  기존 계획대로 진행 시 총 170,400회)
- 예상 소요 시간(직렬 처리 기준) = 170,400 × 0.16s ≈ **27,300초 ≈ 7.6시간**
- 배치 임베딩(여러 조건을 한 번에 CLAP에 통과)으로 임베딩 쪽 오버헤드를 줄이면
  단축 가능하나, pedalboard 렌더링 자체는 조건별 직렬 처리가 불가피 — 실질
  절감폭은 제한적일 것으로 예상. 보수적으로 **6\~9시간**을 Phase 2 예상 소요
  시간으로 보고한다.
- 체크포인트: 지시서 데이터 위생 요건대로 축 단위(15+2개 축)로 캐시를 나눠
  저장하여, 중단 시 완료된 축부터 재개 가능하게 한다(`out/caches/11_phase2_<axis>.npz`).

## 8. 산출 파일

- `out/prereg/11_phase1.md` (본 문서)
- `11_phase1_prereg_checks.py`
- `out/results/11_phase1_prereg_checks.json`
- `out/logs/11_phase1_prereg_checks.log`

---

**본 문서가 승인되면 Phase 2 렌더링(§5의 15축 + 널 축 2, §7의 조건 수·시간
추정)을 시작한다.**

---

## 정정 addendum (2026-08-13, 사람 검토 후)

승인 시점 이후 두 가지가 정정됐다. 원문(§1\~§7)은 최초 승인 기록 보존을 위해
그대로 두고 여기 추가한다.

1. **축 수 정정**: §5의 "15축" 표는 계산 실수였다. item 1(OAT 기준점)에서
   cutoff·q를 `gain=+6dB`/`−6dB` **두 벌**로 렌더링하기로 한 것이 그대로
   반영되면 EQ 단일밴드만 3타입×5(gain 1 + cutoff 2조건 + q 2조건)=15
   서브축이 되고, cascade(1)+distortion(1)+reverb(4)를 더하면 주축 21개,
   널축 2개까지 **총 23축**이다(375조건이 아니라 575조건/소스). 설계 자체는
   변경 없음 — 표기만 정정.
2. **eq_cascade_intensity 밴드 주파수 결함(사람 지시로 발견)**: §3.1에서 정한
   `high_shelf=3500Hz`가 `third_band=4000Hz`보다 낮아 5밴드 순서가
   역전됐었다(하이셸프가 벨 아래에 위치 — 5밴드 파라메트릭 EQ 구조가
   성립하지 않음). 아래로 정정:

   | 밴드 | 이전 | 정정 |
   |---|---|---|
   | third_band | 4000 Hz | **3000 Hz** (Koo 범위 3000\~8000의 하단) |
   | high_shelf | 3500 Hz | **6500 Hz** (Koo 범위 5000\~10000과 Nyquist 8000 사이) |

   순서 보존: `100 < 400 < 2000 < 3000 < 6500 < 8000(Nyquist)`. Koo
   high_shelf 기본값(8000Hz)은 16kHz 소스 Nyquist와 정확히 같아 그대로
   재현이 불가능하다는 한계(§4.2)는 그대로 유지 — 6500Hz는 그 제약 안에서
   고른 근사치다. `11_phase2_eq_cascade_intensity.npz`를 삭제하고 해당
   축만 재렌더링했다(나머지 22축은 캐시 재사용).
