# Phase 0.5 — pedalboard API 실측 + 실무 범위 확정

날짜: 2026-08-12 · 스크립트: `11_phase0_5_api_audit.py` · 원시 로그: `out/results/11_phase0_api_audit.json`, `out/logs/11_phase0_5_api_audit.log`
pedalboard 버전: 0.9.24 (기존 1~10차와 동일)

## 1. 파라미터 시그니처 전수 (pedalboard 0.9.24 실측)

```
HighShelfFilter(cutoff_frequency_hz=440, gain_db=0.0, q=0.7071067690849304)
LowShelfFilter (cutoff_frequency_hz=440, gain_db=0.0, q=0.7071067690849304)
PeakFilter     (cutoff_frequency_hz=440, gain_db=0.0, q=0.7071067690849304)
HighpassFilter (cutoff_frequency_hz=50)
LowpassFilter  (cutoff_frequency_hz=50)
Distortion     (drive_db=25)
Reverb         (room_size=0.5, damping=0.5, wet_level=0.33, dry_level=0.4, width=1.0, freeze_mode=0.0)
```

## 2. 무효과 값 실측 결과 ★ 중요 발견

소스 6개(bass/brass/flute/guitar/keyboard/reed, 각기 다른 패밀리) 기준, dry 대비 CLAP
코사인:

| 후보 무효과 값 | min cos (6소스) | mean cos | 판정 |
|---|---|---|---|
| `HighShelfFilter(gain_db=0)` | 1.00000000 | 1.00000008 | **PASS** |
| `LowShelfFilter(gain_db=0)` | 0.99999994 | 1.00000004 | **PASS** |
| `PeakFilter(gain_db=0)` | 1.00000000 | 1.00000005 | **PASS** |
| `Distortion(drive_db=0)` | 0.90033489 | 0.97858585 | **★ FAIL** |
| `Reverb(wet_level=0, dry_level=1)` | 0.98403323 | 0.98567628 | **★ FAIL** |
| `Reverb(전부 기본값, wet_level=0만)` | 0.99807799 | 0.99834594 | **★ FAIL** |

**EQ 세 필터는 gain_db=0이 진짜 무효과다(0.9999 통과).** distortion과 reverb는
**"파라미터를 무효과처럼 보이는 값으로 두는 것"과 "진짜 dry"가 다르다.**

### 2.1 Distortion — 수학적으로 무효과 값이 존재하지 않는다

pedalboard 공식 문서: `distortion(x) = tanh(x * db_to_gain(drive_db))`.
`drive_db=0` → `db_to_gain(0)=1.0` → `tanh(x)`. 입력 x가 정확히 0이 아닌 한
`tanh(x) ≠ x`(tanh는 원점 근방에서도 3차 이상 비선형이다) — **drive_db를 어떤 유한값으로
둬도 항등함수가 되지 않는다.** 피크가 큰(파형이 tanh의 비선형 구간에 더 많이 걸리는)
소스일수록 dry 대비 코사인이 더 낮다는 것이 min/mean 격차(0.900 vs 0.979)로 나타난다.
drive_db를 매우 음수로 내려도 `tanh(x·gain)→0`(무음)에 수렴할 뿐 항등함수에
수렴하지 않으므로, **음의 방향으로도 무효과 값을 찾을 수 없다.**

→ **결론: distortion의 θ=0 앵커는 기존 관행대로 명시적 bypass(pedalboard를 아예
통과시키지 않음)로만 확보할 수 있다.** 스윕 하한은 "진짜 dry"가 아니라 "가장 약한
실무적 드라이브 값"으로 정의하고, 앵커는 별도 점으로 분리한다(§5).

### 2.2 Reverb — wet_level=0도 진짜 dry가 아니다

원인은 미확정이나(Reverb는 pedalboard 문서에 신호처리 공식이 없음 — JUCE 기반
C++ 구현의 내부 지연 보정·버퍼링 차이로 추정, 확인 안 됨), **wet_level=0으로 둬도
dry 대비 코사인이 0.984~0.998에 그친다.** 이는 6차에서 이미 확립한 관행("이 앵커는
pedalboard를 통과시키지 않고 dry 오디오를 그대로 써서 cos=1.000이 되도록 보장한다")이
왜 필요했는지에 대한 정량적 근거가 된다 — 그때는 원칙으로만 세웠던 것을 이번에 실측으로
확인했다.

## 3. 파라미터 상호 무효화 스캔 결과

| 게이트 조건 | 스윕 대상 | 결과 |
|---|---|---|
| `Reverb.wet_level=0` | room_size | **무효화 확인**(spread=0) — wet_level이 게이트임을 재확인 |
| `Reverb.wet_level=0` | damping | **무효화 확인**(spread=0) |
| `Reverb.freeze_mode=1` | room_size | **무효화 확인**(spread=0) — 기지 결함(6차) 재현, freeze_mode는 반드시 0 고정 |
| `HighShelfFilter.gain_db=0` | cutoff_frequency_hz | **무효화 확인**(spread≈1e-16) — 정상(게이트가 아니라 자명한 항등) |
| `HighShelfFilter.gain_db=0` | q | **무효화 확인**(spread≈1e-7) — 정상 |
| `PeakFilter.gain_db=0` | cutoff_frequency_hz | **무효화 확인**(spread≈1e-16) — 정상 |

EQ 세 필터는 gain_db=0일 때 cutoff·q를 아무리 바꿔도 출력이 바뀌지 않는다 — 이건
"숨은 결함"이 아니라 **gain이 0이면 필터 계수가 항등이 되는 자명한 수학적 결과**이므로
문제 없음. reverb의 `wet_level`·`freeze_mode` 게이트는 기존에 알려진 것과 정확히
일치하며 새 게이트는 발견되지 않았다.

## 4. 실무 범위 표 (사전 등록)

★ 지시서 원칙 1에 따라 아래 범위는 **Phase 2 렌더링의 바깥쪽 경계**일 뿐이다. 범위
안의 특정 구간(예: "±6dB만" vs "±12dB까지")에 대한 판정은 렌더링 후 사후 질의로
낸다 — 지금 좁은 숫자 하나를 확정하려 하지 않는다.

| 파라미터 | 하한(스윕) | 상한 | 근거 | 무효과 값 |
|---|---|---|---|---|
| EQ gain_db (Highshelf/Lowshelf/Peak 공통) | −12 dB | +12 dB | 믹싱 가이드 다수: "부스트는 +3dB 이내가 자연스럽다", "±6dB 초과는 극단적 보정이 아니면 안 씀"([Sonarworks](https://www.sonarworks.com/blog/learn/pro-mastering-tips-stereo-eq-techniques)). 반면 마스터링 EQ 플러그인은 최대 36dB까지 제공하고 Koo et al. 2023 구현(§Phase0 감사)의 기본 클래스 범위는 ±15dB — **출처 간 불일치가 있어 넓게 잡고 사후 질의로 ±6/±9/±12 구간별 R²를 따로 보고한다** |0 dB (dry) — bypass로 확보, §5 |
| EQ q (공통) | 0.1 | 2.0 | Koo et al. 2023 `Equaliser` 클래스 기본 범위(코드 확인, Phase 0 감사)와 "마스터링에서 Q가 2.0을 넘는 경우는 드물다"는 믹싱 가이드가 서로 일치 | q는 게이트 아님(gain=0이면 q 무관, §3) |
| HighShelf cutoff_frequency_hz | 500 Hz | 4000 Hz | **3차/6차에서 이미 확립**: NSynth 소스가 16kHz(Nyquist 8kHz)이므로 8000Hz 부근 셸빙은 대역 자체가 없어 측정 불가였다(4차 R²=0.004). 500~4000Hz로 좁혀 재측정한 전례를 그대로 승계 | 게이트 없음(gain=0이면 자명 무효과) |
| LowShelf cutoff_frequency_hz | 60 Hz | 500 Hz | 로우 셸프의 통상 활용 대역(웜스/저역 정리) — "100Hz에서 Q 0.7의 로우 부스트가 자연스럽다"는 가이드 기준 상하로 여유를 둠. **독립적인 정량 출처를 못 찾음 — 이 사실을 명시하고 넓게 잡음** | 〃 |
| PeakFilter cutoff_frequency_hz | 200 Hz | 8000 Hz | Koo et al. 5밴드 EQ의 벨 3밴드(first/second/third band)가 합쳐서 덮는 영역(200~8000Hz)을 단일 벨 필터의 대표 스윕 구간으로 채택 — Phase 0 감사에서 확인한 실제 구현 근거 | 〃 |
| Distortion drive_db | 0.5 dB | 20 dB | 상한은 Koo et al. 2023 `Distortion` 클래스 기본 범위(0~20dB, 코드 확인). **일반 웹 검색으로는 "약한~강한 디스토션"에 대응하는 독립적인 dB 수치를 찾지 못했다 — 이 사실을 명시한다.** pedalboard 자체 기본값은 25dB(더 높음)이나 Koo 구현을 우선 채택 | **없음(§2.1) — 하한 0.5dB조차 진짜 dry가 아니다.** θ=0 앵커는 bypass로만 확보(§5) |
| Reverb room_size | 0.05 | 0.85 | Koo et al. 2023 `AlgorithmicReverb` 클래스 기본 범위(코드 확인) — pedalboard도 동일 Freeverb 계열이라 같은 정규화 스케일(0~1)로 간주 | 게이트 아님. wet_level=0이면 room_size 변화 무효(§3) |
| Reverb damping | 0.0 | 1.0 | 〃 | 〃 |
| Reverb wet_level | 0.0 | 0.5 | "인서트 리버브의 실무 wet 비율은 12~40%대가 흔하다"는 믹싱 가이드([참고](https://gearspace.com/board/electronic-music-instruments-and-electronic-music-production/817250-percentage-wet-mix-send-vocal-reverb.html)) + 1\~2차에서 이미 0~0.5로 쓴 전례. dry_level은 `1.0 − wet_level`로 고정(1\~10차 관행 유지 — 순수 게이트로 만들기 위함) | 0.0 — 그러나 §2.2에 따라 **wet_level=0 자체도 진짜 dry가 아니므로 bypass로 앵커 확보 필수** |
| Reverb width | 0.0 | 1.0 | Koo et al. 2023 기본 범위 그대로(코드 확인). 스테레오 폭은 저위험 파라미터로 판단, 별도 축소 근거 없음 | 게이트 아님, 무효과 값 미확인(모노 파이프라인에서 원리적으로 무영향이어야 한다고 3\~4차가 가정했으나 6차에서 실제 효과가 있음이 확인된 바 있다 — 이번 차수는 width를 정식 측정 축으로 다룬다, 지시서 §6 명시사항) |
| Reverb freeze_mode | — | — | §3에서 재확인한 게이트. 무조건 0 고정, 스윕 대상 아님 | 0 (고정값 자체가 무효과 값) |

## 5. θ=0 앵커 처리 방침 (Phase 1\~3 공통)

지시서 원칙("하한은 반드시 무효과 값(진짜 dry)이어야 한다")과 §2의 실측 결과를
종합하면:

- **EQ 세 필터**: `gain_db=0`이 곧 진짜 dry다(실측 cos>0.9999). 스윕 격자의 최하단
  점을 `gain_db=0`으로 그대로 둬도 된다.
- **Distortion·Reverb**: 파라미터 값으로는 진짜 dry를 만들 수 없다(§2.1, §2.2).
  1\~10차와 동일하게 **θ=0 앵커를 별도 표본으로 예약해 pedalboard를 통과시키지 않고
  dry 오디오를 그대로 쓴다.** Phase 2 격자의 25레벨은 "θ=0(=bypass) + 실무 범위
  안의 24레벨"로 구성하고, 명시적으로 `cos(e_dry, e_theta0)=1.000`을 각 축마다
  검증해 로그에 남긴다(과거 neutral_check 관행 유지).

## 6. Phase 0 대비 새로 발견한 결함

| # | 결함 | 영향 |
|---|---|---|
| 1 | distortion에 수학적으로 무효과 값이 없음(tanh 특성) | Phase 2 격자 설계에서 θ=0을 파라미터값이 아닌 명시적 bypass로만 처리해야 함 — 이미 관행이었으나 이번에 정량적으로 확인 |
| 2 | reverb wet_level=0도 진짜 dry가 아님(cos 0.984\~0.998) | 위와 동일 처리 필요, 정량적으로 재확인 |
| 3 | EQ 게이트는 자명한 것 외에 새 상호 무효화 관계 없음 | Phase 2/3 설계에 추가 제약 없음 — 안심 |

결함 A(highshelf로 EQ 대표)·B(무작위 7종 서브샘플)·C(단일 범위 R²)는 Phase 0/0.5에서
직접 다루지 않았고 Phase 1\~4에서 해결한다(지시서 원안 그대로).

## 7. 소요 시간

- Phase 0(upstream 코드 감사 + 논문 PDF 확인): 약 25분
- Phase 0.5(API 실측 스크립트 작성·실행 + 범위 조사): 약 20분
- 다음(Phase 1) 예상: EQ 3타입 사전 등록 문서 작성 자체는 짧으나(10\~20분), 그
  직후 Phase 2 렌더링이 14축×25레벨×400소스=140,000 조건이라 가장 오래 걸리는
  단계다 — CLAP forward 1회가 대략 수십 ms\~100ms대(M5 CPU)임을 감안하면 대략
  4\~8시간 규모로 예상(레벨 수·소스 수를 그대로 지시서대로 유지할 경우). 정확한
  추정은 Phase 1에서 소스 400개 선정 후 소규모 타이밍 테스트로 갱신해 보고한다.

## 8. 산출 파일

- `out/results/11_phase0_ranges.md` (본 문서)
- `out/results/11_phase0_api_audit.json` (원시 수치)
- `out/logs/11_phase0_5_api_audit.log`
- `11_phase0_5_api_audit.py` (스크립트, 재실행 가능)
