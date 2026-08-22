# Phase 0 — Upstream 코드 감사: 논문의 "EQ"는 무엇이었는가

날짜: 2026-08-12
방법: 코드를 직접 읽었다. 추측 없음. 확인 안 되는 부분은 "확인 안 됨"으로 명시한다.

## 0. 결론 요약

**결함 A(highshelf 하나로 EQ를 대표시킨 것)는 실제 결함이었다.** TokenSynth 논문이
인용하는 원본 구현("Koo et al. 2023")의 EQ는 **5밴드 파라메트릭 EQ 캐스케이드**(로우
셸프 + 벨 3개 + 하이셸프)다. 우리가 1~10차에서 쓴 `HighShelfFilter` 단일 필터는 이
5개 밴드 중 하나(그것도 원본과 주파수 범위가 다른)만 골라 쓴 것이었다 — 치환 근거가
없었다는 지적이 정확했다.

**Distortion도 구현이 다르다.** 원본은 5가지 모드(hard_clip 기본값·overdrive·
soft_sine·tanh·bit_crusher) 중 선택하는 커스텀 프로세서다. 우리가 쓴
`pedalboard.Distortion`은 이 중 어느 모드와도 정확히 같지 않은 별개의 웨이브셰이핑
구현이다.

**Reverb는 상대적으로 양호하다.** 원본은 Schroeder-Moorer 계열 알고리즘 리버브(comb
8개 + allpass 4개, 이른바 Freeverb 구조)이고, `pedalboard.Reverb` 역시 Freeverb
기반이다. 파라미터 이름(room_size·damping·wet/dry·width)도 대응된다 — 구조적으로는
가장 근접한 재현이었다.

## 1. 계보 — TokenSynth 논문 원문

`arXiv:2502.08939` §IV-A (Dataset) 원문 인용:

> "To enhance timbre diversity, we augmented the audio using a digital effect chain
> with random parameters. **Following Koo et al. [37]**, we randomly applied EQ,
> distortion, and algorithmic reverb with a 0.5 probability, using parameters from
> predefined ranges. Identical effects were applied to each reference-target pair to
> maintain timbre consistency. This process doubled the dataset size."

`[37]`은 J. Koo, M. A. Martínez-Ramírez, W.-H. Liao, S. Uhlich, K. Lee, Y. Mitsufuji,
"Music mixing style transfer: A contrastive learning approach to disentangle audio
effects," ICASSP 2023 (`arXiv:2211.02247`).

★ **TokenSynth 자체 GitHub 저장소(`KyungsuKim42/tokensynth`, main + render 브랜치
전수 확인)에는 augmentation 코드가 없다.** 공개된 것은 추론(inference) 전용 패키지
(`clap.py`, `model.py`, `dac_decoder.py`, `utils.py`)뿐이고, 학습 파이프라인·데이터
증강 스크립트는 저장소에 없다. 즉 **TokenSynth가 실제로 실행한 정확한 글루 코드는
비공개다** — 아래 내용은 전부 "Koo et al." 저장소(`jhtonyKoo/music_mixing_style_transfer`)
에서 확인한 것이며, TokenSynth가 이 라이브러리를 어떻게 호출했는지는 **논문 문장
하나로만 추정 가능**하다.

## 2. Koo et al. 저장소에서 확인한 것 (직접 코드 확인)

출처: `github.com/jhtonyKoo/music_mixing_style_transfer`,
`mixing_style_transfer/mixing_manipulator/common_audioeffects.py`
(원 저작권 표기: Sony Group Corporation, 원본은 `sony/FxNorm-automix`에서 이식·수정).

### 2.1 EQ — `class Equaliser` (라인 370~529)

> "Five band parametric equaliser (two shelves and three central bands)... implemented
> as cascade of five biquad IIR filters... cookbook formulae from RBJ."

기본 파라미터(코드 그대로):

| 밴드 | 타입 | gain 범위 | freq 범위 | Q 범위 |
|---|---|---|---|---|
| low_shelf | 로우 셸프 | −15\~+15 dB | 30\~200 Hz (기본값 80) | 고정 0.707 |
| first_band | 벨(peaking) | −15\~+15 dB | 200\~1000 Hz (기본값 400) | 0.1\~2.0 (기본 0.7) |
| second_band | 벨(peaking) | −15\~+15 dB | 1000\~3000 Hz (기본값 2000) | 0.1\~2.0 (기본 0.7) |
| third_band | 벨(peaking) | −15\~+15 dB | 3000\~8000 Hz (기본값 4000) | 0.1\~2.0 (기본 0.7) |
| high_shelf | 하이 셸프 | −15\~+15 dB | 5000\~10000 Hz (기본값 8000) | 고정 0.707 |

5개 밴드 전부 **직렬 캐스케이드**로 동시에 적용된다(`bands` 인자로 부분집합 선택
가능하나 기본값은 5개 전부). 우리가 1~10차에서 쓴
`HighShelfFilter(cutoff_frequency_hz=4000, gain_db=v)`는 이 5밴드 중 high_shelf
하나만, 그것도 cutoff 기본 범위(5000\~10000Hz)와 다른 값(4000Hz)으로 대체한 것이었다.

### 2.2 Distortion — `class Distortion` (라인 296~369)

기본 파라미터:

| 파라미터 | 기본값 | 범위 |
|---|---|---|
| mode | `hard_clip` | {hard_clip, overdrive, soft_sine, tanh, bit_crusher} |
| threshold | 0.0 dB | −20\~0 dB |
| drive | 0.0 dB | 0\~20 dB |
| colour | 20.0 | 0\~100 |
| bits | 12 | 8\~12 |

기본 모드는 **hard_clip**(단순 클리핑)이며, `pedalboard.Distortion`(항상 동일한
웨이브셰이핑 곡선, drive_db만 조절)과는 다른 알고리즘이다. 5개 모드 중 어느 것을
TokenSynth가 썼는지는 공개 코드로 확인 불가.

### 2.3 Reverb — `class AlgorithmicReverb` (라인 1429~)

Schroeder-Moorer 계열: comb 필터 8개(좌우 각 8개) + allpass 필터 4개(좌우 각 4개)
직렬. Freeverb와 동일 계통 구조(스테레오 spread=23, scalegain=0.2 하드코딩).

| 파라미터 | 기본값 | 범위 |
|---|---|---|
| room_size | 0.5 | 0.05\~0.85 |
| damping | 0.1 | 0\~1.0 |
| dry_mix | 0.9 | 0\~1.0 |
| wet_mix | 0.1 | 0\~1.0 |
| width | 0.7 | 0\~1.0 |

`pedalboard.Reverb`도 Freeverb 기반이며 파라미터명이 거의 1:1 대응한다
(room_size/damping/wet_level/dry_level/width — 우리가 1\~10차에서 실제로 쓴 파라미터
집합과 정확히 일치). **세 이펙트 중 reverb만은 우리 기존 구현이 원본과 구조적으로
가장 가까웠다.**

### 2.4 적용 방식 — `class AugmentationChain.__call__` (라인 91\~194)

각 이펙트는 체인 안에서 **개별 확률로 독립적으로 트리거**된다(`np.random.rand() < p`
를 이펙트마다 따로 굴림) — 즉 한 샘플에 EQ와 distortion이 동시에 걸릴 수도, reverb만
걸릴 수도 있다. 트리거되면 `fx.randomize()`로 해당 프로세서의 전 파라미터를 (구현상
합리적으로 유추하면) 그 파라미터의 [minimum, maximum] 범위에서 무작위 추출한다
(`randomize()` 자체는 외부 의존성 `pymixconsole`에 있어 이 저장소에는 없음 — **정확한
분포 형태는 확인 못 함**, 균등분포로 추정하나 미확인).

이 저장소 자체 학습 스크립트(`data_loader.py`)가 실제로 쓴 기본 확률은
`{eq:0.9, comp:0.9, pan:0.3, imager:0.8, gain:0.5, reverb:악기별 0.01~0.9}`로,
TokenSynth가 인용한 "0.5"와 다르다 — **TokenSynth는 Koo et al.의 프로세서 클래스만
가져다 쓰고 확률·이펙트 조합은 자기 것으로 새로 짰다는 뜻이다.** distortion은 애초에
Koo 저장소의 기본 체인(`create_inst_effects_augmentation_chain`)에 포함되어 있지도
않다 — TokenSynth 저자들이 `Distortion` 클래스를 별도로 가져와 자기 체인에 넣었다.

## 3. 확인 안 된 것 (추측하지 않음)

| 질문 | 상태 |
|---|---|
| TokenSynth가 Koo 클래스의 기본 파라미터 범위를 그대로 썼는가, 자체 범위로 바꿨는가 | **확인 불가** — 논문은 "predefined ranges"라고만 함, 수치 없음 |
| EQ 5밴드를 전부 썼는가, 일부만 썼는가 | **확인 불가** |
| Distortion 5모드 중 어느 것을 썼는가 | **확인 불가** (클래스 기본값은 hard_clip) |
| 이펙트 순서(적용 순서), shuffle 여부 | **확인 불가** |
| "0.5 확률"이 이펙트별 독립인지 전체 체인 1회인지 | Koo 라이브러리 구조상 이펙트별 독립이 자연스러운 해석이나, TokenSynth 자체 문장만으로는 **단정 불가** |
| 파라미터 무작위 추출이 균등분포인지 | **확인 불가** (`pymixconsole.Parameter.randomize()` 비공개/미조사) |
| NSynth 48kHz(우리 파이프라인 기준) vs 이 저장소 기본 44.1kHz 처리 시 리샘플 방식 | **확인 불가**, TokenSynth 학습 시 샘플레이트 불명 |

## 4. Phase 1 설계에 대한 함의

논문이 인용한 EQ가 특정됐으므로(5밴드 파라메트릭, 로우/하이 셸프 + 벨 3개) 사용자
지시서 Phase 1의 "최소 구성"(HighShelf·LowShelf·Peak 3타입)은 이 발견과 정확히
부합한다 — 원본이 실제로 셸프 2개 + 벨 3개를 함께 쓰므로, 이 세 필터 타입 전수 측정이
원본 EQ의 부분집합을 각각 독립적으로 재현하는 가장 근거 있는 설계다. distortion은
pedalboard 기본 구현이 원본과 다른 알고리즘임을 Phase 0.5/1 어디에선가 명시하고
넘어가되, pedalboard가 유일하게 실무에서 접근 가능한 라이브러리이므로 이번 차수는
pedalboard 구현 기준으로 측정하고 이 차이를 결과 해석의 한계로 남긴다(원본 모드
재현은 이번 범위 밖).

## 5. 산출

- 본 문서: `out/results/11_phase0_upstream_audit.md`
- 조사에 사용한 클론(저장소 자체는 커밋하지 않음, 로컬 스크래치에만 존재):
  `KyungsuKim42/tokensynth` (main, render 브랜치),
  `jhtonyKoo/music_mixing_style_transfer`
- 원본 논문 PDF에서 §IV-A 발췌 확인 완료 (`arXiv:2502.08939`)
