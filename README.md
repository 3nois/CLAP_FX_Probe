# CLAP FX Probe

> 원본 TokenSynth 논문(ICASSP 2025) 코드는 [`tokensynth_paper/`](tokensynth_paper/) 폴더로 옮겼습니다 —
> 사용법은 [`tokensynth_paper/README.md`](tokensynth_paper/README.md), 논문 요약은
> [`tokensynth_paper/PAPER_SUMMARY.md`](tokensynth_paper/PAPER_SUMMARY.md) 참고. 이 문서는 현재
> 진행 중인 실험(아래)만 다룹니다.

TokenSynth 논문은 오디오 이펙트(EQ·디스토션·리버브)로 augmentation한 `TokenSynth-Aug`가
이펙트 걸린(wet) 오디오 복제에서 오히려 dry로만 학습한 기본 모델보다 못한 현상을 관찰하고,
그 원인을 "CLAP 임베딩이 오디오 이펙트 정보를 결여했기 때문으로 보인다"고 추정만 했다.
이 하위 프로젝트는 그 추정을 재학습 없이 직접 측정한다.

### 3차 개정 — 왜 "한 번에 하나씩(OAT)"을 버렸는가

reverb의 파라미터는 서로 독립이 아니다. `wet_level`이 다른 모든 파라미터의 **곱셈
게이트**다 — `wet_level=0`이면 `room_size`/`damping`/`width`가 무엇이든 출력은 dry다.
1·2차처럼 OAT(한 번에 하나씩 스윕)로 `damping`을 재려면 `wet_level`을 어딘가 고정해야
하는데, 그 고정값에 따라 "damping 효과"가 완전히 달라진다 — 게이트가 닫혀 있으면 거의
0, 열려 있으면 크게 나온다. 어느 지점을 볼지가 임의 선택이 되어 "CLAP이 damping을
읽는다"는 진술 자체가 정의되지 않는다. 격자 탐색(7^5=16,807 조합/소스)도 비용상
불가능하다.

그래서 파라미터 공간을 **결합(joint) Latin Hypercube 샘플링**하고, 미분 가능한
**대리모델(residual MLP)**을 학습한 뒤 그 **야코비안 J = ∂e'/∂θ**를 분석하는 방식으로
바꿨다. 비용은 OAT와 비슷한데 파라미터 간 상호작용까지 잡힌다.

### 파이프라인 (7단계)

```
01_embed.py         결합 LHS 샘플링 + CLAP 오디오/텍스트 임베딩 추출
      │
02_surrogate.py      residual MLP 대리모델 학습 + H1~H5 위계 재구성
      │
      ├── 03_jacobian.py       야코비안 분석 (게이트 구조, 악기 패밀리별 손잡이) ★핵심
      ├── 04_probe.py          다변량 프로브 + width 음성 통제 + 악기 패밀리 NMI 통제
      ├── 05_text_alignment.py 텍스트-오디오 방향 정렬 검증
      ├── 06_reverse.py        역방향 사상 + cycle consistency + 단사성 진단
      └── 07_subspace.py       (부차) 악기 판별 부분공간 투영
```

`03`~`07`은 `02`가 저장한 `surrogate_model.pt`를 재사용하며 서로 독립적으로 실행
가능하다(순서 무관, 전부 `results.json`에 이어 붙임). `01`, `02`는 반드시 먼저 실행해야
한다.

### 설치

```bash
pip install -e "./tokensynth_paper[probe]"
```

TokenSynth 패키지 정의(`pyproject.toml`)가 `tokensynth_paper/`로 옮겨졌으므로 저장소 루트에서
위 경로로 설치합니다.

> **알려진 문제**: pip가 `torchaudio`를 `torch`와 호환되지 않는 버전(예: torch 2.5.1 +
> torchaudio 2.11.0)으로 설치해 `_torchaudio.abi3.so` 관련 `OSError`가 날 수 있습니다.
> 발생하면 `torch`와 같은 마이너 버전으로 맞춰 재설치하세요: `pip install torchaudio==2.5.1`
> (설치된 `torch` 버전은 `python -c "import torch; print(torch.__version__)"`로 확인).

기존 TokenSynth 의존성(`laion-clap`, `torch` 등)에 더해 `pedalboard`, `scikit-learn`,
`scipy`, `soundfile`, `matplotlib`가 추가로 설치됩니다. 3차 개정은 새 의존성이 없습니다 —
LHS 샘플링은 `scipy.stats.qmc`, 야코비안은 `torch.func`(jacrev/vmap), 부분공간 투영은
`sklearn.discriminant_analysis`를 씁니다.

### 체크포인트

TokenSynth가 사용한 것과 **동일한** CLAP 체크포인트를 사용합니다. `01_embed.py`가 첫 실행 시
자동으로 다운로드하여 `ckpts/`에 캐시합니다 (약 2.2GB, `.gitignore`에 의해 git에는 포함되지 않음).

수동으로 받으려면:
```bash
mkdir -p ckpts
curl -L -o ckpts/music_audioset_epoch_15_esc_90.14.pt \
  https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

### 데이터

[NSynth](https://magenta.tensorflow.org/datasets/nsynth) test split (약 4,096개 wav)을
상정합니다. train split(20GB+)은 불필요합니다. 파일명이 NSynth 규칙
(`{instrument}-{pitch}-{velocity}.wav`)을 따라야 악기/피치/패밀리 메타데이터가 파싱됩니다.
악기 패밀리(`instrument_family`)는 `{family}_{acoustic|electronic|synthetic}_{id}` 규칙에서
source-type 토큰 앞부분을 취해 파싱하므로 `synth_lead`처럼 이름에 밑줄이 있는 패밀리도
올바르게 처리됩니다.

### 파라미터 공간

pedalboard 0.9.24 실제 시그니처로 확인한 파라미터명입니다 (`Reverb`:
room_size/damping/wet_level/dry_level/width/freeze_mode, `Distortion`: drive_db,
`HighShelfFilter`: cutoff_frequency_hz/gain_db/q).

| 이펙트 | 파라미터 | 범위 | 스케일 | 비고 |
|---|---|---|---|---|
| reverb | wet_level | 0.0 → 0.5 | 선형 | 게이트 — 나머지 4개를 곱셈으로 통제 |
| reverb | room_size | 0.0 → 0.9 | 선형 | |
| reverb | damping | 0.0 → 1.0 | 선형 | |
| reverb | **width** | 0.0 → 1.0 | 선형 | **★ 음성 통제** — 모노 파이프라인이라 원리적으로 무영향이어야 함 |
| reverb | freeze_mode | {0,1} | 베르누이(p=0.5) | |
| distortion | drive_db | 0.0 → 15.0 | 선형 | |
| highshelf | gain_db | −9.0 → +9.0 | 선형, signed | 0 중심 대칭 |
| highshelf | cutoff_frequency_hz | 500 → 8000 | 로그 | |
| highshelf | q | 0.3 → 3.0 | 로그 | |

소스당 표본 수: dry 1 + reverb 32 + distortion 8 + highshelf 16 = **57**. 각 이펙트의
표본 중 정확히 1개는 **θ=0 앵커**(모든 파라미터가 "무효과" 값)로 예약되며, 이 앵커는
pedalboard를 통과시키지 않고 dry 오디오를 그대로 써서 `cos(e_dry, e_theta0) = 1.000`이
되도록 보장한다 (`neutral_check` 참고 — 2차의 "무효과 레벨이 진짜 dry가 아닐 수 있다"는
문제가 여기서 구조적으로 해소된다).

### 실행

```bash
# 1. 결합 LHS 샘플링 + CLAP 오디오/텍스트 임베딩 추출 (오디오는 디스크에 쓰지 않음)
#    800소스 × 57조건 ≈ 45,600 임베딩. M5 CPU 기준 약 2.5~3시간.
python 01_embed.py --audio-dir /path/to/nsynth-test/audio --n-sources 800 --out out

# 2. residual MLP 대리모델 학습 + H1~H5 위계 재구성 (surrogate_model.pt 저장)
python 02_surrogate.py --embeddings out/embeddings.npz --out out

# 3~7 (순서 무관, 모두 02가 저장한 surrogate_model.pt를 재사용)
python 03_jacobian.py --embeddings out/embeddings.npz --out out       # ★ 이 실험의 핵심
python 04_probe.py --embeddings out/embeddings.npz --out out          # ★ width 통제 최우선 확인
python 05_text_alignment.py --embeddings out/embeddings.npz --out out
python 06_reverse.py --embeddings out/embeddings.npz --out out
python 07_subspace.py --embeddings out/embeddings.npz --out out       # 부차, 시간 없으면 생략 가능
```

**환경**: 기본 `--device cpu`. Apple Silicon에서 `--device mps`를 쓰려면 먼저
`PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하세요. `01_embed.py`가 압도적으로 오래 걸리는
단계입니다(45,600개 CLAP forward pass). `02`~`07`은 모두 M5 CPU에서 수 분~수십 분
내외입니다(대리모델 학습은 pooled 45,600행에 300 epoch 기준으로 십수 분, 야코비안/부분공간
분석은 batched autograd라 빠름). `04_probe.py`의 `--n-boot`(기본 1000)와
`07_subspace.py`의 `--n-boot`(기본 300)는 부트스트랩 반복 횟수로, 느리면 줄이세요.
GPU 클러스터가 없는 환경을 상정해 **1단계(이 문서)에서는 TokenSynth 자체를 재학습하거나
건드리지 않으며 대리모델(작은 MLP) 추론만 수행**합니다 (TokenSynth를 실제로 통과시키는
검증은 "2단계 — 상한 확인" 참고, 이번 구현 범위 밖).

### 출력

```
out/
├── embeddings.npz          임베딩 + 메타(src_id, effect, instrument_family, is_anchor, theta_raw, theta_norm)
├── text_embeddings.npz     텍스트 임베딩 (과제5용 캡션 쌍)
├── embed_config.json       재현용 설정 — param_space, param_order, theta_slots, n_samples_per_source
├── surrogate_model.pt      학습된 residual MLP 대리모델 (03/05/06/07이 재사용)
├── results.json            모든 수치 (아래 스키마)
│
├── hierarchy.png            H1~H5 위계 사다리 (이펙트별 3분할)
├── surrogate_quality.png    대리모델 신뢰도 — identity/셔플/실제 레이블 held-out 코사인
├── jacobian_gate.png        ★ ‖∂f/∂param‖ vs wet_level — 게이트 구조 검증
├── jacobian_by_family.png   ★ 파라미터별 악기 패밀리 간 야코비안 코사인 — 원래 질문
├── param_profile.png        파라미터별 다변량 프로브 R² (부트스트랩 95% CI)
├── width_control.png        ★ 최우선 확인 — width 음성 통제 (프로브 R² + 야코비안 노름)
├── text_alignment.png       텍스트-오디오 방향 정렬 + 통제 2종(무작위/교차)
├── cycle_consistency.png    정방향→역방향 cycle 코사인 vs 기준선
└── subspace_projection.png  (부차) 이펙트 방향의 악기 판별 부분공간 투영 비율
```

`results.json` 최상위 키: `meta`(실험 버전/샘플링/파라미터 공간), `neutral_check`(θ=0
앵커 검증), `surrogate`(대리모델 신뢰도 + H1~H5), `params`(파라미터별 프로브/야코비안
통계), `controls`(악기 패밀리 NMI), `text_alignment`, `reverse_model`, `subspace`.

### 결과 해석 기준

아래 표들은 코드가 내리지 않는 판정 기준이다. **코드는 수치만 산출한다 — 결론은 사람이
이 표로 내린다.**

#### ① 대리모델을 믿어도 되는가 (모든 분석의 전제)

`03_jacobian.py`부터의 모든 분석은 **실제 CLAP의 미분이 아니라 학습된 근사의 미분**이다.
`surrogate.held_out_cos_real`이 `surrogate.held_out_cos_shuffled`·`held_out_cos_identity`를
확실히 넘지 못하면(`surrogate_quality.png`), 야코비안 해석 전체가 무의미하니 여기서 멈추고
대리모델의 용량·학습을 재검토할 것.

#### ② 게이트 구조 (`jacobian_gate.png`, `params[*].gate_spearman_vs_wet_level`)

`‖∂f/∂room_size‖`, `‖∂f/∂damping‖`, `‖∂f/∂width‖`가 `wet_level`에 대해 Spearman ρ로
얼마나 단조 증가하는지를 본다.

| ρ (wet_level 대비 노름) | 판정 |
|---|---|
| > 0.5, 유의(p<0.05) | 게이트 구조가 대리모델에 실재 — 야코비안 접근 자체가 검증됨 |
| ≈ 0 또는 비유의 | 대리모델이 게이트 구조를 학습하지 못함 — 모델 용량/학습 재검토 필요, 이하 분석 신뢰 낮음 |

이 확인이 이 실험의 **내적 타당성 검증**이다 — 여기서 실패하면 (b) 악기 패밀리 분석도
신뢰할 수 없다.

#### ③ 악기 패밀리별 손잡이 차이 — 원래 질문 (`jacobian_by_family.png`, `params[*].jacobian_family_cosine`)

파라미터별 악기 패밀리 간 야코비안 열 코사인의 평균.

| `cosine_mean` | 판정 | 실용적 귀결 |
|---|---|---|
| > 0.8 | 공통 손잡이로 충분 | 악기 무관하게 파라미터 하나로 조작 가능 |
| 0.5 ~ 0.8 | 대체로 공통이나 일부 예외 | 공통 손잡이 + 악기별 보정 고려 |
| < 0.5 | 악기별 손잡이 필요 | 단일 조작으로는 악기마다 다른 결과 — 악기별 조건화 필요 |

#### ④ width 음성 통제 — 최우선 확인 (`width_control.png`, `params["reverb.width"]`)

| `probe_r2` (width) | 판정 |
|---|---|
| ≈ 0, 셔플과 구분 안 됨 | 파이프라인 정상 (모노 변환이 실제로 stereo 정보를 지웠다) |
| 유의하게 0 초과 | **파이프라인 누수** — 1·2차 결과까지 재검토 대상. `apply_effect`의 모노 처리, pedalboard 채널 처리를 다시 볼 것 |

width는 셔플 통제보다 **강한** 통제다: 셔플은 "레이블이 가짜일 때 못 맞히는지"를 보고,
width는 "**레이블이 진짜인데도** 못 맞혀야 하는지"를 본다.

#### ⑤ H1~H5 위계 (`hierarchy.png`, `surrogate.hierarchy_H1_to_H5`)

| 단계 | 형태 | 의미 |
|---|---|---|
| H1 | Δ = V·θ + b | 선형, θ만, 소스 무관 — J는 상수 |
| H2 | Δ = g(θ)·v | 방향 고정 v, 크기만 θ 의존 |
| H3 | Δ = MLP(θ) | 비선형, θ만, 소스 무관 — J = J(θ) |
| H4 | Δ = M(e_dry)·θ | θ에 선형, 계수(하이퍼네트워크)는 e_dry의 함수 — J = J(e_dry) |
| H5 | Δ = MLP(e_dry, effect, θ) | 완전 비선형 — J = J(e_dry, θ) (실제 대리모델) |

각 칸의 held-out 코사인을 `identity`(e'=e_dry)와 `shuffle_control`과 비교한다. 두
기준선을 모두 이기는 가장 단순한(앞선) 칸이 해당 이펙트의 위계다.

> **알려진 이슈 (미해결)**: 800소스 본 실행에서 세 이펙트 전부 H3(θ만, 비선형)가
> H1(θ만, 선형)보다 뚜렷이 나쁘게 나왔다(예: reverb 0.65 vs 0.98). H3는 이론상 H1을
> 포함하는 상위 모델이라 이래선 안 된다 — "θ만으로는 안 읽힌다"는 실질적 발견이 아니라
> `ThetaOnlyMLP` 학습이 수렴하지 못했을 가능성이 높다(세 이펙트에서 거의 같은 값으로
> 수렴한 게 방증). H4/H5는 정상적으로 identity를 상회하므로 이 문제가 다른 결과에
> 전염되지는 않았지만, H3 자체의 결론은 신뢰하지 말 것 — 후속 조사 필요.

#### ⑥ 다변량 프로브 (`param_profile.png`, `params[*].probe_r2` / `probe_r2_ci95`)

파라미터별 held-out R²(부트스트랩 95% CI)를 그대로 비교한다. CI가 셔플 통제(`≈0`)와
겹치지 않으면 그 파라미터는 임베딩에서 읽힌다는 뜻. 이펙트 간 R² 차이가 유의한지도 CI
겹침 여부로 판단할 것 — std만으로는 판단하지 말 것.

#### ⑦ 텍스트-오디오 방향 정렬 (`text_alignment.png`, `text_alignment.*`)

| 비교 | 기대(캡션 가설이 맞다면) |
|---|---|
| `cos_by_effect[e]` (자기 정렬) | `cos_random_control`보다 뚜렷이 높음 |
| `cos_by_effect[e]` (자기 정렬) | `cos_cross_effect["{e}_vs_{other}"]`(교차 정렬)보다 높음 |

둘 다 성립하면 그 이펙트에 대해 "CLAP이 캡션 개념과 정합적인 방향으로 이펙트를
인코딩한다"는 근거. `highshelf`는 대응하는 정확한 캡션 관용구가 없어 "bright/crisp"
계열로 근사했다 — 정렬도가 낮게 나와도 "정렬이 없다"가 아니라 "근사가 부정확했을
가능성"을 먼저 의심할 것.

#### ⑧ 역방향 사상 (`cycle_consistency.png`, `reverse_model.*`)

- **Cycle consistency**: `reverse_model.cycle_consistency[e]`가
  `reverse_model.cycle_baseline[e]`(=`cos(e_dry, e_wet)`, 아무 처리도 안 했을 때의 값)를
  넘지 못하면 역방향 모델이 무의미하다. `highshelf`는 baseline 자체가 이미 천장(2차 기준
  ≈0.997)이라 개선 여지가 거의 없다는 점을 감안할 것.
- **단사성**: `reverse_model.injectivity_collision_rate`가 유의하게 0보다 크면(서로 다른
  소스가 최근접 이웃으로 자주 충돌하면), 원리적으로 역방향이 불가능한 영역이 있다는 뜻 —
  이 또한 유효한 결과다.

#### ⑨ (부차) 부분공간 투영 (`subspace_projection.png`, `subspace.projection_ratio_by_param`)

★ 이 분석은 "손잡이가 악기마다 다른가"에 답하지 않는다(③이 담당). "왜 TokenSynth가
이 정보를 무시하는가"에 답하는 별개의 2단계 예비 진단이다.

| `projection_ratio`(=‖proj_instrument(v)‖/‖v‖) | 판정 |
|---|---|
| 0에 가까움 | 이펙트 방향이 악기 판별 축과 직교 — TokenSynth가 무시하는 이유는 다른 데 있음 |
| 1에 가까움 | 이펙트 방향이 악기 판별 부분공간 내부 — 악기 정체성 조건화가 이펙트 신호를 "밀어냈을" 가능성 |

### 2단계 — 상한 확인 (이번 구현 범위 밖)

> 진짜 wet 오디오 → CLAP → 임베딩 → TokenSynth → 실제로 wet 소리가 나는가?

이것이 대리모델(H5) 성능의 상한이다. 진짜(실측) wet 임베딩으로도 TokenSynth가 wet을
재현하지 못한다면, 대리모델이 만든 근사 임베딩으로는 당연히 안 된다 — 대리모델이 아무리
`e_wet`에 가까운 벡터를 만들어도, 그 벡터가 실제 오디오에서 나올 수 있는 영역
밖(off-manifold)이면 TokenSynth는 (진짜 오디오에서 나온 임베딩만 보고 학습했으므로)
알아듣지 못한다.

- **낸다** → 임베딩에 정보가 있고 TokenSynth도 그 정보를 읽는다. `TokenSynth-Aug`의 실패는
  학습 문제였다는 뜻.
- **못 낸다** → 조건화(conditioning) 구조 자체를 바꿔야 한다는 뜻.

1단계(이 문서에 구현된 스크립트)의 결과를 본 뒤 진행할 후속 과제로, 이번 구현에는
포함되지 않는다.

### 유지된 것 (2차에서 검증 완료, 3차에서도 변경 없음)

- 스윕 강도(reverb wet 0→0.5, distortion 0→15dB, highshelf ±9dB) — 실무 수준 조정
- 800 소스, src_id 단위 GroupShuffleSplit / 소스 단위 부트스트랩
- 48kHz 리샘플, 피크 정규화 0.7, 무음 제외, 클리핑 방지
- 악기 패밀리 통제(NMI 주 지표, 7클래스 서브샘플)
- residual 파라미터화(e' = e_dry + Δ), identity/셔플 기준선
- 시드 고정, `embed_config.json` 재현 기록

2차의 `abs_param` 병기와 부호별 방향 분리는 이제 야코비안이 자연히 처리한다(부호 있는
파라미터는 J가 부호에 따라 부호를 바꾸는 것 자체가 신호). 다만 2차에서 확인된
`cos(v_+, v_-) = −0.955`와 이번 판의 야코비안 기반 부호 분석이 정합하는지는 검증 항목으로
남아 있다.
