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

### 4차 개정 — 임베딩 재추출 없이 분석만 다시

**3차 임베딩(`out/embeddings.npz`)을 그대로 재사용한다. `01_embed.py`는 이번에 실행하지
않았다.** 아래 7개 과제는 모두 이미 있는 임베딩으로 답할 수 있는 질문들이었다.

1. **H1~H5 위계 폐기 → 입력 마스킹 ablation.** 3차 H3(θ만, 0.650)가 H2(0.974)보다 낮게
   나와 포함 관계(H2⊂H3)가 깨졌다 — 구현 오류였다. 근본 원인: H3(J=J(θ), e_dry 무관)와
   H4(J=J(e_dry), θ 무관)는 애초에 서로를 포함하지 않는 **다이아몬드** 구조였다("사다리"가
   아니었다). `M0`/`M_th`/`M_e`/`M_the` 네 변형(입력 슬롯을 0으로 마스킹, residual
   파라미터화는 유지)으로 대체하고 분산 분해(`d_th`, `d_e`, `d_int`)를 낸다.
2. **★최우선 — NSynth 품질 태그 층화.** "dry" 소스가 사실은 이미 상업 샘플 라이브러리의
   룸 리버브·EQ·컴프레션을 포함하고 있을 수 있다("포화 가설"). `examples.json`의
   `qualities` 필드를 읽어 태그 유무로 층화해 R²를 비교한다(`08_quality_stratified.py`,
   신규 스크립트).
3. **텍스트 정렬 신뢰구간.** 3차 정렬도 수치에 CI가 없었다. 텍스트 쌍 복원추출
   부트스트랩으로 자기 정렬·통제·격차(자기−통제)의 CI를 내고, highshelf의 "bright" 역전이
   유의한지 확인한다.
4. **부분공간 투영 무작위 기준선.** 3차 관측값(0.004~0.008)이 이론적 기댓값
   sqrt(9/512)≈0.133보다 20배 이상 작아 정규화 오류 가능성이 제기됐다. 무작위 단위벡터
   1000개를 같은 부분공간에 투영해 기준선을 만들고 실제값과 비교한다.
5. **게이트 분석 원시 점화.** 3차는 wet_level을 25구간으로 나눠 실효 표본이 25개
   수준이었다(damping p=0.07, 비유의). 테스트셋의 모든 (e_dry, θ) 원시 점을 그대로 써서
   표본을 수천 개로 늘린다.
6. **단사성 지표 정의 점검.** `collision_rate=0.018`(threshold=0.99)과
   `nn_cosine_median=0.9953`(threshold보다 높음)이 모순처럼 보였다 — 정의를 명확히 하고
   같은 소스/다른 소스 최근접 이웃을 분리해 재보고한다.
7. **해상도 바닥 명시.** width(음성 통제, R²=0.0087)와 damping(0.0099)/q(0.0142)/
   cutoff(0.0039)가 통계적으로 구분되지 않았다 — 즉 R²<0.02가 이 실험의 측정 한계다.
   `resolution_floor`를 명시적으로 계산해 그 아래는 "약함"이 아니라 "측정 불가"로 표기한다.

각 과제의 실제 관측값과 판정은 아래 "결과 해석 기준"에 있다.

### 파이프라인 (8단계)

```
01_embed.py               결합 LHS 샘플링 + CLAP 오디오/텍스트 임베딩 추출 (4차는 재실행 안 함)
      │
02_surrogate.py            residual MLP 대리모델 학습 + 입력 마스킹 ablation(M0/M_th/M_e/M_the)
      │
      ├── 03_jacobian.py            야코비안 분석 (게이트 구조, 악기 패밀리별 손잡이) ★핵심
      ├── 04_probe.py               다변량 프로브 + width 음성 통제 + 해상도 바닥 + 악기 패밀리 NMI 통제
      ├── 05_text_alignment.py      텍스트-오디오 방향 정렬 검증 + 부트스트랩 CI
      ├── 06_reverse.py             역방향 사상 + cycle consistency + 단사성 진단(정의 명확화)
      ├── 07_subspace.py            (부차) 악기 판별 부분공간 투영 + 무작위 기준선
      └── 08_quality_stratified.py  ★최우선 — NSynth 품질 태그 층화(포화 가설 검증)
```

`03`~`08`은 `02`가 저장한 `surrogate_model.pt`(`08`은 `embeddings.npz`만)를 재사용하며
서로 독립적으로 실행 가능하다(순서 무관, 전부 `results.json`에 이어 붙임). `01`, `02`는
반드시 먼저 실행해야 한다.

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
`scipy`, `soundfile`, `matplotlib`가 추가로 설치됩니다. 4차 개정은 새 의존성이 없습니다 —
`08_quality_stratified.py`는 NSynth가 이미 제공하는 `examples.json`을 표준 `json` 모듈로
읽을 뿐이다.

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

`08_quality_stratified.py`는 추가로 NSynth의 `examples.json`(파일명이 아니라 이 파일에만
있는 `qualities`/`qualities_str` 필드)이 필요합니다 — test split 다운로드에 기본 포함됩니다.

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
되도록 보장한다 (`neutral_check` 참고 — 4차 재실행에서도 세 이펙트 전부 정확히 1.000).

### 실행

```bash
# 1. 결합 LHS 샘플링 + CLAP 오디오/텍스트 임베딩 추출 (오디오는 디스크에 쓰지 않음)
#    800소스 × 57조건 ≈ 45,600 임베딩. M5 CPU 기준 약 2.5~3시간.
#    ★ 4차에서는 이 단계를 재실행하지 않았다 — 기존 out/embeddings.npz를 그대로 쓴다.
python 01_embed.py --audio-dir /path/to/nsynth-test/audio --n-sources 800 --out out

# 2. residual MLP 대리모델 학습 + 입력 마스킹 ablation (surrogate_model.pt 저장)
#    M0/M_th/M_e/M_the + 셔플 통제, 총 5회 학습. M5 CPU에서 40~50분 내외.
python 02_surrogate.py --embeddings out/embeddings.npz --out out

# 3~8 (순서 무관, 03/05/06/07은 02가 저장한 surrogate_model.pt를 재사용, 08은 embeddings.npz만)
python 03_jacobian.py --embeddings out/embeddings.npz --out out           # ★ 이 실험의 핵심
python 04_probe.py --embeddings out/embeddings.npz --out out              # ★ width 통제 + 해상도 바닥
python 05_text_alignment.py --embeddings out/embeddings.npz --out out
python 06_reverse.py --embeddings out/embeddings.npz --out out
python 07_subspace.py --embeddings out/embeddings.npz --out out           # 부차, 시간 없으면 생략 가능
python 08_quality_stratified.py --examples-json nsynth-test/examples.json --embeddings out/embeddings.npz --out out  # ★최우선
```

**환경**: 기본 `--device cpu`. Apple Silicon에서 `--device mps`를 쓰려면 먼저
`PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하세요. `01_embed.py`가 압도적으로 오래 걸리는
단계입니다(45,600개 CLAP forward pass, 4차에서는 스킵). `02`는 마스킹 변형별로 완전히
새로 학습하므로(5회) 3차의 단일 학습보다 느립니다(M5 CPU 기준 40~50분). `03`~`08`은 모두
M5 CPU에서 수 분 내외입니다(야코비안/부분공간/프로브 분석은 batched autograd·Ridge라 빠름).
`04_probe.py`의 `--n-boot`(기본 1000), `07_subspace.py`의 `--n-boot`(기본 300),
`08_quality_stratified.py`의 부트스트랩은 반복 횟수로, 느리면 줄이세요. GPU 클러스터가
없는 환경을 상정해 **1단계(이 문서)에서는 TokenSynth 자체를 재학습하거나 건드리지 않으며
대리모델(작은 MLP) 추론만 수행**합니다 (TokenSynth를 실제로 통과시키는 검증은 "2단계 —
상한 확인" 참고, 이번 구현 범위 밖).

### 출력

```
out/
├── embeddings.npz            임베딩 + 메타(src_id, effect, instrument_family, is_anchor, theta_raw, theta_norm)
├── text_embeddings.npz       텍스트 임베딩 (텍스트 정렬용 캡션 쌍)
├── embed_config.json         재현용 설정 — param_space, param_order, theta_slots, n_samples_per_source
├── surrogate_model.pt        학습된 residual MLP 대리모델 M_the (03/05/06/07이 재사용)
├── results.json              모든 수치 (아래 스키마)
│
├── ablation.png               ★ 4차 신규 — M0/M_th/M_e/M_the 이펙트별 held-out cos + d_th/d_e/d_int
├── variance_decomposition.png ★ 4차 신규 — d_th/d_e/d_int 그룹 막대 (d_int가 상호작용의 직접 증거)
├── surrogate_quality.png      대리모델 신뢰도 — identity/셔플/실제 레이블 held-out 코사인
├── jacobian_gate.png          ★ 4차 개정 — 원시 점 산점도 + 회귀선(‖∂f/∂param‖ vs wet_level)
├── jacobian_by_family.png     ★ 파라미터별 악기 패밀리 간 야코비안 코사인 — 원래 질문
├── param_profile.png          ★ 4차 개정 — 파라미터별 다변량 프로브 R²(CI) + 해상도 바닥 점선
├── width_control.png          ★ 최우선 확인 — width 음성 통제 + 해상도 바닥 (프로브 R² + 야코비안 노름)
├── text_alignment.png         ★ 4차 개정 — 텍스트-오디오 방향 정렬 + 통제 2종 + 부트스트랩 CI 오차막대
├── cycle_consistency.png      정방향→역방향 cycle 코사인 vs 기준선
├── injectivity.png            ★ 4차 신규 — threshold별 충돌률 + 같은/다른 소스 최근접 이웃 코사인 분리
├── subspace_projection.png    ★ 4차 개정 — (부차) 이펙트 방향의 악기 판별 부분공간 투영 + 무작위 기준선 밴드
└── quality_stratified.png     ★ 4차 신규 — NSynth 품질 태그 유무별 R² 비교 (포화 가설)
```

`results.json` 최상위 키: `meta`(실험 버전/샘플링/파라미터 공간), `neutral_check`(θ=0
앵커 검증), `surrogate`(대리모델 신뢰도, M_the 기준), `ablation`(마스킹 ablation 4변형 +
분산 분해), `params`(파라미터별 프로브/야코비안/해상도 통계), `resolution_floor`,
`jacobian_gate_analysis`, `controls`(악기 패밀리 NMI), `text_alignment`, `reverse_model`,
`subspace`, `quality_stratification`.

### 결과 해석 기준

아래 표들은 코드가 내리지 않는 판정 기준이다. **코드는 수치만 산출한다 — 결론은 사람이
이 표로 내린다.** 괄호 안 수치는 800소스 4차 재실행(3차 임베딩 재사용)의 실제 관측값이다.

#### ① 대리모델을 믿어도 되는가 (모든 분석의 전제)

`03_jacobian.py`부터의 모든 분석은 **실제 CLAP의 미분이 아니라 학습된 근사의 미분**이다.
`surrogate.held_out_cos_real`(M_the, **0.985**)이 `held_out_cos_shuffled`(**0.532**)·
`held_out_cos_identity`(**0.970**)를 확실히 넘지 못하면(`surrogate_quality.png`), 야코비안
해석 전체가 무의미하니 여기서 멈추고 대리모델의 용량·학습을 재검토할 것.

→ **관측**: 셔플(0.532)은 확실히 하회, identity(0.970)는 근소하게 상회(+0.015). CLAP
임베딩이 이펙트와 무관하게 "같은 악기·비슷한 음색"이라는 이유만으로도 코사인이 이미
0.97 근방까지 압축되는 공간이라는 뜻 — 대리모델의 절대적 코사인값 자체보다 identity/셔플
대비 상대적 위치로 판단할 것.

#### ② 입력 마스킹 ablation — H1~H5 위계의 대체 (`ablation.png`, `variance_decomposition.png`, `ablation.*`)

3차 H1~H5는 "θ만 vs e_dry만 vs 둘 다"를 하나의 사다리로 늘어놓았으나, θ만 쓰는 모델과
e_dry만 쓰는 모델은 서로를 포함하지 않는 별개 축이라 사다리가 성립하지 않았다(다이아몬드
구조). 대신 같은 아키텍처·학습설정·시드에서 입력 슬롯만 0으로 마스킹한 네 변형을 비교하고,
`d_th`/`d_e`/`d_int`로 분산을 분해한다.

| 변형 | 마스킹 | 의미 |
|---|---|---|
| `M0` | θ, e_dry 둘 다 차단 | 이펙트 종류만으로 설명되는 부분 |
| `M_th` | e_dry 차단 | θ가 추가로 버는 몫 (파라미터 의존) |
| `M_e` | θ 차단 | e_dry가 추가로 버는 몫 (소스 의존) |
| `M_the` | 전부 사용 | 기존 대리모델과 동일 (상한) |

```
d_th  = M_th  − M0   (파라미터 의존이 버는 몫)
d_e   = M_e   − M0   (소스 의존이 버는 몫)
d_int = (M_the − M0) − d_th − d_e   (상호작용에서만 나오는 몫)
```

| 판정 | 기준 |
|---|---|
| `d_int` ≈ 0 | θ와 e_dry의 기여가 거의 가산적 — 상호작용 약함 |
| `d_int` 뚜렷이 양수 | θ·e_dry 결합에서만 나오는 정보 — "손잡이가 악기마다 다르다"는 ③의 직접 증거. ③의 `jacobian_family_cosine`(0.5~0.8)과 함께 볼 것 |
| `d_int` 음수 | 마스킹 결합이 오히려 방해 — 해당 변형의 최적화가 덜 됐을 가능성(학습 노이즈), 절댓값이 작으면 무시 가능 |

→ **관측** (held-out cos, 이펙트별):

| 이펙트 | M0 | M_th | M_e | M_the | d_th | d_e | d_int | family cosine 범위 |
|---|---|---|---|---|---|---|---|---|
| reverb | 0.9722 | 0.9768 | 0.9761 | 0.9848 | +0.0047 | +0.0040 | **+0.0040** | 0.62~0.73 |
| distortion | 0.9386 | 0.9444 | 0.9570 | 0.9622 | +0.0058 | +0.0184 | **−0.0006** | 0.73 |
| highshelf | 0.9943 | 0.9953 | 0.9942 | 0.9961 | +0.0010 | −0.0001 | **+0.0009** | 0.70~0.77 |

세 이펙트 모두 `d_th`, `d_e`, `d_int`가 절댓값 0.02 미만으로 작다 — 3차 H3<H2 같은
포함 관계 위반은 재현되지 않았다(모두 `M_the`가 최댓값으로 정상적인 순서). `d_int`는
reverb·highshelf에서 작지만 양수, distortion에서는 거의 0(부호는 음수이나 크기가 잡음
수준)이다. `jacobian_family_cosine`이 세 이펙트 모두 0.5~0.8 구간(완전 공통도, 완전
악기별도 아님)에 머무는 것과 방향은 일치하지만, `d_int`의 절대 크기 자체는 이 정도
표본·학습 설정에서 강한 상호작용의 증거로 보기엔 작다 — "약한 상호작용" 정도로 읽는 것이
현재 데이터에 부합한다. 참고: `d_th`/`d_e`/`d_int`에는 부트스트랩 CI를 내지 않았으므로
(단일 시드 point estimate) 이 크기 비교 자체의 통계적 유의성은 별도로 검증되지 않았다는
점을 감안할 것.

#### ③ 게이트 구조 (`jacobian_gate.png`, `jacobian_gate_analysis`) — ★ 신뢰도 경고

`‖∂f/∂room_size‖`, `‖∂f/∂damping‖`, `‖∂f/∂width‖`가 `wet_level`에 대해 Spearman ρ로
얼마나 단조 증가하는지를 본다. 4차는 테스트셋의 원시 (e_dry, θ) 점을 그대로 써서(구간
평균 없음) 표본을 6,000개로 늘렸다.

| ρ (wet_level 대비 노름, 원시 점) | 판정 |
|---|---|
| ≥ 0.7, 유의(p<0.05) | 게이트 구조가 대리모델에 실재 — 야코비안 접근 자체가 검증됨 |
| < 0.7 | 대리모델이 게이트 구조를 강하게 학습하지 못함 — 모델 용량/학습 재검토 필요, 이하 야코비안 기반 결론 전체의 신뢰도가 낮아짐 |

→ **관측** (n=6,000, 전부 p<1e-17로 통계적으로 유의하지만 효과 크기가 작음):

| 파라미터 | Spearman ρ | p-value | 선형 R² |
|---|---|---|---|
| room_size | 0.137 | 2.2e-26 | 0.028 |
| damping | 0.112 | 3.2e-18 | 0.014 |
| width (음성 통제) | 0.143 | 9.7e-29 | 0.008 |

**★ ρ가 세 파라미터 모두 0.7에 크게 못 미친다(0.11~0.14) — 3차의 소표본 결과(room_size
0.44, damping 0.37)보다도 낮다.** 더 심각한 건 음성 통제인 `width`의 ρ(0.143)가
`room_size`(0.137)·`damping`(0.112)보다 오히려 **높다**는 점 — 게이트 구조가 실재한다면
음성 통제가 가장 낮아야 하는데 그렇지 않다. 표본이 커지며 통계적 유의성(p-value)은
확보됐지만, 이는 큰 n에서 흔한 현상이며 효과 크기(ρ, R²)가 여전히 작고 통제와 구분되지
않는다는 사실을 가리지 못한다. **이 결과에 따르면 게이트 구조에 대한 이 대리모델의 학습은
불충분하며, 야코비안 기반 결론(②③⑨) 전체의 신뢰도를 그만큼 낮춰서 읽어야 한다** — 특히
②의 `d_int`나 ③의 낮은 family cosine 차이를 "확립된 사실"이 아니라 "이 대리모델
설정하에서의 잠정 관측"으로 취급할 것. 원인 후보: MLP 용량 부족, 300 epoch 부족, wet_level
자체의 표본 범위(0~0.5로 절반만 사용)가 게이트가 뚜렷해지는 구간을 충분히 덮지 못했을 가능성.

#### ④ 악기 패밀리별 손잡이 차이 — 원래 질문 (`jacobian_by_family.png`, `params[*].jacobian_family_cosine`)

파라미터별 악기 패밀리 간 야코비안 열 코사인의 평균. **③의 신뢰도 경고가 적용된다.**

| `cosine_mean` | 판정 | 실용적 귀결 |
|---|---|---|
| > 0.8 | 공통 손잡이로 충분 | 악기 무관하게 파라미터 하나로 조작 가능 |
| 0.5 ~ 0.8 | 대체로 공통이나 일부 예외 | 공통 손잡이 + 악기별 보정 고려 |
| < 0.5 | 악기별 손잡이 필요 | 단일 조작으로는 악기마다 다른 결과 — 악기별 조건화 필요 |

→ **관측**: 9개 파라미터 전부 0.62~0.77 구간("대체로 공통이나 일부 예외")에 몰려 있다
(`wet_level` 0.618, `room_size` 0.726, `damping` 0.640, `width` 0.634, `drive_db` 0.725,
`gain_db` 0.702, `cutoff_frequency_hz` 0.770, `q` 0.730). 0.8을 넘거나 0.5 밑으로 내려가는
파라미터가 하나도 없어 뚜렷한 구분이 나지 않는다 — ③에서 지적했듯 대리모델의 게이트 학습이
불충분한 것과 같은 근본 원인(용량/학습 부족)일 가능성을 배제할 수 없다.

#### ⑤ width 음성 통제 + 해상도 바닥 — 최우선 확인 (`width_control.png`, `param_profile.png`, `resolution_floor`)

| `probe_r2` (width) | 판정 |
|---|---|
| ≈ 0, 셔플과 구분 안 됨 | 파이프라인 정상 (모노 변환이 실제로 stereo 정보를 지웠다) |
| 유의하게 0 초과 | **파이프라인 누수** — `apply_effect`의 모노 처리, pedalboard 채널 처리를 다시 볼 것 |

width는 셔플 통제보다 **강한** 통제다: 셔플은 "레이블이 가짜일 때 못 맞히는지"를 보고,
width는 "**레이블이 진짜인데도** 못 맞혀야 하는지"를 본다.

→ **관측**: `probe_r2(width) = 0.0087`(95% CI [0.0013, 0.0096]), 셔플 통제는 −0.0055 —
파이프라인은 정상이나 CI 상단이 0을 살짝 넘는다.

**★ 해상도 바닥**: `resolution_floor = 0.0096`(=width R² CI 상단, 유일한 음성 통제
기준). 이 값 아래 R²는 "약함"이 아니라 **"측정 불가(below resolution)"**로 표기한다
(`params[*].measurability` 필드, `param_profile.png`/`width_control.png`의 점선).
`cutoff_frequency_hz`(R²=0.0039)가 이 기준으로 측정 불가에 해당한다. `damping`(0.0099)과
`q`(0.0142)는 바닥을 근소하게 넘어 "resolution 이상"으로 표기되지만, 바닥(0.0096)과의
차이가 각각 0.0003/0.0046으로 작아 — 특히 `damping`은 — 여전히 신중하게 읽을 것(엄격한
하드 임계값이라 바닥 바로 위 값도 실질적으로는 잡음과 크게 다르지 않을 수 있다).

| 파라미터 | R² | 해상도 바닥(0.0096) 대비 |
|---|---|---|
| reverb.width (음성 통제) | 0.0087 | 바닥의 정의 그 자체 |
| highshelf.cutoff_frequency_hz | 0.0039 | 측정 불가 |
| reverb.damping | 0.0099 | 바닥 근소 초과 — 신중히 해석 |
| highshelf.q | 0.0142 | 바닥 초과 |
| reverb.room_size | 0.155 | 바닥 훨씬 초과 |
| reverb.wet_level | 0.056 | 바닥 훨씬 초과 |
| highshelf.gain_db | 0.314 | 바닥 훨씬 초과 |
| distortion.drive_db | 0.699 | 바닥 훨씬 초과 |

> 3차 P23 검정에서 `damping`이 (스펙트럼 우선 가설 등에서) 예측 2위였으나 실측은
> 최하위권이었다 — 그러나 위 표에서 보듯 이 값은 해상도 바닥 바로 위에 걸려 있어,
> "damping이 안 읽힌다"는 이 실험의 확립된 결론이 아니라 **측정 한계 이하일 가능성이
> 크다**. 이를 어떤 가설의 기각 근거로 쓰지 말 것.

#### ⑥ 다변량 프로브 전반 (`param_profile.png`, `params[*].probe_r2` / `probe_r2_ci95`)

파라미터별 held-out R²(부트스트랩 95% CI)를 비교한다. CI가 셔플 통제(`≈0`)·해상도
바닥(⑤)과 겹치지 않으면 그 파라미터는 임베딩에서 읽힌다는 뜻. 이펙트 간 R² 차이가
유의한지도 CI 겹침 여부로 판단할 것 — std만으로는 판단하지 말 것.

→ **관측 요약**: `distortion.drive_db`(0.699)가 압도적 1위, 다음이 `highshelf.gain_db`
(0.314), `reverb.room_size`(0.155), `reverb.wet_level`(0.056) 순. 나머지(`damping`,
`q`, `cutoff_frequency_hz`, `width`, `freeze_mode`)는 해상도 바닥 근방이거나 그 아래다.

#### ⑦ ★최우선 — NSynth 품질 태그 층화 (`quality_stratified.png`, `quality_stratification.*`)

NSynth "dry" 소스가 실은 이미 룸 리버브·디스토션·EQ가 걸린 샘플일 수 있다는 "포화 가설"을
검증한다. 태그 유무로 소스를 층화해 같은 파라미터의 R²를 비교한다.

| 판정 기준 | 의미 |
|---|---|
| 태그 없는 쪽(더 깨끗한 소스) R²가 CI 기준으로 유의하게 높음 | **포화 가설 지지** — 이미 걸린 이펙트 위에 더 걸면 임베딩 변화가 작다. 이펙트 간 순서(distortion≫reverb 등)가 캡션 편향이나 강도 문제가 아니라 "이미 걸려 있어서"일 수 있다는 뜻 — 지금까지의 이펙트 간 순서 결론이 흔들린다 |
| 방향은 일치하나 CI가 겹침 | 방향상 포화 가설과 일관되나, 이 표본 크기로는 통계적으로 확정할 수 없음 |
| 태그 있음/없음 R²가 차이 없거나 반대 | 포화 가설 기각 — dry 소스의 잔여 이펙트가 결과에 유의한 영향을 주지 않음 |
| 층화된 쪽 소스 수 < 20 | "결론 없음" — 표본 부족으로 판단 불가 (코드가 자동으로 이렇게 표시) |

→ **태그 분포** (800소스): reverb 203(25.4%), distortion 181(22.6%), dark 139(17.4%),
long_release 127(15.9%), bright 123(15.4%), fast_decay 120(15.0%), nonlinear_env
90(11.2%), percussive 59(7.4%), multiphonic 38(4.8%), tempo-synced 21(2.6%). NSynth
"dry" 소스의 **4분의 1이 이미 reverb 태그를, 거의 4분의 1이 distortion 태그를 갖고
있다** — "dry"라는 이름과 달리 완전히 깨끗한 소스가 아님을 확인.

→ **층화 결과**:

| 비교 | 태그 있음 | 태그 없음 | 격차(없음−있음) |
|---|---|---|---|
| reverb 태그 → `reverb.wet_level` R² | 0.040 (n=203, CI [−0.010, 0.050]) | 0.061 (n=597, CI [0.041, 0.072]) | +0.021 |
| distortion 태그 → `distortion.drive_db` R² | 0.687 (n=181, CI [0.577, 0.753]) | 0.721 (n=619, CI [0.664, 0.751]) | +0.034 |
| bright 태그 → `highshelf.gain_db` R² | 0.222 (n=123, CI [−0.160, 0.315]) | 0.357 (n=677, CI [0.245, 0.385]) | +0.135 |

**세 비교 모두 방향은 포화 가설과 일치한다**(태그 없는 쪽이 항상 R²가 높다) — 특히
`bright`/`highshelf.gain_db`의 격차(+0.135)가 가장 크다. 다만 세 경우 모두 95% CI가
겹친다(예: reverb는 [−0.010,0.050] vs [0.041,0.072]로 [0.041,0.050] 구간에서 겹침) — 이
표본 크기(태그 있음 쪽 n=123~203)로는 **통계적으로 확정할 수준은 아니다**. 방향의 일관성
자체는 포화 가설에 무게를 싣는 정황 증거이지만, "포화 가설이 입증됐다"고 단정할 근거는
아직 아니다 — 태그 있음 쪽 표본을 늘리거나(더 많은 NSynth 소스 포함) 태그 강도(다중
태그 여부)를 함께 보는 후속 분석이 필요하다.

#### ⑧ 텍스트-오디오 방향 정렬 (`text_alignment.png`, `text_alignment.*`)

| 비교 | 기대(캡션 가설이 맞다면) |
|---|---|
| `gap_self_minus_control_ci[e]` (자기 정렬 − 통제) | CI가 0을 포함하지 않고 상단이 양수 (`significant_positive: true`) |
| `reversal_check[e]` (자기 정렬 vs 최선의 교차 정렬) | 교차가 자기보다 유의하게 높으면(`reversal_significant: true`) "그 캡션은 이 이펙트의 대응어가 아니다"의 증거 |

→ **관측**:

| 이펙트 | 자기 정렬 | 격차(자기−통제) 95% CI | 유의? |
|---|---|---|---|
| reverb | 0.160 | [0.138, 0.286] | **유의 (양수)** |
| distortion | 0.287 | [0.064, 0.226] | **유의 (양수)** |
| highshelf | 0.031 | [−0.162, 0.069] | 비유의 (0 포함) |

reverb·distortion은 캡션 가설과 일치 — 자기 정렬이 통제보다 유의하게 높다. **highshelf는
격차 CI가 0을 포함해, "bright" 방향이 자기 텍스트 정렬에서 통제보다 유의하게 낫다고 말할
수 없다.**

→ **역전 검사** (highshelf의 자기 정렬 0.031 vs 최선의 교차 정렬 — distortion 텍스트
방향과의 정렬 0.138):

`cross − self` 95% CI = **[−0.021, 0.233]** → **0을 포함해 비유의**
(`reversal_significant: false`). 점 추정치만 보면 역전(0.138 > 0.031)처럼 보이지만,
부트스트랩 CI 기준으로는 **이 역전이 통계적으로 유의하다고 확인되지 않았다** — "bright는
하이셸프의 대응어가 아니다"의 증거로 쓰기엔 아직 이르다. highshelf 캡션 근사("bright/
crisp") 자체가 부정확할 가능성과, 표본(텍스트 쌍 20개/그룹)이 작아 검정력이 부족할
가능성을 함께 감안할 것.

#### ⑨ 역방향 사상 (`cycle_consistency.png`, `injectivity.png`, `reverse_model.*`)

- **Cycle consistency**: `reverse_model.cycle_consistency[e]`가
  `reverse_model.cycle_baseline[e]`(=`cos(e_dry, e_wet)`, 아무 처리도 안 했을 때의 값)를
  넘지 못하면 역방향 모델이 무의미하다.
- **단사성**: 아래 "★ 4차 개정 — 단사성 정의" 참고.

→ **cycle consistency 관측**: reverb 0.976 > 기준선 0.968 (통과), distortion 0.963 >
기준선 0.927 (통과), highshelf 0.985 < 기준선 0.994 (**기준선 미달** — highshelf는
기준선 자체가 이미 천장(≈0.994)이라 개선 여지가 거의 없었다는 3차 관측과 일치). 개별
파라미터 복원 R²(`reverse_model.param_r2`)는 `distortion.drive_db`(0.754)만 양수이고
나머지는 전부 음수 — wet 임베딩만으로 개별 파라미터값을 복원하는 것은 대부분 평균 예측보다
못하다.

**★ 4차 개정 — 단사성 정의**: 3차는 `collision_rate=0.018`(threshold=0.99)과
`nn_cosine_median=0.9953`(threshold를 이미 넘는 값)을 나란히 보고해 모순처럼 보였다.
실제로는 모순이 아니라 정의가 불충분히 설명된 것이었다 — **collision은 "최근접 이웃이
'다른 소스'이면서 유사도가 threshold를 넘는 경우"만 센다.** 같은 소스의 다른 θ끼리
최근접 이웃으로 잡히는 것(오히려 정상 — 같은 악기의 다른 이펙트 강도는 임베딩이 가까워야
자연스럽다)은 collision이 아니다.

→ **관측**: 최근접 이웃의 **86.7%**가 같은 소스의 다른 θ다. 같은 소스 최근접 이웃의
코사인 중앙값은 0.996, 다른 소스 최근접 이웃의 코사인 중앙값은 0.956로 뚜렷이 낮다 —
`nn_cosine_median_overall`(0.995)이 높은 이유는 표본의 대부분이 "같은 소스" 경우이기
때문이며, "다른 소스인데도 threshold를 넘는" 진짜 충돌은 threshold=0.99에서 1.8%,
threshold=0.95에서 7.4%, threshold=0.999에서 0.9%로 threshold가 엄격해질수록 줄어드는
정상적인 패턴을 보인다 — 3차의 두 수치는 처음부터 모순이 아니었다.

#### ⑩ (부차) 부분공간 투영 + 무작위 기준선 (`subspace_projection.png`, `subspace.*`)

★ 이 분석은 "손잡이가 악기마다 다른가"에 답하지 않는다(④가 담당). "왜 TokenSynth가
이 정보를 무시하는가"에 답하는 별개의 2단계 예비 진단이다.

| `projection_ratio` vs 무작위 기준선 | 판정 |
|---|---|
| 기준선(같은 부분공간에 무작위 단위벡터를 투영한 분포)과 비슷함(백분위 25~75 근방) | 고차원에서의 우연한 직교성 — 해석 불가 |
| 기준선보다 유의하게 낮음(예: 하위 5% 밖) | CLAP이 이펙트 방향과 악기 판별 방향을 적극적으로 분리 |
| 기준선보다 유의하게 높음 | 이펙트 방향이 악기 판별 부분공간에 쏠려 있음 — 조건화가 이펙트 신호를 밀어냈을 가능성 |

→ **정규화 오류 여부**: 무작위 기준선(1000개, 9차원/512차원 LDA 부분공간) 평균 =
**0.130**, 표준편차 0.031 — 이론적 기댓값 `sqrt(9/512) ≈ 0.133`과 사실상 일치한다.
**정규화 오류는 아니었다.**

→ **실제 관측**: 9개 파라미터 전부 투영 비율이 0.004~0.008로, 무작위 기준선의
**1 백분위수(0.065)보다도 낮다**(z-score ≈ −4.0, 모든 파라미터가 기준선 하위 0%
백분위). 이는 "그냥 고차원 직교성"으로 설명되지 않는다 — 기준선 자체가 이미 이론값과
일치하는 정확한 비교 대상이므로, **이펙트 방향이 악기 판별 부분공간을 무작위 기대보다
훨씬 더 적극적으로 피해 간다는 결론이 이제 통계적으로 뒷받침된다.**

### 2단계 — 상한 확인 (이번 구현 범위 밖)

> 진짜 wet 오디오 → CLAP → 임베딩 → TokenSynth → 실제로 wet 소리가 나는가?

이것이 대리모델(M_the) 성능의 상한이다. 진짜(실측) wet 임베딩으로도 TokenSynth가 wet을
재현하지 못한다면, 대리모델이 만든 근사 임베딩으로는 당연히 안 된다 — 대리모델이 아무리
`e_wet`에 가까운 벡터를 만들어도, 그 벡터가 실제 오디오에서 나올 수 있는 영역
밖(off-manifold)이면 TokenSynth는 (진짜 오디오에서 나온 임베딩만 보고 학습했으므로)
알아듣지 못한다.

- **낸다** → 임베딩에 정보가 있고 TokenSynth도 그 정보를 읽는다. `TokenSynth-Aug`의 실패는
  학습 문제였다는 뜻.
- **못 낸다** → 조건화(conditioning) 구조 자체를 바꿔야 한다는 뜻.

1단계(이 문서에 구현된 스크립트)의 결과를 본 뒤 진행할 후속 과제로, 이번 구현에는
포함되지 않는다.

### 유지된 것 (2차에서 검증 완료, 3·4차에서도 변경 없음)

- 스윕 강도(reverb wet 0→0.5, distortion 0→15dB, highshelf ±9dB) — 실무 수준 조정
- 800 소스, src_id 단위 GroupShuffleSplit / 소스 단위 부트스트랩
- 48kHz 리샘플, 피크 정규화 0.7, 무음 제외, 클리핑 방지
- 악기 패밀리 통제(NMI 주 지표, 7클래스 서브샘플) — 4차 재실행: 전체 10클래스 acc=0.772
  (chance-정규화 0.746) / NMI=0.690, 7클래스 서브샘플 acc=0.891(정규화 0.873) / NMI=0.844
- residual 파라미터화(e' = e_dry + Δ), identity/셔플 기준선
- 시드 고정, `embed_config.json` 재현 기록
- **4차: `out/embeddings.npz` 자체는 3차와 완전히 동일 — 재추출 없음**

2차의 `abs_param` 병기와 부호별 방향 분리는 이제 야코비안이 자연히 처리한다(부호 있는
파라미터는 J가 부호에 따라 부호를 바꾸는 것 자체가 신호). 다만 2차에서 확인된
`cos(v_+, v_-) = −0.955`와 이번 판의 야코비안 기반 부호 분석이 정합하는지는 검증 항목으로
남아 있다.
