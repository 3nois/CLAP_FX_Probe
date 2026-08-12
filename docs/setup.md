# 설정 및 실행

CLAP FX Probe 파이프라인 설치·실행·산출물 구조. 프로젝트 개요는 [../README.md](../README.md) 참고.

## 파이프라인 (8단계)

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

## 설치

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

## 체크포인트

TokenSynth가 사용한 것과 **동일한** CLAP 체크포인트를 사용합니다. `01_embed.py`가 첫 실행 시
자동으로 다운로드하여 `ckpts/`에 캐시합니다 (약 2.2GB, `.gitignore`에 의해 git에는 포함되지 않음).

수동으로 받으려면:
```bash
mkdir -p ckpts
curl -L -o ckpts/music_audioset_epoch_15_esc_90.14.pt \
  https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

## 데이터

[NSynth](https://magenta.tensorflow.org/datasets/nsynth) test split (약 4,096개 wav)을
상정합니다. train split(20GB+)은 불필요합니다. 파일명이 NSynth 규칙
(`{instrument}-{pitch}-{velocity}.wav`)을 따라야 악기/피치/패밀리 메타데이터가 파싱됩니다.
악기 패밀리(`instrument_family`)는 `{family}_{acoustic|electronic|synthetic}_{id}` 규칙에서
source-type 토큰 앞부분을 취해 파싱하므로 `synth_lead`처럼 이름에 밑줄이 있는 패밀리도
올바르게 처리됩니다.

`08_quality_stratified.py`는 추가로 NSynth의 `examples.json`(파일명이 아니라 이 파일에만
있는 `qualities`/`qualities_str` 필드)이 필요합니다 — test split 다운로드에 기본 포함됩니다.

## 파라미터 공간

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

## 실행

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

## 출력

`out/`는 6차 후속까지 누적된 산출물이 40여 개라 **유형별 하위 폴더로 정리**했다
(2026-08-06):

```
out/
├── results/    results.json / results_5.json / results_6.json / results_7.json
├── figures/    모든 *.png (25개)
├── caches/     모든 *.npz — embeddings.npz(gitignore 대상, 대용량), text_embeddings.npz,
│               phase1_fd_cache.npz, phase1_fd_theta_cache.npz, phase3_fd_cache.npz,
│               phase3_base_emb.npz, phase3_solo_emb.npz, oat_emb.npz(gitignore 대상 아님
│               — 2차 데이터 소실 재발 방지용으로 의도적으로 커밋 대상), ultrasonic_null_largeN.npz
├── models/     surrogate_model.pt
└── config/     embed_config.json, oat_emb_meta.json
```

★ **스크립트 기본 경로는 아직 이 구조를 모른다.** 01~20 스크립트는 `--out`/`--embeddings`/
`--cache`/`--results` 등의 기본값이 여전히 평평한 `out/`를 가리킨다 — 옛 스크립트를
그대로 재실행하면 새 산출물이 다시 `out/` 최상위에 떨어진다. 이번 정리는 기존 파일을
옮기기만 했고 스크립트는 건드리지 않았다(요청 범위 밖이라 판단). 특정 스크립트를
재실행할 계획이 있으면 그때 해당 스크립트의 기본 경로만 새 구조에 맞게 고치는 편이
안전하다.

`results.json`(1~4차) 최상위 키: `meta`(실험 버전/샘플링/파라미터 공간), `neutral_check`
(θ=0 앵커 검증), `surrogate`(대리모델 신뢰도, M_the 기준), `ablation`(마스킹 ablation
4변형 + 분산 분해), `params`(파라미터별 프로브/야코비안/해상도 통계), `resolution_floor`,
`jacobian_gate_analysis`, `controls`(악기 패밀리 NMI), `text_alignment`, `reverse_model`,
`subspace`, `quality_stratification`. `results_5/6/7.json`은 각 라운드 절에서 설명한
키로 `results.json` 위에 별도 병합 저장된다(원본 미변경).
