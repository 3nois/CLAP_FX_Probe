# TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrument
![Description](media/figure.png)

<div align="center">

[![Build Status](https://github.com/KyungsuKim42/tokensynth/actions/workflows/test_and_publish.yml/badge.svg)](https://github.com/KyungsuKim42/tokensynth/actions)
[![PyPI version](https://img.shields.io/pypi/v/tokensynth.svg)](https://pypi.org/project/tokensynth/)
[![License](https://img.shields.io/pypi/l/tokensynth.svg)](https://github.com/KyungsuKim42/tokensynth/blob/main/LICENSE)

[Kyungsu Kim](https://scholar.google.com/citations?user=bCMZWFIAAAAJ&hl=en&oi=sra), [Junghyun Koo](https://scholar.google.com/citations?user=9LbxECcAAAAJ&hl=en), [Sungho Lee](https://scholar.google.com/citations?hl=en&user=8yMXL5AAAAAJ), [Haesun Joung](https://scholar.google.com/citations?hl=en&user=yV8xVKoAAAAJ), [Kyogu Lee](https://scholar.google.com/citations?user=Fk4jQFEAAAAJ&hl=en)

📄 [Paper](https://arxiv.org/abs/2502.08939) | 🎵 [Demo Page](http://tinyurl.com/tokensynth-demo)


</div>

###  **Official implementation** of "TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrument", published in **ICASSP 2025**.

TokenSynth is a token-based neural synthesizer that generates polyphonic single-instrument musical audio from MIDI and timbre embeddings, enabling instrument cloning, text-to-instrument synthesis, and timbre manipulation. It uses a decoder-only transformer trained on neural audio tokens with CLAP-based timbre conditioning, allowing for flexible sound design without fine-tuning.

## Installation

To install TokenSynth, simply run:

```bash
pip install tokensynth
```

## Quickstart

```python
from tokensynth import TokenSynth, CLAP, DACDecoder
import audiofile
import torch

# Set file paths
ref_audio = "media/reference_audio.wav"
midi = "media/input_midi.mid"

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
synth = TokenSynth.from_pretrained(aug=True, device=device)
clap = CLAP(device=device)
decoder = DACDecoder(device=device)

with torch.no_grad():
    # Extract timbre embeddings from audio and text
    timbre_audio = clap.encode_audio(ref_audio)
    timbre_text = clap.encode_text("warm smooth electronic bass")
    timbre_audio_text = 0.5 * timbre_audio + 0.5 * timbre_text

    # Generate audio tokens
    tokens_audio = synth.synthesize(timbre_audio, midi, top_k=10)
    tokens_text = synth.synthesize(timbre_text, midi, top_p=0.6, guidance_scale=1.6)
    tokens_audio_text = synth.synthesize(timbre_audio_text, midi, top_p=0.6, guidance_scale=1.6)

    # Decode tokens into audio waveforms
    audio_audio = decoder.decode(tokens_audio) 
    audio_text = decoder.decode(tokens_text)
    audio_audio_text = decoder.decode(tokens_audio_text)

# Save audio files
audiofile.write("media/output_audio.wav", audio_audio.cpu().numpy(), 16000)
audiofile.write("media/output_text.wav", audio_text.cpu().numpy(), 16000)
audiofile.write("media/output_audio_text.wav", audio_audio_text.cpu().numpy(), 16000)
```

You can also run `python quickstart.py` from the project root directory.

## Model Weights
TokenSynth automatically downloads pretrained weights when initialized.  
If you want to manually download the weights, you can find them here:  

[🔗 TokenSynth Pretrained Weights](https://huggingface.co/KyungsuKim/TokenSynth/tree/main)

## Citation

```bibtex
@misc{kim2025tokensynthtokenbasedneuralsynthesizer,
      title={TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrument}, 
      author={Kyungsu Kim and Junghyun Koo and Sungho Lee and Haesun Joung and Kyogu Lee},
      year={2025},
      eprint={2502.08939},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.08939}, 
}
```
## LICENSE

This project is released under the [MIT License](./LICENSE)

### Acknowledgements:
This work utilizes codebase and pretrained weights of [DAC](https://github.com/descriptinc/descript-audio-codec) and [CLAP](https://github.com/LAION-AI/CLAP).

---

## CLAP FX Probe

TokenSynth 논문은 오디오 이펙트(EQ·디스토션·리버브)로 augmentation한 `TokenSynth-Aug`가
이펙트 걸린(wet) 오디오 복제에서 오히려 dry로만 학습한 기본 모델보다 못한 현상을 관찰하고,
그 원인을 "CLAP 임베딩이 오디오 이펙트 정보를 결여했기 때문으로 보인다"고 추정만 했다.
이 하위 프로젝트(`01_embed.py`, `02_analyze.py`, `03_mapping.py`)는 그 추정을 재학습 없이
직접 측정한다.

단, "차이 벡터가 전부 같은 방향인가"라는 단일 질문은 너무 엄격하다 — 방향이 소스마다
달라도 어떤 방법으로든 dry 임베딩에서 wet 임베딩을 예측할 수 있으면 실용적으로 충분하다.
그래서 단일 가설이 아니라 **구조의 위계(H0~H6)**로 측정한다 — 자세한 표는 아래 "결과 해석
기준" 참고.

### 설치

```bash
pip install -e ".[probe]"
```

> **알려진 문제**: pip가 `torchaudio`를 `torch`와 호환되지 않는 버전(예: torch 2.5.1 +
> torchaudio 2.11.0)으로 설치해 `_torchaudio.abi3.so` 관련 `OSError`가 날 수 있습니다.
> 발생하면 `torch`와 같은 마이너 버전으로 맞춰 재설치하세요: `pip install torchaudio==2.5.1`
> (설치된 `torch` 버전은 `python -c "import torch; print(torch.__version__)"`로 확인).

기존 TokenSynth 의존성(`laion-clap`, `torch` 등)에 더해 `pedalboard`, `scikit-learn`,
`scipy`, `soundfile`, `matplotlib`가 추가로 설치됩니다.

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
(`{instrument}-{pitch}-{velocity}.wav`)을 따라야 악기/피치 메타데이터가 파싱됩니다.

### 실행

```bash
# 1. 이펙트 적용 + CLAP 임베딩 추출 (오디오는 디스크에 쓰지 않음)
python 01_embed.py --audio-dir /path/to/nsynth-test/audio --n-sources 300 --out out

# 2. (a)(b)(c) 분석 + 그림 + 공통 통제(레이블 셔플/무작위 벡터/악기 분류 상한)
python 02_analyze.py --embeddings out/embeddings.npz --out out

# 3. (d) 사상 모델(residual MLP) 학습 + H1~H5 위계 사다리 비교
#    out/results.json을 읽어 이어 붙이므로 반드시 2번 다음에 실행할 것
python 03_mapping.py --embeddings out/embeddings.npz --results out/results.json --out out
```

**환경**: 기본 `--device cpu`. Apple Silicon에서 `--device mps`를 쓰려면 먼저
`PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하세요 (CLAP 일부 연산이 MPS에 없어 CPU로 폴백 필요).
`01_embed.py`는 300 소스 기준 M-시리즈 CPU에서 약 10~25분 소요됩니다. `03_mapping.py`는
TokenSynth를 통과시키지 않는 작은 MLP만 학습하므로 M5 CPU에서 수 분이면 끝납니다.
GPU 클러스터가 없는 환경을 상정해 **1단계(이 문서)에서는 TokenSynth 자체를 재학습하거나
건드리지 않으며 추론만 수행**합니다 (TokenSynth를 통과시키는 검증은 "2단계 — 상한 확인"
참고, 이번 구현 범위 밖).

### 출력

```
out/
├── embeddings.npz      임베딩 + 메타(src_id, effect, param_value, instrument, pitch)
├── embed_config.json   재현용 설정 기록
├── results.json        모든 수치 — (a)(b)(c) 측정, 공통 통제, H0~H6 위계(hierarchy), 사상 모델(mapping_model)
├── probe_r2.png         이펙트별 R² vs 셔플 기준선 vs 악기 분류 상한
├── direction_cos.png    ① 방향 일관성(정규화 후 코사인) vs 무작위 벡터, ② 크기-파라미터 Spearman ρ
├── monotonicity.png     파라미터 값 vs 방향 벡터 투영값 산점도
├── mapping_cos.png      사상 모델(H5) 성능 vs identity vs 셔플(동일 용량) 기준선
└── hierarchy.png        H1~H5 위계 사다리 비교 (이펙트별 3분할, identity/셔플 기준선 포함)
```

### 결과 해석 기준 — 구조의 위계 (H0~H6)

"차이 벡터가 전부 같은 방향인가"는 가장 엄격한 질문(H1)이다. 이게 깨져도 실험은 끝나지
않는다 — 아래 표에서 **어느 칸까지 구조가 잡히는지** 찾는 것이 목표다. 코드는 수치만
내고, 최종 판정은 이 표로 사람이 한다. **어느 칸에서 잡히든 유효한 결과다.**

| 단계 | 형태 | 의미 | `results.json`에서 볼 곳 | 실용적 귀결 |
|---|---|---|---|---|
| **H0** | 정보 없음 | 아무 방법으로도 못 읽음 | 아래 모든 지표가 통제 수준 | 별도 이펙트 인코더 필요 |
| **H1** | `e' = e + v` | 상수 벡터 | `hierarchy.<effect>.H1` | 벡터 하나만 더하면 됨 — 사실상 무료 |
| **H2** | `e' = e + f(p)·v` | 방향 고정, 크기만 파라미터 의존 | `hierarchy.<effect>.H2`, `direction_cos.png`②, `magnitude_spearman_rho` | 스칼라 함수 하나만 fit하면 됨 |
| **H3** | `e' = e + Δ(p)` | 파라미터별 방향, 소스와 무관 | `hierarchy.<effect>.H3` | 파라미터→벡터 룩업으로 충분 |
| **H4** | `e' = W·e + b` | 선형 변환, 소스마다 다르게 이동 | `hierarchy.<effect>.H4` | 작은 선형 계층 하나 추가 |
| **H5** | `e' = e + g(e, p)` | 비선형 (residual MLP) | `hierarchy.<effect>.H5`, `mapping_model.held_out_cos_real_labels` | MLP 추론 1회 — 여전히 실용적 |
| **H6** | 정보는 있으나 학습 불가 | 사실상 H0 | H1~H5가 전부 통제 수준에 머무름 | 별도 이펙트 인코더 필요 (H0과 동일 결론) |

**판정 방법**: `hierarchy.<effect>`의 각 칸(H1~H5)을 같은 이펙트의
`hierarchy.<effect>.identity`와 `hierarchy.<effect>.shuffle_control`(또는
`mapping_model.held_out_cos_shuffled_labels`)과 비교한다. **두 기준선을 모두 이기는 가장
앞선(단순한) 칸**이 그 이펙트가 위치한 위계다. 어느 칸도 두 기준선을 못 이기면 H6(=사실상
H0)으로 본다.

- **identity 기준선이 왜 필요한가**: `cos(e_dry, e_wet)`은 이펙트가 약하면 원래도 높게
  나온다. 이 기준선 없이는 "잘 예측했다"는 착시가 생긴다.
- **셔플 기준선의 "동일 용량" 조건**: `03_mapping.py`의 셔플 통제는 실제 모델과 **완전히
  같은 아키텍처·에폭·학습 절차**로, 레이블(=목표 `e_wet`)만 무작위로 섞어 학습한다.
  MLP는 용량이 크면 정보가 없어도 train loss를 낮출 수 있으므로, 반드시 held-out
  코사인으로만 비교하고 절대치가 아니라 **실제 레이블 모델과의 격차**를 신뢰할 것.
  **프로브(사상 모델)가 강력해질수록 이 통제의 중요성도 커진다.**
- **(a)(b)(c) 공통 통제** — `controls.random_vector_cosine_mean`(무작위 벡터, ≈0 기대),
  `probe_r2_shuffled`(레이블 셔플, ≈0 기대), `controls.instrument_classification.accuracy`
  (악기 분류 상한, 논문 기준 90.4%, 표본 수가 훨씬 적어 절대치보다 **이펙트 프로브와의
  상대적 격차**로 해석)는 이 위계 판정 이전에 "측정 자체가 통계적 우연이 아닌지"를
  검증하는 용도다.
- **방향/크기를 반드시 나눠서 볼 것.** `direction_cosine_mean`(정규화 후 코사인, H1의 방향
  일관성)과 `magnitude_spearman_rho`(‖차이 벡터‖-파라미터 상관, H2의 크기 의존성)를
  분리하지 않고 정규화 없이 코사인만 재면, 소스마다 다른 벡터 크기가 방향 불일치로
  오인되어 H2인 경우를 H0으로 잘못 판정하게 된다.
- `highshelf`처럼 파라미터 범위가 0(dry)을 사이에 둔 대칭이면(`-15~+15`), signed
  `monotonicity_spearman_rho`·`magnitude_spearman_rho`가 낮게(혹은 음수로) 나올 수 있다 —
  방향 벡터 투영값/크기가 dry 근방에서 최솟값을 갖는 V자 형태이기 때문이다. 이 경우
  각각의 `..._abs_param` 필드(`|파라미터|` 기준 상관)를 함께 볼 것.

### 2단계 — 상한 확인 (이번 구현 범위 밖)

> 진짜 wet 오디오 → CLAP → 임베딩 → TokenSynth → 실제로 wet 소리가 나는가?

이것이 사상 모델(H5) 성능의 상한이다. 진짜(실측) wet 임베딩으로도 TokenSynth가 wet을
재현하지 못한다면, 사상 모델이 만든 근사 임베딩으로는 당연히 안 된다 — 사상 모델이 아무리
`e_wet`에 가까운 벡터를 만들어도, 그 벡터가 실제 오디오에서 나올 수 있는 영역
밖(off-manifold)이면 TokenSynth는 (진짜 오디오에서 나온 임베딩만 보고 학습했으므로)
알아듣지 못한다.

- **낸다** → 임베딩에 정보가 있고 TokenSynth도 그 정보를 읽는다. `TokenSynth-Aug`의 실패는
  학습 문제였다는 뜻.
- **못 낸다** → 조건화(conditioning) 구조 자체를 바꿔야 한다는 뜻.

1단계(이 문서에 구현된 스크립트)의 결과를 본 뒤 진행할 후속 과제로, 이번 구현에는
포함되지 않는다.
