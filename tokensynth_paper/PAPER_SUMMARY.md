# TokenSynth: A Token-based Neural Synthesizer for Instrument Cloning and Text-to-Instrument

- **저자**: Kyungsu Kim, Junghyun Koo, Sungho Lee, Haesun Joung, Kyogu Lee (Seoul National University, MARG)
- **학회**: ICASSP 2025
- **arXiv**: https://arxiv.org/abs/2502.08939
- **DOI**: https://doi.org/10.1109/ICASSP49660.2025.10888403
- **코드**: https://github.com/KyungsuKim42/tokensynth

## Abstract

신경 오디오 코덱(neural audio codec)의 발전으로 텍스트-투-스피치, 텍스트-투-오디오, 텍스트-투-뮤직 생성 등에서 토큰화된 오디오 표현이 널리 사용되고 있다. 이를 활용하여 저자들은 **TokenSynth**를 제안한다. TokenSynth는 디코더 전용(decoder-only) 트랜스포머를 사용해 MIDI 토큰과 CLAP(Contrastive Language-Audio Pretraining) 임베딩(음색 정보 포함)으로부터 원하는 오디오 토큰을 생성하는 신경 합성기다. 별도의 미세조정(fine-tuning) 없이 악기 복제(instrument cloning), 텍스트-투-악기(text-to-instrument) 합성, 텍스트 기반 음색 조작(text-guided timbre manipulation)을 모두 수행할 수 있다.

## 1. 배경 및 동기

- 기존 음악 생성 모델은 완성된 믹스를 생성하는 데는 강하지만, 사용자 제어성(controllability)이 부족하다.
- 실제 음악 제작은 악기 단위 트랙을 따로 만든 뒤 믹싱하는 방식이므로, 악기를 기본 구성 단위로 다루는 것이 중요하다.
- 음색 제어를 위한 두 접근:
  - **악기 복제(Instrument cloning)**: DDSP 계열 연구에서 시작. 참조 오디오로부터 음색을 추출해 전이. 기존 방법은 대부분 미세조정이 필요하고 단선율(monophonic)에 한정됨.
  - **텍스트-투-악기(Text-to-instrument)**: InstrumentGen 등. 자연어로 음색 제어가 가능하지만, 단일 음(single-note) 샘플 기반이라 다성(polyphonic) 악기에서 표현력이 제한됨.
- TokenSynth는 신경 코덱 언어 모델링을 다성 신경 합성기에 처음으로 적용하여, 제로샷 악기 복제와 텍스트-투-악기를 동시에 지원하는 엔드투엔드 다성 합성기를 목표로 한다.

## 2. 방법 (Method)

### 2.1 표현(Representation)

- **음색 임베딩**: 사전학습된 CLAP 모델(음성-텍스트 대조 학습, CLIP에서 영감)을 사용. 오디오/텍스트 모두에서 임베딩을 추출할 수 있는 공유 임베딩 공간을 가짐. CLAP 임베딩으로 953개 악기를 분류하는 MLP를 학습해 top-1 검증 정확도 90.4%를 달성, CLAP 특징이 음색 조건화에 충분함을 확인. 차원을 맞추기 위해 2-layer MLP 프로젝션 레이어를 사용.
- **MIDI 토큰화**: MT3 방식을 일부 수정해 채택. 노트마다 4종류 토큰(절대 onset 시간 500값·10ms 단위, 절대 offset 시간 500값, 피치 128값, velocity 4값)으로 표현. 노트 수 n개인 MIDI는 길이 M=4n의 토큰 시퀀스.
- **오디오 토큰화**: Descript Audio Codec(DAC, 44kHz)을 사용. Residual Vector Quantization으로 얻은 다중 코드북 토큰을 처리하기 위해 MusicGen의 delay pattern 기법을 적용.

### 2.2 모델 구조 및 학습

- 12-layer, 16-head, 1024 embedding dim, 4096 FFN dim, dropout 0.1의 디코더 전용 트랜스포머, 총 175M 파라미터.
- 매 타임스텝마다 D개의 코드북 토큰을 예측(코드북별 softmax 레이어 사용), next-token prediction cross-entropy loss로 학습.
- 참조 오디오(음색 임베딩 추출용)와 타겟 오디오는 **같은 악기, 다른 연주(performance)** 를 사용하도록 하여, 모델이 음색 임베딩에서 연주 정보가 아닌 음색 정보만 학습하도록 유도.
- 별도로 (a) 무조건(unconditional) 오디오 생성 모델(classifier-free guidance용), (b) 악기 무관 음악 전사(transcription) 모델(합성 정확도 평가용)을 독립적으로 학습 — MusicGen과 달리 파라미터 공유 없음.

### 2.3 추론 및 First-Note Guidance

- 추론 시 CLAP 인코더로 참조 오디오, 텍스트, 혹은 둘의 조합(보간)에서 음색 임베딩을 얻고, top-p(nucleus) 샘플링으로 오디오 토큰 생성.
- Classifier-free guidance를 매 스텝 적용하면 무음 구간에서 원치 않는 비무음 토큰이 샘플링되는 문제 발생.
- 이를 해결하기 위해 **"First-Note Guidance"** 기법 제안: guidance를 합성할 첫 노트의 onset 시점에서만 적용. 학습 데이터의 오디오 클립이 음색을 일관되게 유지하므로, 첫 노트에서 확립된 음색이 전체 합성 구간에 안정적으로 유지됨을 실험적으로 확인.

## 3. 실험 설정

- **데이터셋**: NSynth + Lakh MIDI를 결합한 합성(synthetic) 방식(Kim et al. 2023 방법 채택). 악기당 5초 다성 오디오 10,000개씩 렌더링 → 학습 953만 개(9.53M), 테스트 53만 개(530K) MIDI-오디오 쌍. 추가로 EQ/디스토션/알고리즘 리버브를 0.5 확률로 랜덤 적용해 데이터를 2배로 증강(참조-타겟 쌍에는 동일 이펙트 적용해 음색 일관성 유지).
- **모델**: DAC·CLAP(체크포인트 `music_audioset_epoch_15_esc_90.14.pt`) 모두 고정(freeze). Adam optimizer, lr 1e-4, batch size 8, 1 epoch 학습. 원본 데이터로 학습한 **TokenSynth**(1.2M step), 증강 데이터로 학습한 **TokenSynth-Aug**(2.4M step) 두 버전 비교.
- **평가지표**: Multi-Scale Spectral(MSS) loss(합성 정확도), CLAP score(합성-타겟 간 음색 유사도, 코사인 유사도), F-score(전사 모델 기반 MIDI 준수도).

## 4. 결과

### 4.1 악기 복제(Instrument Cloning)
- 테스트셋 악기당 200개의 참조 오디오-MIDI 쌍으로 합성(top-p=0.95).
- Reference=Target(동일 오디오/노트)일 때가 다른 노트일 때보다 근소하게 우수 → 음역대를 넘어선 음색 외삽(extrapolation)도 잘 작동함을 시사.
- **TokenSynth**가 (wet 오디오로 학습하지 않았음에도) 대부분의 지표에서 **TokenSynth-Aug**보다 우수 — CLAP 모델의 음색 임베딩이 오디오 이펙트 정보를 담지 못하기 때문으로 추정.
- TokenSynth는 spectral loss·CLAP score에서 우수(CLAP score는 Ground Truth보다도 높음), TokenSynth-Aug는 F-score(합성 정확도)에서 우수.
  - 대표 수치(Dry, Ref=Tgt=True): TokenSynth — MSS 0.569 / CLAP 0.860 / F-score 0.643. TokenSynth-Aug — MSS 0.643 / CLAP 0.845 / F-score 0.837.

### 4.2 텍스트-투-악기(Text-to-Instrument)
- InstrumentGen에서 사용된 10개 텍스트 설명 × 200개 MIDI 시퀀스로 합성(top-p=0.6, first-note guidance γ=1.6).
- Ground truth 쌍이 없어 CLAP score·F-score만 평가.
- TokenSynth: CLAP 0.179 / F-score 0.339, TokenSynth-Aug: CLAP 0.159 / F-score 0.8149.
- 악기 복제 과제 대비 CLAP score가 크게 낮은데, 이는 서로 다른 모달리티(오디오 vs 텍스트) 임베딩 간 코사인 유사도가 원래 작아지는 "modality gap" 문제로 설명.

### 4.3 텍스트 기반 음색 조작(Text-guided Timbre Manipulation)
- 오디오 임베딩(e_a)과 텍스트 임베딩(e_t)이 같은 공간에 있으므로 e_α = α·e_t + (1−α)·e_a 형태로 보간(및 다중 임베딩 간 보간/외삽까지) 가능 → 참조 오디오의 음색을 텍스트 방향으로 유연하게 조작.
- 정량 평가가 어려워 GitHub의 오디오 데모로 정성적 확인을 권장.

## 5. 한계 및 결론

**한계**:
- 전체 MIDI 시퀀스가 있어야 토큰 생성이 가능해 실시간(real-time) 합성 불가.
- 오토리그레시브 샘플링의 확률적 특성상 입력 MIDI 노트를 엄격히 따르지 않을 수 있음.
- MIDI 토큰화가 velocity 값 4개만 사용(NSynth 데이터셋 제약)해 세밀한 velocity 제어가 부족.

**결론**: 이러한 한계에도 불구하고 TokenSynth는 미세조정 없이 제로샷 악기 복제, 텍스트-투-악기 합성, 텍스트 기반 음색 조작을 가능하게 하는 강력한 잠재력을 보이며, 증강 데이터 학습이 다양한 음색·오디오 조건에서 성능과 일반화를 향상시킴을 확인했다.

## 저장소 구성 (본 프로젝트에 복사됨)

```
src/tokensynth/
├── __init__.py
├── clap.py          # CLAP 인코더 래퍼 (음색 임베딩 추출)
├── dac_decoder.py    # DAC 디코더 래퍼 (오디오 토큰 → 오디오)
├── model.py          # TokenSynth 디코더 전용 트랜스포머 모델
└── utils.py
quickstart.py          # 간단한 사용 예제
media/                  # 데모용 MIDI/오디오/그림
tests/test_model.py
```

원본 저장소: https://github.com/KyungsuKim42/tokensynth
