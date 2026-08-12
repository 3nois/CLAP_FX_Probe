# 9차 — TokenSynth 연결 (환경 구축, 분석 아님)

TokenSynth 실제 연결 및 오디오 검증 — 프로젝트 개요는 [../README.md](../README.md) 참고.

8차까지 임베딩 단계 검증(B2 방향 cos 0.71\~0.86)이 끝났다. 남은 건 "cos 0.8이 귀에
어떻게 들리는가" — 숫자만으로는 답할 수 없어 실제로 TokenSynth에 넣어 오디오를
뽑아야 한다. 이 절의 작업은 전부 `tokensynth_bridge/`(신규 디렉터리)에서 하며
1\~8차 분석 코드·결과 파일은 건드리지 않는다.

**모델 가중치 경로** (전부 로컬 캐시에 있음, gitignore 대상 — 2차 데이터를 파일명
충돌로 잃은 전례가 있어 경로를 여기 기록해 둔다):

| 파일 | 크기 | 경로 |
|---|---|---|
| `clap_music_audioset_epoch_15_esc_90.14.pt` | 2.35GB | `~/Library/Caches/tokensynth/` |
| `token_synth_aug.pt` | 712MB | 〃 |
| `token_synth_unconditional.pt` | 712MB | 〃 |
| `dac_weights_44khz_8kbps_0.0.1.pt` | 306MB | 〃 |

HuggingFace 저장소 `KyungsuKim/TokenSynth`에서 `tokensynth.utils.download_model()`이
자동 다운로드(`appdirs.user_cache_dir("tokensynth")` 사용) — 이미 전부 있어 이번엔
새로 받은 게 없다.

## Phase 0\~1 — 환경 확인 + 기본 추론 재현

- 저장소는 `tokensynth_paper/`에 이미 미러돼 있고 `pip install -e`로 기존 `.venv`에
  이미 설치되어 있었다(`tokensynth==0.0.4`). `torch==2.5.1`(요구 `>=2.0,<2.6.0` 충족),
  `laion-clap==1.1.6`(정확히 일치) — 충돌 없음, 새 venv 불필요.
- **CLAP 체크포인트 동일성**: `ckpts/music_audioset_epoch_15_esc_90.14.pt`(우리 것)와
  TokenSynth 캐시의 것이 **SHA256 완전 일치**(`fae3e9c0...`) — 1차부터 쓰던 것과
  바이트 단위로 같은 파일임을 확인.
- Phase 1(`tokensynth_bridge/phase1_baseline.py`, CPU): 참조 오디오 조건 합성,
  5.10초 클립. **총 6.4초**(임베딩 0.31s + 토큰생성 4.5s + DAC디코딩 1.6s, \~100
  tok/s) — "몇 분 걸려도 정상"이라던 예상보다 훨씬 빠름. Phase 4 세트 규모를
  넉넉히 잡아도 된다. 저장: `out/audio/phase1_baseline.wav`.
- CLAP 임베딩 노름이 정확히 `1.000000`으로 관측됨(L2 정규화).

## Phase 2 — 임베딩 주입 경로 (★ 이번 작업의 성패 지점)

**2-A. 전처리 대조** (코드 근거, 추측 없음 — `tokensynth_paper/src/tokensynth/clap.py`
`CLAP.encode_audio()` vs `01_embed.py` `load_and_preprocess()`/`embed_batch()`)

| 항목 | 우리(01_embed.py) | TokenSynth(clap.py) | 동일 여부 |
|---|---|---|---|
| 로드 샘플레이트 | 48000 직접 | 16000 → `librosa.resample`로 48000 업샘플 | 다름 |
| 리샘플 경로 | 48k 직접(Nyquist 24kHz 보존) | 16k→48k(사실상 8kHz로 대역제한 후 보간) | 다름 |
| 모노 변환 | `librosa.load(mono=True)` | `librosa.load(mono=True)`(기본값) | 동일 |
| 피크 정규화 | 있음(목표 0.7) | **없음** | 다름 |
| 길이 처리 | 4.0초 고정, zero-pad/truncate | 없음(원본 길이 그대로) | 다름 |
| 무음 제외 | peak<1e-4 시 제외 | 없음 | 다름 |
| **L2 정규화 위치** | `laion_clap.CLAP.get_audio_embedding()`의 `F.normalize`(라이브러리 내부) | **동일 지점**(같은 라이브러리 호출) | **동일** — 어느 래퍼도 직접 정규화하지 않음 |

판정: 일부 다름(정규화 위치만 동일) → 실측 필요. **같은 참조 오디오를 두 경로로
인코딩해 cos = 0.7035**(무작위 기준선 0.003±0.053보다 훨씬 높지만 1.0과는 거리가
멂 — `tokensynth_bridge/phase2a_preprocessing_check.py`). **차이가 실질적이다.**
따라서 8차 학습 임베딩과의 정합성을 위해, 이후 모든 임베딩 추출은 TokenSynth 자체
`clap.encode_audio()`가 아니라 **우리 파이프라인(`01_embed.py`와 동일 전처리)**만
쓴다(`tokensynth_bridge/inject.py`의 `extract_embedding_our_pipeline()`).

**2-B. 임베딩 주입 경로**

세 지점(파일·함수 단위 확인):

1. **CLAP 임베딩 생성** — `clap.py` `CLAP.encode_audio()`/`encode_text()`
2. **transformer 전달** — `model.py` `TokenSynth.forward()`:
   `clap_proj = self.clap_projection(clap_embedding).unsqueeze(1)` 후
   `torch.cat((clap_proj, tok_embedding), dim=1)`
3. **projection layer**(논문 III-A) — `model.py` `TokenSynth.__init__()`:
   `nn.Sequential(nn.Linear(512,1024), nn.ReLU(), nn.Linear(1024, hparams.embed_dim))`
   — 512→1024→1024(=embed_dim), 2-layer MLP로 논문 설명과 정확히 일치.

`TokenSynth.synthesize(clap_embedding, midi_fname, ...)`가 이미 `clap_embedding`을
`torch.Tensor[1,512]`로 직접 받는 공개 API라서 **몽키패치 없이** 임의의 벡터를
주입할 수 있다 — `tokensynth_bridge/inject.py`의 `synthesize_from_embedding()`이 이
경로를 감싼 함수다(`normalize="none"|"unit"|"target"` 옵션으로 주입 전 정규화 여부
선택 가능).

검증(`phase2b_verify_injection.py`, 시드 42 고정):

| 비교 | 결과 |
|---|---|
| 직접 경로 2회(같은 시드) | **완전 동일**(0/3978 토큰 불일치) — 시드 고정이 실제로 재현성을 준다 |
| 직접 경로 vs 주입 경로(같은 임베딩·같은 시드) | **완전 동일**(0/3978 토큰 불일치) |
| 직접 경로 vs 주입 경로(다른 시드) | 다름(정상 — 시드가 작동한다는 반증) |

★ **주입 지점 정상 작동 확인.** 9차 성패를 가르는 지점이 열렸다.

**2-C. 노름 민감도** (`phase2c_norm_sensitivity.py`, 방향 고정·시드 42·같은 MIDI)

| 노름 | 토큰 수 | 길이 | RMS | cos(재임베딩, norm=1.0) |
|---|---|---|---|---|
| 0.6 | 439 | 5.10s | 0.228 | 0.981 |
| 0.8 | 439 | 5.10s | 0.218 | 0.977 |
| 1.0 | 439 | 5.10s | 0.175 | 1.000(기준) |
| 1.2 | 439 | 5.10s | 0.198 | 0.985 |
| 1.5 | 439 | 5.10s | 0.178 | 0.986 |
| 2.0 | 439 | 5.10s | 0.251 | **0.721** |

극단값(0.6, 2.0)에서 토큰 수·길이·RMS 붕괴는 없음(무음·조기종료·이상 반복 없음).
그러나 **노름 2.0에서 재임베딩 코사인이 0.721로 뚜렷하게 하락** — 0.6\~1.5 구간은
전부 0.97 이상으로 견고하다.

★ **판정 — 혼합**: "결과가 거의 같음"(0.6\~1.5)과 "결과가 크게 다름"(2.0) 둘 다
관측됨. 완전 붕괴는 아니므로 β 스윕 범위를 무리하게 제한할 필요는 없지만, 노름이
2.0 근방까지 가면 신뢰도가 떨어진다는 걸 염두에 둬야 한다. **재정규화는 강제하지
않고 Phase 3에서 노름을 그대로 두되(옵션은 유지), 결과 노름이 1.5를 넘는 조합은
"저신뢰 구간"으로 표시하며 리포트한다.** (참고: 8차 데이터의 실제 `‖v_to_dry‖`
평균은 reverb 0.10\~0.22, highshelf 0.11\~0.20, distortion 0.28\~0.51 — distortion을
큰 β로 밀 때가 노름 2.0에 가장 먼저 닿는다.)

산출물: `tokensynth_bridge/{inject.py, phase1_baseline.py,
phase2a_preprocessing_check.py, phase2b_verify_injection.py,
phase2c_norm_sensitivity.py}`, `out/audio/{phase1_baseline.wav,
phase2_norm_*.wav}`, `out/results/results_9_phase2c_norm.json`.

## Phase 3-1/3-2 — 임베딩 공간을 TokenSynth로 통일 + 재학습

Phase 2-A에서 두 경로 임베딩이 cos=0.7035로 실질적으로 다름을 확인했으므로, 이제
TokenSynth에 주입할 것을 전제로 **임베딩 공간 자체를 TokenSynth 것으로 통일**한다.

**3-1** (`phase3_1_reextract.py`): 7차(`19_oat_render.py`)와 완전히 동일한 시드·
렌더링(pedalboard) 코드를 그대로 쓰고, 임베딩 추출 단계만 TokenSynth의
`clap.encode_audio()`가 하는 16kHz↔48kHz 왕복 리샘플로 교체했다(파일 I/O 없이
배치 처리 — 사전에 실제 `encode_audio(파일경로)`와 cos=0.9999999999989999로 동치
확인). 1,200소스×3레벨×3이펙트=10,800회, 11.8분. 저장:
`out/caches/oat_emb_ts.npz`(`oat_emb.npz`는 안 건드림).

**3-2** (`phase3_2_retrain.py`): 8차와 동일 구조·분할·시드로 정방향과 B2만
재학습(B1/B3/C/D는 재학습 대상 아님).

| 이펙트 | 정방향(TS공간) | 정방향(8차) | Δ | B2(TS공간) | B2(8차) | Δ |
|---|---|---|---|---|---|---|
| reverb | 0.775 | 0.776 | −0.001 | 0.712 | 0.714 | −0.003 |
| distortion | 0.839 | 0.860 | −0.021 | 0.815 | 0.823 | −0.008 |
| highshelf | 0.819 | 0.823 | −0.004 | 0.812 | 0.813 | −0.002 |

★ **비슷하다 — 큰 차이 없음(전부 |Δ|<0.1 임계값 이내, 최대는 distortion 정방향
−0.021).** 8차 결과(임베딩 공간이 달라도 방향 예측 가능성)가 TokenSynth 임베딩
공간에서도 그대로 재현된다. B2 모델 가중치 저장:
`out/caches/b2_model_ts_{reverb,distortion,highshelf}.pt`(Phase 3-4에서 재사용).

## Phase 3-3 — OOD 확인   ★ 결과가 애매하다. 청취 확인 필요, 3-4 보류

3소스(bass/vocal/guitar, 조건A 전처리) × 3이펙트, 동일 MIDI(`input_midi.mid`)·시드
42로 (a)원본 wet 오디오, (b)원본 dry(레벨0) 오디오, (c) e_wet 주입 생성,
(d) e_dry_true(레벨0 임베딩) 주입 생성 — 4종 × 9조합 = 36개 wav를 `out/audio/
phase3_3_*.wav`에 저장했다(`phase3_3_ood_check.py`).

**정량 결과 (2026-08-06)**

| 지표 | 값 |
|---|---|
| dry_shift 평균 (cos_to_dry[d] − cos_to_dry[c]) | **−0.0124**, 95% CI [−0.0284, +0.0052] |
| dry_shift 양수인 조합 | 3/9 |
| wet_axis_shift 평균 (cos_to_wet[c] − cos_to_wet[d]) | +0.0380, 7/9 양수 |

**사전 등록 판정 결과: "구분 안 됨"** — dry_shift의 부트스트랩 CI가 0을 포함하고
평균조차 음수(기대와 반대 방향)다. `wet_axis_shift`는 7/9에서 양수라 방향성이
아예 없다고 하기도 애매하다 — 두 지표가 서로 다른 결론을 가리킨다.

★ **더 근본적인 관찰**: 원본 wet-dry 쌍의 코사인은 0.70\~0.99로 자연스러운데(같은
소스의 서로 다른 이펙트 레벨이니 당연히 가깝다), **재생성물은 (c)/(d) 조건 구분과
무관하게 원본 wet에도 dry에도 코사인 0.25\~0.65 수준으로 멀다.** 즉 TokenSynth가
"어느 쪽으로 갔는지"보다 먼저 "원본과 얼마나 닮았는지" 자체가 약하다 — 재구성
충실도가 소스에 따라 크게 갈린다(vocal 0.43\~0.65 > guitar 0.35\~0.50 > bass
0.25\~0.31, 대략 vocal_distortion 원본 cos(wet,dry)=0.696으로 유독 낮은 것도 눈에
띈다).

**가능한 원인 세 갈래(구분 안 됨, 청취·추가 확인 필요)**:
1. TokenSynth가 이 MIDI·이 입력 조건에서 조건화 신호(CLAP 임베딩)에 약하게만
   반응하고 MIDI/자기회귀 사전분포가 지배적이다.
2. 우리 조건A 전처리(피크 0.7, 4초 단일 지속음, NSynth 합성음)가 TokenSynth
   학습분포에서 벗어난 입력이라 재구성 자체가 원래 약하다 — wet/dry 구분과
   무관한 별개 문제.
3. `input_midi.mid`(멜로디/여러 음표)와 소스 오디오(단일 지속음)의 내용 구조가
   달라 CLAP 재임베딩이 음색이 아니라 내용(리듬·음표 수) 차이를 더 크게 반영할
   가능성 — 코사인 지표 자체가 이 조건에서 판별력이 떨어질 수 있다.

★ **이 스크립트는 오디오를 듣지 못한다 — 36개 wav 직접 청취가 필요하다.**
파일명 규칙: `phase3_3_{bass|vocal|guitar}_{reverb|distortion|highshelf}_
{a_orig_wet|b_orig_dry|c_inject_ewet|d_inject_edrytrue}.wav`. 특히 (c)와 (d)를
소스별로 비교해서 "더 dry하게 들리는지" 사람이 직접 판단해야 한다 — 정량 지표가
결론을 못 냈으므로 사전 등록 규칙("구분 안 됨 → 멈추고 보고")에 따라 여기서
멈춘다. 3-4(β 스윕)는 청취 결과를 보고 진행 여부를 정한다.

산출물: `tokensynth_bridge/phase3_3_ood_check.py`, `out/audio/phase3_3_*.wav`(36개),
`out/results/results_9_phase3_3.json`.

## Phase 3-3R — 확장 검증(300생성) + 블라인드 청취 도구

사용자가 Phase 3-3의 wav를 직접 듣고 (c)/(d)가 귀로는 명확히 구분된다고 확인했다.
지표가 잘못된 곳을 재고 있었다 — 재생성물이 원본 wet·dry 양쪽에서 멀다는 사실
(cos 0.25\~0.65) 자체가 재구성 충실도 문제이지 wet/dry 판별 문제가 아니다.
**절대 위치가 아니라 변위 방향**을 봐야 한다:

    v_generated = e_regen(d) − e_regen(c)
    v_original  = e_dry_true − e_wet
    directional_agreement = cos(v_generated, v_original)

50소스(10패밀리×5, `out/caches/oat_emb_ts.npz`에서 추출) × 3이펙트 × 2조건 =
300생성(150쌍)으로 검정력을 키웠다. 임베딩은 Phase 3-1에서 이미 캐시된 TokenSynth
공간 값을 재사용(재추출 안 함), (a)/(b) 원본 wet/dry는 파형만 다시 렌더링해 참조용
wav로 저장. 소요 30.9분 (`phase3_3R_1_generate.py`).

**3-3R-2 결과 (2026-08-07, n=150) — ★ 핵심**

| 구분 | directional_agreement | 95% CI |
|---|---|---|
| **전체** | **+0.0204** | **[−0.0021, +0.0420]** |
| reverb | −0.0046 | [−0.038, +0.028] |
| distortion | +0.0351 | [−0.003, +0.074] |
| highshelf | +0.0306 | [−0.006, +0.066] |
| bass (패밀리) | **+0.0882** | **[0.030, 0.140]** ← CI가 0 배제, 유의 |
| string (패밀리) | **+0.0838** | **[0.031, 0.131]** ← CI가 0 배제, 유의 |
| flute (패밀리) | −0.0793 | [−0.178, +0.005] |

★ **판정(사전 등록 규칙 그대로 적용): "0 근처 — 방향이 무작위."** 전체 CI 하한이
−0.0021로 0을 살짝 포함한다 — 엄밀히는 유의하지 않다. 다만 하한이 0에 극히
가깝고(사실상 경계선), **bass·string 두 패밀리는 개별적으로 CI가 0을 명확히
배제하며 유의하게 양수다.** reverb는 완전히 null에 가깝고 distortion·highshelf는
양의 방향이나 역시 경계선.

**해석**: 전체 평균은 임계값을 살짝 못 넘지만, 효과가 패밀리·이펙트에 따라
이질적일 가능성이 높다(9쌍짜리 Phase 3-3로는 이 이질성 자체를 볼 수 없었다).
이게 바로 블라인드 청취로 검증해야 할 지점이다 — 지표가 실제로 "잘 들리는
정도"를 반영한다면, directional_agreement 상위 구간 쌍이 하위 구간 쌍보다 청취
정답률이 높게 나와야 한다.

(참고, 판정에는 안 씀) `magnitude_ratio` 평균 2.59(생성 변위가 원본 변위보다
훨씬 큼 — 자기회귀 생성 자체의 노이즈가 상당히 섞여 있다는 뜻), `separation`
평균 0.80(조건 c/d의 재생성물끼리는 서로 꽤 비슷함 — MIDI/사전분포가 지배적).
패밀리별 재구성 충실도(`recon_c`,`recon_d`, 판정에 안 씀, 참고용)는 여전히
0.36\~0.52 수준으로 낮다 — TokenSynth가 폴리포닉 음악용인데 NSynth 단일음
4초를 넣는 조건 자체가 학습분포 밖이라는 한계는 여전하다.

산출물: `out/results/results_9_phase3_3R.json`, `out/figures/{phase3_directional,
phase3_by_family}.png`, `out/audio/phase3r_*.wav`(600개).

**3-3R-3/4 — 블라인드 청취 세트 준비 완료**

directional_agreement 기준 상/중/하 3분위 각 8쌍(이펙트·패밀리 균형)씩 24쌍을
`phase3_3R_3_build_blind.py`로 뽑아 `out/audio/blind/`에 준비했다:

- `blind_manifest.json` — 쌍 목록(A/B 파일명만, 조건 정보 없음)
- `answer_key.json` — 정답 + 층 + directional_agreement 값(절대 안 보이게 별도 보관)
- `abx_test.html` — 사용자가 제공한 청취 도구, 수정 없이 그대로 저장

**실행 방법**:
```
cd out/audio/blind
python -m http.server 8000
```
브라우저에서 `http://localhost:8000/abx_test.html` 접속 → 24쌍 청취·응답 →
"응답 저장" 버튼으로 `blind_responses.json` 다운로드 → 그 파일을
`out/audio/blind/blind_responses.json`에 놓고
`tokensynth_bridge/phase3_3R_5_grade.py` 실행.

**3-3R-5 — 블라인드 응답 채점(참고용, 청취 세트는 재구성 품질 문제로 폐기 예정)**

사용자가 24쌍을 전부 청취·응답했으나(`out/audio/blind/blind_responses.json`,
2026-08-07 완료) 자연스러움 응답이 대부분 "하"로 나와 청취 데이터 자체가
오염됐다고 판단, 후속 블라인드 라운드는 중단하고 재구성 품질(F-1\~F-4)을 먼저
고치기로 했다. 다만 이 응답은 참고 데이터로 보존·채점했다:

| 층 | 정답률 |
|---|---|
| high | 6/8 = 0.750 |
| mid | 3/8 = 0.375 |
| low | 1/8 = 0.125 |
| 전체 | 10/24 = 0.417 (유의하지 않음) |

★ **전체 정답률은 낮지만 층별로는 깨끗한 단조 관계**(high>mid>low)이고,
directional_agreement와 정답 여부의 point-biserial 상관 **r=0.573, p=0.0034로
유의하다**. low 구간(지표가 음수/0 근방)에서 사람들이 일관되게 "반대" 방향을
골랐다는 뜻 — 자연스러움 오염과 별개로 **방향 판별 자체는 지표와 유의하게
상관된다.** `out/results/results_9_blind.json`.

## F-0\~F-3 — 재구성 품질 개선 (MIDI 재설계 + 조합 필터 + 파일럿)

**진단**: 재구성 충실도가 낮은(cos 0.25\~0.65) 원인 후보 셋 — ① 전 소스에 같은
MIDI 프레이즈를 써서 음역이 안 맞음 ② 베이스+리버브처럼 실무에 드문 악기-이펙트
조합 ③ NSynth 단일음 4초가 TokenSynth의 폴리포닉 5초 학습분포 밖. ①②를 고치고
③은 구조적이라 이번 범위 밖으로 둔다.

**F-1 (`midi_gen.py`)**: 소스의 NSynth pitch를 중심으로 ±5반음 안에서 4음(각
0.6초, 음 간 0.05초 간격)을 뽑아 짧은 프레이즈를 만들고, 마지막 음 뒤 잔향이
드러날 여백(2.45초, 최소 요구 1.5초 충족)을 남긴다. 소스마다 별도 MIDI 생성,
`tokensynth_bridge/generated_midi/`에 저장.

**F-2 (`phase_f2_filter.py`)**: 이펙트별 허용 패밀리 제한.

| 이펙트 | 허용 패밀리 | 제외 |
|---|---|---|
| reverb | brass,flute,keyboard,mallet,organ,reed,string,vocal | bass, guitar(저역 뭉개짐) |
| distortion | bass,guitar,keyboard,organ,reed | flute,vocal,mallet,string,brass(비주류 조합) |
| highshelf | 전체 | 없음(스펙트럼 조작, 악기 무관) |

**F-3 파일럿 결과 (2026-08-07, 8소스×2MIDI, highshelf 통일, n=16생성)** ★ 통과

| | 기존 MIDI | 신규 MIDI | Δ |
|---|---|---|---|
| cos(e_regen, e_wet) 평균 | 0.4658 | **0.6368** | **+0.1710** |

7/8 소스에서 개선(예외: keyboard −0.16), **Wilcoxon(신규>기존) p=0.0117 — 유의하게
상승.** 판정 기준(유의 상승 → F-4 진행)에 따라 **F-4(전체 재생성)로 진행한다.**
가장 크게 개선된 소스는 brass(+0.373), flute(+0.309), organ(+0.268) — 관악기/오르간처럼
원래 지속음 성격이 강한 소스일수록 "짧은 음+여백" MIDI의 효과가 컸다.

산출물: `out/results/results_9_phase_f3_pilot.json`,
`out/audio/phase_f3_*_midi_{old,new}.wav`(16개),
`tokensynth_bridge/generated_midi/*.mid`(8개).

## F-1/F-3 재수정 — 소스당 MIDI 3변형(상행/하행/지그재그)

"MIDI가 결과를 좌우하는가"를 분리하기 위해 소스당 서로 다른 윤곽 3개를 만들도록
바꿨다(`midi_gen.py`) — 같은 4개 pitch(중심±5반음, 소스당 1회 추첨)를 오름차순/
내림차순/지그재그로만 다르게 배열해 pitch 집합은 고정하고 윤곽만 변수로 남긴다.

**F-3(수정) 파일럿 결과 (2026-08-07, 8소스×(기존1+신규3변형), highshelf 통일,
32생성)** ★ 통과

| | 기존 MIDI | 신규 MIDI(3변형 평균) | Δ |
|---|---|---|---|
| cos(e_regen,e_wet) | 0.4658 | **0.6515** | **+0.1858** |

7/8 소스 개선, **Wilcoxon p=0.0078**(직전 단일-변형판(p=0.012)보다 더 강한
신호). **MIDI 변형 간 분산(소스 내 평균 std=0.039)이 소스 간 분산(std=0.110)보다
작다** — 어떤 변형을 쓰든 결과가 크게 흔들리지 않는다는 뜻, 즉 지금까지 단일
MIDI로 잰 값들도 특별히 불안정하지 않았을 가능성이 높다. 판정 기준(유의 상승 →
F-4 진행)에 따라 **F-4로 진행한다.**

산출물: `out/results/results_9_phase_f3_pilot.json`(갱신),
`out/audio/phase_f3_*_midi_{old,new_v0,new_v1,new_v2}.wav`(32개),
`tokensynth_bridge/generated_midi/*_v{0,1,2}.mid`.

## F-4 — 전체 재생성 (조합 필터 + MIDI 3변형)   ★ 유의한 양성 신호 확정

115개 유효 (소스×이펙트) 조합(F-2 필터로 reverb 40/distortion 25/highshelf 50,
제외 35개) × MIDI 3변형 × 조건 2(c,d) = 690생성, 70.3분(`phase_f4_full.py`).

**결과 (2026-08-07, n=115, MIDI 3변형 평균 기준)**

| 구분 | directional_agreement | 95% CI |
|---|---|---|
| **전체** | **+0.0341** | **[+0.0147, +0.0544]** ← CI가 0을 명확히 배제 |
| reverb (n=40) | −0.0025 | [−0.029, +0.023] — 여전히 null |
| distortion (n=25) | **+0.0635** | **[0.015, 0.112]** ← 유의 |
| highshelf (n=50) | **+0.0486** | **[0.022, 0.078]** ← 유의 |

★ **판정: CI가 0을 넘고 양수 — 조건화가 의도한 방향으로 작동한다.** 3-3R-2의
"0 근처"(전체 CI 하한 −0.002로 경계선)에서 F-4에서는 **명확히 유의한 양성**으로
전환됐다 — MIDI 재설계(F-1)와 악기-이펙트 조합 필터(F-2)가 실제로 측정을
개선했다는 뜻이다. 단, **reverb는 여전히 null이다** — distortion·highshelf와
달리 재구성 품질을 올려도 방향 신호가 안 잡힌다(잔향 자체가 CLAP 임베딩에서
가장 미묘한 신호라는 6차 이전 결과들과 일관).

**패밀리별** (내림차순): vocal +0.082(유의) > guitar +0.056 > brass +0.047 >
bass +0.047 > flute +0.041 > organ +0.040(유의) > reed +0.033 > string +0.027 >
keyboard +0.017 > **mallet −0.044(유의하게 음수)**. mallet만 방향이 뒤집혀 있다 —
타악기 특성상(감쇠가 빠르고 배음 구조가 다름) 별도로 들여다볼 필요가 있다.

**분산 분해(이원분산분석, SS 비율)**:

| 성분 | 비율 |
|---|---|
| 소스 간(source_between) | 42.4% |
| MIDI 변형(midi_variant) | **0.2%** |
| 잔차(residual) | 57.4% |

★ **MIDI 변형이 설명하는 분산은 사실상 0에 가깝다** — 어떤 변형(상행/하행/
지그재그)을 쓰든 결과가 거의 안 바뀐다는 뜻으로, F-3에서 관측한 "변형 간
분산이 작다"는 관찰이 대규모로도 재확인됐다. 즉 **이번 라운드부터는 MIDI
변형을 하나만 써도 측정이 충분히 안정적**이라고 볼 수 있다. 잔차(57%)가 가장
큰 성분인데, 이는 (소스×이펙트) 조합별 고유 변동 — 개별 조합의 특수성(악기와
이펙트의 상호작용)이 남은 변동의 대부분을 차지한다.

산출물: `out/results/results_9_phase_f4.json`(`rows`, `directional_agreement`,
`variance_decomposition`, `meta.exclusions`), `out/figures/phase_f4_directional.png`,
`out/audio/phase_f4_*.wav`(690개).

