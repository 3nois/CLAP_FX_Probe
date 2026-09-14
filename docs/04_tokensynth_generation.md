# 04. TokenSynth 연결 및 오디오 생성 검증 (9차)

[03_direction_prediction.md](03_direction_prediction.md)에서 확인한 임베딩 단계 방향 예측(cos 0.71~0.86)이
실제 TokenSynth 오디오 생성에서도 유지되는지 검증한 단계. 모든 작업은
`tokensynth_bridge/`에서 진행하며 1~8차 코드는 건드리지 않는다. 프로젝트 개요는
[../README.md](../README.md) 참고.

## 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| `tokensynth_bridge/inject.py` | 임의 CLAP 임베딩을 TokenSynth에 직접 주입하는 경로 |
| `phase1_baseline.py` | TokenSynth 기본 추론 재현 |
| `phase2a/2b/2c_*.py` | 전처리 정합성, 주입 지점 검증, 임베딩 노름 민감도 확인 |
| `phase3_1_reextract.py` / `phase3_2_retrain.py` | 임베딩 공간을 TokenSynth 기준으로 통일 후 재학습 |
| `phase3_3_ood_check.py` / `phase3_3R_*.py` | 생성 오디오의 방향 일치도(directional_agreement) 측정 + 블라인드 청취 |
| `phase_f1~f4_*.py` | MIDI/악기-이펙트 조합 개선으로 재구성 품질 향상 |

## 핵심 결과

- 우리 파이프라인과 TokenSynth 자체 CLAP 인코더의 전처리가 달라(cos=0.70) 이후 전 작업에 우리 파이프라인만 사용
- 임베딩 주입 경로 정상 작동 확인(동일 시드·동일 임베딩 → 토큰 완전 일치)
- 초기 재구성 품질이 낮아(cos 0.25~0.65) directional_agreement가 경계선(전체 CI 하한 −0.002)
- MIDI 재설계 + 악기-이펙트 조합 필터(F-4) 후 directional_agreement +0.034 (CI가 0 배제) — 조건화가 의도한 방향으로 작동
  - distortion +0.064, highshelf +0.049 유의, reverb는 여전히 null
- 블라인드 청취로 방향 판별과 지표의 상관 확인(r=0.573, p=0.003)
