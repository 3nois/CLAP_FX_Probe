# 01. 프로브 방법론 확립 (1~4차)

CLAP 임베딩에 오디오 이펙트 정보가 존재하는지 처음 측정하고, 방법론을 OAT 스윕에서
결합 LHS 샘플링 + 야코비안 분석으로 바꾼 단계. 프로젝트 개요는 [../README.md](../README.md) 참고.

## 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| `01_embed.py` | dry/wet 오디오 CLAP 임베딩 추출 |
| `02_surrogate.py` | 임베딩 변화를 근사하는 미분 가능 대리모델(residual MLP) 학습 |
| `03_jacobian.py` | 대리모델의 야코비안 J = ∂e'/∂θ 계산 |
| `04_probe.py` | 파라미터별 Ridge 프로브 held-out R² 측정 |
| `05_text_alignment.py` | 텍스트 캡션과 이펙트 방향의 정렬도 측정 |
| `06_reverse.py` | wet 임베딩에서 파라미터/dry 방향 복원 시도 |
| `07_subspace.py` | 이펙트 방향과 악기 판별 부분공간의 직교성 측정 |
| `08_quality_stratified.py` | NSynth "dry" 태그의 잔여 이펙트 오염 여부 층화 검증 |

## 핵심 결과

- 대리모델 신뢰도 확보: held-out cos 0.985 (셔플 0.532, identity 0.970 대비)
- 이펙트 정보는 존재하나 약함 — `distortion.drive_db` R²=0.699가 최강, 다수 파라미터는 해상도 바닥 근방
- 악기 패밀리별 손잡이 방향 코사인 0.62~0.77 — 완전 공통도 완전 개별도 아닌 부분 공유
- 이펙트 방향은 악기 판별 부분공간과 무작위 기대보다도 더 직교(z≈−4)
- ⚠ 대리모델의 게이트(wet_level) 학습이 불충분(Spearman ρ 0.11~0.14)해 야코비안 기반 결론의 신뢰도가 낮다는 경고 발생 — [02_resolution_and_family.md](02_resolution_and_family.md)에서 유한차분으로 재검증
