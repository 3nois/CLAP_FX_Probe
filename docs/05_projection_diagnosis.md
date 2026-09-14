# 05. Projection Layer 진단 (9차 후속)

[04_tokensynth_generation.md](04_tokensynth_generation.md)에서 확인된 격차(임베딩 단계 cos 0.71~0.86 vs
생성 오디오 방향 일치도 cos 0.03~0.06)의 원인을 TokenSynth의 `clap_projection`
(512→1024→1024) 층에서 찾는 단계. 재학습·재렌더링 없이 기존 체크포인트와
`out/caches/oat_emb_ts.npz`만 사용. 프로젝트 개요는 [../README.md](../README.md) 참고.

## 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| `tokensynth_bridge/phase_d_projection_diagnostic.py` | projection 층 통과 전/후의 변위 크기·프로브 R²·부분공간 직교성 대조 |

## 핵심 결과

- 변위 크기는 projection 통과 후 오히려 커짐(감쇠율 1.38~1.63) — 단순 축소가 아님
- 이펙트 방향 프로브 R²는 통과 후 평균 23% 하락, 단 이펙트마다 다름: reverb −41%, highshelf −37%, distortion −6%
- 악기 패밀리 판별력은 오히려 향상(NMI 0.80→0.92, 정확도 0.85→0.95)
- 부분공간 직교성(z≈−4)은 유지되며 통과 후 더 강해짐
- 결론: "악기 정보는 능동적으로 강화되고, 이펙트 정보는 종류별로 선택적으로 약화된다" — 단순 압축으로는 설명 안 되는 패턴
