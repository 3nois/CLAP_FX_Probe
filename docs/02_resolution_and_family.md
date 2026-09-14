# 02. 해상도 바닥 확정 + 소스 고유성 검증 (5~7차)

[01_probe_baseline.md](01_probe_baseline.md)에서 나온 대리모델 신뢰도 문제를 유한차분(FD) 야코비안으로
우회하고, 렌더링 조건 버그를 고친 뒤 이펙트 간 정보량 순서와 "손잡이가 소스마다
다른가"를 확정한 단계. 프로젝트 개요는 [../README.md](../README.md) 참고.

## 스크립트 역할

| 스크립트 | 역할 |
|---|---|
| `12_freeze_probe.py` | `freeze_mode` 버그(reverb 표본 48.7% 무효화) 확인·수정 |
| `13_fd_phase3_render.py` | 유한차분용 평가점 렌더링 + 점별 야코비안 캐시 |
| `14_polarity_probe.py` | 극성 반전으로 파이프라인 대칭성(비버그) 확인 |
| `15_resolution_floor.py` / `16_resolution_floor_v3.py` | 초음파 통제축 기반 해상도 바닥 재정의 |
| `17_solo_probe_comparison.py` | 조건·N·체인 구성 효과 분리 |
| `18_ultrasonic_null_largeN.py` | 큰 N 널 분포로 해상도 바닥 최종 확정 |
| `19_oat_render.py` / `20_family_cosine_oat.py` | 대리모델 없이 within/between/family cosine 직접 산출 |

## 핵심 결과

- `freeze_mode=1` 버그 수정 — reverb 프로브 R²가 3.4~4.8배 상승
- `width`는 음성 통제가 아님(R²=0.066, CI가 0 배제)으로 재분류
- 대리모델 야코비안은 불신(FD와 코사인 0.418) — 이후 모든 야코비안 분석은 FD 사용
- 최종 해상도 바닥(N=12,800 큰 널)으로 확정한 이펙트 간 순서: distortion(0.699) > reverb(0.226 평균) > highshelf(0.111 평균), `highshelf.cutoff_frequency_hz`만 진짜 측정 불가
- within > between이 3개 이펙트 전부에서 유의(gap CI가 0 배제) — 손잡이는 소스 고유, 악기군은 그 차이의 일부만 설명
- family cosine 0.70~0.75 — 대리모델 없이도 01_probe_baseline.md의 값과 일치, "부분 공유" 결론 확정
