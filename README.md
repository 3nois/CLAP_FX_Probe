# CLAP FX Probe

TokenSynth-Aug 성능 저하(wet 복제 실패)의 추정 원인 — "CLAP 임베딩의 이펙트 정보 결여" — 을 재학습 없이 직접 측정.

## 최종 논문

[`out/apply/김성현_CLAP음색임베딩_오디오이펙트정보분석.pdf`](out/apply/김성현_CLAP음색임베딩_오디오이펙트정보분석.pdf)

## 결론

- 이펙트 정보는 존재하나 약함: 악기 AMI 0.77 vs 이펙트 AMI 0.19 (4.06~5.29배 차)
- 손잡이 방향은 임베딩만으로 예측 가능: within>between 전 축 성립, 방향 예측 코사인 다수 기준선 초과
- 생성(TokenSynth) 경로를 통과하면 방향 정보 대부분 소거 — projection 층에서 입력 방향의 약 81%가 1차 미분상 소멸
- 원인 후보 3가지: projection 층 유효계수, 자기회귀 잔차, 조건 채널 경쟁 (완화 후에도 최대 ~1.77배 개선에 그쳐 실용 수준 미달)
- 생성기를 우회한 검색(retrieval) 경로에서는 동일 방향 정보가 하류 과제(M1/M2/M3)에 유의하게 전달됨
- 한계: NSynth 1종·이펙트 3종 한정, JND는 15축 중 8축만 정밀 측정, Q5 평가는 CLAP 자기참조 구조

상세 수치와 그림은 최종 논문 Ⅲ~Ⅳ장 참고.

## 문서 구조

| 문서 | 내용 |
|---|---|
| [`out/apply/김성현_CLAP음색임베딩_오디오이펙트정보분석.pdf`](out/apply/김성현_CLAP음색임베딩_오디오이펙트정보분석.pdf) | 최종 논문 (PDF) |
| [`docs/논문_CLAP음색임베딩의_오디오이펙트정보.md`](docs/논문_CLAP음색임베딩의_오디오이펙트정보.md) | 최종 논문 (Markdown) |
| [`docs/setup.md`](docs/setup.md) | 설치, 체크포인트, 데이터, 파라미터 공간, 실행 명령, `out/` 산출물 구조 |
| [`docs/01_probe_baseline.md`](docs/01_probe_baseline.md) | 1~4차 — 프로브 방법론 확립, 정보 존재 확인 |
| [`docs/02_resolution_and_family.md`](docs/02_resolution_and_family.md) | 5~7차 — 해상도 바닥 확정, 소스 고유성(within/between) 검증 |
| [`docs/03_direction_prediction.md`](docs/03_direction_prediction.md) | 8차 — 손잡이 방향 예측(정방향/역방향/LOFO) |
| [`docs/04_tokensynth_generation.md`](docs/04_tokensynth_generation.md) | 9차 — TokenSynth 실제 연결, 오디오 생성·검증 |
| [`docs/05_projection_diagnosis.md`](docs/05_projection_diagnosis.md) | 9차 후속 — projection layer 정보 손실 진단 |
| [`docs/report_figures.md`](docs/report_figures.md) | 보고서용 핵심 그림 2개의 데이터 출처와 해설 |
