# CLAP FX Probe

> 원본 TokenSynth 논문(ICASSP 2025) 코드는 [`tokensynth_paper/`](tokensynth_paper/) 폴더로 옮겼습니다 —
> 사용법은 [`tokensynth_paper/README.md`](tokensynth_paper/README.md), 논문 요약은
> [`tokensynth_paper/PAPER_SUMMARY.md`](tokensynth_paper/PAPER_SUMMARY.md) 참고. 이 문서는 현재
> 진행 중인 실험(아래)만 다룹니다.

TokenSynth 논문은 오디오 이펙트(EQ·디스토션·리버브)로 augmentation한 `TokenSynth-Aug`가
이펙트 걸린(wet) 오디오 복제에서 오히려 dry로만 학습한 기본 모델보다 못한 현상을 관찰하고,
그 원인을 "CLAP 임베딩이 오디오 이펙트 정보를 결여했기 때문으로 보인다"고 추정만 했다.
이 하위 프로젝트는 그 추정을 재학습 없이 직접 측정한다.

## 결론

CLAP 임베딩은 악기 정체성은 강하게 담지만 이펙트 정보는 3.4\~9.2배 약하게만 담는다
([1\~2차](docs/round1-4.md)). 그 약한 신호로도 손잡이 방향은 코사인 0.71\~0.86으로 상당히
정확하게 예측된다([7차](docs/round6-7.md)\~[8차](docs/round8.md)). 그러나 그 방향으로
임베딩을 이동시켜 실제 TokenSynth 오디오를 생성하면 방향 일치도가 코사인
0.03\~0.06(무작위에 가까움)까지 무너진다([9차](docs/round9.md)) — projection layer를
진단해보면 이펙트 정보가 이펙트마다 다르게 선택적으로 약화되고 악기 정보는 오히려
강화되는, 단순 "압축"으로는 설명 안 되는 패턴이 나온다([10차](docs/round10.md)).
**정보가 없는 것이 아니라, 예측까지는 되는데 TokenSynth를 통과하면서 거의 다 소거된다.**

![그림 1 — 악기 정체성 vs 이펙트 정보량](out/figures/report_fig1_instrument_vs_effect.png)
![그림 2 — 임베딩 단계 vs 오디오 단계](out/figures/report_fig2_embedding_vs_audio.png)

두 그림의 수치 출처와 상세 설명은 [`docs/report_figures.md`](docs/report_figures.md) 참고.

## 문서 구조

실행 방법과 각 라운드의 상세 방법론·수치·판정 기준은 `docs/`에 라운드별로 분리되어 있다.

| 문서 | 내용 |
|---|---|
| [`docs/setup.md`](docs/setup.md) | 설치, 체크포인트, 데이터, 파라미터 공간, 실행 명령, `out/` 산출물 구조 |
| [`docs/round1-4.md`](docs/round1-4.md) | 1\~4차 — 정보 존재 확인, 지표 통일, 야코비안 전환, 결과 해석 기준 ①\~⑩ |
| [`docs/round6-7.md`](docs/round6-7.md) | 6차 후속(5\~7차) — 유한차분 검증, freeze_mode 결함, 해상도 바닥 재정의, family cosine |
| [`docs/round8.md`](docs/round8.md) | 8차 — 손잡이 방향 예측(정방향/역방향/LOFO) |
| [`docs/round9.md`](docs/round9.md) | 9차 — TokenSynth 실제 연결, 오디오 생성·검증, 재구성 품질 개선 |
| [`docs/round10.md`](docs/round10.md) | 9차 후속 D — projection layer 진단 |
| [`docs/report_figures.md`](docs/report_figures.md) | 보고서용 핵심 그림 2개의 데이터 출처와 해설 |

빠르게 실행하려면 [`docs/setup.md`](docs/setup.md)의 설치·실행 절만 보면 된다.
