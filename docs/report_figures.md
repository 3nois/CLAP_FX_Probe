# 보고서용 핵심 그림

전체 프로젝트를 요약하는 두 그림. 프로젝트 개요는 [../README.md](../README.md) 참고.

두 그림 모두 기존 `out/results/*.json`에서 읽은 값만 쓴다 (재계산 없음, 생성:
`23_report_figures.py`).

![그림 1 — 악기 정체성 vs 이펙트 정보량](../out/figures/report_fig1_instrument_vs_effect.png)

**그림 1.** 악기 패밀리와 세 이펙트를 동일한 7클래스 분류 프로브·동일 NMI 지표로 비교했다
(출처: `out/results/results_2.json` — `controls.instrument_family_7class_subsampled.nmi`,
`effects.*.probe_nmi`). CLAP 임베딩은 악기 정체성(NMI 0.844)은 강하게 인코딩하지만
이펙트 정보는 그보다 3.4~9.2배 약하게만 담는다 — distortion이 상대적으로 가장 잘
읽히고(0.250), reverb·highshelf는 거의 바닥 수준이다(0.096, 0.092). 지표를 통일하지
않고 R²(회귀)와 accuracy(분류)를 섞어 비교하면 이 결론 자체가 성립하지 않는다는 점이
1차의 실제 오류였다.

![그림 2 — 임베딩 단계 vs 오디오 단계](../out/figures/report_fig2_embedding_vs_audio.png)

**그림 2 — 이 연구의 결론.** 소스 임베딩만으로 손잡이 방향을 예측하면 코사인
0.71~0.82(31~45도)로 상당히 정확하다(출처: `out/results/results_8.json` —
`reverse_b2.*.mlp.cos_mean`). 그러나 그 예측 방향으로 임베딩을 이동시켜 실제 오디오를
생성한 뒤 방향 일치도를 재면 코사인 −0.003~0.064(86~88도)로 무작위(90도)에 가깝게
무너진다(출처: `out/results/results_9_phase_f4.json` —
`directional_agreement.by_effect`, n=115). distortion·highshelf는 통계적으로
유의한 양의 신호가 있지만(95% CI가 0을 배제) reverb는 CI가 0을 포함해 null이다. 즉
**정보가 없는 것이 아니라, 예측까지는 되는데 TokenSynth를 통과하면서 거의 다
소거된다** — "유의하지만 손잡이로 쓰기엔 부족하다"가 이 프로젝트 전체의 결론이다.
