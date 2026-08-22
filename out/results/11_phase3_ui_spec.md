# 3-4. 도구 사양 분류

분류 기준(사전 확정, `out/prereg/11_phase3.md`): 주효과≥50%&상호작용<30%→**독립 노출**; 주효과<30%&상호작용≥50%→**조건부 노출**; 그 외→**보류**.

| 파라미터 | 주효과 분산비율 | 그 축 관련 상호작용 합 | 분류 | 근거 |
|---|---|---|---|---|
| highshelf_gain_cutoff_q::gain | 48.9% | 17.5% | **보류(애매)** | 주효과 48.9% vs 상호작용합 17.5% |
| highshelf_gain_cutoff_q::cutoff | 7.9% | 9.6% | **보류(애매)** | 주효과 7.9% vs 상호작용합 9.6% |
| highshelf_gain_cutoff_q::q | 18.9% | 13.5% | **보류(애매)** | 주효과 18.9% vs 상호작용합 13.5% |
| lowshelf_gain_cutoff_q::gain | 33.7% | 21.7% | **보류(애매)** | 주효과 33.7% vs 상호작용합 21.7% |
| lowshelf_gain_cutoff_q::cutoff | 18.1% | 22.0% | **보류(애매)** | 주효과 18.1% vs 상호작용합 22.0% |
| lowshelf_gain_cutoff_q::q | 9.6% | 16.6% | **보류(애매)** | 주효과 9.6% vs 상호작용합 16.6% |
| peak_gain_cutoff_q::gain | 60.8% | 15.3% | **독립 노출** | 주효과 60.8% vs 상호작용합 15.3% |
| peak_gain_cutoff_q::cutoff | 6.7% | 9.2% | **보류(애매)** | 주효과 6.7% vs 상호작용합 9.2% |
| peak_gain_cutoff_q::q | 12.0% | 10.8% | **보류(애매)** | 주효과 12.0% vs 상호작용합 10.8% |
| reverb_wet_room_damping_width::wet_level | 58.6% | 11.4% | **독립 노출** | 주효과 58.6% vs 상호작용합 11.4% |
| reverb_wet_room_damping_width::room_size | 26.0% | 10.8% | **보류(애매)** | 주효과 26.0% vs 상호작용합 10.8% |
| reverb_wet_room_damping_width::damping | 0.6% | 0.8% | **보류(애매)** | 주효과 0.6% vs 상호작용합 0.8% |
| reverb_wet_room_damping_width::width | 1.9% | 1.5% | **보류(애매)** | 주효과 1.9% vs 상호작용합 1.5% |

집계: 독립 노출 2개, 조건부 노출 0개, 보류 11개 (총 13개 파라미터, 3-D+ 격자에 포함된 것만 — 2-D 전용 파라미터인 damping/width는 reverb 4축 격자에 포함되어 있으므로 이미 반영됨).
