# 3-1. 손잡이 회전 지도 — 결과

사전 등록: `out/prereg/11_phase3.md`. 12개 검정(6쌍×2방향).

| 쌍 | focus | context | rot_context 최대(deg, 95%CI) | 위치 | rot_source(deg, 95%CI) | 판정 |
|---|---|---|---|---|---|---|
| reverb_wet_room | wet_level | room_size | 79.6° [79.2,80.0] | room_size=0.05 | 86.5° [86.4,86.6] | context 부차적 (Branch A) |
| reverb_wet_room | room_size | wet_level | 90.0° [90.0,90.0] | wet_level=0 | 86.8° [86.7,86.9] | context 우세 (Branch B) |
| reverb_wet_damping | wet_level | damping | 76.2° [75.6,76.7] | damping=1 | 86.2° [86.1,86.3] | context 부차적 (Branch A) |
| reverb_wet_damping | damping | wet_level | 90.0° [90.0,90.0] | wet_level=0 | 89.9° [89.9,89.9] | context 우세 (Branch B) |
| reverb_room_damping | room_size | damping | 80.1° [79.6,80.5] | damping=1 | 86.7° [86.6,86.8] | context 부차적 (Branch A) |
| reverb_room_damping | damping | room_size | 89.8° [89.7,89.8] | room_size=0.05 | 89.9° [89.9,89.9] | context 부차적 (Branch A) |
| highshelf_gain_cutoff | gain | cutoff | 83.3° [83.1,83.4] | cutoff=4e+03 | 86.6° [86.6,86.7] | context 부차적 (Branch A) |
| highshelf_gain_cutoff | cutoff | gain | 90.0° [90.0,90.0] | gain=-2.5 | 90.0° [90.0,90.0] | 대등 (Branch B) |
| lowshelf_gain_cutoff | gain | cutoff | 88.5° [88.3,88.6] | cutoff=30 | 89.5° [89.4,89.5] | context 부차적 (Branch A) |
| lowshelf_gain_cutoff | cutoff | gain | 90.0° [90.0,90.0] | gain=15 | 90.0° [90.0,90.0] | context 우세 (Branch B) |
| peak_gain_cutoff | gain | cutoff | 87.9° [87.8,88.1] | cutoff=200 | 86.2° [86.1,86.3] | context 우세 (Branch B) |
| peak_gain_cutoff | cutoff | gain | 90.0° [90.0,90.0] | gain=-15 | 90.0° [90.0,90.0] | 대등 (Branch B) |

**12개 중 Branch B 판정: 6개**

## 전체 판정: **Branch B** (사전 등록 집계 규칙: 하나라도 Branch B면 전체 Branch B)
