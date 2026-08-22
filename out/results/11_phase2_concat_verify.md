# Phase 2 concat 무결성 검증 (400 + 800 → 1,200)

17개 확장 축 + bypass 전부에 대해 `out/caches/11_phase2_<axis>.npz`(400, src_id 0~399)와
`out/caches/11_phase2ext_<axis>.npz`(800, src_id 400~1199)를 concat했을 때
src_id 합집합이 정확히 {0..1199}인지, 중복·누락이 없는지, theta_raw 격자가
일치하는지 검증했다.

| 축 | base N | ext N | 합 | 중복 | 누락 | 초과 | theta 일치 | 판정 |
|---|---|---|---|---|---|---|---|---|
| distortion_drive_db | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| reverb_wet_level | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| reverb_room_size | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| reverb_damping | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| reverb_width | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| highshelf_gain | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| lowshelf_gain | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| peak_gain | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| highshelf_cutoff_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| lowshelf_cutoff_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| peak_cutoff_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| highshelf_q_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| lowshelf_q_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| peak_q_gp6 | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| eq_cascade_intensity | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| null_12k_gain | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| null_15k_gain | 400 | 800 | 1200 | 없음 | 0 | 0 | 일치 | OK |
| bypass | 400 | 800 | 1200 | 없음 | 0 | 0 | — | OK |

**종합: PASS — 17축 + bypass 전부 1,200개 src_id가 정확히 0~1199 무결하게 concat됨.**

gn6 축 6개(`{highshelf,lowshelf,peak}_{cutoff,q}_gn6`)는 지시대로 확장분을 렌더링하지
않았으므로 기존 400소스만 존재한다(보조 분석용, 주 수치에는 미사용).
