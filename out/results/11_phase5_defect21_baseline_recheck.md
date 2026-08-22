# 결함 21 재검증 — 고정방향 베이스라인 raw-평균 vs 단위벡터-평균 (2026-08-22)

판정 기준: |raw 평균 cos − 단위평균 cos| < 0.01 이면 실질 영향 없음(단위평균 채택), 아니면 개별 표시 + between 기준선 재검토 대상.

## 1. 21_handle_predict_phase1 — OAT 기반 A/B1/B2 (reverb/distortion/highshelf)

| 이펙트 | 조건 | global_mean(raw) | global_mean(단위평균) | 차이 | family_oracle(raw) | family_oracle(단위평균) | 차이 | 실질영향 |
|---|---|---|---|---|---|---|---|---|
| reverb | A(forward) | 0.5017 | 0.5050 | +0.0033 | 0.5770 | 0.5816 | +0.0047 | 실질영향없음 |
| reverb | B1(reverse,known) | 0.5017 | 0.5050 | +0.0033 | 0.5770 | 0.5816 | +0.0047 | 실질영향없음 |
| distortion | A(forward) | 0.5372 | 0.5390 | +0.0018 | 0.6023 | 0.6057 | +0.0034 | 실질영향없음 |
| distortion | B1(reverse,known) | 0.5372 | 0.5390 | +0.0018 | 0.6023 | 0.6057 | +0.0034 | 실질영향없음 |
| highshelf | A(forward) | 0.5213 | 0.5227 | +0.0014 | 0.6079 | 0.6089 | +0.0010 | 실질영향없음 |
| highshelf | B1(reverse,known) | 0.5213 | 0.5227 | +0.0014 | 0.6079 | 0.6089 | +0.0010 | 실질영향없음 |

## 2. 11_phase5_q3q4 — 25레벨 5축 x 7구간 forward/B1/B2

| 축 | 구간 | 조건 | global(raw) | global(단위평균) | 차이 | family(raw) | family(단위평균) | 차이 |
|---|---|---|---|---|---|---|---|---|
| distortion_drive_db | 전범위 | forward | 0.5509 | 0.5517 | +0.0008 | 0.6361 | 0.6377 | +0.0016 |
| distortion_drive_db | 전범위 | B1 | 0.5509 | 0.5517 | +0.0008 | 0.6361 | 0.6377 | +0.0016 |
| distortion_drive_db | 전범위 | B2 | 0.5124 | 0.5149 | +0.0025 | 0.5809 | 0.5848 | +0.0039 |
| distortion_drive_db | 하위1/3 | forward | 0.5107 | 0.5120 | +0.0012 | 0.5812 | 0.5844 | +0.0032 |
| distortion_drive_db | 하위1/3 | B1 | 0.5107 | 0.5120 | +0.0012 | 0.5812 | 0.5844 | +0.0032 |
| distortion_drive_db | 하위1/3 | B2 | 0.4971 | 0.4987 | +0.0016 | 0.5624 | 0.5658 | +0.0035 |
| distortion_drive_db | 중위1/3 | forward | 0.4431 | 0.4448 | +0.0017 | 0.5159 | 0.5190 | +0.0031 |
| distortion_drive_db | 중위1/3 | B1 | 0.4431 | 0.4448 | +0.0017 | 0.5159 | 0.5190 | +0.0031 |
| distortion_drive_db | 중위1/3 | B2 | 0.4259 | 0.4283 | +0.0024 | 0.4916 | 0.4956 | +0.0040 |
| distortion_drive_db | 상위1/3 | forward | 0.3827 | 0.3851 | +0.0023 | 0.4812 | 0.4846 | +0.0034 |
| distortion_drive_db | 상위1/3 | B1 | 0.3827 | 0.3851 | +0.0023 | 0.4812 | 0.4846 | +0.0034 |
| distortion_drive_db | 상위1/3 | B2 | 0.3566 | 0.3596 | +0.0030 | 0.4495 | 0.4534 | +0.0039 |
| distortion_drive_db | 인접-하 | forward | 0.4709 | 0.4718 | +0.0009 | 0.5304 | 0.5341 | +0.0037 |
| distortion_drive_db | 인접-하 | B1 | 0.4709 | 0.4718 | +0.0009 | 0.5304 | 0.5341 | +0.0037 |
| distortion_drive_db | 인접-중 | forward | 0.3826 | 0.3847 | +0.0021 | 0.4412 | 0.4451 | +0.0039 |
| distortion_drive_db | 인접-중 | B1 | 0.3826 | 0.3847 | +0.0021 | 0.4412 | 0.4451 | +0.0039 |
| distortion_drive_db | 인접-상 | forward | 0.3061 | 0.3093 | +0.0032 | 0.3883 | 0.3928 | +0.0044 |
| distortion_drive_db | 인접-상 | B1 | 0.3061 | 0.3093 | +0.0032 | 0.3883 | 0.3928 | +0.0044 |
| reverb_room_size | 전범위 | forward | 0.5737 | 0.5756 | +0.0019 | 0.6570 | 0.6573 | +0.0003 |
| reverb_room_size | 전범위 | B1 | 0.5737 | 0.5756 | +0.0019 | 0.6570 | 0.6573 | +0.0003 |
| reverb_room_size | 전범위 | B2 | 0.4477 | 0.4498 | +0.0021 | 0.5195 | 0.5240 | +0.0046 |
| reverb_room_size | 하위1/3 | forward | 0.4147 | 0.4147 | +0.0000 | 0.4959 | 0.4951 | +0.0007 |
| reverb_room_size | 하위1/3 | B1 | 0.4147 | 0.4147 | +0.0000 | 0.4959 | 0.4951 | +0.0007 |
| reverb_room_size | 하위1/3 | B2 | 0.3589 | 0.3586 | +0.0003 | 0.4376 | 0.4371 | +0.0005 |
| reverb_room_size | 중위1/3 | forward | 0.4988 | 0.5008 | +0.0021 | 0.5728 | 0.5726 | +0.0002 |
| reverb_room_size | 중위1/3 | B1 | 0.4988 | 0.5008 | +0.0021 | 0.5728 | 0.5726 | +0.0002 |
| reverb_room_size | 중위1/3 | B2 | 0.4544 | 0.4557 | +0.0013 | 0.5227 | 0.5229 | +0.0003 |
| reverb_room_size | 상위1/3 | forward | 0.5241 | 0.5282 | +0.0041 | 0.6071 | 0.6114 | +0.0042 |
| reverb_room_size | 상위1/3 | B1 | 0.5241 | 0.5282 | +0.0041 | 0.6071 | 0.6114 | +0.0042 |
| reverb_room_size | 상위1/3 | B2 | 0.4872 | 0.4938 | +0.0066 | 0.5624 | 0.5694 | +0.0070 |
| reverb_room_size | 인접-하 | forward | 0.3298 | 0.3301 | +0.0003 | 0.3976 | 0.3964 | +0.0013 |
| reverb_room_size | 인접-하 | B1 | 0.3298 | 0.3301 | +0.0003 | 0.3976 | 0.3964 | +0.0013 |
| reverb_room_size | 인접-중 | forward | 0.4451 | 0.4478 | +0.0026 | 0.5088 | 0.5086 | +0.0002 |
| reverb_room_size | 인접-중 | B1 | 0.4451 | 0.4478 | +0.0026 | 0.5088 | 0.5086 | +0.0002 |
| reverb_room_size | 인접-상 | forward | 0.4459 | 0.4526 | +0.0067 | 0.5204 | 0.5255 | +0.0051 |
| reverb_room_size | 인접-상 | B1 | 0.4459 | 0.4526 | +0.0067 | 0.5204 | 0.5255 | +0.0051 |
| highshelf_gain | 전범위 | forward | 0.5086 | 0.5094 | +0.0008 | 0.6138 | 0.6127 | +0.0011 |
| highshelf_gain | 전범위 | B1 | 0.5086 | 0.5094 | +0.0008 | 0.6138 | 0.6127 | +0.0011 |
| highshelf_gain | 전범위 | B2 | 0.4291 | 0.4322 | +0.0030 | 0.5398 | 0.5420 | +0.0022 |
| highshelf_gain | 하위1/3 | forward | 0.4145 | 0.4146 | +0.0000 | 0.5472 | 0.5462 | +0.0010 |
| highshelf_gain | 하위1/3 | B1 | 0.4145 | 0.4146 | +0.0000 | 0.5472 | 0.5462 | +0.0010 |
| highshelf_gain | 하위1/3 | B2 | 0.3859 | 0.3867 | +0.0008 | 0.5228 | 0.5213 | +0.0015 |
| highshelf_gain | 중위1/3 | forward | 0.4665 | 0.4676 | +0.0011 | 0.5623 | 0.5615 | +0.0008 |
| highshelf_gain | 중위1/3 | B1 | 0.4665 | 0.4676 | +0.0011 | 0.5623 | 0.5615 | +0.0008 |
| highshelf_gain | 중위1/3 | B2 | 0.4514 | 0.4531 | +0.0018 | 0.5460 | 0.5465 | +0.0005 |
| highshelf_gain | 상위1/3 | forward | 0.4738 | 0.4763 | +0.0025 | 0.5801 | 0.5807 | +0.0006 |
| highshelf_gain | 상위1/3 | B1 | 0.4738 | 0.4763 | +0.0025 | 0.5801 | 0.5807 | +0.0006 |
| highshelf_gain | 상위1/3 | B2 | 0.4648 | 0.4668 | +0.0020 | 0.5668 | 0.5662 | +0.0007 |
| highshelf_gain | 인접-하 | forward | 0.3741 | 0.3794 | +0.0053 | 0.5180 | 0.5189 | +0.0009 |
| highshelf_gain | 인접-하 | B1 | 0.3741 | 0.3794 | +0.0053 | 0.5180 | 0.5189 | +0.0009 |
| highshelf_gain | 인접-중 | forward | 0.4487 | 0.4499 | +0.0012 | 0.5432 | 0.5423 | +0.0009 |
| highshelf_gain | 인접-중 | B1 | 0.4487 | 0.4499 | +0.0012 | 0.5432 | 0.5423 | +0.0009 |
| highshelf_gain | 인접-상 | forward | 0.4555 | 0.4584 | +0.0029 | 0.5620 | 0.5631 | +0.0011 |
| highshelf_gain | 인접-상 | B1 | 0.4555 | 0.4584 | +0.0029 | 0.5620 | 0.5631 | +0.0011 |
| lowshelf_gain | 전범위 | forward | 0.3861 | 0.4149 | +0.0288 | 0.4800 | 0.5023 | +0.0222 |
| lowshelf_gain | 전범위 | B1 | 0.3861 | 0.4149 | +0.0288 | 0.4800 | 0.5023 | +0.0222 |
| lowshelf_gain | 전범위 | B2 | 0.3455 | 0.3731 | +0.0276 | 0.4386 | 0.4619 | +0.0233 |
| lowshelf_gain | 하위1/3 | forward | 0.3422 | 0.3641 | +0.0218 | 0.4357 | 0.4574 | +0.0217 |
| lowshelf_gain | 하위1/3 | B1 | 0.3422 | 0.3641 | +0.0218 | 0.4357 | 0.4574 | +0.0217 |
| lowshelf_gain | 하위1/3 | B2 | 0.3348 | 0.3506 | +0.0158 | 0.4253 | 0.4411 | +0.0158 |
| lowshelf_gain | 중위1/3 | forward | 0.3394 | 0.3738 | +0.0343 | 0.4313 | 0.4588 | +0.0275 |
| lowshelf_gain | 중위1/3 | B1 | 0.3394 | 0.3738 | +0.0343 | 0.4313 | 0.4588 | +0.0275 |
| lowshelf_gain | 중위1/3 | B2 | 0.3240 | 0.3548 | +0.0307 | 0.4147 | 0.4392 | +0.0245 |
| lowshelf_gain | 상위1/3 | forward | 0.3766 | 0.4024 | +0.0258 | 0.4573 | 0.4801 | +0.0228 |
| lowshelf_gain | 상위1/3 | B1 | 0.3766 | 0.4024 | +0.0258 | 0.4573 | 0.4801 | +0.0228 |
| lowshelf_gain | 상위1/3 | B2 | 0.3575 | 0.3839 | +0.0263 | 0.4384 | 0.4627 | +0.0243 |
| lowshelf_gain | 인접-하 | forward | 0.3226 | 0.3329 | +0.0103 | 0.4130 | 0.4263 | +0.0132 |
| lowshelf_gain | 인접-하 | B1 | 0.3226 | 0.3329 | +0.0103 | 0.4130 | 0.4263 | +0.0132 |
| lowshelf_gain | 인접-중 | forward | 0.3087 | 0.3326 | +0.0240 | 0.3930 | 0.4133 | +0.0203 |
| lowshelf_gain | 인접-중 | B1 | 0.3087 | 0.3326 | +0.0240 | 0.3930 | 0.4133 | +0.0203 |
| lowshelf_gain | 인접-상 | forward | 0.3638 | 0.3796 | +0.0158 | 0.4387 | 0.4527 | +0.0140 |
| lowshelf_gain | 인접-상 | B1 | 0.3638 | 0.3796 | +0.0158 | 0.4387 | 0.4527 | +0.0140 |
| peak_gain | 전범위 | forward | 0.4993 | 0.5012 | +0.0019 | 0.5800 | 0.5836 | +0.0037 |
| peak_gain | 전범위 | B1 | 0.4993 | 0.5012 | +0.0019 | 0.5800 | 0.5836 | +0.0037 |
| peak_gain | 전범위 | B2 | 0.4383 | 0.4448 | +0.0065 | 0.5236 | 0.5318 | +0.0083 |
| peak_gain | 하위1/3 | forward | 0.4310 | 0.4334 | +0.0024 | 0.5308 | 0.5310 | +0.0002 |
| peak_gain | 하위1/3 | B1 | 0.4310 | 0.4334 | +0.0024 | 0.5308 | 0.5310 | +0.0002 |
| peak_gain | 하위1/3 | B2 | 0.4206 | 0.4232 | +0.0026 | 0.5200 | 0.5211 | +0.0011 |
| peak_gain | 중위1/3 | forward | 0.4447 | 0.4471 | +0.0024 | 0.5297 | 0.5337 | +0.0040 |
| peak_gain | 중위1/3 | B1 | 0.4447 | 0.4471 | +0.0024 | 0.5297 | 0.5337 | +0.0040 |
| peak_gain | 중위1/3 | B2 | 0.4298 | 0.4343 | +0.0045 | 0.5166 | 0.5216 | +0.0050 |
| peak_gain | 상위1/3 | forward | 0.4807 | 0.4832 | +0.0025 | 0.5668 | 0.5723 | +0.0055 |
| peak_gain | 상위1/3 | B1 | 0.4807 | 0.4832 | +0.0025 | 0.5668 | 0.5723 | +0.0055 |
| peak_gain | 상위1/3 | B2 | 0.4626 | 0.4661 | +0.0035 | 0.5463 | 0.5529 | +0.0065 |
| peak_gain | 인접-하 | forward | 0.4111 | 0.4134 | +0.0022 | 0.5096 | 0.5098 | +0.0002 |
| peak_gain | 인접-하 | B1 | 0.4111 | 0.4134 | +0.0022 | 0.5096 | 0.5098 | +0.0002 |
| peak_gain | 인접-중 | forward | 0.4302 | 0.4326 | +0.0023 | 0.5143 | 0.5187 | +0.0044 |
| peak_gain | 인접-중 | B1 | 0.4302 | 0.4326 | +0.0023 | 0.5143 | 0.5187 | +0.0044 |
| peak_gain | 인접-상 | forward | 0.4734 | 0.4764 | +0.0030 | 0.5579 | 0.5644 | +0.0066 |
| peak_gain | 인접-상 | B1 | 0.4734 | 0.4764 | +0.0030 | 0.5579 | 0.5644 | +0.0066 |

## 3. 20_family_cosine_oat — 과제 C (family 평균 코사인, split-half 보정)

| 이펙트 | 패밀리쌍 | cross_cosine(raw) | cross_cosine(단위평균) | 차이 |
|---|---|---|---|---|
| reverb | bass\|brass | 0.6089 | 0.6403 | +0.0315 |
| reverb | bass\|flute | 0.6435 | 0.6820 | +0.0384 |
| reverb | bass\|guitar | 0.8469 | 0.8581 | +0.0112 |
| reverb | bass\|keyboard | 0.8999 | 0.9191 | +0.0192 |
| reverb | bass\|mallet | 0.7844 | 0.7800 | +0.0045 |
| reverb | bass\|organ | 0.6299 | 0.6866 | +0.0568 |
| reverb | bass\|reed | 0.6872 | 0.7066 | +0.0194 |
| reverb | bass\|string | 0.7791 | 0.7602 | +0.0190 |
| reverb | bass\|vocal | 0.6011 | 0.6147 | +0.0137 |
| reverb | brass\|flute | 0.8249 | 0.8274 | +0.0025 |
| reverb | brass\|guitar | 0.6239 | 0.6272 | +0.0033 |
| reverb | brass\|keyboard | 0.6774 | 0.6880 | +0.0106 |
| reverb | brass\|mallet | 0.4535 | 0.4589 | +0.0054 |
| reverb | brass\|organ | 0.7161 | 0.7433 | +0.0272 |
| reverb | brass\|reed | 0.8553 | 0.8540 | +0.0013 |
| reverb | brass\|string | 0.7427 | 0.7877 | +0.0449 |
| reverb | brass\|vocal | 0.7156 | 0.6758 | +0.0398 |
| reverb | flute\|guitar | 0.7114 | 0.7107 | +0.0007 |
| reverb | flute\|keyboard | 0.7312 | 0.7357 | +0.0045 |
| reverb | flute\|mallet | 0.4822 | 0.4919 | +0.0098 |
| reverb | flute\|organ | 0.8930 | 0.8900 | +0.0030 |
| reverb | flute\|reed | 0.7650 | 0.7905 | +0.0255 |
| reverb | flute\|string | 0.6293 | 0.6989 | +0.0696 |
| reverb | flute\|vocal | 0.7193 | 0.6617 | +0.0575 |
| reverb | guitar\|keyboard | 0.9119 | 0.8978 | +0.0141 |
| reverb | guitar\|mallet | 0.6917 | 0.6789 | +0.0128 |
| reverb | guitar\|organ | 0.7132 | 0.7337 | +0.0205 |
| reverb | guitar\|reed | 0.6951 | 0.6987 | +0.0035 |
| reverb | guitar\|string | 0.6355 | 0.6588 | +0.0233 |
| reverb | guitar\|vocal | 0.6640 | 0.6203 | +0.0437 |
| reverb | keyboard\|mallet | 0.8090 | 0.8110 | +0.0020 |
| reverb | keyboard\|organ | 0.7083 | 0.7181 | +0.0098 |
| reverb | keyboard\|reed | 0.7792 | 0.7832 | +0.0040 |
| reverb | keyboard\|string | 0.7387 | 0.7519 | +0.0132 |
| reverb | keyboard\|vocal | 0.6789 | 0.6406 | +0.0383 |
| reverb | mallet\|organ | 0.4556 | 0.4701 | +0.0145 |
| reverb | mallet\|reed | 0.6051 | 0.5972 | +0.0079 |
| reverb | mallet\|string | 0.6400 | 0.5861 | +0.0539 |
| reverb | mallet\|vocal | 0.4776 | 0.4730 | +0.0045 |
| reverb | organ\|reed | 0.6578 | 0.6953 | +0.0375 |
| reverb | organ\|string | 0.5931 | 0.6749 | +0.0818 |
| reverb | organ\|vocal | 0.7506 | 0.7333 | +0.0174 |
| reverb | reed\|string | 0.7623 | 0.7966 | +0.0343 |
| reverb | reed\|vocal | 0.6656 | 0.6252 | +0.0404 |
| reverb | string\|vocal | 0.5674 | 0.5610 | +0.0064 |
| distortion | bass\|brass | 0.5799 | 0.6004 | +0.0205 |
| distortion | bass\|flute | 0.6221 | 0.6403 | +0.0182 |
| distortion | bass\|guitar | 0.8829 | 0.8964 | +0.0135 |
| distortion | bass\|keyboard | 0.7927 | 0.8104 | +0.0177 |
| distortion | bass\|mallet | 0.6572 | 0.6706 | +0.0134 |
| distortion | bass\|organ | 0.6137 | 0.6372 | +0.0236 |
| distortion | bass\|reed | 0.5975 | 0.6073 | +0.0097 |
| distortion | bass\|string | 0.6767 | 0.6788 | +0.0021 |
| distortion | bass\|vocal | 0.5707 | 0.5931 | +0.0224 |
| distortion | brass\|flute | 0.8072 | 0.8080 | +0.0007 |
| distortion | brass\|guitar | 0.7094 | 0.7133 | +0.0039 |
| distortion | brass\|keyboard | 0.7642 | 0.7728 | +0.0086 |
| distortion | brass\|mallet | 0.6538 | 0.6605 | +0.0067 |
| distortion | brass\|organ | 0.7675 | 0.7699 | +0.0025 |
| distortion | brass\|reed | 0.8821 | 0.8793 | +0.0028 |
| distortion | brass\|string | 0.8428 | 0.8503 | +0.0075 |
| distortion | brass\|vocal | 0.7656 | 0.7416 | +0.0240 |
| distortion | flute\|guitar | 0.6976 | 0.7025 | +0.0049 |
| distortion | flute\|keyboard | 0.7129 | 0.7239 | +0.0111 |
| distortion | flute\|mallet | 0.6522 | 0.6603 | +0.0081 |
| distortion | flute\|organ | 0.8697 | 0.8726 | +0.0029 |
| distortion | flute\|reed | 0.7564 | 0.7375 | +0.0189 |
| distortion | flute\|string | 0.7775 | 0.7814 | +0.0038 |
| distortion | flute\|vocal | 0.6896 | 0.6520 | +0.0376 |
| distortion | guitar\|keyboard | 0.9210 | 0.9234 | +0.0024 |
| distortion | guitar\|mallet | 0.8050 | 0.8053 | +0.0003 |
| distortion | guitar\|organ | 0.7156 | 0.7219 | +0.0062 |
| distortion | guitar\|reed | 0.7158 | 0.6987 | +0.0171 |
| distortion | guitar\|string | 0.7392 | 0.7373 | +0.0019 |
| distortion | guitar\|vocal | 0.6199 | 0.6041 | +0.0158 |
| distortion | keyboard\|mallet | 0.8573 | 0.8580 | +0.0008 |
| distortion | keyboard\|organ | 0.7512 | 0.7600 | +0.0088 |
| distortion | keyboard\|reed | 0.7640 | 0.7666 | +0.0027 |
| distortion | keyboard\|string | 0.7928 | 0.7890 | +0.0038 |
| distortion | keyboard\|vocal | 0.6844 | 0.6711 | +0.0133 |
| distortion | mallet\|organ | 0.6578 | 0.6586 | +0.0008 |
| distortion | mallet\|reed | 0.5984 | 0.5960 | +0.0024 |
| distortion | mallet\|string | 0.6508 | 0.6587 | +0.0079 |
| distortion | mallet\|vocal | 0.5897 | 0.5650 | +0.0247 |
| distortion | organ\|reed | 0.7673 | 0.7588 | +0.0086 |
| distortion | organ\|string | 0.7627 | 0.7644 | +0.0017 |
| distortion | organ\|vocal | 0.7733 | 0.7352 | +0.0381 |
| distortion | reed\|string | 0.7916 | 0.7884 | +0.0032 |
| distortion | reed\|vocal | 0.7730 | 0.7524 | +0.0206 |
| distortion | string\|vocal | 0.6946 | 0.6524 | +0.0422 |
| highshelf | bass\|brass | 0.6831 | 0.6896 | +0.0065 |
| highshelf | bass\|flute | 0.7637 | 0.7708 | +0.0071 |
| highshelf | bass\|guitar | 0.9242 | 0.9269 | +0.0028 |
| highshelf | bass\|keyboard | 0.9101 | 0.9132 | +0.0031 |
| highshelf | bass\|mallet | 0.8068 | 0.8175 | +0.0108 |
| highshelf | bass\|organ | 0.7966 | 0.7883 | +0.0083 |
| highshelf | bass\|reed | 0.5326 | 0.5222 | +0.0103 |
| highshelf | bass\|string | 0.6825 | 0.6973 | +0.0148 |
| highshelf | bass\|vocal | 0.4886 | 0.4989 | +0.0103 |
| highshelf | brass\|flute | 0.8113 | 0.8069 | +0.0044 |
| highshelf | brass\|guitar | 0.6443 | 0.6503 | +0.0061 |
| highshelf | brass\|keyboard | 0.7101 | 0.7138 | +0.0037 |
| highshelf | brass\|mallet | 0.5952 | 0.5933 | +0.0019 |
| highshelf | brass\|organ | 0.6735 | 0.6683 | +0.0051 |
| highshelf | brass\|reed | 0.7586 | 0.7472 | +0.0114 |
| highshelf | brass\|string | 0.6906 | 0.6938 | +0.0032 |
| highshelf | brass\|vocal | 0.5400 | 0.5509 | +0.0109 |
| highshelf | flute\|guitar | 0.7441 | 0.7439 | +0.0002 |
| highshelf | flute\|keyboard | 0.8134 | 0.8123 | +0.0010 |
| highshelf | flute\|mallet | 0.6946 | 0.6921 | +0.0025 |
| highshelf | flute\|organ | 0.8078 | 0.8117 | +0.0039 |
| highshelf | flute\|reed | 0.7573 | 0.7470 | +0.0103 |
| highshelf | flute\|string | 0.7176 | 0.7251 | +0.0076 |
| highshelf | flute\|vocal | 0.5883 | 0.6038 | +0.0155 |
| highshelf | guitar\|keyboard | 0.9499 | 0.9508 | +0.0009 |
| highshelf | guitar\|mallet | 0.8505 | 0.8566 | +0.0062 |
| highshelf | guitar\|organ | 0.7830 | 0.7768 | +0.0062 |
| highshelf | guitar\|reed | 0.5639 | 0.5548 | +0.0091 |
| highshelf | guitar\|string | 0.6590 | 0.6754 | +0.0164 |
| highshelf | guitar\|vocal | 0.4388 | 0.4536 | +0.0149 |
| highshelf | keyboard\|mallet | 0.8715 | 0.8863 | +0.0148 |
| highshelf | keyboard\|organ | 0.8006 | 0.7939 | +0.0067 |
| highshelf | keyboard\|reed | 0.6335 | 0.6272 | +0.0063 |
| highshelf | keyboard\|string | 0.7111 | 0.7285 | +0.0173 |
| highshelf | keyboard\|vocal | 0.4751 | 0.5018 | +0.0267 |
| highshelf | mallet\|organ | 0.6935 | 0.6894 | +0.0041 |
| highshelf | mallet\|reed | 0.5198 | 0.5144 | +0.0054 |
| highshelf | mallet\|string | 0.5795 | 0.6008 | +0.0213 |
| highshelf | mallet\|vocal | 0.4510 | 0.4639 | +0.0129 |
| highshelf | organ\|reed | 0.5851 | 0.5615 | +0.0236 |
| highshelf | organ\|string | 0.6710 | 0.6560 | +0.0150 |
| highshelf | organ\|vocal | 0.4834 | 0.4758 | +0.0075 |
| highshelf | reed\|string | 0.8010 | 0.7888 | +0.0123 |
| highshelf | reed\|vocal | 0.5907 | 0.5983 | +0.0076 |
| highshelf | string\|vocal | 0.6311 | 0.6335 | +0.0024 |

## 결론

세 영역 전체 최대 차이 = 0.0818 (21/OAT=0.0047, q3q4=0.0343, 20/과제C=0.0818).

임계값(0.01) 이상인 항목이 있다 — 위 표에서 해당 행을 확인하고, 영향을 받는 축/구간에 대해서는 Q4의 between 기준선(값 자체는 이미 `unit(v)` 기반이라 이 결함과 무관하지만, B2가 그 기준선을 넘는지 여부의 해석은) 재검토가 필요하다.

**참고**: Q3/Q4 표의 'between 기준선' 열 자체는 `20_family_cosine_oat.bootstrap_within_between`이 `unit(v)`로 이미 정규화한 벡터의 소스쌍별 코사인을 평균한 값이다 — raw-평균 문제와 무관한 경로이므로 재계산이 필요 없음을 확인했다.

## 결함 21

> 고정방향 베이스라인(전역평균/패밀리평균오라클/family 평균 코사인)을 raw(비정규화) 벡터를 먼저 소스 간 평균한 뒤 정규화해서 구했다. 목표 지표(소스별 코사인의 평균)를 최대화하는 상수 방향은 단위벡터의 평균이지 raw 평균이 아니므로, 이 방식은 최적보다 열등한(낮은 cos) 베이스라인을 만든다 — 편향의 방향이 B2(학습 모델)가 베이스라인을 이기는 폭을 과대평가하게 만드는 쪽이었다. 단위벡터-평균 버전을 추가해 나란히 비교했다(위 결론 참고).
