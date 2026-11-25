# 설계 05 1단계 결과 요약 (자동 생성, `d05.py summarize`)

모든 팔은 후보의 3스텝을 실제 simulator로 실행하는 planner다. ORA = 사전학습 가치(설계 04), ORA-VR = 실제 return에 맞춘 가치, ORA-VRP = 그 가치 − 데이터에서 먼 상태 벌점, ORA-VC = a + b × 사전학습 가치(눈금만 고침, §4a). PRE = 적응 없는 planner(설계 02). 평가 시드 2.

## 가치 맞추기 (확인용 에피소드, t ≤ 400)

| 셀 | 맞추기 전 상관 / 기울기 | 맞춘 뒤 상관 / 기울기 | 평균 V 전 → 뒤 (목표 평균) | 학습 상태 평균 V: 절반 / 3/4 / 끝 | σ_G | d₀ | F0 | 눈금 고침 a, b (확인용 상관 / 기울기) |
|---|---|---|---|---|---|---|---|---|
| cheetah-run__friction_0.15 | 0.93 / 3.63 | 0.94 / 1.08 | 142.9 → 75.1 (73.5) | 118.4 / 118.3 / 118.3 | 47.1 | 2.27 | 통과 | -352.7, 3.01 (0.93 / 1.20) |
| quadruped-run__motor_0.35 | 0.72 / 1.21 | 0.95 / 1.17 | 187.8 → 118.3 (118.2) | 118.4 / 118.3 / 118.3 | 4.1 | 1.21 | (보고만) | -31.9, 0.80 (0.97 / 1.19) |

## 바뀐 조건 return (넘어짐 사건 / 넘어져 있는 시간 비율)

| 셀 | PRE | ORA | ORA-VR | ORA-VRP | ORA-VC |
|---|---|---|---|---|---|
| cheetah-run__friction_0.15 | 453 (0.0 / 0.00) | 276 (0.0 / 0.00) | 11 (0.1 / 0.04) | 40 (0.0 / 0.00) | 307 (0.0 / 0.00) |
| quadruped-run__motor_0.35 | 554 (2.1 / 0.05) | 569 (0.5 / 0.09) | 292 (0.5 / 0.45) | 301 (0.5 / 0.45) | 596 (0.6 / 0.03) |

## 에피소드 짝 차이 (평균 [bootstrap 95 % 구간])

| 셀 | 팔 | 팔 − ORA | 팔 − PRE | 초기 관측 일치 |
|---|---|---|---|---|
| cheetah-run__friction_0.15 | ORA-VR | -265 [-335, -203] | -442 [-491, -389] | 최대 차이 0.0e+00 |
| cheetah-run__friction_0.15 | ORA-VRP | -236 [-305, -173] | -413 [-469, -355] | 최대 차이 0.0e+00 |
| cheetah-run__friction_0.15 | ORA-VC | +31 [-46, +111] | -146 [-210, -83] | 최대 차이 0.0e+00 |
| cheetah-run__friction_0.15 | ORA-VRP − ORA-VC | -267 [-320, -215] | – | – |
| cheetah-run__friction_0.15 | ORA-VR − ORA-VC | -296 [-348, -242] | – | – |
| cheetah-run__friction_0.15 | ORA-VR − ORA-VRP | -29 [-47, -14] | – | – |
| quadruped-run__motor_0.35 | ORA-VR | -277 [-380, -181] | -262 [-362, -165] | 최대 차이 0.0e+00 |
| quadruped-run__motor_0.35 | ORA-VRP | -269 [-375, -170] | -254 [-356, -154] | 최대 차이 0.0e+00 |
| quadruped-run__motor_0.35 | ORA-VC | +26 [-1, +74] | +41 [+8, +84] | 최대 차이 0.0e+00 |
| quadruped-run__motor_0.35 | ORA-VRP − ORA-VC | -295 [-399, -190] | – | – |
| quadruped-run__motor_0.35 | ORA-VR − ORA-VC | -303 [-404, -202] | – | – |
| quadruped-run__motor_0.35 | ORA-VR − ORA-VRP | -8 [-12, -4] | – | – |

## 동작 (모든 에피소드 기록)

| 셀 | 팔 | 속도 (m/s) | 2 m/s에 처음 닿는 스텝(중앙값, cheetah만) | 부드러움 / 노력 / 흔들림 | 계획 가치 − 실현 return | d/d₀ 평균 (벌점 > 0인 스텝 비율) | 계획 시간(초/스텝) |
|---|---|---|---|---|---|---|---|
| cheetah-run__friction_0.15 | ORA | 2.76 | 196 | 5.06 / 3.14 / 1.48 | 94.2 | – | 0.19 |
| cheetah-run__friction_0.15 | ORA-VR | 0.11 | 501 | 9.85 / 2.97 / 1.54 | 96.8 | – | 0.29 |
| cheetah-run__friction_0.15 | ORA-VRP | 0.40 | 318 | 12.20 / 3.12 / 1.69 | 80.7 | 1.10 (0.70) | 0.30 |
| cheetah-run__friction_0.15 | ORA-VC | 3.07 | 160 | 5.79 / 3.35 / 1.52 | 27.1 | – | 0.31 |
| quadruped-run__motor_0.35 | ORA | 1.25 | – | 11.49 / 6.20 / 1.38 | 73.2 | – | 0.42 |
| quadruped-run__motor_0.35 | ORA-VR | 0.27 | – | 22.00 / 5.51 / 1.10 | 90.7 | – | 0.43 |
| quadruped-run__motor_0.35 | ORA-VRP | 0.43 | – | 21.77 / 5.68 / 1.06 | 84.6 | 1.64 (0.96) | 0.43 |
| quadruped-run__motor_0.35 | ORA-VC | 1.24 | – | 11.56 / 6.16 / 1.37 | 1.0 | – | 0.43 |

## 판정

- F0 맞추기 타당성 (cheetah 마찰): 예
- ORA-VR: G1 아니오, G2 아니오, cheetah 마찰에서 PRE보다 확실히 높음(보고만) 아니오
- ORA-VRP: G1 아니오, G2 아니오, cheetah 마찰에서 PRE보다 확실히 높음(보고만) 아니오
- ORA-VC: G1 아니오, G2 예, cheetah 마찰에서 PRE보다 확실히 높음(보고만) 아니오
- 관문 통과: 아니오
- → G1을 만족하는 팔: 없음

## 예측 대조

- P1 F0 통과: 예
- P2 cheetah 마찰 ORA-VR − ORA > 0: 아니오
- P3 cheetah 마찰 ORA-VRP − ORA > 0: 아니오
- P4 G1 만족하는 팔 있음: 아니오
- P5 cheetah 마찰에서 PRE보다 확실히 높은 팔 있음: 아니오
- P6 quadruped 모터 ORA-VRP G2: 아니오
- P7 관문 통과: 아니오
- P8 cheetah 마찰 ORA-VR이 2 m/s에 닿는 스텝 < ORA: 아니오
- P9 cheetah 마찰 ORA-VC − ORA > 0: 아니오
- P10 quadruped 모터 ORA-VC G2: 예
- P11 quadruped 모터 ORA-VR 넘어져 있는 시간 > ORA: 예
