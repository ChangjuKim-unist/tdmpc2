# 문헌조사: 합의 planner의 위치, 학습한 모델 병목, 문제 설정과 기준선 (2026-11-26, GPU 없음, 등록 안 함)

설계 06 결과(정확한 dynamics와의 합의는 통과, 학습한 모델과의 합의는 quadruped 모터에서 실패) 뒤에 한 조사다. 세 갈래를 따로 조사했다. 읽은 깊이를 논문마다 적는다.

- **방법 절 읽음:** 방법 절이나 해당 절을 본문으로 읽음.
- **초록만:** arXiv 기록의 초록만 봄.
- **미확인:** 기록을 직접 열지 못함. 논문에 인용하기 전에 반드시 다시 확인한다.

서지는 조사 중 arXiv·출판사 기록에서 가져왔다. 초안의 bib에 넣을 때는 항목마다 기록과 다시 대조한다(RSS는 지어낸 인용을 desk reject한다).

## 1. 합의 규칙은 우리 것인가

**판정: 같은 규칙을 쓴 논문은 찾지 못했다. 형태는 알려진 틀의 특수형이다.** 같은 규칙이 발표되지 않았다는 쪽의 신뢰도는 중간(약 70 %)이다. Petrik·ARMOR·CARL을 인용한 논문 추적은 하지 못했다(가장 큰 구멍).

### 1.1 알려진 틀과의 관계

- Petrik et al. 2016의 목적식은 max_π min_{ξ∈Ξ} [ρ(π, ξ) − ρ(π_B, ξ)]다. π를 후보 행동열, Ξ를 {사전학습, 적응 후}, π_B를 "후보 512개의 평균"으로 놓으면 우리 점수와 같은 꼴이다. ARMOR(Bhardwaj et al. 2023)와 ATAC(Cheng et al. 2022)의 relative pessimism도 같은 계열이다.
- **보장은 넘어오지 않는다.**
  - 그 정리들은 참 모델이 집합 안에 있다고 가정한다. 우리 두 모델은 둘 다 틀렸다.
  - 보장의 내용은 "기준 π_B보다 나쁘지 않다"이다. 우리 기준은 매 반복 바뀌는 후보 평균이라, 넘어와도 "평균 후보보다 낫다"뿐이다. 적응 없는 planner가 기준이 아니다.
  - 우리 J_x는 낡은 terminal Q를 쓰므로 적응한 모델 아래의 온전한 return이 아니다.
- **평균을 빼는 이유가 여기서 나온다(식에서 유도, 논문에 있는 내용 아님).** 기준을 "sim 모델이 가장 좋아하는 계획"으로 잡으면 모든 후보의 A_s ≤ 0이고 그 계획의 점수가 0이라, 답이 항상 sim 계획이 된다. 적응 이득이 0이다. 후보 평균을 기준으로 잡는 것이 이 퇴화를 피한다.
- **넘어오는 논증:** 절대 비관(최솟값)보다 상대 비관이 덜 보수적이다. 두 계획에 공통인 모델 오차가 상쇄되기 때문이다(Petrik §3, §5.1).

### 1.2 가까운 논문과의 차이

| 논문 | 모델 집합 | 결합 방식 | 우리와의 차이 | 읽은 깊이 |
|---|---|---|---|---|
| Petrik, Chow, Ghavamzadeh, "Safe Policy Improvement by Minimizing Robust Baseline Regret", 2016, arXiv 1607.03842 | 오차 한계가 있는 모델 집합 | 기준 정책 대비 최악 차이 | 학습 단계, 참 모델 포함 가정, 기준이 정책 | 방법 절 읽음 (부록 증명 제외) |
| Bhardwaj, Xie, Boots, Jiang, Cheng, "Adversarial Model for Offline Reinforcement Learning" (ARMOR), 2023, arXiv 2302.11048 | 데이터에 맞는 모델 전체 | 기준 정책 대비 최악 차이, 모델을 적대적으로 학습 | 학습 필요, 조정 값 3개 | 방법 절 읽음 |
| Zhang, Cheung, Finn, Levine, Jayaraman, "Cautious Adaptation For Reinforcement Learning in Safety-Critical Settings" (CARL), 2020, arXiv 2008.06622 | 확률 ensemble 하나 | 후보마다 하위 분위 평균(절대 비관) | 사전·사후 두 모델 아님, 상대 점수 아님, 조정 값 있음 | 방법 절 읽음 |
| Aswani, Gonzalez, Sastry, Tomlin, "Provably Safe and Robust Learning-Based Model Predictive Control" (LBMPC), 2011, arXiv 1107.2487 | 옛 선형 모델 + 학습 모델 | 비용은 학습 모델, 제약은 옛 모델 | 제약으로 결합, 오차 한계를 알아야 함 | 방법 절 읽음 |
| Feng et al., "Finetuning Offline World Models in the Real World" (FOWM), 2023, arXiv 2310.16029 / Remonda et al. (PIA), 2026, arXiv 2610.26757 | dynamics 하나 | planner 점수에서 λ · Q ensemble 표준편차를 뺌 | 옛 dynamics를 남기지 않음, 조정 값 λ | 방법 절 읽음 (FOWM은 일부) |
| Hadfield-Menell et al., "Inverse Reward Design", 2017, arXiv 1711.02827 (부록) | 보상 가설 여럿 | argmax min_i (w_iᵀφ − c_i), 기준점 c_i 필요를 명시 | 가설이 보상, 기준이 log-sum-exp. 식이 가장 닮음 | 방법 절 읽음 (부록) |
| Kurutach et al., "Model-Ensemble Trust-Region Policy Optimization", 2018, arXiv 1802.10592 | 모델 K개 | 새 정책이 나아지는 모델 비율이 70 % 아래면 멈춤 | 학습 중 정지 규칙 | 방법 절 읽음 (한 단락) |
| Kalaria et al., "Agile Mobility with Rapid Online Adaptation via Meta-learning and Uncertainty-aware MPPI", 2024, arXiv 2410.06565 | 모델 하나 | MPPI 비용에 예측 분산 벌점 | 단일 모델 | 방법 절 읽음 |
| Xu et al., "Cross-Domain Policy Adaptation via Value-Guided Data Filtering" (VGDF), 2023, arXiv 2305.17625 | 타깃 dynamics ensemble | 두 도메인의 가치가 맞는 데이터만 학습에 씀 | 데이터 선택이지 계획 채점이 아님 | 방법 절 읽음 |
| Wendl et al., "Safe Exploration via Policy Priors" (SOOPER), 2026, arXiv 2601.19612 | 보정된 모델 집합 | 비관적 비용이 예산을 넘으면 안전한 정책으로 전환 | 정책 전환, 보정된 불확실성 필요 | 방법 절 읽음 |
| Subramanian et al., "On the Practical Design of Tube-enhanced Multi-stage Nonlinear Model Predictive Control", 2022, arXiv 2204.09607 | 시나리오 트리 | 시나리오 비용의 가중합 + 모든 시나리오에서 제약 | 가중합과 제약 | 방법 절 읽음 |

- **앞선 기록의 인용 오류 둘:** ARMOR의 1저자는 Bhardwaj다(Cheng 아님). "Steimle et al. 2021"은 MPC가 아니라 multi-model MDP 논문이다(초록만). Dadhe & Engell 2008은 2차 인용으로만 봤다(미확인).
- **Petrik 2016의 저자 순서:** arXiv는 Petrik, Chow, Ghavamzadeh인데 NeurIPS proceedings는 다른 순서일 수 있다. 인용 전에 proceedings를 확인한다.

### 1.3 예상 반론과 답 (가능성 순)

1. **"원소 둘짜리 relative pessimism일 뿐이다"** (Petrik, ARMOR, ATAC). 계보를 스스로 밝힌다. 차이는 참 모델 포함 가정 없이 계획 채점 시점에 하고, 자연스러운 기준이 퇴화해서 후보 평균을 쓴다는 것이다. 보장은 실험적이라고 쓴다.
2. **"TD-MPC 적응의 planner 벌점은 이미 있다"** (FOWM, PIA). 직접 비교가 필요하다. 설계 02의 sim 혼합 + 벌점 팔(MIXPEN1)이 이 기준선이다. §4.1 참고.
3. **"적응 중 비관적 계획은 CARL이 했다."** 절대 비관은 평소 성능을 깎는다는 것을 숫자로 보여야 한다.
4. **"옛 모델로 안전, 새 모델로 성능은 2011년 아이디어다"** (LBMPC). 오차 한계 없이 점수 수준에서 대칭으로 결합한다. 그만큼 보장이 없음을 인정한다.
5. **"ensemble 최솟값과 뭐가 다른가."** 평균을 빼지 않은 최솟값 팔을 ablation으로 넣는다(설계 06 §8의 2단계에 이미 있음).

**쓸 수 있는 표현:** "알려진 원리(상대 비관)의 새 적용: 적응 전·후 두 모델, 계획 채점 시점, 조정 값 없음". "새 원리"라고 쓰지 않는다. "guarantee", "safe"는 쓰지 않고 구간이 붙은 실험 기준으로 쓴다.

## 2. 학습한 모델 병목을 푸는 장치

**판정: 문헌이 가장 지지하는 것은 적응한 planner로 데이터를 다시 모아 다시 맞추는 방식(반복 수집)이다.** 다만 우리 예산(5,000 스텝)에서, 넘어짐이 실패인 다리 로봇으로 확인한 논문은 없다.

| 장치 | 가장 좋은 근거 | 약점 | 우리 진단과의 맞음 |
|---|---|---|---|
| 같은 예산을 출발·회복 구간에 다시 배분 | 간접 근거뿐. 같은 예산에서 짧은 에피소드와 긴 에피소드를 비교한 논문을 찾지 못함 | Ross & Bagnell에서 고정된 탐색 분포만 쓰는 방식은 세 분포 중 하나에서만 됨. 일어나는 데이터와 다시 넘어지는 데이터는 다름 | 증상(6.8 %)과는 맞지만 근거가 약함 |
| 반복 수집 (planner가 가는 상태의 데이터를 모아 합침) | Ross & Bagnell 2012: 라운드마다 절반은 탐색 분포, 절반은 현재 정책. 모든 탐색 분포에서 고정 방식보다 나음. SimDist(실기 Go2), PIA(실차), FOWM(실기 팔)이 같은 구조 | 실기에서 실패를 실제로 겪어야 함. 실기 예산이 3–36분으로 우리보다 큼. 다시 맞춘 뒤 방문 분포가 또 움직일 수 있음 | 직접 맞음. 우리 실패는 수집 분포와 planner 분포의 차이(6.8 % 대 26 %)다 |
| 겨냥한 탐색 | TERRA: 모델 오차가 클 곳을 MPPI 보너스로 찾아감. 탐색 이득은 작음(성공률 0.91 대 0.85) | 오차가 큰 곳은 다리 로봇에서 넘어지는 곳이다 | 일부 맞음. Go1에서 위험한 방향 |
| 모르는 곳에서 적응한 쪽이 빠짐 | PIA: Q ensemble 벌점을 빼면 실패 9 → 20. RWM-U: 벌점 크기의 쓸 수 있는 범위가 좁음 | latent ensemble의 불확실성은 보정이 나쁨. 판별기도 모자란 그 상태의 데이터가 필요함 | 합의 planner가 이미 이 계열이다 |
| 사전학습 쪽으로 묶는 규제 | TERRA: 저차원 적응 0.91 대 전체 적응 0.68 | 덮이지 않은 상태의 오차를 잰 논문 없음. 이미 sim을 절반 섞고 있음 | 미확인 |

- **가장 강한 반론:** 반복 수집을 우리 예산에서 넘어짐으로 확인한 논문이 없다. Go1에서는 넘어지는 planner를 일부러 돌려야 한다.
- **우리 흐름과 겹치는 선행:** "적응 전·후 합의 채점 + 실패 상태 겨냥 데이터"를 합친 논문은 찾지 못했다. 가장 가까운 것은 PIA다(sim 사전학습 TD-MPC2, 벌점 planner, sim 절반 혼합의 온라인 적응, 실차).

읽은 논문: Ross & Bagnell, "Agnostic System Identification for Model-Based Reinforcement Learning", 2012, arXiv 1203.1007 (방법 절 읽음). Levy et al., TERRA, 2026, arXiv 2610.38657 (본문 읽음). Levy et al., SimDist, 2026, arXiv 2603.15759, RSS 2026 (방법 절 읽음). Remonda et al., PIA, 2026, arXiv 2610.26757 (방법 절 읽음). Feng et al., FOWM, 2023, arXiv 2310.16029 (방법 절 읽음). Li, Krause, Hutter, "Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots", 2025, arXiv 2504.16680 (실기 절만). Rothfuss et al., "Bridging the Sim-to-Real Gap with Bayesian Inference", 2024, arXiv 2403.16644 (실험 절). Smith et al., "Legged Robots that Keep on Learning", 2021, arXiv 2110.05457 (일부). 나머지(CLAW 2609.12278, DreamTIP 2604.02911, Biased Dreams 2604.25416, model preconditions 2206.05573·2401.04007 등)는 초록만.

## 3. 문제 설정: 우리 발견은 이미 발표됐는가

| 발견 | 판정 | 가장 가까운 보고 |
|---|---|---|
| dynamics fine-tuning이 예측 오차는 줄이지만 return은 못 살리거나 적응 없음보다 나쁘다 | 부분적으로 발표됨 | ReDRAW §5.1.2: dynamics만 fine-tuning한 기준선이 결국 overfit한다고 씀(Dreamer 구조, 적응 없음 아래로 내려가는지는 미확인). 예측 오차와 계획 성능의 분리는 Objective Mismatch(2002.04523, 초록만)와 2026년 JEPA 논문 둘(2610.19375, 2610.05550, 초록만) |
| 알려진 가치 쪽 처방이 안 통한다 | 부분적으로 발표됨 | SimDist §V-E: 가치를 풀면 catastrophic forgetting. 적응 뒤 Q 최솟값이 실패한다는 보고는 찾지 못함 |
| 정확한 dynamics + 낡은 terminal 가치가 적응 없음보다 나쁘다 | 실험은 찾지 못함. 개념은 예견됨 | 2601.21306 §4: 완벽한 모델과 가치로도 search가 실패할 수 있다(원인은 다름). LOOP(2008.10066)와 Bhardwaj et al.(2012.05909)의 상계: 모델 오차가 0이어도 가치 오차 항이 남는다. Self-Consistent Models and Values(2110.12840, 초록만) |

- **새로워 보이는 부분:** TD-MPC2(MPPI + terminal Q)에서 적응 없음 아래로 떨어지는 것, 알려진 처방 5종이 절반 이하만 고친다는 체계적 비교, 정확한 dynamics 실험.
- **약점:** 셋째 발견의 근거는 6개 조건 중 1개(276 대 453)다.
- **반대 증거:** SimDist(RSS 2026)는 같은 방식(dynamics만 fine-tuning, 가치 고정)으로 실기에서 단조 개선을 보고한다. 차이는 planning horizon(5 대 3), 반복 수집 대 한 번 모은 데이터다.

## 4. 기준선과 반론

### 4.1 이미 돌린 기준선과의 비교 (설계 02·06의 숫자)

FOWM·PIA 방식의 벌점 planner에 sim 혼합을 더한 팔(MIXPEN1)과 학습한 모델 합의(CON-MIX)의 바뀐 조건 return.

| 조건 | 적응 없음 | sim 혼합 + 벌점 | 학습한 모델 합의 | 합의 − 벌점 |
|---|---|---|---|---|
| walker 질량 | 573 | 821 | 719 | −102 [−124, −79] |
| walker 마찰 | 471 | 533 | 498 | −35 [−92, +2] |
| quadruped 마찰 | 509 | 776 | 737 | −40 [−50, −29] |
| cheetah 질량 | 571 | 786 | 691 | −96 [−162, −49] |
| quadruped 모터 | 554 | 548 | 506 | −41 [−109, +23] |
| cheetah 마찰 | 453 | 116 | 383 | +267 [+211, +322] |

- 지금의 학습한 모델 합의는 이 기준선을 5개 조건에서 못 넘는다. 넘는 것은 cheetah 마찰 하나다.
- 평소 조건 return은 둘 다 적응 없음의 0.97–1.01배다.
- 정확한 dynamics와의 합의(830, 624, 894, 812, 591, 572)는 6개 조건 모두 이 기준선보다 높다. 차이는 학습한 모델에서 생긴다.

### 4.2 아직 안 돌린 기준선

| 기준선 | 무엇을 적응 | 이식 난이도 |
|---|---|---|
| PIA 방식: sim + 실데이터 replay로 TD-MPC2 전체 loss | 모델과 가치 전부 | 낮음–중간 (실제 보상 필요) |
| 반복 수집 (SimDist 방식) | dynamics만, 라운드마다 다시 수집 | 낮음 |
| 충분한 예산의 상상 다시 맞추기 (ReDRAW 방식) | 가치·정책 | 중간 |
| CARL: dynamics ensemble의 하위 분위 계획 | ensemble 필요 | 중간 |
| system ID 뒤 sim에서 다시 학습 / domain randomisation·RMA | 실데이터 fine-tuning 없음 | 중간–높음 |

### 4.3 가장 아픈 반론 셋

1. **"SimDist는 같은 방식으로 된다."** 반복 수집을 통제한 직접 비교가 없으면 구현 문제로 읽힌다.
2. **"방법이 정확한 dynamics에서만 된다."** 학습한 모델로는 한 조건에서 실패했고 알려진 기준선을 못 넘는다. 실기 결과와 형식적 보장이 없다.
3. **"더 쉬운 길이 있다."** 질량·마찰·모터 세 값이면 system ID나 domain randomisation으로 풀 수 있다는 지적.

## 5. 확인하지 못한 것

- Petrik·ARMOR·CARL을 인용한 논문 추적(Semantic Scholar 제한으로 실패).
- 유료 제어 문헌(min-max regret MPC)의 본문, 특허.
- §3의 논문 대부분은 초록만 봤다. 본문을 처음부터 끝까지 읽은 논문은 TERRA 하나다.
- 2026년 하반기 논문은 웹 검색 위주라 빠진 것이 있을 수 있다.
- 같은 예산에서 에피소드 길이를 비교한 연구가 없다는 것은 검색 약 10회의 결과다.
