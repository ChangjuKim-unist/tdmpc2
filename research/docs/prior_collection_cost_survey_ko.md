# 문헌조사: 착취를 찾아가는 수집과 그 비용 (2026-11-27)

설계 07 뒤의 문헌조사다(설계 07 §8 셋째 줄, 설계 08 §0). GPU를 쓰지 않았다. 질문은 셋이다.

1. "적응 모델만 쓰는 planner로 모으면 모델이 고쳐진다"는 이미 발표됐는가?
2. 라운드 수와 라운드당 데이터 양의 교환에 대해 알려진 결과가 있는가?
3. 그런 수집의 비용(넘어짐, 잃은 return)을 줄이는 방법은 무엇이 있고, 무엇이 비어 있는가?

**읽은 깊이와 주의.** "본문"은 방법 절의 해당 대목을 읽었다는 뜻이고, 끝까지 통독한 논문은 없다. venue는 BREMEN, DaD, Abbeel 2006만 확인했고 나머지는 arXiv 기록만 대조했다. 검색은 약 30회이고 인용 추적은 하지 않았다. **논문에 인용하기 전에 각 문헌을 그 문헌의 기록과 다시 대조한다**(RSS는 잘못된 참고문헌이 있으면 심사 없이 거절한다).

## 0. 결론

- **적응 모델 수집의 이득은 새롭지 않다.** Ross & Bagnell 2012의 Algorithm 1이 그 구조이고, SimDist가 실기 quadruped에서 같은 일을 한다. Amigo et al. 2026은 착취하는 정책을 일부러 실기에 올려 모델을 고친다고 본문에 적었다.
- **"수집은 공격적으로, 배치는 보수적으로"도 발표돼 있다**(OOO, COPlanner, SOOPER).
- **찾지 못한 것은 비용 쪽이다.** 착취 수집의 비용을 재고 줄이는 연구, 그리고 "착취가 드러나면 sim planner로 넘기고 그때까지의 전이는 남긴다"는 trigger는 찾지 못했다(신뢰도 낮음–중간, "없다"가 아니라 "못 찾았다").

## 1. 착취를 수집 신호로 쓰는 것

- **Ross & Bagnell 2012 (본문, Alg. 1).** 라운드마다 표본의 절반은 현재 모델의 최적 정책으로, 절반은 탐색 분포로 모은다. 모델을 다시 맞추고 정책을 다시 푼다. 설계 07의 적응 수집 팔은 이 알고리즘의 라운드 2회짜리다(탐색 분포 = 적응 없는 planner의 첫 5 에피소드). 차이: 보수적인 배치 planner가 따로 없고, 수집 중 실패 비용을 다루지 않으며, 헬리콥터 simulation만 있다.
- **Amigo et al. 2026 (본문 §3.5, App. A.2).** world model 안에서 RL이 착취를 찾으면 그 정책을 실기 humanoid에 올려 겨냥한 데이터를 모으고 모델을 fine-tuning한다. 차이: 절차로만 적혀 있고 통제 비교, 수집 비용 측정, 보수적 배치가 없다. manipulation이라 넘어짐이 비용이 아니다.
- **SimDist (본문, Alg. 1).** MPPI와 적응 모델로 on-policy rollout을 모으고 dynamics만 다시 맞추기를 반복한다. 수집자와 배치자가 같고, 수집 중 넘어짐은 보고되지 않았다.
- **Ibarz et al. 2021 (본문 §5.6), Nagabandi et al. 2017 (본문 §IV-D).** 모델이 착취되면 실세계 데이터를 더 모아 다시 학습한다는 처방과, on-policy 집계 비율의 ablation이다.
- **보수적으로 모으는 쪽:** FOWM (본문 §3.2)과 PIA (본문 §IV)는 수집과 배치 모두 불확실성 벌점 planner를 쓴다. 설계 07의 결과(합의 수집 +21, 적응 수집 +69)와 반대 방향의 선택이다.
- **낙관으로 모으는 쪽:** COPlanner (본문)는 같은 다단계 불확실성을 model rollout에는 벌점으로, 실환경 행동 선택에는 보너스로 쓴다. TERRA (본문 일부)는 보상 가중 오차 예측기를 MPPI 보너스로 쓴다. H-UCRL, MAX, Plan2Explore는 초록만 봤다. 차이: 이들의 신호는 불확실성 추정이고, 우리 신호는 planner가 실제로 고른 계획(모델이 낙관적으로 틀린 곳)이다.
- **분리 원리:** OOO (본문 §4)는 낙관 정책으로 모으고 끝에 비관 목적으로 정책을 다시 학습한다. model-free이고 수집 비용은 다루지 않는다.
- **ASID (본문 §3–4.1):** sim에서 Fisher information을 최대화하는 탐색 정책으로 실기 1 에피소드를 모은다. 매개변수 식별용이고 착취와 무관하다.
- **초록만 본 것:** Talvitie 2017, DaD, Abbeel et al. 2006, Fröhlich et al. 2022. on-policy 오차를 고치지만 수집자를 고르는 문제는 아니다.

## 2. 라운드 수 대 라운드당 데이터 (deployment-efficient RL)

- **BREMEN (본문 §3, §5.1).** 배치 횟수와 배치당 전이 수를 나눠 센다. 처음부터 학습하는 설정이고 배치당 100k–200k 전이라 우리(라운드당 2,500 스텝)와 규모가 다르다. 같은 총량에서 분할 수를 비교한 분석은 본문에서 찾지 못했다.
- **Huang et al. 2022 (본문 Thm 4.1, App. F.1).** linear MDP 이론이다. 배치 수를 늘리면 배치당 필요한 궤적 수가 줄지만 빨리 포화한다. 저자들은 이 교환의 정확한 형태를 미해결로 적었다.
- **낮은 switching cost 계열 (초록만):** Gao et al. 2021, Bai et al. 2019.
- **쓸 수 있는 결론:** 이론은 "라운드를 늘리면 이득이 있으나 빨리 포화한다"까지다. 우리 규모에서 같은 총량의 분할 수를 비교한 실험 결과는 찾지 못했다. 직접 쓸 기준선은 Ross & Bagnell의 (라운드 수, 라운드당 표본) 조합과 SimDist의 "몇 에피소드마다 다시 맞추기"다.

## 3. 수집 비용을 줄이는 방법

후보 장치는 "적응 모델 planner가 착취를 드러낼 때까지 두고, 그 뒤 sim planner에 넘긴다"이다.

- **SOOPER (본문 §4, Alg. 1, Thm 1).** 구조가 가장 가깝다. 학습 정책이 행동하다가 (지금까지 실현된 누적 cost) + (prior 정책의 비관적 cost 가치)가 예산을 넘으면 sim 또는 offline에서 얻은 보수적 prior 정책으로 넘긴다. 넘기기 전의 전이는 모델 학습에 남긴다. 차이: trigger가 앞을 보는 cost 상계이고(관측된 예측 오차나 return 하락이 아니다), cost 함수와 보정된 불확실성이 필요하며(우리 latent 모델에는 둘 다 없다), 탐색은 내재 보상으로 한다(착취를 쓰지 않는다).
- **Yang et al. 2022 (본문 §IV).** 실기 quadruped에서 상태 기반 trigger(넘어짐 직전)에 들어가면 recovery 정책으로 넘기고, 근사 dynamics rollout으로 미래가 안전하면 학습 정책에 돌려준다. 넘어짐 비용에는 바로 쓸 수 있다. cheetah의 return 손실 같은 비용에는 trigger가 없다.
- **DAgger의 gate 계열 (초록만):** SafeDAgger, EnsembleDAgger, ThriftyDAgger. novice가 행동하다가 불일치·분산·위험 기준에서 expert로 넘긴다. expert를 sim planner로 읽으면 같은 틀이다. 다만 모으는 것이 행동 label이다.
- **그 밖 (초록만):** Leave No Trace(reset 정책의 가치로 조기 중단), Recovery RL·SAILR·Protective Policy Transfer·SLowRL(안전 critic 또는 추정기로 전환), Neural Simplex·LBMPC(두 제어기 또는 두 모델로 안전과 성능을 나눔), Domberg & Schildbach 2026(world model 예측 잔차로 fine-tuning을 시작), LaGrassa & Kroemer(모델을 믿을 영역을 추정).
- **그 trigger가 발표됐는가:** 찾지 못했다. 부품(전환 구조, 잔차 감시, 전이 보존)은 각각 있다.
- **설계상 주의(추론, 문헌 아님):** 고칠 정보가 trigger 앞에 있는지 뒤(넘어지는 구간)에 있는지는 실험으로만 알 수 있다.

## 4. 다음 설계에 주는 것

**넣어야 할 기준선**

1. 적응 없는 수집, 합의 수집, 적응 모델 5 에피소드 수집(셋째는 "Ross & Bagnell / SimDist 방식"으로 부른다).
2. 용량(1–2 에피소드)과 라운드 분할(같은 총량). 이 둘이 이미 풀면 새 장치는 필요 없다. 설계 08이 먼저 잰다.
3. 불확실성 벌점 수집(FOWM, PIA 방식).
4. 불확실성 보너스 수집(COPlanner, TERRA 방식).
5. 상태 기반 안전 전환(Yang et al. 방식).
6. 같은 착취 스텝 수를 무작위 시점에서 끊는 대조(trigger의 효과와 용량의 효과를 가르기 위한 것).

**찾지 못한 것**

- 착취 수집과 분포 내 수집을 같은 예산에서 통제 비교하고, 수집 비용을 넘어짐과 return으로 같이 보고한 연구.
- 몇 에피소드 규모에서 용량 또는 라운드 분할에 따른 비용–이득 곡선.
- 관측된 예측 오차나 return 하락으로 착취 에피소드를 끊고 sim planner에 넘기는 trigger.
- 위의 것들과 두 모델 합의 배치의 조합.

**"그냥 반복 MBRL / DAgger다"라는 지적의 근거**

Ross & Bagnell의 Alg. 1이 "절반은 현재 모델의 최적 정책, 절반은 탐색 분포"다. SimDist는 같은 것을 MPPI와 quadruped로 했고, Amigo et al.은 착취 정책을 실기에 올려 고친다고 적었다. 배치를 보수적으로 나누는 것은 OOO와 COPlanner다. 그래서 설계 07의 +69는 알려진 방법의 재현으로 읽힌다. 기여가 되려면 비용 쪽이어야 하고, 용량과 라운드 분할이 그 비용을 이미 없애지 못한다는 것을 먼저 보여야 한다.

## 5. 확인하지 못한 것

- 인용 추적(Ross & Bagnell, SOOPER, SimDist를 인용한 논문).
- 제어 학술지 쪽(plant–model mismatch 감시, 원조 Simplex).
- BREMEN의 그림 수치. PILCO와 GAT는 확인하지 않아 적지 않았다.

## 6. 참고 문헌 (arXiv 번호는 조사 때 기록과 대조한 것, 인용 전 재확인 필요)

**본문을 읽은 것**

- Ross, Bagnell. Agnostic System Identification for Model-Based Reinforcement Learning. arXiv 1203.1007
- Amigo, Khorrambakht, Mansard, Righetti. Coupled Local and Global World Models for Efficient First Order RL. arXiv 2602.06219
- Levy et al. Simulation Distillation. arXiv 2603.15759
- Wendl, As, Prajapat, Pollak, Coros, Krause. Safe Exploration via Policy Priors (SOOPER). arXiv 2601.19612
- Wang et al. COPlanner. arXiv 2310.07220
- Mark et al. Offline Retraining for Online RL (OOO). arXiv 2310.08558
- Feng, Hansen, Xiong, Rajagopalan, Wang. Finetuning Offline World Models in the Real World. arXiv 2310.16029
- Yang, Zhang, Luu, Ha, Tan, Yu. Safe Reinforcement Learning for Legged Locomotion. arXiv 2203.02638
- Matsushima, Furuta, Matsuo, Nachum, Gu. Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization (BREMEN). ICLR 2021, arXiv 2006.03647
- Huang, Chen, Zhao, Qin, Jiang, Liu. Towards Deployment-Efficient Reinforcement Learning: Lower Bound and Optimality. arXiv 2202.06450 (본문 일부)
- Ibarz, Tan, Finn, Kalakrishnan, Pastor, Levine. How to Train Your Robot with Deep Reinforcement Learning: Lessons We've Learned. arXiv 2102.02915
- Nagabandi, Kahn, Fearing, Levine. Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning. arXiv 1708.02596
- Remonda et al. Simulation Priors for Rapid Adaptation in Model-Based RL for Autonomous Racing (PIA). arXiv 2610.26757
- Levy, Bouton, Hassler, Nayar, Fridovich-Keil. Explore Where It Matters (TERRA). arXiv 2610.38657
- Memmel, Wagenmaker, Zhu, Yin, Fox, Gupta. ASID. arXiv 2404.12308

**초록만 본 것**

- Hoque et al. ThriftyDAgger. arXiv 2109.08273
- Zhang, Cho. SafeDAgger. arXiv 1605.06450
- Menda, Driggs-Campbell, Kochenderfer. EnsembleDAgger. arXiv 1807.08364
- Thananjeyan et al. Recovery RL. arXiv 2010.15920
- Eysenbach, Gu, Ibarz, Levine. Leave No Trace. arXiv 1711.06782
- Wagener, Boots, Cheng. SAILR. arXiv 2106.09110
- Yu, Liu, Turk. Protective Policy Transfer. arXiv 2012.06662
- Daneshmand et al. SLowRL. arXiv 2603.17092
- Phan et al. Neural Simplex Architecture. arXiv 1908.00528
- Aswani, Gonzalez, Sastry, Tomlin. LBMPC. arXiv 1107.2487
- Domberg, Schildbach. Self-adapting Robotic Agents through Online Continual RL with World Model Feedback. arXiv 2603.04029
- LaGrassa, Kroemer. Learning Model Preconditions for Planning with Multiple Models. arXiv 2206.05573
- LaGrassa, Lee, Kroemer. Task-Oriented Active Learning of Model Preconditions. arXiv 2401.04007
- Gao, Xie, Du, Yang. Linear MDP with Low Switching Cost. arXiv 2101.00494
- Bai, Xie, Jiang, Wang. Provably Efficient Q-Learning with Low Switching Cost. arXiv 1905.12849
- Curi, Berkenkamp, Krause. H-UCRL. arXiv 2006.08684
- Shyam, Jaśkowski, Gomez. MAX. arXiv 1810.12162
- Sekar et al. Plan2Explore. arXiv 2005.05960
- Talvitie. Self-Correcting Models for Model-Based Reinforcement Learning. arXiv 1612.06018
- Fröhlich, Lefarov, Zeilinger, Berkenkamp. On-Policy Model Errors in Reinforcement Learning. arXiv 2110.07985

**venue 기록으로만 확인 (초록 수준)**

- Venkatraman, Hebert, Bagnell. Improving Multi-step Prediction of Learned Time Series Models. AAAI 2015
- Abbeel, Quigley, Ng. Using Inaccurate Models in Reinforcement Learning. ICML 2006
