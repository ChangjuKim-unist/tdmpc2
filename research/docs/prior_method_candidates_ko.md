# 선행 연구 확인: 방법 후보 넷과 진단 (2026-11-25, GPU 없음)

설계 04가 도는 동안 방법 후보를 미리 확인했다. 검색: arXiv 본문 약 60편, 웹 검색. 이전 연구(SimDist 저장소 `docs/research_log_ko.md`)의 같은 방향 시도도 함께 적는다.

## 결론

- **진단("적응은 평균 오차를 줄이지만 행동에 따른 오차는 거의 못 줄인다, 상태당 행동이 하나라서")은 2026-11에 이미 발표됐다.** 설정은 JEPA goal-reaching이다. 우리 것으로 남는 부분은 물리 변화의 sim-to-real 적응, terminal Q를 쓰는 planner, 적응이 적응 없음보다 나빠지는 해, 그리고 그 해가 어떤 상태에서 나는지다.
- **후보 넷 모두 이웃 설정에 발표가 있다.** 그대로 내면 새로움이 얇다. 두 후보는 이전 연구에서 이미 부정적 결과가 있었다.

## 1. 진단과 겹치는 연구

| 논문 | 한 것 | 우리와의 차이 |
|---|---|---|
| sft-JEPA (2611.29453, 2026-11-22) | simulator를 되돌려 같은 상태에서 행동 9개를 실행하고 오차를 공통 부분과 행동 의존 부분으로 나눔. test-time adaptation이 공통 부분은 ×0.46, 행동 의존 부분은 ×0.93 | JEPA, 물리 변화 없음, 가치 없음 |
| Fork-JEPA (2611.28350, 2026-11-20) | 행동 반응은 정책이 상태 안에서 바꾸지 않는 방향으로는 식별되지 않는다(Prop. 1). 같은 상태의 두 행동 차이를 simulator 복원으로 재서 학습 | 처음부터 학습, 실측 fork가 필요 |
| Zhang et al. (2607.22430) | on-policy 오차 δ가 반사실 오차 δ/ρ로 커진다. 정책 잡음 σ가 0이면 닫힌 고리 사상만 정해진다 | 이론 |
| GALILEO (2206.04890) | 행동 정책 데이터로 배운 모델은 반사실 질의에서 반대 반응을 낸다 | 적응 설정 아님 |
| Few-shot residual (2611.13592) | 예측 오차는 비슷하게 줄었는데 성공률은 6 % 대 72 % | – |
| TERRA (2610.38657, SimDist 저자) | "Adapting the full dynamics model substantially degrades performance" | 보상 가중 손실 + LoRA |

- 문헌에 없는 것: fine-tuning이 행동 효과를 **손상**한다는 증거, 그리고 dynamics가 정확해질수록 낡은 terminal 가치와 어긋나 해가 된다는 측정.
- 계획 순위로 world model을 평가하는 지표는 여러 편에 있다(AD-WM 2609.30264, pico-WAM 2611.01039, ARC-Bench 2609.05461). 지표는 새롭지 않다.

## 2. 후보별 판정

| 후보 | 판정 | 가장 가까운 연구 | 남는 차이 | 이전 연구의 결과 |
|---|---|---|---|---|
| 행동 통로만 적응(사전학습 dynamics는 얼리고 행동 변환 + 상태만의 잔차) | 이웃 설정에 발표 | Sandwich-Residuals 2609.21740(얼린 JEPA + 행동 잔차 + CEM), Reflexive World Models 2505.15589(얼린 TD-MPC2 + 보정 행동, 가치 차이 상한 정리), GAT 계열, ASAP 2502.01143, L1-MBRL 2403.14860 | terminal Q가 있는 planner 안에서 쓰는 것, "이 범위의 변화면 사전학습 가치가 유효하고 나머지가 가치의 낡음을 잰다"는 논증 | SimDist 설계 30 QA: 오프라인 예측 오차 회복 0.42로 불통과. 폐루프는 안 돌림 |
| 사전학습 모델의 행동 간 차이를 anchor로 두는 fine-tuning | 그대로는 못 찾음 | Fork-JEPA(실측 차이로 회귀), 감도 하한(2606.14585) | – | 모터 변화처럼 행동 효과 자체가 바뀌면 anchor가 틀린다 |
| 알려진 행동 흔들기를 도구 변수로 행동 효과 식별 | 이웃 설정에 발표, 원리는 교과서 | ASID 2404.12308, SPI-Active 2505.14266(simulator 파라미터 식별), TERRA(탐색 보너스) | 사전학습 latent world model의 행동 효과를 적은 데이터로 식별 | 설계 01: 흔들기 0.05에서도 return 최대 −17.7 % |
| 기준 planner의 계획보다 확실히 나을 때만 벗어나기(짝 차이에 비관) | 이웃 설정에 발표 | Petrik et al. 1607.03842(기준 대비 후회 최소화, 같은 불확실성 실현에서 비교), ARMOR 2302.11048(relative pessimism), ME-TRPO(멤버별 개선 투표), SPOWL 2506.04828(TD-MPC2에서 계획과 정책 중 Q가 큰 쪽) | 매 스텝 MPC에서 쓰는 것, fine-tuning한 world model 집합 | SimDist 설계 30 QS: 시드 5개 불일치는 오차의 1–4 %, 신호 0/4칸. 설계 33: 두 모델 전환은 보존만 되고 이득 못 냄 |

## 3. 가치 쪽으로 갈 때 이미 있는 것

- FOWM(2310.16029): 적은 실데이터에서 in-sample TD(expectile V) + Q 불확실성 벌점 + offline/online 반반 샘플링.
- PIA(2610.26757): TD-MPC2 전체를 sim + 실데이터 replay로 갱신, Q 표준편차 벌점, 데이터 출처 context.
- ReDRAW(2504.02252): 적응한 모델 안의 상상으로 actor-critic 다시 학습(전문가 40K 스텝, 상상 길이 40).
- 우리가 아직 안 돌린 알려진 기준선: sim과 실데이터를 섞어 가치·정책까지 함께 갱신하는 것(PIA 방식). 설계 01의 FT-A는 실데이터만 썼고 무너졌다. 설계 02는 FOWM에서 벌점과 섞기만 가져왔다.

## 4. 심사자가 요구할 기준선

- 적응 없음, dynamics만 fine-tuning(SimDist), 전체 fine-tuning, sim 섞기, LoRA(TERRA), 잔차(ReDRAW), Sandwich-Residuals.
- 수준 불확실성 계열: FOWM/PIA 벌점, 앙상블 최솟값, dynamics 불일치 벌점.
- 기준으로 물러나는 계열: 짝짓지 않은 검정, 평균 차이만(SPOWL 규칙), 멤버 투표(ME-TRPO 규칙), 상태 기반 회복 전환(Yang et al. 2203.02638).

## 5. 알려진 위험

- 같은 초기값에서 fine-tuning한 앙상블은 다양성이 없다(2303.03374, 2611.23118: 1 시그마 범위가 결과의 4.6 %만 덮음). 이전 연구 설계 30과 같은 결론이다.
- 짝 차이는 계획 사이의 공통 오차만 지운다. 멤버가 공유하는 행동 의존 편향과 공유하는 terminal Q의 오차는 남는다.
- 마찰·질량 변화는 행동 변환으로 대부분 표현되지 않는다.

## 출처

sft-JEPA 2611.29453, Fork-JEPA 2611.28350, Zhang et al. 2607.22430, GALILEO 2206.04890, 2611.13592, TERRA 2610.38657, Sandwich-Residuals 2609.21740, Reflexive World Models 2505.15589, SGAT 2008.01281, ASAP 2502.01143, L1-MBRL 2403.14860, 2610.04896, AD-WM 2609.30264, pico-WAM 2611.01039, ARC-Bench 2609.05461, VaGraM 2204.01464, ASID 2404.12308, SPI-Active 2505.14266, Petrik et al. 1607.03842, ARMOR 2302.11048, ATAC 2202.02446, ME-TRPO 1802.10592, SPOWL 2506.04828, BROIL 2007.12315, 1611.06426, Yang et al. 2203.02638, ABS 2401.17583, UNISafe 2505.00779, 2303.03374, 2611.23118, MSG 2205.13703, FOWM 2310.16029, PIA 2610.26757, ReDRAW 2504.02252

## 6. 추가 확인 (2026-11-25 오후, 설계 04의 핵심 두 셀 결과 뒤): 가치 쪽 방법

설계 04에서 "정확한 dynamics여도 나쁘다"가 나온 뒤, 그 진단과 가치 쪽 방법 후보 하나를 확인했다. "본문"은 논문 본문에서 읽은 것, "초록"은 초록만 읽은 것이다.

### 6.1 진단: 정확한 dynamics + 낡은 terminal 가치가 적응 없음보다 나쁘다

| 논문 | 한 것 | 우리와의 차이 |
|---|---|---|
| SGFT (2502.02705, 본문) | 실제 H스텝 + sim 가치를 terminal로 쓴다(TD-MPC2 판 포함). 조건으로 "V_sim is improvable with respect to M_real"을 두고 성립한다고 가정 | 가정이 깨지는 경우를 재지 않았다. 설계 04는 그 반례다 |
| SimDist (2603.15759, 본문) | "freezing reward and value models can cap performance … may require selectively updating value functions" | 한계로만 언급. 해가 된다는 측정 없음 |
| Hamrick et al. (2011.04021, 본문) | MuZero에서 학습 모델을 simulator로 교체. simulator가 조금 낫다 | 물리 변화 없음. simulator 쪽이 나빠지지 않았다 |
| Bhardwaj et al. (2012.05909) | H스텝 MPC의 오차 한계: 모델 오차 항 + 가치 오차 항 | 식뿐이다 |
| HuRL (2106.02757, 본문) | improvable heuristic 개념, "Pessimistic heuristics are good heuristics" | 이론, model-free |
| PIA (2610.26757, 본문) | "value function estimates may reflect simulator-prior bias" | 한 문장 |

- 판정: 이웃 설정에 발표. "바뀐 조건의 정확한 dynamics + sim 가치가 완전히 낡은 planner보다 나쁘다"는 측정은 찾지 못했다.
- 예상 반박: "SGFT의 improvable 조건 위반일 뿐이다." 우리 쪽 답: 그 조건이 실제로 깨지고, 깨지면 적응 없음보다 나빠진다는 것을 처음 잰다.

### 6.2 방법 후보: dynamics fine-tuning + 데이터를 모은 planner의 실제 return으로 맞춘 terminal 가치

| 논문 | 한 것 | 우리와의 차이 |
|---|---|---|
| LMPC (1609.01387, 1901.08184, 초록) | 이전 주행의 실현 비용을 terminal 비용으로, 지나간 상태를 안전 집합으로 쓴다. 모델 식별 병행. 주행마다 나빠지지 않는다는 보장 | 기제가 같다. 반복 과제, sim 사전학습 가치 없음, world model 아님 |
| SAVED (1905.13402, 본문) | 학습 dynamics + 현재 정책의 terminal 비용 + 안전 집합 밀도 제약 + CEM | 시연에서 시작, 이후 TD. sim-to-real 아님 |
| 2610.38731 (본문) | MPPI가 쓰는 critic을 planner 자신의 rollout의 Monte Carlo return으로 fine-tuning | dynamics 적응·sim anchor·비관 없음 |
| MBOP (2008.05556) | offline: 학습 모델 + 행동 복제 prior + 데이터 return으로 맞춘 가치 + MPPI | 구조가 같다. sim 사전학습과 물리 변화 없음 |
| one-step RL (2106.08909, 초록) | 행동 정책의 Q + 한 번 개선 | model-free |
| SimDist ablation (본문) | "Unfreezing the value function … causes catastrophic forgetting" | TD로 푼 것. 실제 return 평가는 안 해 봄 |

- 판정: 이웃 설정에 발표. 기제(실현 return을 terminal로 + 데이터 밖 금지)는 LMPC가 원형이다. 찾지 못한 것은 sim 사전학습 world model 적응에서 낡은 sim 가치를 데이터를 모은 planner의 실제 가치로 바꾸고 sim 상태에 anchor를 두는 조합이다.
- 확인 못 한 것: ROSMO(2210.05980), MuZero Unplugged, LS3(2107.04775), Bertsekas(2108.10315, 모델이 바뀌어도 offline terminal 비용을 유지한 재계획이 견고하다는 반대 방향 주장).
- 심사자가 요구할 기준선: sim + 실데이터 replay의 TD 가치 fine-tuning(PIA/FOWM), 적응한 모델 안 상상으로 다시 학습(ReDRAW), 분해(dynamics만, 가치만, anchor·비관 없는 판).
