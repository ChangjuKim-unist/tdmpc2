# 선행 연구 확인: "적응한 모델 안에서 가치를 다시 맞추기"는 우리 것인가 (2026-11-24)

설계 02 뒤의 후보 방법을 등록하기 전에 확인했다. GPU 실험 없음. 검색: arXiv(키워드 12회, SimDist 인용 그래프), 웹 4회, SimDist 원문, 이전 조사 보고서(simdist `reports/값 함수 선택적 적응 novelty 검토.md`, 2026-10-12).

## 결론

- **후보 방법은 우리 것이 아니다.** "sim에서 배운 latent world model의 dynamics만 적은 실데이터로 고치고, 그 모델 안의 상상 rollout으로 가치와 정책을 다시 배운다"는 ReDRAW(2504.02252)가 이미 했다.
- **SimDist 저자들은 가치 갱신을 future work로 적었지만, 아직 발표한 것은 없다**(2026-11-24 기준).
- **우리 진단("모델이 정확해져도 planner가 나빠지고, 원인은 가치 쪽")도 dynamics 변화가 없는 경우에는 이미 발표됐다**(2601.21306).

## 1. SimDist 저자들이 한 것과 안 한 것

| 항목 | 원문 근거 |
|---|---|
| 가치·reward·encoder를 얼리고 dynamics만 fine-tuning | 2603.15759 §IV-B: "finetunes only the environment-specific dynamics model while freezing the encoder, reward, and value functions" |
| 가치를 풀면 무너진다는 ablation | §V-E: "Unfreezing the value function ... reintroduces long-horizon credit assignment from limited real-world data and causes catastrophic forgetting" |
| future work | 결론: "freezing reward and value models can cap performance when the transferred value function saturates ... may require selectively updating value functions in addition to dynamics" |
| 인용 논문 16편 | 가치를 갱신한 논문 없음. 저자 후속 TERRA(2610.38657, reward 가중 예측 손실 + 탐색), CLAW(2609.12278, hypernetwork LoRA)는 둘 다 가치를 건드리지 않는다 |
| 같은 저자들의 SGFT(2502.02705) | sim 가치를 일부러 얼려 두고 탐색 안내로만 쓴다. "sim 가치는 고치지 않는다"가 이 그룹의 일관된 입장이다 |

- 선점 위험: 이 그룹이 가장 먼저 할 가능성이 높다. arXiv에 없다는 것이 진행 중이 아니라는 뜻은 아니다.

## 2. 후보 방법과 가장 가까운 선행 연구

| 선행 연구 | 한 것 | 후보와의 차이 |
|---|---|---|
| ReDRAW (2504.02252) | sim에서 Dreamer식 latent world model 사전학습 → 적은 실데이터로 latent dynamics 잔차만 학습(실제 reward 없음) → 고친 모델 안의 상상 rollout으로 actor-critic 학습 → 배포. DMC 4개 + 실기 | 배포가 정책이다(planner 아님). 그 밖은 같다 |
| GAT(2017), neural-augmented sim(2018), ASAP(2502.01143) | 실데이터로 시뮬레이터를 고친 뒤 그 안에서 정책을 다시 학습 | latent world model이 아니라 시뮬레이터 |
| FOWM (2310.16029) | TD-MPC를 적은 실데이터로 fine-tuning. Q를 실데이터 TD로 갱신 + Q 앙상블 벌점 + 균형 샘플링 | 가치 학습 신호가 실데이터(상상 아님). 설계 02에서 벌점·섞기를 시험함 |
| PIA (2610.26757, TD-MPC2 저자 포함) | sim 궤적으로 world model 사전학습 → 적은 실제 상호작용으로 적응, 불확실성 벌점 planner. 실차 | 같은 문제 설정(TD-MPC2 + sim prior + 적은 데이터). 가치를 상상으로 맞추는지는 초록에 없음(본문 미확인) |
| MOBODY (2506.08460) | dynamics가 다른 source 데이터 + 적은 target 데이터. target dynamics를 배워 그 rollout으로 정책 학습, target Q 가중 | offline, model-free 평가 |
| QWM (2608.17163) | 반대 입장: 상상 rollout으로 가치를 배우면 누적 편향이 커지므로, 가치는 실데이터로만 배우고 world model은 탐색에만 쓴다 | 후보 방법이 실패할 이유를 말하는 쪽 |

## 3. 우리 진단과 겹치는 연구

- **"The Surprising Difficulty of Search in MBRL" (2601.21306):** 모델이 매우 정확해도 탐색(planner)이 성능을 해칠 수 있고, 모델·가치 정확도보다 가치 과대평가 편향이 더 중요하다고 보고했다. 처방은 가치 앙상블의 최솟값이다. dynamics 변화가 없는 설정이다.
  - 설계 02의 관찰(자기 상태에서 모델이 더 정확한데 성능이 나쁨, 계획 가치의 대부분이 terminal Q)과 같은 방향이다.
  - 설계 02의 FOWM 벌점(평균 − λ·표준편차)은 이 처방과 비슷한 종류다. 6셀 중 5셀에서 보통 fine-tuning보다 나았지만 3셀은 못 풀었다. 최솟값 자체는 아직 안 돌렸다.
- **PL-MPC (2609.39751):** TD-MPC의 terminal 가치를 불일치 인식 추정으로 바꾼다. dynamics 변화 없음.

## 4. 이전 연구(SimDist 저장소)의 같은 방향 시도

- 설계 1(가치 적응): 순위 상관은 0.20 → 0.46–0.50으로 고쳐졌지만 넘어짐은 −13 %뿐.
- 설계 22: 가치 head를 다시 맞추면 planner가 그것을 악용해 넘어짐 증가(2.50 → 2.69–5.19).
- 설계 37: 끝 latent 교정 head, 적응 조건 0/3.
- 이 저장소 설계 01 FT-A(실데이터로 Q까지 학습): walker 마찰 return 40.
- 넷 다 가치 학습 신호가 **실데이터**였다. 적응한 모델 안의 **상상 rollout**으로 가치를 맞추는 것(ReDRAW 방식)은 한 번도 안 돌렸다.

## 5. 남는 빈자리

- 발표 안 된 것: planner(MPPI + terminal 가치)를 쓰는 world model에서, dynamics 변화 뒤 가치를 고치는 방법과 그 실패 방식. SimDist는 "가치를 풀면 무너진다"와 "가치를 얼리면 포화한다" 사이를 열어 두었다.
- 다만 "상상으로 다시 맞춘다"를 그대로 가져오면 ReDRAW를 planner에 붙인 것이라 새로움이 약하다. 방법의 새로움은 ReDRAW 방식이 이 설정에서 **왜 안 되는지**가 먼저 드러나야 그 실패를 겨냥해 만들 수 있다. 될 경우에는 그것이 기준선이 되고 우리 자리는 더 좁아진다.

## 출처

- SimDist: https://arxiv.org/abs/2603.15759 , https://roboticsconference.org/program/papers/17/
- ReDRAW: https://arxiv.org/abs/2504.02252
- FOWM: https://arxiv.org/abs/2310.16029
- PIA: https://arxiv.org/abs/2610.26757
- Search difficulty: https://arxiv.org/abs/2601.21306
- PL-MPC: https://arxiv.org/abs/2609.39751
- QWM: https://arxiv.org/abs/2608.17163
- MOBODY: https://arxiv.org/abs/2506.08460
- SGFT: https://arxiv.org/abs/2502.02705
- TERRA: https://arxiv.org/abs/2610.38657 , CLAW: https://arxiv.org/abs/2609.12278
- JEPA-TTT: https://arxiv.org/abs/2610.00722 , AdaJEPA: https://arxiv.org/abs/2606.32026
