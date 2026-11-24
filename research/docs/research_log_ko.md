# 연구 기록 (TD-MPC2)

설계마다 한 줄. 등록 → 결과 순서로 채운다.

| 번호 | 날짜 | 문서 | 내용 | 결과 | 결과 위치 |
|---|---|---|---|---|---|
| 0 | 2026-11-24 | (준비, 등록 없음) | gpu1에 공식 TD-MPC2 설치, 공개 단일 과제 체크포인트(seed 1) 재현 확인. 10 에피소드 평가 | walker-run 870.9 (논문 868.1), quadruped-run 955.2 (951.5), cheetah-run 843.4 (848.3) → 1 % 안에서 재현. dog-run 756.3 (859.1)은 낮아 기준 과제에서 뺌. 평가 1회 약 2분 | research/results/repro/ |
| 1 | 2026-11-24 | research/docs/design01_adapt_failure_recurrence_ko.md | (등록) K2: TD-MPC2(walker·quadruped·cheetah-run)를 물리 변화 조건에서 실데이터 10 에피소드로 fine-tuning하면 잃은 return을 못 되찾거나 넘어짐이 느는가, 그때 수준 오차만 주고 행동 효과 오차는 덜 주는가. K1: 다른 보폭의 비슷한 상태끼리 짝지은 ±δ 흔들기로 실측 fork 목표의 신호 대 잡음비가 1 이상인가 | K2a 있다(4/6 셀: ρ 0.18, −0.66, 0.02, −1.16), K2b 그렇다(4/6, r_L 0.46–0.84, r_C 0.16–0.44), K1 아니다(0/6, R_c 최고 1.10, 흔들기 비용 큼) → 보폭 반복 fork 접음. 등록 안 함: C는 실패·성공 셀을 가르지 못하고, 평소 조건 return이 fine-tuning 단계 수에 따라 6/6 셀에서 단조 하락 | /mnt/newssd/cjkim/td/d01/, research/results/d01/ |
| 2 | 2026-11-24 | research/docs/design02_selfstate_damage_remedies_ko.md | (등록, 사용자 결정으로 설계 01 §7 대신) 같은 6셀에서 새 평가 시드로 FT-D2000 실패 재현(R). A1: FT-D2000 planner가 실제로 가는 상태에서 FT 모델 오차가 사전학습보다 큰가. A2: FT planner가 고른 계획의 예측 가치가 실제보다 더 부풀려지는가. B: 알려진 처방(적은 단계, 조기 종료, sim 섞기, 가중치 anchor, FOWM 불확실성 벌점) 중 하나가 실패를 막는가, 이득까지 내는가 | (실행 전) | /mnt/newssd/cjkim/td/d02/, research/results/d02/ |
