# 연구 작업 규칙 (TD-MPC2 기반, 2026-11-24 새 출발)

이 저장소는 공식 TD-MPC2(upstream `nicklashansen/tdmpc2`, 332f19e) 위에서 하는 연구용이다. 연구 문서, 스크립트, 결과는 모두 `research/` 아래에 둔다. 공식 코드(`tdmpc2/`)를 고칠 때는 기본값에서 동작이 바뀌지 않게 옵션으로만 넣는다.

## 문제

적은 실제 데이터로 world model을 적응하면 평균 예측 오차는 준다. 그런데 planner가 남은 오차를 파고들고, 그 오차가 horizon을 따라 쌓인다. 그래서 적응이 이득 대신 실패(넘어짐)를 만든다. 이를 막는 적응 방법을 만든다(이득 + 안전 + 보존).

- 순서: DMControl에서 방법을 확인한 뒤, 잘되면 Go1(실기)로 옮긴다. 2026-11-27 사용자 결정으로 Go1 시뮬레이션은 지금부터 병행한다(코드는 `~/simdist`, 실행은 gpu4의 `/data1/cjkim/simdist_d14`, 설계 문서와 기록은 이 저장소).
- 목표 학회: RSS 2027. extended abstract 마감 2026-12-04, 최종본 2027-04-16.
- 이전 연구 저장소: `~/simdist` (fork `ChangjuKim-unist/simdist`, branch go1-support). 결과는 필요할 때만 꺼내 쓴다. 요약은 `research/docs/background_ko.md`.

## 실험 규칙

- **사전 등록:** GPU 실험 전에 설계 문서(`research/docs/designNN_*_ko.md`)를 쓰고 commit + push한다. 등록 문서에는 질문, 고정 설정, 팔, 측정, 판정 기준, 예측, 다음 단계 규칙이 들어간다. 결과를 본 뒤 기준을 바꾸지 않고, 결과마다 방향을 틀지 않는다. 등록하지 않은 추가 분석에는 "등록 안 함"이라고 표시한다.
- **연구 기록:** 설계마다 `research/docs/research_log_ko.md`에 한 줄을 쓴다.
- **사용자 결정:** 사용자가 고정한 설정은 몰래 바꾸지 않는다. 충돌하면 숫자로 보이고 묻는다. 실패 뒤에는 분석 논문으로 돌리자고 하지 말고 다음 방법 아이디어를 낸다. 제안은 하나로 모은다.
- **지표:** 점수 하나만 보지 않는다. 넘어짐, 부드러움, 노력, 흔들림 같은 동작 품질 지표도 같이 본다.
- **삭제:** 명시적 승인 없이는 하지 않는다. gpu1의 docker 컨테이너를 kill하는 것도 같다.
- **git:** push는 `mine`(git@github.com:ChangjuKim-unist/tdmpc2.git)으로만 한다. `upstream`으로는 절대 push하지 않는다. commit 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`를 붙인다. 큰 파일(npz, pt, 체크포인트)은 커밋하지 않는다.
- **보고 방식:** 짧은 한국어로 쓴다. 숫자마다 무엇을 재는지 말한다. 표 하나에는 비교 하나만 넣는다. 실험 코드 이름만 덜렁 쓰지 않는다. 영어 기술 용어(planner, fine-tuning, world model)는 그대로 둔다.

## gpu1

- **공유 서버:** GPU당 11 GB 이하로 쓴다. 쓰기 전에 `nvidia-smi`를 확인한다.
- **접속:** `ssh gpu1 '…' 2>&1 | grep -v Ubuntu`. 백그라운드 실행은 `(nohup … < /dev/null &)`.
- **저장소 위치:** gpu1 `~/tdmpc2`, 컨테이너 안 `/workspace/simdist`.
- **실행:** `cd ~/tdmpc2 && GPU=N research/scripts/td.sh tdmpc2/evaluate.py task=… model_size=5 checkpoint=… save_video=false`
  - 이미지: `pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel` (Python 3.11, torch 2.7.1).
  - 라이브러리: `/mnt/newssd/cjkim/td_pylibs`(고정 버전), 빠진 순수 파이썬 의존성은 `td_pylibs_deps`에서 채운다.
  - 렌더링: 꺼져 있다(`MUJOCO_GL=disable`). 영상이 필요하면 EGL을 따로 준비해야 한다.
- **공식 단일 과제 체크포인트:** `/mnt/newssd/cjkim/tdmpc2_ckpt/{task}-{1,2,3}.pt`, model_size 5. 과제는 walker-run, quadruped-run, dog-run, cheetah-run, humanoid-walk, hopper-hop.
- **큰 결과 파일:** `/mnt/newssd/cjkim/td/` 아래에 둔다.
