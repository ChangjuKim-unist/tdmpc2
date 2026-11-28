"""Design 17 (research/docs/design17_veto_on_dip_checkpoints_ko.md): consensus on the saved checkpoints of full fine-tuning.

Evaluation only, no training. The learners are design 13's RAW checkpoints (quadruped motor, seeds 1-3, steps 10k-60k), whose own
evaluation returns are already known. Two questions:
  A  does the consensus of the frozen pretrained agent and a collapsed learner come back to the frozen agent's level?
  B  can a signal measured on the consensus agent's own episodes (the learner never acts alone) tell collapsed from recovered learners?

Arms per checkpoint (no exploration noise, seed 3000 + s, two episodes per seed; the same initial states as design 12):
  SOLO   the learner alone, official planner
  CON    design 12's consensus (official MPPI loop, every candidate scored by both agents, min of the two relative scores)
The frozen agent alone (PRE) is design 12's deploy_PRE arm (same code path, same seeds).

Signals (registered direction: larger = learner less ready), from each arm's recorded episodes:
  S1  share of steps where the frozen agent's relative score is the lower one at the chosen candidate (CON only)
  S2  learner / frozen ratio of the mean |episode-mean 100-step value residual| (design 11's q_stats) on the same episodes
  S3  learner / frozen ratio of the mean Q-ensemble spread at the visited state and executed action on the same episodes

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d17.py run --seed 1 --step 10000 --arm CON [--seed_from 1 --seed_to 16 --eps 2]
  GPU=N research/scripts/td.sh research/scripts/d17.py summarize --res_dir /mnt/newssd/cjkim/td/d17/res
Smoke tests must pass --out_root (another folder). The official TD-MPC2 code is imported, not modified.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d07 import fresh, stack
from d11 import q_stats
from d12 import Dual, interval, fmt

CELL = 'quadruped-run__motor_0.35'
D12, D13 = '/mnt/newssd/cjkim/td/d12', '/mnt/newssd/cjkim/td/d13'
OUT = '/mnt/newssd/cjkim/td/d17'
SEEDS, STEPS = (1, 2, 3), (10000, 20000, 30000, 40000, 50000, 60000)
DYN_REF, MARGIN, DEPTH, AUC_PASS = 571.0, -25.0, 0.25, 0.9


def ckpt(seed, step):
	return f'{D13}/{CELL}/RAW_s{seed}/step{step}.pt'


def labels():
	"""Registered labels from design 13's own evaluations (10 episodes): collapsed if below (1 - 0.25) x the run's pre-training evaluation,
	recovered if at or above the dynamics-only reference 571, otherwise unlabelled."""
	out = {}
	for s in SEEDS:
		ev = {e['step']: e['ret'] for e in json.load(open(f'{D13}/{CELL}/RAW_s{s}/log.json'))['evals']}
		for k in STEPS:
			r = ev[k]
			out[s, k] = dict(solo_d13=r, pre_d13=ev[0], label='collapsed' if r < (1 - DEPTH) * ev[0] else ('recovered' if r >= DYN_REF else 'between'))
	return out


def signals(learner, frozen, recs):
	d = stack(recs)
	ql, qf = q_stats(learner, d), q_stats(frozen, d)
	rl, rf = float(ql['resid'].abs().mean()), float(qf['resid'].abs().mean())
	ul, uf = float(ql['u_vis'].mean()), float(qf['u_vis'].mean())
	return dict(resid_learner=rl, resid_frozen=rf, S2=rl / max(rf, 1e-6), u_learner=ul, u_frozen=uf, S3=ul / max(uf, 1e-6))


def cmd_run(a):
	task, cond = cell_parts(CELL)
	folder = f'{a.out_root}/{CELL}/RAW_s{a.seed}_step{a.step}/{a.arm}'
	os.makedirs(folder, exist_ok=True)
	cfg, env, frozen = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	learner = fresh(cfg)
	learner.load(ckpt(a.seed, a.step))
	actor = learner if a.arm == 'SOLO' else Dual(frozen, learner, 'CON')
	for s in range(a.seed_from, a.seed_to + 1):
		if os.path.exists(f'{folder}/s{s:02d}.json'):
			continue
		t0 = time.time()
		n0 = len(actor.binds) if isinstance(actor, Dual) else 0
		eps, recs = run_cond(actor, env, shifter, domain, cond, a.eps, eval_mode=True, seed=3000 + s, record=True)
		with torch.no_grad():
			sig = signals(learner, frozen, recs)
		if isinstance(actor, Dual):
			sig['S1'] = float(np.mean(actor.binds[n0:]))
		out = dict(cell=CELL, run_seed=a.seed, step=a.step, arm=a.arm, seed=s, args=vars(a), summary=summary(eps), episodes=eps, signals=sig, seconds=time.time() - t0)
		dump(f'{folder}/s{s:02d}.json', out)
		print(a.seed, a.step, a.arm, s, {k: round(v, 2) for k, v in summary(eps).items()}, {k: round(v, 3) for k, v in sig.items()}, f'{time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 17 §4)

def auroc(pos, neg):
	"""P(signal of a collapsed checkpoint > signal of a recovered one), ties count one half."""
	pos, neg = np.asarray(pos, float), np.asarray(neg, float)
	if not len(pos) or not len(neg):
		return float('nan')
	return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def cmd_summarize(a):
	L, S = ['# 설계 17 결과 요약 (자동 생성, `d17.py summarize`)\n'], dict(checkpoints={})
	L.append('return은 seed 16개 × 에피소드 2개(seed마다 평균), 구간은 seed 단위 t 구간 95 %다. 적응 없음(얼린 agent 단독)은 설계 12의 같은 seed 값이다. '
			 f'꼬리표는 설계 13의 평가(10 에피소드)로 미리 정했다: 학습 전 평가의 {1 - DEPTH:.2f}배 미만이면 "꺼짐", {DYN_REF:.0f} 이상이면 "회복".\n')
	pre = {int(os.path.basename(f)[1:3]): float(np.mean([e['ret'] for e in json.load(open(f))['episodes']]))
		   for f in glob.glob(f'{D12}/{CELL}/deploy_PRE/s*.json')}
	pre_m = {k: float(np.mean([np.mean([e[k] for e in json.load(open(f))['episodes']]) for f in glob.glob(f'{D12}/{CELL}/deploy_PRE/s*.json')]))
			 for k in ('ret', 'fallen_frac', 'effort')}
	S['pre'] = pre_m
	L.append(f'- 적응 없음: return {pre_m["ret"]:.0f}, 넘어져 있는 시간 {pre_m["fallen_frac"]:.2f}, 노력 {pre_m["effort"]:.2f} (seed {len(pre)}개)\n')
	lab = labels()
	L.append('| 체크포인트 | 꼬리표 (설계 13 단독) | 단독 return | 합의 return | 합의 − 적응 없음 | 해 없음 (얼린 agent 대비) | 합의 − 단독 | 넘어져 있는 시간 단독 / 합의 | 노력 단독 / 합의 | S1 | S2 | S3 |')
	L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
	verdA = []
	for s in SEEDS:
		for k in STEPS:
			R = {}
			for arm in ('SOLO', 'CON'):
				for f in glob.glob(f'{a.out_root}/{CELL}/RAW_s{s}_step{k}/{arm}/s*.json'):
					R.setdefault(arm, {})[int(os.path.basename(f)[1:3])] = json.load(open(f))
			if 'CON' not in R:
				continue
			mv = lambda arm, key='ret': {q: float(np.mean([e[key] for e in r['episodes']])) for q, r in R.get(arm, {}).items()}
			con, solo = mv('CON'), mv('SOLO')
			seeds = sorted(set(con) & set(pre))
			g = interval([con[q] - pre[q] for q in seeds])
			hm = '통과' if g['t'][0] > MARGIN else ('실패' if g['t'][1] < MARGIN else '미결')
			cs = sorted(set(con) & set(solo))
			c = interval([con[q] - solo[q] for q in cs]) if cs else None
			sig = {n: float(np.mean([r['signals'][n] for r in R['CON'].values()])) for n in ('S1', 'S2', 'S3')}
			sig_solo = {n: float(np.mean([r['signals'][n] for r in R['SOLO'].values()])) for n in ('S2', 'S3')} if 'SOLO' in R else {}
			m = lambda arm, key: float(np.mean(list(mv(arm, key).values()))) if arm in R else float('nan')
			lb = lab[s, k]['label']
			S['checkpoints'][f's{s}_{k}'] = dict(**lab[s, k], n=len(seeds), solo=m('SOLO', 'ret'), con=m('CON', 'ret'), con_vs_pre=g, harm=hm, con_vs_solo=c,
												 fallen=[m('SOLO', 'fallen_frac'), m('CON', 'fallen_frac')], effort=[m('SOLO', 'effort'), m('CON', 'effort')],
												 signals=sig, signals_solo=sig_solo)
			if lb == 'collapsed':
				verdA.append(hm)
			name = {'collapsed': '꺼짐', 'recovered': '회복', 'between': '사이'}[lb]
			L.append(f'| seed {s}, {k // 1000}k | {name} ({lab[s, k]["solo_d13"]:.0f}) | {m("SOLO", "ret"):.0f} | {m("CON", "ret"):.0f} | {fmt(g)} | {hm} | {"–" if c is None else fmt(c)} | '
					 f'{m("SOLO", "fallen_frac"):.2f} / {m("CON", "fallen_frac"):.2f} | {m("SOLO", "effort"):.2f} / {m("CON", "effort"):.2f} | {sig["S1"]:.3f} | {sig["S2"]:.3f} | {sig["S3"]:.3f} |')
	nA = sum(1 for v in lab.values() if v['label'] == 'collapsed')
	A = '통과' if len(verdA) == nA and all(v == '통과' for v in verdA) else ('실패' if any(v == '실패' for v in verdA) else '미결')
	S['A'] = dict(verdict=A, per_checkpoint=verdA, expected=nA)
	L.append(f'\n## A. 꺼진 체크포인트에 합의를 씌우면 얼린 agent 수준으로 돌아오는가\n\n- 꺼진 체크포인트 {nA}개 중 잰 것 {len(verdA)}개: ' + ', '.join(verdA) + f' → **{A}**')
	rec = [v for v in S['checkpoints'].values() if v['label'] == 'recovered']
	if rec:
		kept = [(v['con'] - pre_m['ret']) / (v['solo'] - pre_m['ret']) for v in rec if v['solo'] - pre_m['ret'] > 25]
		L.append(f'- 회복한 체크포인트 {len(rec)}개(보고): 단독 평균 {np.mean([v["solo"] for v in rec]):.0f}, 합의 평균 {np.mean([v["con"] for v in rec]):.0f}, '
				 f'합의가 지킨 이득의 몫 (합의 − 적응 없음) / (단독 − 적응 없음) 평균 {np.mean(kept) if kept else float("nan"):.2f}')
	L.append('\n## B. 합의 에피소드에서 잰 신호가 꺼진 학습자와 회복한 학습자를 가르는가\n')
	L.append(f'AUROC = 꺼진 체크포인트의 신호가 회복한 체크포인트의 신호보다 클 확률(같으면 절반). {AUC_PASS} 이상이면 "가른다".\n')
	L.append('| 신호 | 꺼짐 평균 | 회복 평균 | AUROC | 판정 | 단독 에피소드에서 잰 AUROC (보고) |')
	L.append('|---|---|---|---|---|---|')
	S['B'] = {}
	for n, desc in (('S1', 'S1 얼린 쪽이 막은 비율'), ('S2', 'S2 가치 잔차 비 (학습자 / 얼린)'), ('S3', 'S3 Q 앙상블 폭 비 (학습자 / 얼린)')):
		pos = [v['signals'][n] for v in S['checkpoints'].values() if v['label'] == 'collapsed']
		neg = [v['signals'][n] for v in S['checkpoints'].values() if v['label'] == 'recovered']
		au = auroc(pos, neg)
		vd = '가른다' if au >= AUC_PASS else '못 가른다'
		au_solo = auroc([v['signals_solo'].get(n, np.nan) for v in S['checkpoints'].values() if v['label'] == 'collapsed'],
						[v['signals_solo'].get(n, np.nan) for v in S['checkpoints'].values() if v['label'] == 'recovered']) if n != 'S1' else float('nan')
		S['B'][n] = dict(collapsed=float(np.mean(pos)) if pos else None, recovered=float(np.mean(neg)) if neg else None, auroc=au, verdict=vd, auroc_solo=au_solo)
		L.append(f'| {desc} | {np.mean(pos) if pos else float("nan"):.3f} | {np.mean(neg) if neg else float("nan"):.3f} | {au:.2f} | {vd} | {"–" if n == "S1" else f"{au_solo:.2f}"} |')
	S['B_any'] = any(v['verdict'] == '가른다' for v in S['B'].values())
	L.append(f'\n- **가르는 신호가 하나라도 있나: {"예" if S["B_any"] else "아니오"}**')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--seed', type=int, choices=SEEDS, required=True)
	p.add_argument('--step', type=int, choices=STEPS, required=True)
	p.add_argument('--arm', choices=['SOLO', 'CON'], required=True)
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=16)
	p.add_argument('--eps', type=int, default=2)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)
