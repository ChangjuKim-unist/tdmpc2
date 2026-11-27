"""Design 11 (research/docs/design11_min_rule_and_q_disagreement_ko.md): is the min rule the working part of the consensus planner,
and does the stale Q ensemble disagree at the states the true dynamics reaches.

  run      exact-dynamics level (part A): MEANORA = design 06's CON-ORA with the min of the two relative scores replaced by their average.
           Same cell / checkpoint / protocol / seeds as design 09's A1, so the episodes pair with PRE, ORA and CON-ORA.
  learned  learned-model level (part C): the average rule with the adapted model M2 that design 10 used (arm MEAN next to design 10's arms).
  qdis     forward only on stored trajectories (part B): std of the Q ensemble and the value at the true 3-step end state
           versus the sim model's 3-step end state, std at the visited states, and a 100-step value residual.
  summarize

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d11.py run --cell cheetah-run__friction_0.15 [--seed_from 1 --seed_to 16 --eps 2 --workers 10]
  GPU=N research/scripts/td.sh research/scripts/d11.py learned --cell walker-run__friction_0.15 --seed 1
  GPU=N research/scripts/td.sh research/scripts/d11.py qdis
  GPU=N research/scripts/td.sh research/scripts/d11.py summarize --res_dir /mnt/newssd/cjkim/td/d11/res
Smoke tests must pass --out_root (another folder); `run --timing 20` only times 20 planning calls and prints the code check.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from common import math as tdmath

import d06
from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import CELLS

D02, D04, D06, D07, D09, D10 = (f'/mnt/newssd/cjkim/td/d{n}' for n in ('02', '04', '06', '07', '09', '10'))
OUT = '/mnt/newssd/cjkim/td/d11'
CF = 'cheetah-run__friction_0.15'
EVAL_EPS, NOM_EPS = 10, 3
MARGIN, V0_RATIO, EQ = -25.0, 0.9, 25.0
R_LO, R_HI = 1.2, 1.5
SHORT = {'walker-run__mass_1.5': 'walker 질량', 'walker-run__friction_0.15': 'walker 마찰', 'quadruped-run__friction_0.15': 'quadruped 마찰',
		 'cheetah-run__mass_2.0': 'cheetah 질량', 'quadruped-run__motor_0.35': 'quadruped 모터', CF: 'cheetah 마찰'}
TQ = {15: 2.1314, 31: 2.0395, 19: 2.0930, 79: 1.9905, 4: 2.7764, 1: 12.7062}   # t quantile 0.975 (two-sided 95 %)


def mean_rule(js, jx):
	"""Drop-in for d06.consensus: the average of the two relative scores instead of their minimum."""
	a_s = js - js.mean(0, keepdim=True)
	a_x = jx - jx.mean(0, keepdim=True)
	return 0.5 * (a_s + a_x), a_s, a_x


# ----------------------------------------------------------------------------- part A: exact dynamics

def cmd_run(a):
	from d04 import _winit
	from d09 import make_h, run_dir
	task, cond = cell_parts(a.cell)
	folder = run_dir(a.out_root, a.cell, 1, 3, 'deploy', 'MEANORA')
	os.makedirs(folder, exist_ok=True)
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, cond))  # before any CUDA use
	cfg, env, agent = make_h(task, 1, 3)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	min_rule = d06.consensus
	d06.consensus = mean_rule                                                # OracleCon.act looks the rule up at call time
	actor = d06.OracleCon(agent, env, pool, a.workers)
	if a.timing:
		shifter.set(cond)
		obs = env.reset()
		tic = time.time()
		for i in range(a.timing):
			actor.act(obs, t0=i == 0, eval_mode=True)
		print(a.cell, 'MEANORA', f'{(time.time() - tic) / a.timing:.3f} s/plan', flush=True)
		state = phys_of(env).get_state().copy()                              # code check on random candidates
		acts = torch.rand(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda') * 2 - 1
		z = agent.model.encode(obs.to('cuda').unsqueeze(0), None).repeat(cfg.num_samples, 1)
		torch.manual_seed(0)
		jx = actor.values(state, acts)
		torch.manual_seed(0)
		js = d06.plan_value(agent, z, acts, actor.dyn0)
		v_mean, v_min = mean_rule(js, jx)[0], min_rule(js, jx)[0]
		ref = 0.5 * (js + jx)
		print(f'check: mean rule − (0.5 (js + jx) − its mean) max abs {float((v_mean - (ref - ref.mean())).abs().max()):.6f} | '
			  f'mean rule ≥ min rule everywhere: {bool((v_mean >= v_min - 1e-6).all())} | same top plan as min rule: {int(v_mean.argmax()) == int(v_min.argmax())} | '
			  f'sd js {float(js.std()):.3f} jx {float(jx.std()):.3f}', flush=True)
		return
	for s in range(a.seed_from, a.seed_to + 1):
		if os.path.exists(f'{folder}/s{s:02d}.json'):
			continue
		t0, n0 = time.time(), len(actor.sim_binds)
		eps, recs = run_cond(actor, env, shifter, domain, cond, a.eps, eval_mode=True, seed=3000 + s, record=True)
		np.savez_compressed(f'{folder}/rec_s{s:02d}.npz', state0=np.stack([np.asarray(r['state'][0]) for r in recs]),
							rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]),
							obs=np.stack([np.asarray(r['obs'], np.float32) for r in recs]), act=np.stack([np.asarray(r['act'], np.float32) for r in recs]))
		dump(f'{folder}/s{s:02d}.json', dict(cell=a.cell, seed=s, args=vars(a), summary=summary(eps), episodes=eps,
											 sim_binds=float(np.mean(actor.sim_binds[n0:])), seconds=time.time() - t0))
		print(a.cell, 'MEANORA', s, {k: round(v, 2) for k, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- part C: learned model (design 10's M2)

def cmd_learned(a):
	from d07 import fresh
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder = f'{a.out_root}/learned/{a.cell}/s{s:02d}'
	f = f'{folder}/eval_MEAN.json'
	if os.path.exists(f):
		return
	src = f'{a.d10_root}/{a.cell}/s{s:02d}/M2_dyn.pt'
	if not os.path.exists(src):
		src = f'{a.d07_root}/{a.cell}/s{s:02d}/M2_CTRL_dyn.pt'               # the two design-07 cells (design 10 reused these)
	cfg, env, agent0 = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	dyn0 = copy.deepcopy(agent0.model._dynamics).eval()
	for p in dyn0.parameters():
		p.requires_grad_(False)
	ag = fresh(cfg)
	ag.model._dynamics.load_state_dict(torch.load(src))
	d06.consensus = mean_rule                                                # ConAgent._est looks the rule up at call time
	con = d06.ConAgent(ag, dyn0)
	out = dict(M2=src)
	for c, n in (('shift', a.eval_eps), ('nominal', a.nom_eps)):
		real = cond if c == 'shift' else 'nominal'
		t0 = time.time()
		eps, recs = run_cond(con, env, shifter, domain, real, n, eval_mode=True, seed=3000 + s, record=(c == 'shift'))
		out[c] = dict(cond=real, summary=summary(eps), episodes=eps, seconds=time.time() - t0)
		if c == 'shift':
			os.makedirs(folder, exist_ok=True)
			np.savez_compressed(f'{folder}/rec_MEAN.npz', rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
			out['sim_binds'] = float(np.nanmean(con.sim_binds))
		print(a.cell, s, 'MEAN', c, {k: round(v, 2) for k, v in out[c]['summary'].items()}, f'{out[c]["seconds"]:.0f}s', flush=True)
	dump(f, out)


# ----------------------------------------------------------------------------- part B: Q ensemble at true vs model end states (forward only)

def sources(cell):
	"""Stored trajectories with observations and actions, checkpoint 1. name -> (protocol note, loader)."""
	task = cell_parts(cell)[0]

	def npz(path, pre=''):
		def f():
			if not os.path.exists(path):
				return None
			d = np.load(path)
			return {k: d[pre + k] for k in ('obs', 'act', 'rew')} if pre + 'obs' in d.files else None
		return f

	def d07_pre():
		ds = [np.load(p) for p in (f'{D07}/{cell}/s{s:02d}/data1.npz' for s in range(1, 17)) if os.path.exists(p)]
		return {k: np.concatenate([d[k] for d in ds]) for k in ('obs', 'act', 'rew')} if ds else None

	return {
		'shift_PRE': ('적응 없음, 바뀐 조건 (설계 02 기록, 잡음 없음)', npz(f'{D02}/{cell}/rec_PRE.npz', 'shift_')),
		'shift_PREc': ('적응 없음, 바뀐 조건 (설계 07 첫 수집, 잡음 켬)', d07_pre),
		'shift_ORA': ('정확한 dynamics만, 바뀐 조건 (설계 04 기록)', npz(f'{D04}/{cell}/rec_ORA.npz')),
		'shift_CONORA': ('정확한 dynamics + sim 합의, 바뀐 조건 (설계 06 기록)', npz(f'{D06}/{cell}/rec_CONORA.npz')),
		'nom_PRE': ('적응 없음, 평소 조건 (설계 02 기록)', npz(f'{D02}/{cell}/rec_PRE.npz', 'nominal_')),
		'nom_ORA': ('정확한 dynamics만, 평소 조건 (설계 04 기록)', npz(f'{D04}/nominal/{task}/rec_ORA.npz')),
		'nom_CONORA': ('정확한 dynamics + sim 합의, 평소 조건 (설계 06 기록)', npz(f'{D06}/nominal/{task}/rec_CONORA.npz')),
	}


@torch.no_grad()
def q_stats(agent, d, K=100):
	"""Per-episode means. u = std over the Q ensemble members (two-hot inverted), q = their mean."""
	cfg, m, g = agent.cfg, agent.model, float(agent.discount)
	obs = torch.as_tensor(d['obs'], dtype=torch.float32, device='cuda')
	act = torch.as_tensor(d['act'], dtype=torch.float32, device='cuda')
	rew = torch.as_tensor(d['rew'], dtype=torch.float32, device='cuda')
	E, T = act.shape[:2]
	H = cfg.horizon
	n = T - H + 1

	def qu(z, a):
		q = tdmath.two_hot_inv(m.Q(z, a, None, return_type='all'), cfg).squeeze(-1)   # members x N
		return q.mean(0), q.std(0)

	torch.manual_seed(0)
	Z = m.encode(obs.reshape(E * (T + 1), -1), None)
	q_pi, u_pi = (v.reshape(E, T + 1) for v in qu(Z, m.pi(Z, None)[0]))      # at every visited state with the policy action (what the planners' terminal value uses)
	Z = Z.reshape(E, T + 1, -1)
	q_a, u_a = (v.reshape(E, T) for v in qu(Z[:, :T].reshape(E * T, -1), act.reshape(E * T, -1)))   # at the visited state with the executed action
	z = Z[:, :n].reshape(E * n, -1)
	for h in range(H):
		z = m.next(z, act[:, h:h + n].reshape(E * n, -1), None)
	q_m, u_m = (v.reshape(E, n) for v in qu(z, m.pi(z, None)[0]))            # the sim model's H-step end state of the executed actions
	q_t, u_t = q_pi[:, H:H + n], u_pi[:, H:H + n]                            # the true H-step end state of the same actions
	disc = g ** torch.arange(K, device='cuda', dtype=torch.float32)
	nk = T - K + 1
	ret = torch.stack([(rew[:, t:t + K] * disc).sum(1) for t in range(nk)], 1)
	resid = q_a[:, :nk] - (ret + g ** K * q_pi[:, K:K + nk])                 # > 0: the value promised more than the next K steps and the value after them delivered
	return dict(u_true=u_t.mean(1), u_model=u_m.mean(1), ratio=u_t.mean(1) / u_m.mean(1), dq=(q_m - q_t).mean(1), q_true=q_t.mean(1),
				u_vis=u_a.mean(1), u_vis_first100=u_a[:, :100].mean(1), resid=resid.mean(1), ret=rew.sum(1))


def cmd_qdis(a):
	for cell in (a.cells.split(',') if a.cells else CELLS):
		cfg, env, agent = make(cell_parts(cell)[0], compile_=False)
		res = {}
		for name, (note, load) in sources(cell).items():
			d = load()
			if d is None:
				continue
			st = q_stats(agent, d)
			res[name] = dict(note=note, episodes=int(len(d['obs'])), **{k: v.cpu().numpy().tolist() for k, v in st.items()})
			print(cell, name, len(d['obs']), {k: round(float(v.mean()), 3) for k, v in st.items()}, flush=True)
		dump(f'{a.out_root}/qdis/{cell}.json', res)
		del env, agent


# ----------------------------------------------------------------------------- summary (registered analysis, design 11 §5)

def interval(x):
	x = np.asarray(x, float)
	n = len(x)
	if n < 2:
		return dict(mean=float(x.mean()) if n else float('nan'), t=[float('nan')] * 2, n=n)
	if n - 1 in TQ:
		q = TQ[n - 1]
	else:
		from scipy import stats
		q = float(stats.t.ppf(0.975, n - 1))
	h = q * x.std(ddof=1) / np.sqrt(n)
	return dict(mean=float(x.mean()), t=[float(x.mean() - h), float(x.mean() + h)], n=n)


def fmt(iv, d=1, sign=True):
	p = '+' if sign else ''
	if not np.isfinite(iv['t'][0]):
		return f'{iv["mean"]:{p}.{d}f} (n {iv["n"]})'
	return f'{iv["mean"]:{p}.{d}f} [{iv["t"][0]:{p}.{d}f}, {iv["t"][1]:{p}.{d}f}]'


def min_works(iv):
	"""Verdict on (min rule − average rule)."""
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return 'min 규칙이 일한다' if lo > 0 else ('min 규칙의 이득 없음' if hi < EQ else '미결')


def harm(iv):
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return '통과' if lo > MARGIN else ('실패' if hi < MARGIN else '미결')


def cmd_summarize(a):
	import d09
	L, S = ['# 설계 11 결과 요약 (자동 생성, `d11.py summarize`)\n'], {}
	K = ('ret', 'falls', 'fallen_frac', 'smooth', 'effort', 'angvel_rms')

	# part A
	L.append('## A. 정확한 dynamics 수준: min 규칙 대 평균 규칙 (cheetah 마찰, 체크포인트 1, 호라이즌 3, 배치 규약, seed 16개 × 에피소드 2개)\n')
	seeds = list(range(1, 17))
	ns = argparse.Namespace(out_root=a.d09_root, d07_root=a.d07_root, d08_root=a.d08_root)
	arms = {'적응 없음': d09.get(ns, CF, 1, 3, 'deploy', 'PRE', seeds, 2), '정확한 dynamics만': d09.get(ns, CF, 1, 3, 'deploy', 'ORA', seeds, 2),
			'합의 (min)': d09.get(ns, CF, 1, 3, 'deploy', 'CONORA', seeds, 2),
			'평균': d09.get(argparse.Namespace(out_root=a.out_root, d07_root='', d08_root=''), CF, 1, 3, 'deploy', 'MEANORA', seeds, 2)}
	if all(v is not None for v in arms.values()):
		L += d09.HEAD
		for n, d in arms.items():
			L.append(d09.row(n, d))
		sb = [json.load(open(f'{d09.run_dir(a.out_root, CF, 1, 3, "deploy", "MEANORA")}/s{s:02d}.json'))['sim_binds'] for s in seeds]
		L.append(f'\n- 평균 규칙에서 sim 쪽 상대 점수가 더 낮았던 스텝 비율: {np.mean(sb):.2f}\n')
		L.append('| 비교 (같은 에피소드끼리 짝, seed 단위) | 평균 [t 구간 95 %] | 판정 |')
		L.append('|---|---|---|')
		sd = lambda x, y: interval(arms[x]['ret'].mean(1) - arms[y]['ret'].mean(1))
		iv = sd('합의 (min)', '평균')
		S['A'] = dict(min_minus_mean=iv, verdict=min_works(iv), mean_minus_pre=sd('평균', '적응 없음'), mean_minus_ora=sd('평균', '정확한 dynamics만'),
					  ret={n: float(d['ret'].mean()) for n, d in arms.items()})
		L.append(f'| 합의 (min) − 평균 | {fmt(iv)} | **{min_works(iv)}** |')
		L.append(f'| 평균 − 적응 없음 | {fmt(S["A"]["mean_minus_pre"])} | {d09.three(S["A"]["mean_minus_pre"])} |')
		L.append(f'| 평균 − 정확한 dynamics만 | {fmt(S["A"]["mean_minus_ora"])} | {d09.three(S["A"]["mean_minus_ora"])} |')
	else:
		L.append('(결과 없음: ' + ', '.join(n for n, v in arms.items() if v is None) + ')')

	# part B
	L.append('\n## B. Q 앙상블 불일치: 실제 3스텝 끝 상태 대 sim 모델의 3스텝 끝 상태 (저장된 궤적, forward만)\n')
	L.append('u는 Q 앙상블 다섯 망의 표준편차다. 비는 에피소드마다 (실제 끝 상태의 u 평균) / (sim 모델 끝 상태의 u 평균)이고, 구간은 에피소드 단위 t 구간 95 %다. '
			 f'주 판정은 cheetah 마찰의 "적응 없음, 바뀐 조건 (잡음 켬)" 줄이다: 구간 전체가 {R_LO} 아래면 "차이 없음", {R_HI} 위면 "차이 큼", 그 밖은 "미결".\n')
	S['B'] = {}
	for cell in [CF] + [c for c in CELLS if c != CF]:
		f = f'{a.out_root}/qdis/{cell}.json'
		if not os.path.exists(f):
			continue
		R = json.load(open(f))
		L.append(f'\n### {SHORT[cell]}\n')
		L.append('| 궤적 | 에피소드 | return | u 비 (실제 끝 / 모델 끝) | Q(모델 끝) − Q(실제 끝) | 방문 상태의 u (전체 / 첫 100 스텝) | 100스텝 가치 잔차 |')
		L.append('|---|---|---|---|---|---|---|')
		S['B'][cell] = {}
		for name, r in R.items():
			iv = {k: interval(r[k]) for k in ('ratio', 'dq', 'u_vis', 'u_vis_first100', 'resid', 'ret')}
			S['B'][cell][name] = iv
			L.append(f'| {r["note"]} | {r["episodes"]} | {iv["ret"]["mean"]:.0f} | {fmt(iv["ratio"], 2, False)} | {fmt(iv["dq"], 2)} | '
					 f'{iv["u_vis"]["mean"]:.2f} / {iv["u_vis_first100"]["mean"]:.2f} | {fmt(iv["resid"], 2)} |')
		if cell == CF and 'shift_PREc' in S['B'][cell]:
			lo, hi = S['B'][cell]['shift_PREc']['ratio']['t']
			S['B']['verdict'] = '차이 없음' if hi < R_LO else ('차이 큼' if lo > R_HI else '미결')
			L.append(f'\n- **주 판정 (cheetah 마찰, 적응 없음의 궤적): {S["B"]["verdict"]}**')

	# part C
	L.append('\n## C. 학습한 모델 수준: 평균 규칙 (설계 10과 같은 적응 모델, 같은 평가 seed)\n')
	L.append('| 조건 | seed 수 | 적응 없음 | 합의 (min) | 평균 | 평균 − 적응 없음 | 해 없음 | 합의 − 평균 | 판정 | 평소 조건 (적응 없음 대비) | 넘어짐 사건 | 부드러움 / 노력 / 흔들림 |')
	L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
	S['C'], hs, v0 = {}, [], True
	for cell in CELLS:
		rows = {}
		for s in range(1, 33):
			fs = dict(MEAN=f'{a.out_root}/learned/{cell}/s{s:02d}/eval_MEAN.json', PRE=f'{a.d10_root}/{cell}/s{s:02d}/eval_PRE.json', CON=f'{a.d10_root}/{cell}/s{s:02d}/eval_CON.json')
			if all(os.path.exists(p) for p in fs.values()):
				rows[s] = {k: json.load(open(p)) for k, p in fs.items()}
		if not rows:
			continue
		sm = lambda arm, c='shift', k='ret': np.array([np.mean([e[k] for e in rows[s][arm][c]['episodes']]) for s in sorted(rows)])
		g, dm = interval(sm('MEAN') - sm('PRE')), interval(sm('CON') - sm('MEAN'))
		nr = float(sm('MEAN', 'nominal').mean() / sm('PRE', 'nominal').mean())
		S['C'][cell] = dict(n=len(rows), pre=float(sm('PRE').mean()), con=float(sm('CON').mean()), mean=float(sm('MEAN').mean()), gain=g, harm=harm(g),
							con_minus_mean=dm, verdict=min_works(dm), nominal_ratio=nr, **{k: float(sm('MEAN', 'shift', k).mean()) for k in K[1:]})
		hs.append(harm(g))
		v0 = v0 and nr >= V0_RATIO
		L.append(f'| {SHORT[cell]} | {len(rows)} | {sm("PRE").mean():.0f} | {sm("CON").mean():.0f} | {sm("MEAN").mean():.0f} | {fmt(g)} | {harm(g)} | {fmt(dm)} | {min_works(dm)} | {nr:.2f} | '
				 f'{sm("MEAN", "shift", "falls").mean():.1f} | {sm("MEAN", "shift", "smooth").mean():.2f} / {sm("MEAN", "shift", "effort").mean():.2f} / {sm("MEAN", "shift", "angvel_rms").mean():.2f} |')
	if len(hs) == len(CELLS):
		n = '통과' if all(v == '통과' for v in hs) else ('실패' if any(v == '실패' for v in hs) else '미결')
		S['C']['N'], S['C']['V0'] = n, bool(v0)
		S['C']['mean_gain'] = float(np.mean([S['C'][c]['gain']['mean'] for c in CELLS]))
		S['C']['mean_con_minus_mean'] = float(np.mean([S['C'][c]['con_minus_mean']['mean'] for c in CELLS]))
		L.append(f'\n- **평균 규칙의 N "해 없음": {n}**, 보존 V0: {"예" if v0 else "아니오"}, 여섯 조건 평균 이득 {S["C"]["mean_gain"]:+.1f}, 여섯 조건 평균 (합의 − 평균) {S["C"]["mean_con_minus_mean"]:+.1f}')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', default=CF)
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=16)
	p.add_argument('--eps', type=int, default=2)
	p.add_argument('--workers', type=int, default=10)
	p.add_argument('--timing', type=int, default=0)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('learned')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--eval_eps', type=int, default=EVAL_EPS)
	p.add_argument('--nom_eps', type=int, default=NOM_EPS)
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--d10_root', default=D10)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('qdis')
	p.add_argument('--cells', default='')
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--d08_root', default='/mnt/newssd/cjkim/td/d08')
	p.add_argument('--d09_root', default=D09)
	p.add_argument('--d10_root', default=D10)
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, learned=cmd_learned, qdis=cmd_qdis, summarize=cmd_summarize)[a.cmd](a)
