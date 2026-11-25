"""Design 06 (research/docs/design06_consensus_planner_ko.md): consensus planner (no training).

Every MPPI candidate is scored by two sides and the smaller centered value is used:
  J_s = plan value under the pretrained (sim) latent dynamics, J_x = plan value under the adapted side,
  score = min(J_s - mean_candidates(J_s), J_x - mean_candidates(J_x)).
Arms: CONORA (adapted side = exact simulator rollout of design 04), CONMIX / COND2000 (adapted side = fine-tuned dynamics
of design 02 / 01). CONPRE (adapted side = the pretrained dynamics itself) exists only as a code check for smoke tests.

Commands (run from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d06.py eval_ora --cell walker-run__friction_0.15 --cond shift|nominal [--eps 20] [--workers 40]
  GPU=N research/scripts/td.sh research/scripts/d06.py eval --cell walker-run__friction_0.15 --arm CONMIX|COND2000 [--eps 20]
  GPU=N research/scripts/td.sh research/scripts/d06.py summarize
All commands take --out_root (default /mnt/newssd/cjkim/td/d06); smoke tests must pass another folder.
The official TD-MPC2 code is imported, not modified. Reward, Q, policy and encoder stay pretrained.
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

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import CELLS, EVAL_EPS, EVAL_SEED, model_path
from d04 import GAMMA, NOM_EPS, Oracle, _ep, _winit, paired_ci, speed

D02 = '/mnt/newssd/cjkim/td/d02'
D04 = '/mnt/newssd/cjkim/td/d04'
OUT = '/mnt/newssd/cjkim/td/d06'
DYN_CELLS = ['walker-run__mass_1.5', 'walker-run__friction_0.15', 'quadruped-run__friction_0.15', 'cheetah-run__mass_2.0']
VAL_CELLS = ['cheetah-run__friction_0.15', 'quadruped-run__motor_0.35']
CF, QF = 'cheetah-run__friction_0.15', 'quadruped-run__friction_0.15'
LEARNED = {'CONMIX': 'MIX', 'COND2000': 'D2000', 'CONPRE': 'PRE'}
JUDGED = ['CONORA', 'CONMIX']
NAME = {'CONORA': 'CON-ORA', 'CONMIX': 'CON-MIX', 'COND2000': 'CON-D2000'}
NOM_CELL = {'walker-run': 'walker-run__friction_0.15', 'quadruped-run': 'quadruped-run__motor_0.35', 'cheetah-run': 'cheetah-run__friction_0.15'}
V0_RATIO, C2_NEED = 0.9, 3


# ----------------------------------------------------------------------------- consensus score (design 06 §3)

def consensus(js, jx):
	a_s = js - js.mean(0, keepdim=True)
	a_x = jx - jx.mean(0, keepdim=True)
	return torch.minimum(a_s, a_x), a_s, a_x


@torch.no_grad()
def plan_value(agent, z, actions, dyn):
	"""Official _estimate_value (non-episodic, single task) with the latent dynamics network given explicitly."""
	cfg, m = agent.cfg, agent.model
	assert not cfg.episodic and not cfg.multitask
	G, disc = 0, 1
	for t in range(cfg.horizon):
		reward = tdmath.two_hot_inv(m.reward(z, actions[t], None), cfg)
		z = dyn(torch.cat([z, actions[t]], dim=-1))
		G = G + disc * reward
		disc = disc * agent.discount
	action, _ = m.pi(z, None)
	return G + disc * m.Q(z, action, None, return_type='avg')


# ----------------------------------------------------------------------------- CON-ORA: exact dynamics as the adapted side

class OracleCon(Oracle):
	"""Design-04 oracle planner (official MPPI loop) scored by the consensus of the sim model and the exact rollout."""

	def __init__(self, agent, env, pool, workers):
		super().__init__(agent, env, pool, workers)
		self.dyn0 = agent.model._dynamics                                 # pretrained (nothing is loaded on top)
		self.jx_sel, self.js_sel, self.sim_binds = [], [], []

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		cfg, m = self.cfg, self.m
		tic = time.time()
		state = self.p.get_state().copy()
		z = m.encode(obs.to('cuda').unsqueeze(0), None)
		pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device='cuda')
		_z = z.repeat(cfg.num_pi_trajs, 1)
		for t in range(cfg.horizon - 1):
			pi_actions[t], _ = m.pi(_z, None)
			_z = m.next(_z, pi_actions[t], None)
		pi_actions[-1], _ = m.pi(_z, None)
		zs = z.repeat(cfg.num_samples, 1)
		mean = torch.zeros(cfg.horizon, cfg.action_dim, device='cuda')
		std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device='cuda')
		if not t0:
			mean[:-1] = self.prev_mean[1:]
		actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda')
		actions[:, :cfg.num_pi_trajs] = pi_actions
		for _ in range(cfg.iterations):
			r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device='cuda')
			actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
			jx = self.values(state, actions).nan_to_num(0)                 # exact 3 steps + pretrained Q (design 04)
			js = plan_value(self.agent, zs, actions, self.dyn0).nan_to_num(0)
			value, a_s, a_x = consensus(js, jx)
			elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			max_value = elite_value.max(0).values
			score = torch.exp(cfg.temperature * (elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(cfg.min_std, cfg.max_std)
		idx = tdmath.gumbel_softmax_sample(score.squeeze(1))
		a, s0 = torch.index_select(elite_actions, 1, idx).squeeze(1)[0], std[0]
		if not eval_mode:
			a = a + s0 * torch.randn(cfg.action_dim, device='cuda')
		self.prev_mean.copy_(mean)
		sel = elite_idxs[idx]
		self.vsel.append(float(jx[sel].squeeze()))                         # adapted-side plan value of the chosen plan
		self.jx_sel.append(float(jx[sel].squeeze()))
		self.js_sel.append(float(js[sel].squeeze()))
		self.sim_binds.append(float((a_s[sel] < a_x[sel]).squeeze()))
		self.plan_time.append(time.time() - tic)
		return a.clamp(-1, 1).cpu()


def cmd_eval_ora(a):
	task, cond = cell_parts(a.cell)
	real = cond if a.cond == 'shift' else 'nominal'
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, real))  # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	ora = OracleCon(agent, env, pool, a.workers)
	t0 = time.time()
	eps, recs = run_cond(ora, env, shifter, domain, real, a.eps, seed=EVAL_SEED, record=True)
	res = dict(cond=real, summary=summary(eps), episodes=eps, plan_time=float(np.mean(ora.plan_time)),
			   sim_binds=float(np.mean(ora.sim_binds)), seconds=time.time() - t0)
	rec = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in ('obs', 'state', 'act', 'rew')}
	for k, v in (('vsel', ora.vsel), ('jx', ora.jx_sel), ('js', ora.js_sel), ('sim_binds', ora.sim_binds)):
		rec[k] = np.asarray(v, np.float32).reshape(len(recs), -1)
	folder = f'{a.out_root}/{a.cell}' if a.cond == 'shift' else f'{a.out_root}/nominal/{task}'
	dump(f'{folder}/eval_CONORA.json', res)
	np.savez_compressed(f'{folder}/rec_CONORA.npz', **rec)
	print(a.cell, a.cond, 'CONORA', res['summary'], f'sim_binds {res["sim_binds"]:.2f} plan {res["plan_time"]:.3f}s/step total {res["seconds"]:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- CON-MIX / CON-D2000: fine-tuned dynamics as the adapted side

class ConAgent:
	"""Official planner (agent._plan, not compiled) with _estimate_value replaced by the consensus score."""

	def __init__(self, agent, dyn0):
		self.agent, self.dyn0 = agent, dyn0
		self.jx_sel, self.js_sel, self.sim_binds, self.plan_time = [], [], [], []
		self._last = None
		agent._estimate_value = self._est

	@torch.no_grad()
	def _est(self, z, actions, task):
		jx = plan_value(self.agent, z, actions, self.agent.model._dynamics).nan_to_num(0)
		js = plan_value(self.agent, z, actions, self.dyn0).nan_to_num(0)
		value, a_s, a_x = consensus(js, jx)
		self._last = (actions[0].clone(), jx, js, a_s, a_x)
		return value

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		tic = time.time()
		a = self.agent.act(obs, t0=t0, eval_mode=eval_mode)
		a0, jx, js, a_s, a_x = self._last                                 # candidates of the last MPPI iteration
		d = (a0 - a.to(a0.device)).abs().sum(-1)
		sel = int(d.argmin())                                             # eval_mode: the returned action is a candidate's first action
		self.jx_sel.append(float(jx[sel]))
		self.js_sel.append(float(js[sel]))
		self.sim_binds.append(float(a_s[sel] < a_x[sel]) if float(d[sel]) < 1e-5 else float('nan'))
		self.plan_time.append(time.time() - tic)
		return a


def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	dyn0 = copy.deepcopy(agent.model._dynamics).eval()                     # pretrained (sim) dynamics, kept aside
	for p in dyn0.parameters():
		p.requires_grad_(False)
	src = LEARNED[a.arm]
	if src != 'PRE':
		agent.load(model_path(a.cell, src))
	diff = float(sum((p - q).abs().sum() for p, q in zip(agent.model._dynamics.parameters(), dyn0.parameters())))
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	con = ConAgent(agent, dyn0)
	res, rec = dict(dyn_param_absdiff=diff), {}
	for c in a.conds.split(','):
		real = cond if c == 'shift' else 'nominal'
		n0, t0 = len(con.sim_binds), time.time()
		eps, recs = run_cond(con, env, shifter, domain, real, a.eps, seed=EVAL_SEED, record=True)
		res[c] = dict(cond=real, summary=summary(eps), episodes=eps, sim_binds=float(np.nanmean(con.sim_binds[n0:])),
					  plan_time=float(np.mean(con.plan_time[n0:])), seconds=time.time() - t0)
		for k in ('obs', 'state', 'act', 'rew'):
			rec[f'{c}_{k}'] = np.stack([np.asarray(r[k]) for r in recs])
		for k, v in (('jx', con.jx_sel), ('js', con.js_sel), ('sim_binds', con.sim_binds)):
			rec[f'{c}_{k}'] = np.asarray(v[n0:], np.float32).reshape(len(recs), -1)
		print(a.cell, a.arm, c, res[c]['summary'], f'sim_binds {res[c]["sim_binds"]:.2f} {res[c]["seconds"]:.0f}s', flush=True)
	dump(f'{a.out_root}/{a.cell}/eval_{a.arm}.json', res)
	np.savez_compressed(f'{a.out_root}/{a.cell}/rec_{a.arm}.npz', **rec)


# ----------------------------------------------------------------------------- summary (registered criteria, design 06 §6)

def first_reach(obs, v=2.0):
	"""cheetah: median over episodes of the first step with forward speed > v (episode length if never). Same as design 05."""
	s = obs[:, :, 8]
	return float(np.median([int(np.argmax(x > v)) if (x > v).any() else s.shape[1] for x in s]))


def joint_speed(obs, n=100):
	"""cheetah: mean absolute joint velocity (rad/s) over the first n steps."""
	return float(np.abs(obs[:, :n, 11:17]).mean())


def gap(rew, vsel):
	G, acc = np.zeros_like(rew), np.zeros(len(rew))
	for t in range(rew.shape[1] - 1, -1, -1):
		acc = rew[:, t] + GAMMA * acc
		G[:, t] = acc
	return float((vsel[:, 50:251] - G[:, 50:251]).mean())


def _load(path):
	return json.load(open(path)) if os.path.exists(path) else None


def cmd_summarize(a):
	L, S = [], dict(cells={}, nominal={})
	yn = lambda v: '(없음)' if v is None else ('예' if v else '아니오')
	fmt = lambda d: '(없음)' if d is None else f'{d["diff"]:+.0f} [{d["ci"][0]:+.0f}, {d["ci"][1]:+.0f}]'
	L.append('# 설계 06 결과 요약 (자동 생성, `d06.py summarize`)\n')
	L.append('합의 planner: 후보마다 sim 모델(사전학습 dynamics)과 적응한 쪽의 계획 가치를 내고, 후보 평균을 뺀 상대 점수 중 작은 쪽으로 고른다. '
			 'CON-ORA = 적응한 쪽이 정확한 dynamics(simulator 실행), CON-MIX = sim 혼합 fine-tuning 모델, CON-D2000 = 보통 fine-tuning 모델(보고만). '
			 'PRE = 적응 없음, D2000·MIX·MIXPEN1 = 설계 02, ORA = 설계 04. 평가 시드 2.\n')
	ev, rec = {}, {}
	for c in CELLS:
		d2 = {k: _load(f'{a.d02_root}/{c}/eval_{k}.json') for k in ('PRE', 'D2000', 'MIX', 'MIXPEN1')}
		for k, v in d2.items():
			if v:
				ev[c, k], ev[c, k, 'nom'] = v['shift'], v['nominal']
		o = _load(f'{a.d04_root}/{c}/eval_ORA.json')
		if o:
			ev[c, 'ORA'] = o
		o = _load(f'{a.out_root}/{c}/eval_CONORA.json')
		if o:
			ev[c, 'CONORA'] = o
			rec[c, 'CONORA'] = np.load(f'{a.out_root}/{c}/rec_CONORA.npz')
		for arm in ('CONMIX', 'COND2000'):
			o = _load(f'{a.out_root}/{c}/eval_{arm}.json')
			if o:
				ev[c, arm], ev[c, arm, 'nom'] = o['shift'], o['nominal']
				rec[c, arm] = np.load(f'{a.out_root}/{c}/rec_{arm}.npz')
	arms = ['CONORA', 'CONMIX', 'COND2000']

	L.append('## 바뀐 조건 return (넘어짐 사건 / 넘어져 있는 시간 비율)\n')
	cols = ['PRE', 'D2000', 'MIX', 'MIXPEN1', 'ORA'] + arms
	L.append('| 셀 | ' + ' | '.join(NAME.get(k, k) for k in cols) + ' |')
	L.append('|---|' + '---|' * len(cols))
	for c in CELLS:
		row = []
		for k in cols:
			s = ev[c, k]['summary'] if (c, k) in ev else None
			row.append(f'{s["ret"]:.0f} ({s["falls"]:.1f} / {s["fallen_frac"]:.2f})' if s else '(없음)')
		L.append(f'| {c}{" (가치)" if c in VAL_CELLS else ""} | ' + ' | '.join(row) + ' |')

	def pair(c, x, y, nom=False):
		kx, ky = ((c, x, 'nom'), (c, y, 'nom')) if nom else ((c, x), (c, y))
		if kx not in ev or ky not in ev:
			return None
		rx, ry = _ep(ev[kx], 'ret'), _ep(ev[ky], 'ret')
		n = min(len(rx), len(ry))
		m, ci = paired_ci(rx[:n], ry[:n])
		return dict(diff=m, ci=ci, n=n)

	REFS = {'CONORA': ['PRE', 'ORA'], 'CONMIX': ['PRE', 'MIX', 'MIXPEN1'], 'COND2000': ['PRE', 'D2000']}
	for arm in arms:
		L.append(f'\n## {NAME[arm]}: 에피소드 짝 차이 (평균 [bootstrap 95 % 구간])\n')
		L.append('| 셀 | ' + ' | '.join(f'{NAME[arm]} − {r}' for r in REFS[arm]) + ' |')
		L.append('|---|' + '---|' * len(REFS[arm]))
		for c in CELLS:
			e = S['cells'].setdefault(c, {}).setdefault(arm, {})
			if (c, arm) in ev:
				e['summary'] = ev[c, arm]['summary']
				e['sim_binds'] = ev[c, arm].get('sim_binds')
			for r in REFS[arm]:
				e[r] = pair(c, arm, r)
			L.append(f'| {c} | ' + ' | '.join(fmt(e[r]) for r in REFS[arm]) + ' |')

	L.append('\n## 평소 조건 return (V0: PRE 평소 조건의 0.9배 이상)\n')
	L.append('| 과제 | PRE (20 에피소드) | CON-ORA | 비율 | 통과 |')
	L.append('|---|---|---|---|---|')
	v0 = {arm: {} for arm in arms}
	for task, c in NOM_CELL.items():
		pre = ev[c, 'PRE', 'nom']['summary']['ret'] if (c, 'PRE', 'nom') in ev else None
		o = _load(f'{a.out_root}/nominal/{task}/eval_CONORA.json')
		ok = None if (o is None or pre is None) else bool(o['summary']['ret'] >= V0_RATIO * pre)
		v0['CONORA'][task] = ok
		S['nominal'][task] = dict(pre=pre, conora=o['summary'] if o else None, n=len(o['episodes']) if o else 0, ok=ok)
		L.append(f'| {task} | {"(없음)" if pre is None else f"{pre:.0f}"} | ' + (f'{o["summary"]["ret"]:.0f} ({len(o["episodes"])} 에피소드) | {o["summary"]["ret"] / pre:.2f}' if o and pre else '(없음) | –') + f' | {yn(ok)} |')
	L.append('\n| 셀 | PRE | D2000 | MIX | CON-MIX | CON-MIX 비율 | 통과 | CON-D2000 | CON-D2000 비율 |')
	L.append('|---|---|---|---|---|---|---|---|---|')
	for c in CELLS:
		g = lambda k: ev[c, k, 'nom']['summary']['ret'] if (c, k, 'nom') in ev else None
		pre = g('PRE')
		f = lambda v: '(없음)' if v is None else f'{v:.0f}'
		ra = lambda v: '–' if (v is None or pre is None) else f'{v / pre:.2f}'
		for arm in ('CONMIX', 'COND2000'):
			v0[arm][c] = None if (g(arm) is None or pre is None) else bool(g(arm) >= V0_RATIO * pre)
			S['cells'][c][arm]['nominal'] = dict(ret=g(arm), pre=pre, ok=v0[arm][c], d2000=g('D2000'), mix=g('MIX'))
		L.append(f'| {c} | {f(pre)} | {f(g("D2000"))} | {f(g("MIX"))} | {f(g("CONMIX"))} | {ra(g("CONMIX"))} | {yn(v0["CONMIX"][c])} | {f(g("COND2000"))} | {ra(g("COND2000"))} |')

	L.append('\n## 동작 (기록된 에피소드: 설계 02 팔은 2개, 나머지는 전부)\n')
	L.append('| 셀 | 팔 | 속도 (m/s) | 부드러움 / 노력 / 흔들림 | sim 쪽이 점수를 정한 스텝 비율 | 고른 계획의 J_x − 실현 return | 계획 시간(초/스텝) |')
	L.append('|---|---|---|---|---|---|---|')
	R = {}
	for c in CELLS:
		dom = domain_of(cell_parts(c)[0])
		p = f'{a.d02_root}/{c}/rec_PRE.npz'
		if os.path.exists(p):
			z = np.load(p)
			R[c, 'PRE'] = dict(obs=z['shift_obs'], state=z['shift_state'], rew=z['shift_rew'])
		p = f'{a.d04_root}/{c}/rec_ORA.npz'
		if os.path.exists(p):
			z = np.load(p)
			R[c, 'ORA'] = dict(obs=z['obs'], state=z['state'], rew=z['rew'], vsel=z['vsel'])
		if (c, 'CONORA') in rec:
			z = rec[c, 'CONORA']
			R[c, 'CONORA'] = dict(obs=z['obs'], state=z['state'], rew=z['rew'], vsel=z['jx'])
		for arm in ('CONMIX', 'COND2000'):
			if (c, arm) in rec:
				z = rec[c, arm]
				R[c, arm] = dict(obs=z['shift_obs'], state=z['shift_state'], rew=z['shift_rew'], vsel=z['shift_jx'])
		for k in ['PRE', 'ORA'] + arms:
			if (c, k) not in R or (c, k) not in ev:
				continue
			r, s = R[c, k], ev[c, k]['summary']
			sp = speed(dom, r['state'], r['rew'])
			g = gap(r['rew'], r['vsel']) if 'vsel' in r else None
			sb, pt = ev[c, k].get('sim_binds'), ev[c, k].get('plan_time')
			if k in arms:
				S['cells'][c][k].update(speed=sp, gap=g)
			L.append(f'| {c} | {NAME.get(k, k)} | {sp:.2f} | {s["smooth"]:.2f} / {s["effort"]:.2f} / {s["angvel_rms"]:.2f} | '
					 f'{"–" if sb is None else f"{sb:.2f}"} | {"–" if g is None else f"{g:.1f}"} | {"–" if pt is None else f"{pt:.2f}"} |')

	L.append('\n## cheetah 마찰의 출발\n')
	L.append('| 팔 | 첫 100 스텝 관절 속도 (rad/s) | 2 m/s에 처음 닿는 스텝 (중앙값) | 에피소드 수 |')
	L.append('|---|---|---|---|')
	S['startup'] = {}
	for k in ['PRE', 'ORA'] + arms:
		if (CF, k) in R:
			o = R[CF, k]['obs']
			S['startup'][k] = dict(joint_speed=joint_speed(o), first_reach=first_reach(o), n=len(o))
			L.append(f'| {NAME.get(k, k)} | {joint_speed(o):.2f} | {first_reach(o):.0f} | {len(o)} |')

	L.append('\n## 판정\n')
	V = {}
	for arm in JUDGED:
		hi = {c: (S['cells'][c][arm]['PRE']['ci'][1] if S['cells'][c][arm].get('PRE') else None) for c in CELLS}
		lo = {c: (S['cells'][c][arm]['PRE']['ci'][0] if S['cells'][c][arm].get('PRE') else None) for c in CELLS}
		done = all(v is not None for v in hi.values())
		c1 = bool(all(hi[c] >= 0 for c in CELLS)) if done else None
		ngain = sum(1 for c in DYN_CELLS if lo[c] is not None and lo[c] > 0)
		c2 = bool(ngain >= C2_NEED) if done else None
		vv = list(v0[arm].values())
		v0ok = None if any(v is None for v in vv) else bool(all(vv))
		ok = None if None in (c1, c2, v0ok) else bool(c1 and c2 and v0ok)
		V[arm] = dict(V0=v0ok, C1=c1, C2=c2, n_gain=ngain, passed=ok,
					  harm=[c for c in CELLS if hi[c] is not None and hi[c] < 0])
		L.append(f'- {NAME[arm]}: V0 {yn(v0ok)}, C1 해 없음 {yn(c1)}' + (f' (해가 난 셀: {", ".join(V[arm]["harm"])})' if V[arm]['harm'] else '')
				 + f', C2 이득 {yn(c2)} (4셀 중 {ngain}셀), **통과 {yn(ok)}**')
	S['verdict'] = V

	L.append('\n## 예측 대조\n')
	ci = lambda c, arm, r, i: (S['cells'][c][arm].get(r) or {}).get('ci', [None, None])[i]
	gt0 = lambda v: None if v is None else bool(v > 0)
	su = S['startup']
	nd = [(S['cells'][c]['COND2000'].get('nominal') or {}) for c in CELLS]
	p13 = None if any(d.get('ret') is None or d.get('d2000') is None for d in nd) else bool(sum(d['ret'] > d['d2000'] for d in nd) >= 5)
	chk = {
		'P1 CON-ORA V0 통과': V['CONORA']['V0'],
		'P2 cheetah 마찰 CON-ORA − ORA > 0': gt0(ci(CF, 'CONORA', 'ORA', 0)),
		'P3 cheetah 마찰 CON-ORA − PRE 위 끝 ≥ 0': None if ci(CF, 'CONORA', 'PRE', 1) is None else bool(ci(CF, 'CONORA', 'PRE', 1) >= 0),
		'P4 cheetah 마찰 CON-ORA 첫 100 스텝 관절 속도 > ORA': None if 'CONORA' not in su or 'ORA' not in su else bool(su['CONORA']['joint_speed'] > su['ORA']['joint_speed']),
		'P5 CON-ORA C1': V['CONORA']['C1'],
		'P6 CON-ORA C2': V['CONORA']['C2'],
		'P7 CON-ORA 통과': V['CONORA']['passed'],
		'P8 quadruped 마찰 CON-ORA − ORA < 0': None if ci(QF, 'CONORA', 'ORA', 1) is None else bool(ci(QF, 'CONORA', 'ORA', 1) < 0),
		'P9 cheetah 마찰 CON-MIX − MIX > 0': gt0(ci(CF, 'CONMIX', 'MIX', 0)),
		'P10 CON-MIX C1': V['CONMIX']['C1'],
		'P11 CON-MIX C2': V['CONMIX']['C2'],
		'P12 CON-MIX 통과': V['CONMIX']['passed'],
		'P13 CON-D2000 평소 조건 return > D2000인 셀 ≥ 5': p13,
	}
	for k, v in chk.items():
		L.append(f'- {k}: {yn(v)}')
	S['pred'] = chk
	os.makedirs(f'{a.out_root}/summary', exist_ok=True)
	open(f'{a.out_root}/summary/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.out_root}/summary/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	for name in ('eval_ora', 'eval', 'summarize'):
		p = sub.add_parser(name)
		p.add_argument('--out_root', default=OUT)
		if name != 'summarize':
			p.add_argument('--cell', required=True)
			p.add_argument('--eps', type=int, default=None)
		if name == 'eval_ora':
			p.add_argument('--cond', choices=['shift', 'nominal'], required=True)
			p.add_argument('--workers', type=int, default=40)
		if name == 'eval':
			p.add_argument('--arm', choices=list(LEARNED), required=True)
			p.add_argument('--conds', default='shift,nominal')
		if name == 'summarize':
			p.add_argument('--d02_root', default=D02)
			p.add_argument('--d04_root', default=D04)
	a = ap.parse_args()
	if a.cmd == 'eval_ora' and a.eps is None:
		a.eps = EVAL_EPS if a.cond == 'shift' else NOM_EPS
	if a.cmd == 'eval' and a.eps is None:
		a.eps = EVAL_EPS
	dict(eval_ora=cmd_eval_ora, eval=cmd_eval, summarize=cmd_summarize)[a.cmd](a)
