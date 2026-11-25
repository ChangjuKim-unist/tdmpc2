"""Design 04 (research/docs/design04_oracle_dynamics_ko.md): oracle-dynamics planner (measurement, no training).

Commands (run from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d04.py eval --cell walker-run__friction_0.15 --cond shift|nominal [--eps 20] [--workers 40]
  GPU=N research/scripts/td.sh research/scripts/d04.py summarize
The official MPPI loop is copied; only candidate evaluation changes: each candidate's 3 actions run in the true simulator
from the restored current state (true rewards), terminal value = pretrained Q at the encoded true state.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from common import math as tdmath
from common.parser import parse_cfg
from envs import make_env

from d01 import CFG_PATH, CKPT, Shifter, _cuda, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import CELLS, EVAL_EPS, EVAL_SEED, Restorer2

D02 = '/mnt/newssd/cjkim/td/d02'
OUT = os.environ.get('D04_OUT', '/mnt/newssd/cjkim/td/d04')
H_CELLS = ['cheetah-run__friction_0.15', 'quadruped-run__motor_0.35']
NOM_EPS = 5
GAMMA = 0.99  # agent.discount for episode length 500
REF = ['PRE', 'D2000', 'MIX', 'MIXPEN1']

# ----------------------------------------------------------------------------- workers (CPU only, one simulator each)

_W = None


def _winit(task, cond):
	global _W
	os.nice(10)
	torch.set_num_threads(1)
	cfg = OmegaConf.load(CFG_PATH)
	cfg.task, cfg.model_size, cfg.checkpoint = task, 5, CKPT.format(task=task)
	cfg.compile, cfg.save_video, cfg.enable_wandb = False, False, False
	cfg = parse_cfg(cfg)
	env = make_env(cfg)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	_W = Restorer2(env)


def _wroll(args):
	state, seqs = args
	R = np.empty((len(seqs), seqs.shape[1]), np.float32)
	O = []
	for i, s in enumerate(seqs):
		outs, rews = _W.rollout_r(state, s)
		R[i] = rews
		O.append(outs[-1])
	return R, np.asarray(O, np.float32)


# ----------------------------------------------------------------------------- planner

class Oracle:
	"""Official _plan with candidate values from the true simulator."""

	def __init__(self, agent, env, pool, workers):
		self.agent, self.cfg, self.m = agent, agent.cfg, agent.model
		self.p, self.pool, self.W = phys_of(env), pool, workers
		self.gamma = float(agent.discount)
		self.prev_mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device='cuda')
		self.plan_time, self.vsel = [], []

	@torch.no_grad()
	def values(self, state, actions):
		seqs = actions.permute(1, 0, 2).cpu().numpy().astype(np.float32)  # N x H x A
		chunks = [c for c in np.array_split(seqs, self.W) if len(c)]
		outs = self.pool.map(_wroll, [(state, c) for c in chunks], chunksize=1)
		R = np.concatenate([o[0] for o in outs])
		O = np.concatenate([o[1] for o in outs])
		disc = self.gamma ** np.arange(self.cfg.horizon)
		G = torch.as_tensor(R @ disc, dtype=torch.float32, device='cuda').unsqueeze(1)
		z = self.m.encode(_cuda(O), None)
		action, _ = self.m.pi(z, None)
		return G + self.gamma ** self.cfg.horizon * self.m.Q(z, action, None, return_type='avg')

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
		mean = torch.zeros(cfg.horizon, cfg.action_dim, device='cuda')
		std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device='cuda')
		if not t0:
			mean[:-1] = self.prev_mean[1:]
		actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda')
		actions[:, :cfg.num_pi_trajs] = pi_actions
		for _ in range(cfg.iterations):
			r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device='cuda')
			actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
			value = self.values(state, actions).nan_to_num(0)
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
		self.vsel.append(float(elite_value[idx].squeeze()))
		self.plan_time.append(time.time() - tic)
		return a.clamp(-1, 1).cpu()


def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	real = cond if a.cond == 'shift' else 'nominal'
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, real))  # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	ora = Oracle(agent, env, pool, a.workers)
	t0 = time.time()
	eps, recs = run_cond(ora, env, shifter, domain, real, a.eps, seed=EVAL_SEED, record=True)
	res = dict(cond=real, summary=summary(eps), episodes=eps, plan_time=float(np.mean(ora.plan_time)), seconds=time.time() - t0)
	rec = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in ('obs', 'state', 'act', 'rew')}
	rec['vsel'] = np.asarray(ora.vsel, np.float32).reshape(len(recs), -1)
	folder = f'{OUT}/{a.cell}' if a.cond == 'shift' else f'{OUT}/nominal/{task}'
	os.makedirs(folder, exist_ok=True)
	dump(f'{folder}/eval_ORA.json', res)
	np.savez_compressed(f'{folder}/rec_ORA.npz', **rec)
	print(a.cell, a.cond, res['summary'], f'plan {res["plan_time"]:.3f}s/step total {res["seconds"]:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- summary (registered criteria, design 04 §5)

def _ep(ev, k):
	return np.array([e[k] for e in ev['episodes']])


def paired_ci(x, y, n=2000, seed=4):
	d = np.asarray(x) - np.asarray(y)
	bs = np.random.RandomState(seed)
	m = [d[bs.randint(0, len(d), len(d))].mean() for _ in range(n)]
	return float(d.mean()), [float(v) for v in np.percentile(m, [2.5, 97.5])]


def speed(domain, rec_state, rec_rew):
	"""Mean forward speed (m/s) over recorded episodes; quadruped: only steps with torso upright > 0.9."""
	if domain == 'cheetah':
		return float((5 * rec_rew).mean())
	if domain == 'walker':
		return float(rec_state[:, :, 10].mean())
	up = 1 - 2 * (rec_state[:, :-1, 4] ** 2 + rec_state[:, :-1, 5] ** 2)
	ok = up > 0.9
	if not ok.any():
		return float('nan')
	return float(((rec_rew[ok] / 2 / ((1 + up[ok]) / 2) - 0.5) / 0.1).mean())


def cmd_summarize(a):
	out_root, d02_root = a.out_root or OUT, a.d02_root or D02
	cells = a.cells.split(',') if a.cells else CELLS
	L, S = [], dict(cells={})
	L.append('# 설계 04 결과 요약 (자동 생성, `d04.py summarize`)\n')
	L.append('ORA = 후보 평가를 simulator로 한 planner(정확한 3스텝 dynamics + 사전학습 Q). PRE, D2000, MIX, MIXPEN1은 설계 02의 평가다. 평가 시드 2.\n')
	# V0
	L.append('## V0 타당성: 평소 조건 return\n')
	L.append('| 과제 | PRE (20 에피소드) | ORA | 비율 | 통과(≥ 0.9) |')
	L.append('|---|---|---|---|---|')
	v0 = {}
	for task in sorted({cell_parts(c)[0] for c in cells}):
		cell0 = [c for c in CELLS + cells if cell_parts(c)[0] == task][0]
		pre = json.load(open(f'{d02_root}/{cell0}/eval_PRE.json'))['nominal']['summary']['ret']
		f = f'{out_root}/nominal/{task}/eval_ORA.json'
		if not os.path.exists(f):
			v0[task] = None
			L.append(f'| {task} | {pre:.0f} | (없음) | – | – |')
			continue
		o = json.load(open(f))
		v0[task] = bool(o['summary']['ret'] >= 0.9 * pre)
		S[f'v0/{task}'] = dict(pre=pre, ora=o['summary']['ret'], ok=v0[task], n=len(o['episodes']))
		L.append(f'| {task} | {pre:.0f} | {o["summary"]["ret"]:.0f} ({len(o["episodes"])} 에피소드) | {o["summary"]["ret"] / pre:.2f} | {"예" if v0[task] else "아니오"} |')
	# returns
	L.append('\n## 바뀐 조건 return (넘어짐 사건 / 넘어져 있는 시간 비율)\n')
	L.append('| 셀 | ' + ' | '.join(REF) + ' | ORA |')
	L.append('|---|' + '---|' * (len(REF) + 1))
	ev = {}
	for c in cells:
		row = []
		for arm in REF:
			e = json.load(open(f'{d02_root}/{c}/eval_{arm}.json'))['shift']
			ev[c, arm] = e
			row.append(f'{e["summary"]["ret"]:.0f} ({e["summary"]["falls"]:.1f} / {e["summary"]["fallen_frac"]:.2f})')
		f = f'{out_root}/{c}/eval_ORA.json'
		if os.path.exists(f):
			e = json.load(open(f))
			ev[c, 'ORA'] = e
			row.append(f'{e["summary"]["ret"]:.0f} ({e["summary"]["falls"]:.1f} / {e["summary"]["fallen_frac"]:.2f})')
		else:
			row.append('(없음)')
		L.append(f'| {c}{" (H)" if c in H_CELLS else ""} | ' + ' | '.join(row) + ' |')
	# paired differences
	L.append('\n## ORA와의 에피소드 짝 차이 (평균 [bootstrap 95 % 구간])\n')
	L.append('| 셀 | ORA − PRE | ORA − D2000 | ORA − MIXPEN1 | 초기 관측 일치 |')
	L.append('|---|---|---|---|---|')
	for c in cells:
		if (c, 'ORA') not in ev:
			continue
		o = _ep(ev[c, 'ORA'], 'ret')
		n = len(o)
		row, S['cells'][c] = [], {}
		for arm in ('PRE', 'D2000', 'MIXPEN1'):
			m, ci = paired_ci(o, _ep(ev[c, arm], 'ret')[:n])
			S['cells'][c][arm] = dict(diff=m, ci=ci)
			row.append(f'{m:+.0f} [{ci[0]:+.0f}, {ci[1]:+.0f}]')
		ro, rp = np.load(f'{out_root}/{c}/rec_ORA.npz'), np.load(f'{d02_root}/{c}/rec_PRE.npz')
		k = min(2, len(ro['obs']))
		same = float(np.abs(ro['obs'][:k, 0] - rp['shift_obs'][:k, 0]).max())
		S['cells'][c]['init_obs_maxdiff'] = same
		S['cells'][c]['ora'] = ev[c, 'ORA']['summary']
		L.append(f'| {c}{" (H)" if c in H_CELLS else ""} | ' + ' | '.join(row) + f' | 최대 차이 {same:.1e} |')
	# behaviour
	L.append('\n## 동작 (기록된 에피소드: 설계 02 팔은 2개, ORA는 전부)\n')
	L.append('| 셀 | 속도 PRE | 속도 D2000 | 속도 ORA | 부드러움 / 노력 / 흔들림 ORA | 계획 가치 − 실현 return ORA | 계획 시간(초/스텝) |')
	L.append('|---|---|---|---|---|---|---|')
	for c in cells:
		if (c, 'ORA') not in ev:
			continue
		dom = domain_of(cell_parts(c)[0])
		sp = []
		for arm in ('PRE', 'D2000'):
			r = np.load(f'{d02_root}/{c}/rec_{arm}.npz')
			sp.append(speed(dom, r['shift_state'], r['shift_rew']))
		ro = np.load(f'{out_root}/{c}/rec_ORA.npz')
		sp.append(speed(dom, ro['state'], ro['rew']))
		rew, T = ro['rew'], ro['rew'].shape[1]
		G = np.zeros_like(rew)
		acc = np.zeros(len(rew))
		for t in range(T - 1, -1, -1):
			acc = rew[:, t] + GAMMA * acc
			G[:, t] = acc
		gap = float((ro['vsel'][:, 50:251] - G[:, 50:251]).mean())
		s = ev[c, 'ORA']['summary']
		S['cells'][c].update(speed=dict(PRE=sp[0], D2000=sp[1], ORA=sp[2]), gap=gap)
		L.append(f'| {c} | {sp[0]:.2f} | {sp[1]:.2f} | {sp[2]:.2f} | {s["smooth"]:.2f} / {s["effort"]:.2f} / {s["angvel_rms"]:.2f} | {gap:.1f} | {ev[c, "ORA"]["plan_time"]:.2f} |')
	# verdicts
	L.append('\n## 판정\n')
	hv = [c for c in H_CELLS if c in cells]
	usable = all(v0.get(cell_parts(c)[0]) for c in hv) and all((c, 'ORA') in ev for c in hv) and len(hv) == len(H_CELLS)
	if not usable:
		L.append('- D1, D2: 판정 불가 (V0 실패 또는 H 셀 결과 없음)')
		S['D1'] = S['D2'] = None
	else:
		d1 = all(S['cells'][c]['D2000']['ci'][0] > 0 and S['cells'][c]['PRE']['ci'][1] >= 0 for c in hv)
		d2 = any(S['cells'][c]['PRE']['ci'][1] < 0 for c in hv)
		S['D1'], S['D2'] = bool(d1), bool(d2)
		L.append(f'- D1 정확한 dynamics면 해가 사라진다: {"그렇다" if d1 else "아니다"}')
		L.append(f'- D2 정확한 dynamics여도 나쁘다: {"그렇다" if d2 else "아니다"}')
		if not d1 and not d2:
			L.append('- → 섞임')
	L.append('\n## 예측 대조\n')
	g = lambda c, arm, i: S['cells'].get(c, {}).get(arm, {}).get('ci', [None, None])[i]
	chk = {
		'P1 V0 3과제 통과': all(v for v in v0.values()) if v0 and None not in v0.values() else None,
		'P2 cheetah 마찰 ORA − D2000 > 0': None if g('cheetah-run__friction_0.15', 'D2000', 0) is None else g('cheetah-run__friction_0.15', 'D2000', 0) > 0,
		'P3 cheetah 마찰 ORA − PRE 위 끝 ≥ 0': None if g('cheetah-run__friction_0.15', 'PRE', 1) is None else g('cheetah-run__friction_0.15', 'PRE', 1) >= 0,
		'P4 quadruped 모터 넘어져 있는 시간 < D2000': None if ('quadruped-run__motor_0.35', 'ORA') not in ev else
			ev['quadruped-run__motor_0.35', 'ORA']['summary']['fallen_frac'] < ev['quadruped-run__motor_0.35', 'D2000']['summary']['fallen_frac'],
		'P5 D1': S['D1'], 'P6 D2': S['D2'],
		'P7 질량 2셀 ORA − D2000 위 끝 ≥ 0': None if any(g(c, 'D2000', 1) is None for c in ('walker-run__mass_1.5', 'cheetah-run__mass_2.0')) else
			all(g(c, 'D2000', 1) >= 0 for c in ('walker-run__mass_1.5', 'cheetah-run__mass_2.0')),
		'P8 quadruped 마찰 ORA ≥ MIX': None if ('quadruped-run__friction_0.15', 'ORA') not in ev else
			ev['quadruped-run__friction_0.15', 'ORA']['summary']['ret'] >= ev['quadruped-run__friction_0.15', 'MIX']['summary']['ret'],
		'P9 walker 마찰 ORA − D2000 > 0': None if g('walker-run__friction_0.15', 'D2000', 0) is None else g('walker-run__friction_0.15', 'D2000', 0) > 0,
	}
	for k, v in chk.items():
		L.append(f'- {k}: {"(없음)" if v is None else ("예" if v else "아니오")}')
	S['pred'] = {k: (None if v is None else bool(v)) for k, v in chk.items()}
	os.makedirs(f'{out_root}/summary', exist_ok=True)
	open(f'{out_root}/summary/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{out_root}/summary/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('eval')
	p.add_argument('--cell', required=True)
	p.add_argument('--cond', choices=['shift', 'nominal'], default='shift')
	p.add_argument('--eps', type=int, default=None)
	p.add_argument('--workers', type=int, default=40)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=None)
	p.add_argument('--d02_root', default=None)
	p.add_argument('--cells', default=None)
	a = ap.parse_args()
	if a.cmd == 'eval':
		if a.eps is None:
			a.eps = EVAL_EPS if a.cond == 'shift' else NOM_EPS
		cmd_eval(a)
	else:
		cmd_summarize(a)
