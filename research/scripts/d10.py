"""Design 10 (research/docs/design10_deployment_devices_ko.md): deployment devices on the same adapted model.

One process = one (cell, seed). The adapted model M2 is the control arm of design 07: 10 real episodes collected by the
un-adapted planner (5 with seed 1000 + s, 5 with seed 2000 + s), sim-mix fine-tuning of the dynamics network (torch seed s).
For the two design-07 cells the stored M2 (and the stored PRE / CON evaluations) are reused.
Deployment devices, all with the same M2, no exploration noise, seed 3000 + s:
  PRE       un-adapted planner (reference)
  PLAIN     ordinary planner with M2
  PEN<l>    ordinary planner with M2 and the FOWM penalty of design 02 (l x std of the Q ensemble)
  CON       consensus of the sim model and M2 (design 06)
  CONPEN<l> consensus whose adapted side is the penalised value (sim side unchanged)

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d10.py run --cell walker-run__friction_0.15 --seed 1 [--arms PRE,PLAIN,PEN1,CON,CONPEN1]
  GPU=N research/scripts/td.sh research/scripts/d10.py summarize --res_dir /mnt/newssd/cjkim/td/d10/res
Smoke tests must pass --out_root (another folder) and may shorten with --r_eps / --eval_eps / --nom_eps / --ft_steps; `--check` only compares values.
The official TD-MPC2 code is imported, not modified. Reward, Q, policy and encoder stay pretrained.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch

from common import math as tdmath

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import CELLS, load_sim, pen_value
from d06 import ConAgent, consensus, plan_value
from d07 import FT_STEPS, R_EPS, finetune_mix, fresh, stack

D07 = '/mnt/newssd/cjkim/td/d07'
OUT = '/mnt/newssd/cjkim/td/d10'
CORE = ['PRE', 'PLAIN', 'PEN1', 'CON', 'CONPEN1']
EXTRA = ['PEN0.3', 'PEN3', 'CONPEN0.3', 'CONPEN3']
DEVICES = ['PLAIN', 'PEN0.3', 'PEN1', 'PEN3', 'CON', 'CONPEN0.3', 'CONPEN1', 'CONPEN3']
EVAL_EPS, NOM_EPS = 10, 3
MARGIN, V0_RATIO = -25.0, 0.9
TQ = {15: 2.1314, 31: 2.0395}                                               # t quantile 0.975 (two-sided 95 %)
SHORT = {'walker-run__mass_1.5': 'walker 질량', 'walker-run__friction_0.15': 'walker 마찰', 'quadruped-run__friction_0.15': 'quadruped 마찰',
		 'cheetah-run__mass_2.0': 'cheetah 질량', 'quadruped-run__motor_0.35': 'quadruped 모터', 'cheetah-run__friction_0.15': 'cheetah 마찰'}
DNAME = {'PRE': '적응 없음', 'PLAIN': '장치 없음', 'CON': '합의'}
for _l in ('0.3', '1', '3'):
	DNAME[f'PEN{_l}'] = f'벌점 {_l}'
	DNAME[f'CONPEN{_l}'] = f'합의 + 벌점 {_l}'


@torch.no_grad()
def pen_plan_value(agent, z, actions, dyn, lam):
	"""d06.plan_value with design 02's penalty (lam x std of the Q ensemble at every step and at the terminal value)."""
	cfg, m = agent.cfg, agent.model
	G, disc = 0, 1
	for t in range(cfg.horizon):
		reward = tdmath.two_hot_inv(m.reward(z, actions[t], None), cfg)
		u = tdmath.two_hot_inv(m.Q(z, actions[t], None, return_type='all'), cfg).std(0)
		z = dyn(torch.cat([z, actions[t]], dim=-1))
		G = G + disc * (reward - lam * u)
		disc = disc * agent.discount
	action, _ = m.pi(z, None)
	u = tdmath.two_hot_inv(m.Q(z, action, None, return_type='all'), cfg).std(0)
	return G + disc * (m.Q(z, action, None, return_type='avg') - lam * u)


class ConPen(ConAgent):
	"""Consensus planner whose adapted side is the penalised plan value; the sim side is the plain value of the sim model."""

	def __init__(self, agent, dyn0, lam):
		self.lam = lam
		super().__init__(agent, dyn0)

	@torch.no_grad()
	def _est(self, z, actions, task):
		jx = pen_plan_value(self.agent, z, actions, self.agent.model._dynamics, self.lam).nan_to_num(0)
		js = plan_value(self.agent, z, actions, self.dyn0).nan_to_num(0)
		value, a_s, a_x = consensus(js, jx)
		self._last = (actions[0].clone(), jx, js, a_s, a_x)
		return value


def lam_of(arm):
	return float(arm[6:] if arm.startswith('CONPEN') else arm[3:])


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder, src = f'{a.out_root}/{a.cell}/s{s:02d}', f'{a.d07_root}/{a.cell}/s{s:02d}'
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent0 = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	dyn0 = copy.deepcopy(agent0.model._dynamics).eval()                    # pretrained (sim) dynamics, kept aside
	for p in dyn0.parameters():
		p.requires_grad_(False)
	reuse = os.path.exists(f'{src}/done.json') and not a.no_reuse
	d07 = json.load(open(f'{src}/done.json')) if reuse else None

	# adapted model M2 (10 episodes by the un-adapted planner)
	if os.path.exists(f'{folder}/M2_dyn.pt'):
		sd = torch.load(f'{folder}/M2_dyn.pt')
	elif reuse:
		sd = torch.load(f'{src}/M2_CTRL_dyn.pt')
		dump(f'{folder}/collect.json', dict(reused=src, round1=d07['round1']['summary'], round2=d07['CTRL']['round2']['summary'], M2=d07['CTRL']['M2']))
	else:
		t0, data, info = time.time(), [], {}
		for r, base in ((1, 1000), (2, 2000)):
			eps, recs = run_cond(agent0, env, shifter, domain, cond, a.r_eps, eval_mode=False, seed=base + s, record=True)
			data.append(stack(recs))
			info[f'round{r}'] = summary(eps)
		data = {k: np.concatenate([d[k] for d in data], 0) for k in ('obs', 'state', 'act', 'rew')}
		np.savez_compressed(f'{folder}/data.npz', **data)
		m2, log = finetune_mix(cfg, data, load_sim(task), s, a.ft_steps)
		sd = {n: v.detach().clone() for n, v in m2.model._dynamics.state_dict().items()}
		torch.save(sd, f'{folder}/M2_dyn.pt')
		info['M2'] = log[-1] if log else None
		dump(f'{folder}/collect.json', info)
		print(a.cell, s, 'collect', round(info['round1']['ret'], 1), round(info['round2']['ret'], 1), 'M2', info['M2'], f'{time.time() - t0:.0f}s', flush=True)

	def actor_of(arm):
		if arm == 'PRE':
			return agent0, None
		ag = fresh(cfg)
		ag.model._dynamics.load_state_dict(sd)
		if arm == 'PLAIN':
			return ag, None
		if arm == 'CON':
			con = ConAgent(ag, dyn0)
			return con, con
		if arm.startswith('CONPEN'):
			con = ConPen(ag, dyn0, lam_of(arm))
			return con, con
		ag._estimate_value = pen_value(ag, lam_of(arm))
		return ag, None

	if a.check:                                                             # code check on random candidates at a reset state
		ag = fresh(cfg)
		ag.model._dynamics.load_state_dict(sd)
		shifter.set(cond)
		z = ag.model.encode(env.reset().to('cuda').unsqueeze(0), None).repeat(cfg.num_samples, 1)
		acts = torch.rand(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda') * 2 - 1
		val = lambda f: (torch.manual_seed(0), f())[1]
		v_plain = val(lambda: plan_value(ag, z, acts, ag.model._dynamics))
		v_pen0 = val(lambda: pen_plan_value(ag, z, acts, ag.model._dynamics, 0.0))
		v_pen1 = val(lambda: pen_plan_value(ag, z, acts, ag.model._dynamics, 1.0))
		v_d02 = val(lambda: pen_value(ag, 1.0)(z, acts, None))
		v_off = val(lambda: ag._estimate_value(z, acts, None))
		print('check: plain − penalised(0) max abs', float((v_plain - v_pen0).abs().max()), '| penalised(1) − design 02 penalty max abs', float((v_pen1 - v_d02).abs().max()),
			  '| plain − official value max abs', float((v_plain - v_off).abs().max()), '| penalty mean', float((v_plain - v_pen1).mean()), 'value sd', float(v_plain.std()), flush=True)
		return

	for arm in a.arms.split(','):
		f = f'{folder}/eval_{arm}.json'
		if os.path.exists(f):
			continue
		if reuse and arm in ('PRE', 'CON'):                                  # the same model, planner and evaluation seed as design 07
			r = d07['PRE'] if arm == 'PRE' else d07['CTRL']['eval']
			out = {c: dict(cond=r[c]['cond'], episodes=r[c]['episodes'][:n], summary=summary(r[c]['episodes'][:n])) for c, n in (('shift', a.eval_eps), ('nominal', a.nom_eps))}
			out.update(reused=src, sim_binds=None if arm == 'PRE' else d07['CTRL']['sim_binds'])
			dump(f, out)
			continue
		actor, con = actor_of(arm)
		out = {}
		for c, n in (('shift', a.eval_eps), ('nominal', a.nom_eps)):
			real = cond if c == 'shift' else 'nominal'
			t0 = time.time()
			eps, recs = run_cond(actor, env, shifter, domain, real, n, eval_mode=True, seed=3000 + s, record=(c == 'shift'))
			out[c] = dict(cond=real, summary=summary(eps), episodes=eps, seconds=time.time() - t0)
			if c == 'shift':
				np.savez_compressed(f'{folder}/rec_{arm}.npz', rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
				out['sim_binds'] = float(np.nanmean(con.sim_binds)) if con is not None and con.sim_binds else None
			print(a.cell, s, arm, c, {k: round(v, 2) for k, v in out[c]['summary'].items()}, f'{out[c]["seconds"]:.0f}s', flush=True)
		dump(f, out)
	print(a.cell, s, 'done', a.arms, flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 10 §5)

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


def fmt(iv):
	if not np.isfinite(iv['t'][0]):
		return f'{iv["mean"]:+.1f} (seed {iv["n"]}개)'
	return f'{iv["mean"]:+.1f} [{iv["t"][0]:+.1f}, {iv["t"][1]:+.1f}]'


def harm(iv):
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return '통과' if lo > MARGIN else ('실패' if hi < MARGIN else '미결')


def cmd_summarize(a):
	L, S = [], dict(cells={})
	L.append('# 설계 10 결과 요약 (자동 생성, `d10.py summarize`)\n')
	L.append('모든 장치가 같은 적응 모델(적응 없는 planner가 모은 실데이터 10 에피소드, sim 혼합 fine-tuning)을 쓴다. 값은 바뀐 조건 return이고, 구간은 seed 단위 t 구간 95 %다. "해 없음"은 (장치 − 적응 없음)의 구간 아래 끝이 −25 위일 때 통과, 위 끝이 −25 아래일 때 실패다.\n')
	K = ('ret', 'falls', 'fallen_frac', 'smooth', 'effort', 'angvel_rms')
	gain, nomr, val = {}, {}, {}
	for cell in CELLS:
		R = {}
		for f in sorted(glob.glob(f'{a.out_root}/{cell}/s*/eval_*.json')):
			s, arm = int(f.split('/')[-2][1:]), os.path.basename(f)[5:-5]
			R.setdefault(arm, {})[s] = json.load(open(f))
		if 'PRE' not in R:
			continue
		sm = lambda arm, s, c='shift', k='ret': float(np.mean([e[k] for e in R[arm][s][c]['episodes']]))
		L.append(f'\n## {SHORT[cell]} ({cell})\n')
		L.append('| 장치 | seed 수 | return | − 적응 없음 | 해 없음 | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 | 평소 조건 return (적응 없음 대비) | sim 쪽이 정한 비율 |')
		L.append('|---|---|---|---|---|---|---|---|---|')
		S['cells'][cell] = {}
		for arm in ['PRE'] + DEVICES:
			if arm not in R:
				continue
			seeds = sorted(set(R[arm]) & set(R['PRE']))
			m = {k: float(np.mean([sm(arm, s, 'shift', k) for s in seeds])) for k in K}
			iv = interval([sm(arm, s) - sm('PRE', s) for s in seeds])
			nom, nom0 = float(np.mean([sm(arm, s, 'nominal') for s in seeds])), float(np.mean([sm('PRE', s, 'nominal') for s in seeds]))
			sb = [R[arm][s].get('sim_binds') for s in seeds]
			sb = [v for v in sb if v is not None]
			S['cells'][cell][arm] = dict(n=len(seeds), **m, diff=iv, nominal=nom, nominal_ratio=nom / nom0, sim_binds=float(np.mean(sb)) if sb else None)
			val[cell, arm] = {s: sm(arm, s) for s in seeds}
			if arm != 'PRE':
				gain[cell, arm], nomr[cell, arm] = iv, nom / nom0
			L.append(f'| {DNAME[arm]} | {len(seeds)} | {m["ret"]:.0f} | {"–" if arm == "PRE" else fmt(iv)} | {"–" if arm == "PRE" else harm(iv)} | {m["falls"]:.1f} / {m["fallen_frac"]:.2f} | '
					 f'{m["smooth"]:.2f} / {m["effort"]:.2f} / {m["angvel_rms"]:.2f} | {nom:.0f} ({nom / nom0:.2f}) | {"–" if not sb else f"{np.mean(sb):.2f}"} |')
	cells = [c for c in CELLS if c in S['cells']]

	L.append('\n## 주 판정 N "해 없음"과 보존 V0 (장치마다)\n')
	L.append('| 장치 | ' + ' | '.join(SHORT[c] for c in cells) + ' | N | V0 (평소 조건 0.9배 이상) | 여섯 조건 평균 이득 | 가장 나쁜 조건의 이득 |')
	L.append('|---|' + '---|' * (len(cells) + 4))
	S['N'], S['V0'], S['mean_gain'], S['worst_gain'] = {}, {}, {}, {}
	for arm in DEVICES:
		if not all((c, arm) in gain for c in cells) or len(cells) < len(CELLS):
			row = [harm(gain[c, arm]) if (c, arm) in gain else '(없음)' for c in cells]
			L.append(f'| {DNAME[arm]} | ' + ' | '.join(row) + ' | (조건 부족) | | | |')
			continue
		h = [harm(gain[c, arm]) for c in cells]
		n = '통과' if all(v == '통과' for v in h) else ('실패' if any(v == '실패' for v in h) else '미결')
		v0 = all(nomr[c, arm] >= V0_RATIO for c in cells)
		mg, wg = float(np.mean([gain[c, arm]['mean'] for c in cells])), min(gain[c, arm]['mean'] for c in cells)
		S['N'][arm], S['V0'][arm], S['mean_gain'][arm], S['worst_gain'][arm] = n, bool(v0), mg, float(wg)
		L.append(f'| {DNAME[arm]} | ' + ' | '.join(f'{v} ({gain[c, arm]["mean"]:+.0f})' for v, c in zip(h, cells)) + f' | **{n}** | {"예" if v0 else "아니오"} | {mg:+.1f} | {wg:+.1f} |')

	def pair(x, y, c):
		if (c, x) not in val or (c, y) not in val:
			return None
		seeds = sorted(set(val[c, x]) & set(val[c, y]))
		return interval([val[c, x][s] - val[c, y][s] for s in seeds])

	L.append('\n## 보조 C: 조합 (합의 + 벌점 1)과 두 부모\n')
	L.append('| 조건 | 조합 − 합의 | 조합 − 벌점 1 | 합의 − 벌점 1 |')
	L.append('|---|---|---|---|')
	S['C'], below = {}, False
	for c in cells:
		ivs = [pair('CONPEN1', 'CON', c), pair('CONPEN1', 'PEN1', c), pair('CON', 'PEN1', c)]
		if None in ivs:
			continue
		S['C'][c] = dict(vs_con=ivs[0], vs_pen=ivs[1], con_vs_pen=ivs[2])
		better = ivs[0] if ivs[2]['mean'] >= 0 else ivs[1]                    # the comparison with the better parent in this condition
		below = below or (np.isfinite(better['t'][1]) and better['t'][1] < 0)
		L.append(f'| {SHORT[c]} | {fmt(ivs[0])} | {fmt(ivs[1])} | {fmt(ivs[2])} |')
	if all(k in S['mean_gain'] for k in ('CONPEN1', 'CON', 'PEN1')):
		mg = S['mean_gain']
		wins = mg['CONPEN1'] > mg['CON'] and mg['CONPEN1'] > mg['PEN1'] and not below
		S['combo_wins'] = bool(wins)
		L.append(f'\n- 여섯 조건 평균 이득: 조합 {mg["CONPEN1"]:+.1f}, 합의 {mg["CON"]:+.1f}, 벌점 1 {mg["PEN1"]:+.1f}. 더 좋은 부모보다 구간 전체가 아래인 조건: {"있음" if below else "없음"}.')
		L.append(f'- **"조합이 둘을 이긴다": {"그렇다" if wins else "아니다"}**')

	L.append('\n## 보조 S: 벌점 계수에 대한 민감도 (조건별 평균 return)\n')
	L.append('| 조건 | 벌점 0.3 / 1 / 3 | 가장 좋은 계수 | 계수 사이 최대 차이 | 합의 + 벌점 0.3 / 1 / 3 | 계수 사이 최대 차이 |')
	L.append('|---|---|---|---|---|---|')
	S['S'] = {}
	for c in cells:
		g = lambda arm: S['cells'][c].get(arm, {}).get('ret')
		pv, cv = [g(f'PEN{l}') for l in ('0.3', '1', '3')], [g(f'CONPEN{l}') for l in ('0.3', '1', '3')]
		if None in pv or None in cv:
			continue
		best = ('0.3', '1', '3')[int(np.argmax(pv))]
		S['S'][c] = dict(pen=pv, conpen=cv, best=best, pen_range=max(pv) - min(pv), conpen_range=max(cv) - min(cv))
		L.append(f'| {SHORT[c]} | ' + ' / '.join(f'{v:.0f}' for v in pv) + f' | {best} | {max(pv) - min(pv):.0f} | ' + ' / '.join(f'{v:.0f}' for v in cv) + f' | {max(cv) - min(cv):.0f} |')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--arms', default=','.join(CORE + EXTRA))
	p.add_argument('--r_eps', type=int, default=R_EPS)
	p.add_argument('--eval_eps', type=int, default=EVAL_EPS)
	p.add_argument('--nom_eps', type=int, default=NOM_EPS)
	p.add_argument('--ft_steps', type=int, default=FT_STEPS)
	p.add_argument('--no_reuse', action='store_true')
	p.add_argument('--check', action='store_true')
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	if a.cmd == 'run':
		cmd_run(a)
	else:
		cmd_summarize(a)
