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
		raise SystemExit('summarize: added with the registration commit')
