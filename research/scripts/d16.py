"""Design 16 (research/docs/design16_benchmark_extension_ko.md): benchmark extension.

New conditions (outside the simulator's parameter set, which is global friction x global motor x torso mass):
  leg_<k>    one leg weakened: gain and bias of that leg's three actuators x k (the operation of the existing motor
             condition, on one leg only). quadruped: front-left leg (actuators 0-2); cheetah: back leg (bthigh, bshin, bfoot)
  delay_<k>  the executed action is the one decided k decision steps earlier (zeros for the first k steps of an episode).
             Declared as irreducible model error: the pending actions are not in the observation.

Commands (from ~/tdmpc2 on gpu1; one process = one job):
  screen   --task quadruped-run --kind leg            level ladder, un-adapted planner, 2 seeds x 5 episodes (not used for any verdict)
  select                                              registered level rule -> {out}/cells.json
  run      --cell quadruped-run__leg_0.3 --seed 1     adapted model M2 (design 10 procedure) and the deployment arms
           [--arms PRE,PLAIN,PEN1,CON,CONC0,CONC1,CONPEN1]; on the six existing cells only CONC1 (design 10's M2)
  exact    --cell ... --seed 1 --planner ORA|CONORA   exact dynamics (the wrapper is the exact model; delay restores its queue)
  fit      --cell ... --seed 1                        F, identification: one-step prediction error on the 10 real episodes over a grid
  retrain  --cell ... --seed 1                        F, retraining from the pretrained checkpoint in the identified simulator (no real data)
  sgft     --cell ... --seed 1 --variant SGFT|WARM    SGFT (public TD-MPC2 code ported) / SGFT-warm (ours: same shaping, pretrained start)
  summarize
Smoke tests must pass --out_root (another folder) and may shorten the step / episode counts. `run --check` only compares values.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from common.buffer import Buffer
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

from d01 import CFG_PATH, CKPT, Shifter, cell_parts, domain_of, dump, make, parse_cond, phys_of, reseed_env, run_cond, run_episode, summary
from d02 import Restorer2, load_sim, pen_value
from d04 import Oracle
from d06 import ConAgent, OracleCon, plan_value
from d07 import FT_STEPS, R_EPS, finetune_mix, fresh, stack
from d10 import ConPen, interval, fmt, harm
from d13 import to_td

OUT = '/mnt/newssd/cjkim/td/d16'
D07, D10, D11, D13 = ('/mnt/newssd/cjkim/td/' + d for d in ('d07', 'd10', 'd11', 'd13'))
OLD = ['walker-run__mass_1.5', 'walker-run__friction_0.15', 'quadruped-run__friction_0.15', 'cheetah-run__mass_2.0',
	   'quadruped-run__motor_0.35', 'cheetah-run__friction_0.15']
F_OLD = ['quadruped-run__motor_0.35', 'walker-run__friction_0.15']                 # F and SGFT also run on these two
NEW_KINDS = [('quadruped-run', 'leg'), ('cheetah-run', 'leg'), ('walker-run', 'delay'), ('cheetah-run', 'delay')]
LEVELS = {'leg': [0.5, 0.3, 0.15, 0.0], 'delay': [1, 2]}                               # weakest first
LEG = {'quadruped-run': [0, 1, 2], 'cheetah-run': [0, 1, 2]}
BAND = (0.4, 0.75)
EVAL_EPS, NOM_EPS, EXACT_EPS = 10, 3, 2
NEW_ARMS = ['PRE', 'PLAIN', 'PEN1', 'CON', 'CONC0', 'CONC1', 'CONPEN1']
GRID = dict(friction=[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0], motor=[0.25, 0.35, 0.5, 0.7, 0.85, 1.0], mass=[0.7, 1.0, 1.5, 2.0, 3.0])
N_FIT, NEAR = 3000, 1.1
F_STEPS, F_SEED_STEPS, F_EVAL_FREQ = 100_000, 5_000, 20_000
BUDGETS = [5_000, 20_000, 50_000, 100_000]
MARGIN = -25.0
ANAME = {'PRE': '적응 없음', 'PLAIN': 'SGFT식 배치 planner', 'PEN1': '벌점 1', 'CON': '합의 c = 0.5', 'CONC0': '합의 c = 0 (평균)', 'CONC1': '합의 c = 1',
		 'CONPEN1': '합의 + 벌점 1', 'ORA': '정확한 dynamics만', 'CONORA': '정확한 dynamics 합의', 'F': 'F (식별 + 다시 학습)'}


# ----------------------------------------------------------------------------- condition wrappers

class DelayEnv:
	"""Outermost wrapper: with delay k the action executed at decision step t is the one decided at t - k (zeros before)."""

	def __init__(self, env):
		self.env, self.delay, self.queue = env, 0, []                      # no __getattr__: d01.dmw_of walks .env

	@property
	def action_space(self):
		return self.env.action_space

	def rand_act(self):
		return self.env.rand_act()

	def reset(self, **kw):
		self.queue = [torch.zeros_like(self.env.rand_act()) for _ in range(self.delay)]
		return self.env.reset(**kw)

	def step(self, action):
		if self.delay == 0:
			return self.env.step(action)
		self.queue.append(action)
		return self.env.step(self.queue.pop(0))

	def pending(self, A):
		return np.stack([q.numpy() for q in self.queue]).astype(np.float32) if self.queue else np.zeros((0, A), np.float32)


class Shifter16(Shifter):
	"""d01.Shifter plus leg_<k>, delay_<k> and F's combined global set 'fit:<friction>:<motor>:<mass>'."""

	def __init__(self, physics, task, denv=None):
		super().__init__(physics)
		self.leg, self.denv = LEG.get(task), denv

	def set(self, cond):
		if self.denv is not None:
			self.denv.delay = 0
		if isinstance(cond, str) and cond.startswith('fit:'):
			f, k, ms = (float(v) for v in cond.split(':')[1:])
			super().set('nominal')
			m, n, o = self.p.model, self.p.named.model, self.o
			m.geom_friction[:, 0] = o['fric'] * f
			m.actuator_gainprm[:, 0] = o['gain'] * k
			m.actuator_biasprm[:, 0:3] = o['bias'] * k
			n.body_mass['torso'] = o['mass'] * ms
			n.body_inertia['torso'] = o['inertia'] * ms
			return
		kind, k = parse_cond(cond)
		super().set(cond if kind in (None, 'friction', 'motor', 'mass') else 'nominal')
		if kind == 'leg':
			i = self.leg
			self.p.model.actuator_gainprm[i, 0] = self.o['gain'][i] * k
			self.p.model.actuator_biasprm[i, 0:3] = self.o['bias'][i] * k
		elif kind == 'delay':
			assert self.denv is not None, 'delay needs the DelayEnv wrapper'
			self.denv.delay = int(k)
		elif kind not in (None, 'friction', 'motor', 'mass'):
			raise ValueError(cond)


def true_params(cond):
	"""The condition expressed in F's parameter set, or None when it is outside it."""
	kind, k = parse_cond(cond)
	p = dict(friction=1.0, motor=1.0, mass=1.0)
	if kind is None:
		return p
	if kind in p:
		p[kind] = k
		return p
	return None


def make16(task, compile_=False, seed=1):
	cfg, env, agent = make(task, compile_=compile_, seed=seed)
	env = DelayEnv(env)
	return cfg, env, agent, Shifter16(phys_of(env), task, env)


# ----------------------------------------------------------------------------- screening (§2.1, level choice; no verdict)

def cmd_screen(a):
	cfg, env, agent, sh = make16(a.task)
	domain = domain_of(a.task)
	out = {}
	for cond in ['nominal'] + [f'{a.kind}_{v:g}' for v in LEVELS[a.kind]]:
		eps = []
		for s in (1, 2):
			e, _ = run_cond(agent, env, sh, domain, cond, a.eps, seed=5000 + s)
			eps += e
		out[cond] = dict(summary=summary(eps), episodes=eps)
		print(a.task, cond, {k: round(v, 2) for k, v in out[cond]['summary'].items()}, flush=True)
	dump(f'{a.out_root}/screen/{a.task}__{a.kind}.json', out)


def pick_level(res, kind):
	r0 = res['nominal']['summary']['ret']
	q = [(v, res[f'{kind}_{v:g}']['summary']['ret'] / r0) for v in LEVELS[kind]]
	inside = [v for v, r in q if BAND[0] <= r <= BAND[1]]
	if inside:
		return inside[0], q
	below = [v for v, r in q if r < BAND[0]]
	return (below[0] if below else None), q


def cmd_select(a):
	cells, L = [], []
	for task, kind in NEW_KINDS:
		f = f'{a.out_root}/screen/{task}__{kind}.json'
		if not os.path.exists(f):
			L.append(f'{task} {kind}: 선별 없음')
			continue
		v, q = pick_level(json.load(open(f)), kind)
		L.append(f'{task} {kind}: ' + ', '.join(f'{lv:g} → {r:.2f}' for lv, r in q) + f' → {"뺀다" if v is None else f"{v:g}"}')
		if v is not None:
			cells.append(f'{task}__{kind}_{v:g}')
	dump(f'{a.out_root}/cells.json', dict(cells=cells, lines=L))
	print('\n'.join(L), '\ncells', cells)


# ----------------------------------------------------------------------------- deployment arms on the adapted model (§3)

def mix_rule(c):
	"""mean - c |difference| of the centred scores; c = 0.5 is exactly design 06's min (kept as torch.minimum)."""
	def rule(js, jx):
		a_s, a_x = js - js.mean(0, keepdim=True), jx - jx.mean(0, keepdim=True)
		v = torch.minimum(a_s, a_x) if c == 0.5 else 0.5 * (a_s + a_x) - c * (a_s - a_x).abs()
		return v, a_s, a_x
	return rule


class ConC(ConAgent):
	def __init__(self, agent, dyn0, c):
		self.rule = mix_rule(c)
		super().__init__(agent, dyn0)

	@torch.no_grad()
	def _est(self, z, actions, task):
		jx = plan_value(self.agent, z, actions, self.agent.model._dynamics).nan_to_num(0)
		js = plan_value(self.agent, z, actions, self.dyn0).nan_to_num(0)
		value, a_s, a_x = self.rule(js, jx)
		self._last = (actions[0].clone(), jx, js, a_s, a_x)
		return value


def m2_source(cell, s, out_root):
	"""(dynamics state dict path, data loader) for the cell and seed: design 10 / 07 for the six existing cells, design 16 otherwise."""
	if cell in OLD:
		p = f'{D10}/{cell}/s{s:02d}/M2_dyn.pt'
		if os.path.exists(p):
			return p, lambda: dict(np.load(f'{D10}/{cell}/s{s:02d}/data.npz'))
		src = f'{D07}/{cell}/s{s:02d}'                                       # the two design-07 cells reused by design 10
		return f'{src}/M2_CTRL_dyn.pt', lambda: cat_data([dict(np.load(f'{src}/data1.npz')), dict(np.load(f'{src}/data2_CTRL.npz'))])
	f = f'{out_root}/{cell}/s{s:02d}'
	return f'{f}/M2_dyn.pt', lambda: dict(np.load(f'{f}/data.npz'))


def cat_data(ds):
	return {k: np.concatenate([d[k] for d in ds], 0) for k in ('obs', 'state', 'act', 'rew')}


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder = f'{a.out_root}/{a.cell}/s{s:02d}'
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent0, sh = make16(task)
	domain = domain_of(task)
	dyn0 = copy.deepcopy(agent0.model._dynamics).eval()
	for p in dyn0.parameters():
		p.requires_grad_(False)
	path, _ = m2_source(a.cell, s, a.out_root)
	if not os.path.exists(path):
		assert a.cell not in OLD, f'missing design 10 model {path}'
		t0, data, info = time.time(), [], {}
		for r, base in ((1, 1000), (2, 2000)):                                # design 10: 5 + 5 episodes by the un-adapted planner, noise on
			eps, recs = run_cond(agent0, env, sh, domain, cond, a.r_eps, eval_mode=False, seed=base + s, record=True)
			data.append(stack(recs))
			info[f'round{r}'] = summary(eps)
		data = cat_data(data)
		np.savez_compressed(f'{folder}/data.npz', **data)
		m2, log = finetune_mix(cfg, data, load_sim(task), s, a.ft_steps)
		torch.save({n: v.detach().clone() for n, v in m2.model._dynamics.state_dict().items()}, path)
		info['M2'] = log[-1] if log else None
		dump(f'{folder}/collect.json', info)
		print(a.cell, s, 'collect', round(info['round1']['ret'], 1), round(info['round2']['ret'], 1), 'M2', info['M2'], f'{time.time() - t0:.0f}s', flush=True)
	sd = torch.load(path)

	def actor_of(arm):
		if arm == 'PRE':
			return agent0, None
		ag = fresh(cfg)
		ag.model._dynamics.load_state_dict(sd)
		if arm == 'PLAIN':
			return ag, None
		if arm == 'PEN1':
			ag._estimate_value = pen_value(ag, 1.0)
			return ag, None
		if arm == 'CONPEN1':
			con = ConPen(ag, dyn0, 1.0)
			return con, con
		con = ConC(ag, dyn0, {'CON': 0.5, 'CONC0': 0.0, 'CONC1': 1.0}[arm])
		return con, con

	if a.check:                                                              # c = 0.5 must equal design 06's min; c = 0 the mean of design 11
		from d06 import consensus
		js, jx = torch.randn(512, 1, device='cuda'), torch.randn(512, 1, device='cuda') * 3 + 1
		v05, v0 = mix_rule(0.5)(js, jx)[0], mix_rule(0.0)(js, jx)[0]
		a_s, a_x = js - js.mean(0), jx - jx.mean(0)
		gen = 0.5 * (a_s + a_x) - 0.5 * (a_s - a_x).abs()
		print('check: c=0.5 vs design 06 max abs', float((v05 - consensus(js, jx)[0]).abs().max()), '| general formula at c=0.5 vs min max abs', float((gen - v05).abs().max()),
			  '| c=0 vs mean max abs', float((v0 - 0.5 * (a_s + a_x)).abs().max()), flush=True)
		sh.set(cond)
		obs = env.reset()
		for arm in a.arms.split(','):
			act, _ = actor_of(arm)
			print('check', arm, 'action', act.act(obs, t0=True, eval_mode=True)[:3].tolist(), flush=True)
		return

	for arm in a.arms.split(','):
		f = f'{folder}/eval_{arm}.json'
		if os.path.exists(f):
			continue
		actor, con = actor_of(arm)
		out = dict(M2=path)
		for c, n in (('shift', a.eval_eps), ('nominal', a.nom_eps)):
			real = cond if c == 'shift' else 'nominal'
			t0 = time.time()
			eps, recs = run_cond(actor, env, sh, domain, real, n, eval_mode=True, seed=3000 + s, record=(c == 'shift'))
			out[c] = dict(cond=real, summary=summary(eps), episodes=eps, seconds=time.time() - t0)
			if c == 'shift':
				np.savez_compressed(f'{folder}/rec_{arm}.npz', rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
				out['sim_binds'] = float(np.nanmean(con.sim_binds)) if con is not None and con.sim_binds else None
			print(a.cell, s, arm, c, {k: round(v, 2) for k, v in out[c]['summary'].items()}, f'{out[c]["seconds"]:.0f}s', flush=True)
		dump(f, out)
	print(a.cell, s, 'done', a.arms, flush=True)


# ----------------------------------------------------------------------------- exact dynamics (§2.4)

_W = None


def _winit16(task, cond):
	global _W
	os.nice(10)
	torch.set_num_threads(1)
	cfg = OmegaConf.load(CFG_PATH)
	cfg.task, cfg.model_size, cfg.checkpoint = task, 5, CKPT.format(task=task)
	cfg.compile, cfg.save_video, cfg.enable_wandb = False, False, False
	cfg = parse_cfg(cfg)
	env = make_env(cfg)
	sh = Shifter16(phys_of(env), task)
	if not (isinstance(cond, str) and cond.startswith('delay')):
		sh.set(cond)                                                         # delay: physics nominal, the queue comes with every call
	env.reset()
	_W = Restorer2(env)


def _wroll16(args):
	state, pending, seqs = args
	H = seqs.shape[1]
	R, O = np.empty((len(seqs), H), np.float32), []
	for i, s in enumerate(seqs):
		outs, rews = _W.rollout_r(state, np.concatenate([pending, s], 0)[:H])  # executed: pending actions first
		R[i] = rews
		O.append(outs[-1])
	return R, np.asarray(O, np.float32)


class ExactMixin:
	@torch.no_grad()
	def values(self, state, actions):
		from d01 import _cuda
		pending = self.denv.pending(self.cfg.action_dim)
		seqs = actions.permute(1, 0, 2).cpu().numpy().astype(np.float32)
		chunks = [c for c in np.array_split(seqs, self.W) if len(c)]
		outs = self.pool.map(_wroll16, [(state, pending, c) for c in chunks], chunksize=1)
		R = np.concatenate([o[0] for o in outs])
		O = np.concatenate([o[1] for o in outs])
		G = torch.as_tensor(R @ (self.gamma ** np.arange(self.cfg.horizon)), dtype=torch.float32, device='cuda').unsqueeze(1)
		z = self.m.encode(_cuda(O), None)
		action, _ = self.m.pi(z, None)
		return G + self.gamma ** self.cfg.horizon * self.m.Q(z, action, None, return_type='avg')


class Ora16(ExactMixin, Oracle):
	pass


class ConOra16(ExactMixin, OracleCon):
	pass


def cmd_exact(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	f = f'{a.out_root}/{a.cell}/s{s:02d}/eval_{a.planner}.json'
	if os.path.exists(f):
		return
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit16, initargs=(task, cond))  # before any CUDA use
	cfg, env, agent, sh = make16(task)
	domain = domain_of(task)
	pl = (Ora16 if a.planner == 'ORA' else ConOra16)(agent, env, pool, a.workers)
	pl.denv = env
	t0 = time.time()
	eps, _ = run_cond(pl, env, sh, domain, cond, a.eps, eval_mode=True, seed=3000 + s)
	out = dict(shift=dict(cond=cond, summary=summary(eps), episodes=eps, seconds=time.time() - t0), plan_time=float(np.mean(pl.plan_time)))
	if a.planner == 'CONORA':
		out['sim_binds'] = float(np.mean(pl.sim_binds))
	dump(f, out)
	print(a.cell, s, a.planner, {k: round(v, 2) for k, v in out['shift']['summary'].items()}, f'{out["shift"]["seconds"]:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- F: identification (§2.3)

_FD = None


def _finit(task):
	global _W, _SH
	os.nice(10)
	torch.set_num_threads(1)
	cfg = OmegaConf.load(CFG_PATH)
	cfg.task, cfg.model_size, cfg.checkpoint = task, 5, CKPT.format(task=task)
	cfg.compile, cfg.save_video, cfg.enable_wandb = False, False, False
	cfg = parse_cfg(cfg)
	env = make_env(cfg)
	_SH = Shifter16(phys_of(env), task)
	env.reset()
	_W = Restorer2(env)


def _fjob(combo):
	"""Mean over transitions and observation components of the squared one-step error, each component divided by the sd of its real change."""
	_SH.set('fit:%g:%g:%g' % combo)
	S, A, O1, sd = _FD
	err = np.empty(len(S))
	for i in range(len(S)):
		pred = _W.rollout_r(S[i], A[i:i + 1])[0][-1]
		err[i] = np.mean(((pred - O1[i]) / sd) ** 2)
	return combo, float(np.mean(err)), float(np.median(err))


def cmd_fit(a):
	global _FD
	task, cond = cell_parts(a.cell)
	s = a.seed
	_, load = m2_source(a.cell, s, a.out_root)
	d = load()
	E, T = d['act'].shape[:2]
	rng = np.random.RandomState(100 + s)
	idx = rng.choice(E * T, size=min(a.n_fit, E * T), replace=False)
	e, t = idx // T, idx % T
	S, A = d['state'][e, t], d['act'][e, t].astype(np.float32)
	O0, O1 = d['obs'][e, t].astype(np.float32), d['obs'][e, t + 1].astype(np.float32)
	sd = np.maximum((O1 - O0).std(0), 1e-3).astype(np.float32)
	_FD = (S, A, O1, sd)
	combos = [(f, k, m) for f in GRID['friction'] for k in GRID['motor'] for m in GRID['mass']]
	t0 = time.time()
	with mp.get_context('fork').Pool(a.workers, initializer=_finit, initargs=(task,)) as pool:
		res = pool.map(_fjob, combos, chunksize=1)
	res.sort(key=lambda r: r[1])
	best, emin = res[0][0], res[0][1]
	near = [dict(friction=c[0], motor=c[1], mass=c[2], err=e) for c, e, _ in res if e <= NEAR * emin]
	tp = true_params(cond)
	tp_err = None
	if tp is not None:
		hit = [e for c, e, _ in res if c == (tp['friction'], tp['motor'], tp['mass'])]
		tp_err = hit[0] if hit else None
	out = dict(cell=a.cell, seed=s, n=len(S), best=dict(friction=best[0], motor=best[1], mass=best[2], err=emin), true=tp, true_err=tp_err,
			   near=near, grid=[dict(friction=c[0], motor=c[1], mass=c[2], err=e, median=md) for c, e, md in res], seconds=time.time() - t0)
	dump(f'{a.out_root}/F/{a.cell}/s{s:02d}/fit.json', out)
	print(a.cell, s, 'fit best', out['best'], 'true', tp, 'true err', tp_err, 'near', len(near), f'{out["seconds"]:.0f}s', flush=True)


# ----------------------------------------------------------------------------- F: retraining in the identified simulator (§2.3)

def cmd_retrain(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder = f'{a.out_root}/F/{a.cell}/s{s:02d}'
	if os.path.exists(f'{folder}/eval_F.json'):
		return
	b = json.load(open(f'{folder}/fit.json'))['best']
	sim = 'fit:%g:%g:%g' % (b['friction'], b['motor'], b['mass'])
	cfg, env, agent, sh = make16(task, compile_=True, seed=s)
	cfg.steps = a.steps + a.seed_steps
	cfg.buffer_size = cfg.steps + 1000
	buffer, domain = Buffer(cfg), domain_of(task)
	log = dict(cell=a.cell, sim=sim, args=vars(a), evals=[], train=[])
	step, t0, last = 0, time.time(), {}

	def evaluate():                                                          # record only: in the real condition
		reseed_env(env, 4000 + len(log['evals']))
		sh.set(cond)
		eps = [run_episode(agent, env, domain, True)[0] for _ in range(a.eval_eps)]
		sh.set(sim)
		log['evals'].append(dict(step=step, seconds=time.time() - t0, **summary(eps)))
		dump(f'{folder}/log.json', log)
		print(a.cell, s, 'F eval (real)', step, {k: round(v, 2) for k, v in log['evals'][-1].items()}, flush=True)

	sh.set(sim)
	while step <= cfg.steps:
		if step % a.eval_freq == 0:
			evaluate()
		obs, done, R = env.reset(), False, 0.0
		tds = [to_td(env, obs)]
		while not done:
			action = agent.act(obs, t0=len(tds) == 1, eval_mode=False)       # design 07 §9: the pretrained planner seeds, no burst
			obs, reward, done, info = env.step(action)
			tds.append(to_td(env, obs, action, reward))
			R += float(reward)
			if step >= a.seed_steps and buffer.num_eps > 0:
				last = agent.update(buffer)
			step += 1
		buffer.add(torch.cat(tds))
		log['train'].append(dict(step=step, ret=R, **{k: float(v) for k, v in last.items()}))
	agent.save(f'{folder}/final.pt')
	dump(f'{folder}/log.json', log)
	out = dict(sim=sim, seconds=time.time() - t0)
	for c, n in (('shift', a.final_eps), ('nominal', a.nom_eps)):
		real = cond if c == 'shift' else 'nominal'
		eps, _ = run_cond(agent, env, sh, domain, real, n, eval_mode=True, seed=3000 + s)
		out[c] = dict(cond=real, summary=summary(eps), episodes=eps)
		print(a.cell, s, 'F', c, {k: round(v, 2) for k, v in out[c]['summary'].items()}, flush=True)
	dump(f'{folder}/eval_F.json', out)


# ----------------------------------------------------------------------------- SGFT (§2.2)

def cmd_sgft(a):
	"""Port of the TD-MPC2 trainer of github.com/WEIRDLabUW/sgft (online_trainer.py with cfg.sgft): the official online loop
	(cfg.seed_steps random actions, then seed_steps updates at once, then one update per step), every stored reward replaced by
	r + discount * V(s') - V(s), V(x) = base.Q(z, base.pi(z) sampled, 'min' of two random members), base = frozen pretrained agent.
	SGFT starts from random initialisation (their train.py loads no checkpoint); WARM (ours) starts from the pretrained checkpoint."""
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder = f'{a.out_root}/SGFT/{a.cell}/{a.variant}_s{s:02d}'
	if os.path.exists(f'{folder}/done.json'):
		return
	os.makedirs(folder, exist_ok=True)
	cfg, env, base, sh = make16(task, compile_=False, seed=s)
	for p in base.model.parameters():
		p.requires_grad_(False)
	ccfg = copy.deepcopy(cfg)
	ccfg.compile = True
	torch.manual_seed(s)
	agent = TDMPC2(ccfg)
	if a.variant == 'WARM':
		agent.load(ccfg.checkpoint)
	seed_steps = a.seed_steps or cfg.seed_steps
	ccfg.steps = a.steps
	ccfg.buffer_size = a.steps + 1000
	buffer, domain = Buffer(ccfg), domain_of(task)
	budgets = sorted(b for b in a.budgets if b <= a.steps)
	log = dict(cell=a.cell, variant=a.variant, seed_steps=seed_steps, args=vars(a), evals=[], train=[])
	sh.set(cond)

	@torch.no_grad()
	def V(o):
		z = base.model.encode(o.to('cuda').unsqueeze(0), None)
		return float(base.model.Q(z, base.model.pi(z, None)[0], None))

	def evaluate(step):
		t1 = time.time()
		eps, _ = run_cond(agent, env, sh, domain, cond, a.eval_eps, eval_mode=True, seed=3000 + s)
		e = dict(step=step, seconds=time.time() - t1, **summary(eps), episodes=eps)
		if step == budgets[-1]:
			nom, _ = run_cond(agent, env, sh, domain, 'nominal', a.nom_eps, eval_mode=True, seed=3000 + s)
			e['nominal'] = dict(summary=summary(nom), episodes=nom)
			sh.set(cond)
		log['evals'].append(e)
		agent.save(f'{folder}/step{step}.pt')
		dump(f'{folder}/log.json', log)
		print(a.cell, s, a.variant, 'eval', step, {k: round(v, 2) for k, v in e.items() if not isinstance(v, (dict, list))}, flush=True)

	step, done, t0, last, tds, R = 0, True, time.time(), {}, None, 0.0
	nxt = 0
	while step <= a.steps:
		if done:
			if step > 0:
				buffer.add(torch.cat(tds))
				log['train'].append(dict(step=step, ret=R, **{k: float(v) for k, v in last.items()}))
			if nxt < len(budgets) and step >= budgets[nxt]:
				evaluate(budgets[nxt])
				nxt += 1
				if nxt == len(budgets):
					break
			obs, R = env.reset(), 0.0
			tds = [to_td(env, obs)]
		action = agent.act(obs, t0=len(tds) == 1) if step > seed_steps else env.rand_act()
		prev = obs
		obs, reward, done, info = env.step(action)
		R += float(reward)
		shaped = float(reward) + float(agent.discount) * V(obs) - V(prev)
		tds.append(to_td(env, obs, action, shaped))
		if step >= seed_steps:
			for _ in range(seed_steps if step == seed_steps else 1):
				last = agent.update(buffer)
		step += 1
	dump(f'{folder}/done.json', dict(seconds=time.time() - t0, budgets=budgets))
	print(a.cell, s, a.variant, 'done', f'{time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- summary (§4, §5)

def load_eval(path):
	return json.load(open(path)) if os.path.exists(path) else None


def ep_mean(r, c='shift', k='ret'):
	return float(np.mean([e[k] for e in r[c]['episodes']]))


def arm_vals(a, cell, arm):
	"""seed -> evaluation dict, from design 16 first, design 10 for the six existing cells."""
	out = {}
	for f in glob.glob(f'{a.out_root}/{cell}/s*/eval_{arm}.json'):
		out[int(f.split('/')[-2][1:])] = json.load(open(f))
	if cell in OLD and not out:
		pat = f'{D11}/learned/{cell}/s*/eval_MEAN.json' if arm == 'CONC0' else f'{D10}/{cell}/s*/eval_{arm}.json'   # c = 0 is design 11 C
		for f in glob.glob(pat):
			out[int(f.split('/')[-2][1:])] = json.load(open(f))
	return out


def quality(rs, c='shift'):
	K = ('falls', 'fallen_frac', 'smooth', 'effort', 'angvel_rms')
	return {k: float(np.mean([ep_mean(r, c, k) for r in rs])) for k in K}


def cmd_summarize(a):
	L, S = ['# 설계 16 결과 요약 (자동 생성, `d16.py summarize`)\n'], dict(cells={})
	sel = load_eval(f'{a.out_root}/cells.json')
	new = sel['cells'] if sel else []
	if sel:
		L.append('## 0단계 선별 (판정에 쓰지 않음; 적응 없는 planner의 바뀐 조건 return / 평소 조건 return)\n')
		L += [f'- {x}' for x in sel['lines']]
	S['new'] = new

	L.append('\n## 조건별 배치 팔 (바뀐 조건 return, (팔 − 적응 없음)의 seed 단위 t 구간 95 %)\n')
	gain, con_mean = {}, {}
	for cell in new + OLD:
		R = {arm: arm_vals(a, cell, arm) for arm in NEW_ARMS + ['ORA', 'CONORA']}
		if not R['PRE']:
			continue
		L.append(f'\n### {cell}\n')
		L.append('| 팔 | seed 수 | return | − 적응 없음 | 해 없음 | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 | 평소 조건 return |')
		L.append('|---|---|---|---|---|---|---|---|')
		S['cells'][cell] = {}
		for arm, rr in R.items():
			if not rr:
				continue
			seeds = sorted(set(rr) & set(R['PRE']))
			n_ep = len(rr[seeds[0]]['shift']['episodes'])                    # exact arms: the first 2 episodes of the same seeds
			pre = {s: float(np.mean([e['ret'] for e in R['PRE'][s]['shift']['episodes'][:n_ep]])) for s in seeds}
			v = {s: ep_mean(rr[s]) for s in seeds}
			iv = interval([v[s] - pre[s] for s in seeds])
			q = quality([rr[s] for s in seeds])
			nom = [ep_mean(rr[s], 'nominal') for s in seeds if 'nominal' in rr[s]]
			S['cells'][cell][arm] = dict(n=len(seeds), ret=float(np.mean(list(v.values()))), diff=iv, harm=harm(iv), nominal=float(np.mean(nom)) if nom else None, **q, by_seed=v)
			if arm != 'PRE':
				gain[cell, arm] = iv
			if arm == 'CON':
				con_mean[cell] = float(np.mean(list(v.values())))
			L.append(f'| {ANAME[arm]} | {len(seeds)} | {np.mean(list(v.values())):.0f} | {"–" if arm == "PRE" else fmt(iv)} | {"–" if arm == "PRE" else harm(iv)} | '
					 f'{q["falls"]:.1f} / {q["fallen_frac"]:.2f} | {q["smooth"]:.2f} / {q["effort"]:.2f} / {q["angvel_rms"]:.2f} | {"–" if not nom else f"{np.mean(nom):.0f}"} |')

	L.append('\n## F (식별 + 다시 학습)\n')
	L.append('| 조건 | seed | 고른 값 (마찰, 모터, 질량) | 참값 | 오차가 최솟값의 1.1배 안인 조합 수 | F return (seed별) | 합의 c = 0.5 seed 평균 | 평소 조건 return |')
	L.append('|---|---|---|---|---|---|---|---|')
	F = {}
	for cell in F_OLD + new:
		for f in sorted(glob.glob(f'{a.out_root}/F/{cell}/s*/fit.json')):
			fit, s = json.load(open(f)), int(f.split('/')[-2][1:])
			ev = load_eval(os.path.dirname(f) + '/eval_F.json')
			r = ep_mean(ev) if ev else None
			if r is not None:
				F.setdefault(cell, {})[s] = r
			b, t = fit['best'], fit['true']
			ts = '밖' if t is None else '%g, %g, %g' % (t['friction'], t['motor'], t['mass'])
			rs = '–' if r is None else '%.0f' % r
			ns = '–' if not ev else '%.0f' % ep_mean(ev, 'nominal')
			L.append(f'| {cell} | {s} | {b["friction"]:g}, {b["motor"]:g}, {b["mass"]:g} | {ts} | {len(fit["near"])} | {rs} | {con_mean.get(cell, float("nan")):.0f} | {ns} |')
	S['F'] = F
	q_in = [cell for cell in F_OLD if cell in F and cell in con_mean]
	if len(q_in) == len(F_OLD):
		win = all(np.mean(list(F[c].values())) > con_mean[c] for c in F_OLD)
		S['Q1_in'] = bool(win)
		L.append(f'\n- **Q1-안: {"F가 이긴다" if win else "F가 이기지 못한다"}** (F의 seed 평균 대 합의의 seed 평균: ' + ', '.join(f'{c} {np.mean(list(F[c].values())):.0f} 대 {con_mean[c]:.0f}' for c in F_OLD) + ')')
	q_out = [c for c in new if c in F and c in con_mean]
	if q_out and len(q_out) == len(new):
		lose = [c for c in new if len(F[c]) >= 3 and all(v < con_mean[c] for v in F[c].values())]
		S['Q1_out'] = lose
		L.append(f'- **Q1-밖: {"F가 지는 조건이 있다 (" + ", ".join(lose) + ")" if lose else "F가 지는 조건이 없다"}**')

	L.append('\n## Q2: 바뀐 조건 넷에서 "해 없음" (구간 아래 끝 > −25)\n')
	L.append('| 팔 | ' + ' | '.join(new) + ' | 판정 |')
	L.append('|---|' + '---|' * (len(new) + 1))
	S['Q2'] = {}
	for arm in ['CON', 'CONC0', 'CONC1', 'CONPEN1', 'PLAIN', 'PEN1']:
		if not new or not all((c, arm) in gain for c in new):
			continue
		h = [harm(gain[c, arm]) for c in new]
		v = '통과' if all(x == '통과' for x in h) else ('실패' if any(x == '실패' for x in h) else '미결')
		S['Q2'][arm] = v
		L.append(f'| {ANAME[arm]} | ' + ' | '.join(f'{x} ({gain[c, arm]["mean"]:+.0f})' for x, c in zip(h, new)) + f' | {"**" + v + "**" if arm == "CON" else v} |')

	L.append('\n## 보조: SGFT 학습 곡선 (바뀐 조건 return, 실데이터 스텝별 seed 평균)\n')
	L.append('| 조건 | 팔 | ' + ' | '.join(f'{b // 1000}k' for b in BUDGETS) + ' | 마지막 평소 조건 | 마지막 넘어져 있는 시간 |')
	L.append('|---|---|' + '---|' * (len(BUDGETS) + 2))
	S['SGFT'] = {}
	for cell in F_OLD + new:
		for var in ('SGFT', 'WARM'):
			logs = [json.load(open(f)) for f in sorted(glob.glob(f'{a.out_root}/SGFT/{cell}/{var}_s*/log.json'))]
			if not logs:
				continue
			row = []
			for b in BUDGETS:
				v = [e['ret'] for g in logs for e in g['evals'] if e['step'] == b]
				row.append(f'{np.mean(v):.0f} ({len(v)})' if v else '–')
			last = [g['evals'][-1] for g in logs if g['evals'] and 'nominal' in g['evals'][-1]]
			S['SGFT'][f'{cell}/{var}'] = [[e['step'], e['ret']] for g in logs for e in g['evals']]
			L.append(f'| {cell} | {"SGFT (공개 코드 이식)" if var == "SGFT" else "SGFT-warm (우리 변형)"} | ' + ' | '.join(row) + ' | '
					 + (f'{np.mean([e["nominal"]["summary"]["ret"] for e in last]):.0f} | {np.mean([e["fallen_frac"] for e in last]):.2f} |' if last else '– | – |'))
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)

	def P(name):
		p = sub.add_parser(name)
		p.add_argument('--out_root', default=OUT)
		return p
	p = P('screen')
	p.add_argument('--task', required=True)
	p.add_argument('--kind', choices=list(LEVELS), required=True)
	p.add_argument('--eps', type=int, default=5)
	P('select')
	p = P('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--arms', default=','.join(NEW_ARMS))
	p.add_argument('--r_eps', type=int, default=R_EPS)
	p.add_argument('--eval_eps', type=int, default=EVAL_EPS)
	p.add_argument('--nom_eps', type=int, default=NOM_EPS)
	p.add_argument('--ft_steps', type=int, default=FT_STEPS)
	p.add_argument('--check', action='store_true')
	p = P('exact')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--planner', choices=['ORA', 'CONORA'], required=True)
	p.add_argument('--eps', type=int, default=EXACT_EPS)
	p.add_argument('--workers', type=int, default=40)
	p = P('fit')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--n_fit', type=int, default=N_FIT)
	p.add_argument('--workers', type=int, default=24)
	p = P('retrain')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--steps', type=int, default=F_STEPS)
	p.add_argument('--seed_steps', type=int, default=F_SEED_STEPS)
	p.add_argument('--eval_freq', type=int, default=F_EVAL_FREQ)
	p.add_argument('--eval_eps', type=int, default=5)
	p.add_argument('--final_eps', type=int, default=EVAL_EPS)
	p.add_argument('--nom_eps', type=int, default=NOM_EPS)
	p = P('sgft')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--variant', choices=['SGFT', 'WARM'], required=True)
	p.add_argument('--steps', type=int, default=BUDGETS[-1])
	p.add_argument('--budgets', type=int, nargs='+', default=BUDGETS)
	p.add_argument('--seed_steps', type=int, default=0, help='0 = the official cfg.seed_steps (as in the public code)')
	p.add_argument('--eval_eps', type=int, default=EVAL_EPS)
	p.add_argument('--nom_eps', type=int, default=NOM_EPS)
	p = P('summarize')
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(screen=cmd_screen, select=cmd_select, run=cmd_run, exact=cmd_exact, fit=cmd_fit, retrain=cmd_retrain, sgft=cmd_sgft, summarize=cmd_summarize)[a.cmd](a)
