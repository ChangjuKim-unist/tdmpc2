"""Design 02 (research/docs/design02_selfstate_damage_remedies_ko.md).

Commands (run from ~/tdmpc2 on gpu1: GPU=N research/scripts/td.sh research/scripts/d02.py <cmd> ...):
  calib_anc   --cell C                 ANC lambda_c from design 01 checkpoints
  collect_sim --task T                 PRE in nominal physics, training mode, 20 episodes (seed 202)
  finetune    --cell C --mode es|mix|anc
  eval        --cell C --arm PRE|D500|D2000|ES|MIX|ANC|PEN1|PEN3|MIXPEN1
  probe       --cell C                 A1 (3-step errors on S_data, S_nom, S_own) and A2 (plan value gaps)
  summarize
Design 01 data and checkpoints are read only. The official TD-MPC2 code is imported, not modified.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import d01
from d01 import (make, phys_of, Shifter, cell_parts, domain_of, run_cond, summary, dump, Restorer, _cuda, enc, nxt,
				 N_TRAIN, N_HELD)
from common import math as tdmath

D01 = '/mnt/newssd/cjkim/td/d01'
OUT = '/mnt/newssd/cjkim/td/d02'
OFFICIAL_OUT = OUT
CELLS = ['walker-run__mass_1.5', 'walker-run__friction_0.15', 'quadruped-run__motor_0.35',
		 'quadruped-run__friction_0.15', 'cheetah-run__mass_2.0', 'cheetah-run__friction_0.15']
FAIL_CELLS = ['walker-run__friction_0.15', 'quadruped-run__motor_0.35', 'quadruped-run__friction_0.15',
			  'cheetah-run__friction_0.15']  # design 01 K2a failures (fixed before this design)
MODEL_ARMS = ['PRE', 'D500', 'D2000', 'ES', 'MIX', 'ANC']
REMEDY_ARMS = ['D500', 'ES', 'MIX', 'ANC', 'PEN1', 'PEN3', 'MIXPEN1']
ALL_ARMS = ['PRE', 'D2000'] + REMEDY_ARMS
PEN = {'PEN1': ('D2000', 1.0), 'PEN3': ('D2000', 3.0), 'MIXPEN1': ('MIX', 1.0)}
EVAL_SEED, EVAL_EPS = 2, 20
N_SIM, SIM_SEED, FT_SEED = 20, 202, 11
STEPS = dict(es=8000, mix=2000, anc=2000)
ARM_OF_MODE = dict(es='ES', mix='MIX', anc='ANC')


# ----------------------------------------------------------------------------- data, models, losses

def load_d01(cell):
	d = np.load(f'{D01}/{cell}/data.npz')
	return {k: d[k] for k in d.files}


def load_sim(task):
	d = np.load(f'{OUT}/sim/{task}.npz')
	return {k: d[k] for k in d.files}


def model_path(cell, arm):
	return f'{D01}/{cell}/{arm}.pt' if arm in ('D500', 'D2000') else f'{OUT}/{cell}/{arm}.pt'


def load_model_arm(agent, cfg, cell, arm):
	agent.load(cfg.checkpoint)
	if arm != 'PRE':
		agent.load(model_path(cell, arm))


def make_buffer(cfg, data, n):
	from tensordict.tensordict import TensorDict
	from common.buffer import Buffer
	cfg.buffer_size = 20_000
	buf = Buffer(cfg)
	obs = torch.from_numpy(data['obs'][:n]).float()
	act = torch.from_numpy(data['act'][:n]).float()
	rew = torch.from_numpy(data['rew'][:n]).float()
	E, T1 = obs.shape[:2]
	td = TensorDict(obs=obs, action=torch.cat([torch.full((E, 1, act.shape[-1]), float('nan')), act], 1),
					reward=torch.cat([torch.full((E, 1), float('nan')), rew], 1), terminated=torch.zeros(E, T1),
					batch_size=(E, T1))
	buf.load(td)
	return buf


def windows(data, eps, H=3):
	"""All length-H windows of the given episodes: obs (H+1) x N x D, act H x N x A."""
	obs, act = data['obs'][list(eps)], data['act'][list(eps)]
	T = act.shape[1]
	idx = np.arange(T - H + 1)
	O = np.stack([obs[:, idx + h] for h in range(H + 1)], 0).reshape(H + 1, -1, obs.shape[-1])
	A = np.stack([act[:, idx + h] for h in range(H)], 0).reshape(H, -1, act.shape[-1])
	return _cuda(O), _cuda(A)


def consistency(m, obs, action, cfg):
	"""Official consistency loss (design 01 FT-D); obs (H+1) x B x D, action H x B x A."""
	with torch.no_grad():
		next_z = m.encode(obs[1:], None)
		z = m.encode(obs[0], None)
	loss = 0
	for t in range(cfg.horizon):
		z = m.next(z, action[t], None)
		loss = loss + F.mse_loss(z, next_z[t]) * cfg.rho ** t
	return loss / cfg.horizon


def pen_value(agent, lam):
	"""FOWM eq. 4: subtract lam * std of the Q ensemble at every step of the planned value."""
	cfg, m = agent.cfg, agent.model

	@torch.no_grad()
	def _est(z, actions, task):
		G, disc = 0, 1
		for t in range(cfg.horizon):
			reward = tdmath.two_hot_inv(m.reward(z, actions[t], task), cfg)
			u = tdmath.two_hot_inv(m.Q(z, actions[t], task, return_type='all'), cfg).std(0)
			z = m.next(z, actions[t], task)
			G = G + disc * (reward - lam * u)
			disc = disc * agent.discount
		action, _ = m.pi(z, task)
		q = m.Q(z, action, task, return_type='avg')
		u = tdmath.two_hot_inv(m.Q(z, action, task, return_type='all'), cfg).std(0)
		return G + disc * (q - lam * u)
	return _est


# ----------------------------------------------------------------------------- preparation

def cmd_calib_anc(a):
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	m = agent.model
	O, A = windows(load_d01(a.cell), range(N_TRAIN))
	th0 = [p.detach().clone() for p in m._dynamics.parameters()]
	with torch.no_grad():
		l_pre = float(consistency(m, O, A, cfg))
		agent.load(model_path(a.cell, 'D2000'))
		l_ft = float(consistency(m, O, A, cfg))
		d2 = float(sum(((p - q) ** 2).sum() for p, q in zip(m._dynamics.parameters(), th0)))
	res = dict(l_pre=l_pre, l_d2000=l_ft, dtheta2=d2, lam=(l_pre - l_ft) / d2)
	assert res['lam'] > 0, res
	dump(f'{OUT}/{a.cell}/anc.json', res)
	print(a.cell, res, flush=True)


def cmd_collect_sim(a):
	cfg, env, agent = make(a.task)
	shifter, domain = Shifter(phys_of(env)), domain_of(a.task)
	eps, recs = run_cond(agent, env, shifter, domain, 'nominal', a.sim_eps, eval_mode=False, seed=SIM_SEED, record=True)
	data = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in ('obs', 'act', 'rew')}
	os.makedirs(f'{OUT}/sim', exist_ok=True)
	np.savez_compressed(f'{OUT}/sim/{a.task}.npz', **data)
	dump(f'{OUT}/sim/{a.task}.json', dict(summary=summary(eps), episodes=eps))
	print('sim', a.task, {k: v.shape for k, v in data.items()}, summary(eps), flush=True)


def cmd_finetune(a):
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	torch.manual_seed(FT_SEED)
	data = load_d01(a.cell)
	buf = make_buffer(cfg, data, N_TRAIN)
	m = agent.model
	params = list(m._dynamics.parameters())
	opt = torch.optim.Adam(params, lr=cfg.lr)
	steps = a.steps or STEPS[a.mode]
	log = []
	if a.mode == 'es':
		Ov, Av = windows(data, range(N_TRAIN, N_TRAIN + N_HELD))
		with torch.no_grad():
			v0 = float(consistency(m, Ov, Av, cfg))
		best = dict(val=v0, step=0, sd=copy.deepcopy(m.state_dict()))
		log.append(dict(step=0, val=v0))
	elif a.mode == 'mix':
		sbuf = make_buffer(cfg, load_sim(task), N_SIM)
	elif a.mode == 'anc':
		lam = json.load(open(f'{OUT}/{a.cell}/anc.json'))['lam']
		th0 = [p.detach().clone() for p in params]
	t0 = time.time()
	for step in range(1, steps + 1):
		obs, action, _, _, _ = buf.sample()
		loss = consistency(m, obs, action, cfg)
		total = loss
		if a.mode == 'mix':
			sobs, saction, _, _, _ = sbuf.sample()
			loss_s = consistency(m, sobs, saction, cfg)
			total = 0.5 * (loss + loss_s)
		elif a.mode == 'anc':
			anchor = sum(((p - q) ** 2).sum() for p, q in zip(params, th0))
			total = loss + lam * anchor
		total.backward()
		torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
		opt.step()
		opt.zero_grad(set_to_none=True)
		if step % 100 == 0:
			e = dict(step=step, consistency=float(loss))
			if a.mode == 'mix':
				e['sim'] = float(loss_s)
			if a.mode == 'anc':
				e['anchor'] = float(anchor)
			if a.mode == 'es':
				with torch.no_grad():
					e['val'] = float(consistency(m, Ov, Av, cfg))
				if e['val'] < best['val']:
					best = dict(val=e['val'], step=step, sd=copy.deepcopy(m.state_dict()))
			log.append(e)
	res = dict(log=log, seconds=time.time() - t0)
	if a.mode == 'es':
		m.load_state_dict(best['sd'])
		res.update(best_step=best['step'], best_val=best['val'])
	arm = ARM_OF_MODE[a.mode]
	agent.save(f'{OUT}/{a.cell}/{arm}.pt')
	dump(f'{OUT}/{a.cell}/finetune_{arm}.json', res)
	print(a.cell, arm, log[-1], {k: v for k, v in res.items() if k.startswith('best')}, flush=True)


# ----------------------------------------------------------------------------- evaluation

def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task)
	model_arm, lam = PEN.get(a.arm, (a.arm, 0.0))
	if model_arm != 'PRE':
		agent.load(model_path(a.cell, model_arm))
	if lam > 0:
		agent._estimate_value = pen_value(agent, lam)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	res, rec = {}, {}
	for c in ('shift', 'nominal'):
		real = cond if c == 'shift' else 'nominal'
		t0 = time.time()
		eps, recs = run_cond(agent, env, shifter, domain, real, a.eps, seed=EVAL_SEED, record=True)
		res[c] = dict(cond=real, summary=summary(eps), episodes=eps)
		for k in ('obs', 'state', 'act', 'rew'):
			rec[f'{c}_{k}'] = np.stack([np.asarray(r[k]) for r in recs[:2]])
		print(a.cell, a.arm, c, res[c]['summary'], f'{time.time() - t0:.0f}s', flush=True)
	dump(f'{OUT}/{a.cell}/eval_{a.arm}.json', res)
	np.savez_compressed(f'{OUT}/{a.cell}/rec_{a.arm}.npz', **rec)


# ----------------------------------------------------------------------------- probes (restore is measurement only)

class Restorer2(Restorer):
	def rollout_r(self, state, seq):
		with self.p.reset_context():
			self.p.set_state(state)
		self.denv._step_count = 0
		self.denv._reset_next_step = False
		outs, rews = [], []
		for act in seq:
			act = np.clip(act, -1, 1).astype(self.dmw.action_spec_dtype)
			r = 0.0
			for _ in range(2):
				ts = self.dmw.env.step(act)
				r += float(ts.reward or 0.0)
			outs.append(self.dmw._obs_to_array(ts.observation).numpy())
			rews.append(r)
		return outs, rews


def disc_return(rews, gamma):
	return float(np.sum(rews * gamma ** np.arange(len(rews))))


def anchors_data(data, A):
	"""Design 01 K2b anchors and alternatives (same RNG), from held-out episodes."""
	rng = np.random.RandomState(0)
	idx = [(e, t) for e in range(N_TRAIN, N_TRAIN + N_HELD) for t in rng.choice(np.arange(50, 497), 100, replace=False)]
	noise = rng.randn(len(idx), 8, 3, A)
	an = [dict(state=data['state'][e, t], obs=data['obs'][e, t], act=data['act'][e, t:t + 3]) for e, t in idx]
	return an, noise


def anchors_rec(rec, prefix, A, seed, gamma):
	rng = np.random.RandomState(seed)
	an = []
	for e in range(2):
		for t in rng.choice(np.arange(50, 251), 100, replace=False):
			an.append(dict(state=rec[f'{prefix}_state'][e, t], obs=rec[f'{prefix}_obs'][e, t],
						   act=rec[f'{prefix}_act'][e, t:t + 3], G=disc_return(rec[f'{prefix}_rew'][e, t:], gamma)))
	return an, rng.randn(len(an), 8, 3, A)


def true_rollouts(rs, an, noise):
	seqs, t1, t3 = [], [], []
	for i, x in enumerate(an):
		S = np.concatenate([x['act'][None], np.clip(x['act'][None] + 0.3 * noise[i], -1, 1)], 0)
		o1, o3 = [], []
		for s in S:
			outs = rs.rollout(x['state'], s)
			o1.append(outs[0])
			o3.append(outs[2])
		seqs.append(S); t1.append(o1); t3.append(o3)
	obs_t = np.asarray([x['obs'] for x in an])
	return obs_t, np.asarray(seqs), np.asarray(t1), np.asarray(t3)


@torch.no_grad()
def errs(m, obs_t, seqs, true1, true3):
	"""Per-anchor numerators/denominators of level error L and action-effect error C (design 01 §4)."""
	N, K1 = seqs.shape[:2]
	zt = enc(m, obs_t)
	z1 = enc(m, true1.reshape(N * K1, -1)).view(N, K1, -1)
	z3 = enc(m, true3.reshape(N * K1, -1)).view(N, K1, -1)
	z = zt[:, None].expand(N, K1, zt.shape[-1]).reshape(N * K1, -1)
	S = _cuda(seqs).reshape(N * K1, 3, -1)
	preds = []
	for h in range(3):
		z = nxt(m, z, S[:, h])
		preds.append(z.view(N, K1, -1))
	out = {}
	for name, pred, true in (('1', preds[0], z1), ('3', preds[2], z3)):
		dp, dt = pred[:, 1:] - pred[:, :1], true[:, 1:] - true[:, :1]
		out[name] = {k: v.cpu().numpy() for k, v in dict(
			lev_num=((pred - true) ** 2).sum(-1).sum(1), lev_den=((true - zt[:, None]) ** 2).sum(-1).sum(1),
			con_num=((dp - dt) ** 2).sum(-1).sum(1), con_den=(dt ** 2).sum(-1).sum(1)).items()}
	return out


@torch.no_grad()
def plan_seq(agent, obs):
	"""Copy of the official _plan (t0=True, eval_mode=True) that returns the chosen elite sequence (H x A)."""
	cfg, m = agent.cfg, agent.model
	z = m.encode(_cuda(obs)[None], None)
	pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device='cuda')
	_z = z.repeat(cfg.num_pi_trajs, 1)
	for t in range(cfg.horizon - 1):
		pi_actions[t], _ = m.pi(_z, None)
		_z = m.next(_z, pi_actions[t], None)
	pi_actions[-1], _ = m.pi(_z, None)
	z = z.repeat(cfg.num_samples, 1)
	mean = torch.zeros(cfg.horizon, cfg.action_dim, device='cuda')
	std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device='cuda')
	actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda')
	actions[:, :cfg.num_pi_trajs] = pi_actions
	for _ in range(cfg.iterations):
		r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device='cuda')
		actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
		value = agent._estimate_value(z, actions, None).nan_to_num(0)
		elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
		max_value = elite_value.max(0).values
		score = torch.exp(cfg.temperature * (elite_value - max_value))
		score = score / score.sum(0)
		mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
		std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
		std = std.clamp(cfg.min_std, cfg.max_std)
	idx = tdmath.gumbel_softmax_sample(score.squeeze(1))
	return torch.index_select(elite_actions, 1, idx).squeeze(1).cpu().numpy()


@torch.no_grad()
def qbar(m, cfg, z):
	mu = m.pi(z, None)[1]['mean']
	return tdmath.two_hot_inv(m.Q(z, mu, None, return_type='all'), cfg).mean(0).squeeze(-1)


@torch.no_grad()
def plan_values(m, cfg, gamma, obs_t, seqs):
	"""Predicted 3-step value: sum_h gamma^h r_hat + gamma^3 Qbar(z_3, mu_pi(z_3)); seqs N x K x 3 x A."""
	N, K = seqs.shape[:2]
	z = enc(m, obs_t)[:, None].expand(N, K, -1).reshape(N * K, -1)
	S = _cuda(seqs).reshape(N * K, 3, -1)
	V = torch.zeros(N * K, device='cuda')
	for h in range(3):
		V += gamma ** h * tdmath.two_hot_inv(m.reward(z, S[:, h], None), cfg).squeeze(-1)
		z = m.next(z, S[:, h], None)
	V += gamma ** 3 * qbar(m, cfg, z)
	return V.view(N, K).cpu().numpy()


def boot_ratio(num_a, den_a, num_b, den_b, n=1000, seed=1):
	bs = np.random.RandomState(seed)
	N = len(num_a)
	out = []
	for _ in range(n):
		i = bs.randint(0, N, N)
		out.append((num_a[i].sum() / den_a[i].sum()) / (num_b[i].sum() / den_b[i].sum()))
	return np.percentile(out, [2.5, 97.5]).tolist()


def boot_diff(x, y, n=2000, seed=2):
	bs = np.random.RandomState(seed)
	out = [x[bs.randint(0, len(x), len(x))].mean() - y[bs.randint(0, len(y), len(y))].mean() for _ in range(n)]
	return np.percentile(out, [2.5, 97.5]).tolist()


def cmd_probe(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	rs = Restorer2(env)
	gamma, A, m = float(agent.discount), cfg.action_dim, agent.model
	data = load_d01(a.cell)
	recs = {arm: dict(np.load(f'{OUT}/{a.cell}/rec_{arm}.npz')) for arm in MODEL_ARMS}
	sets = {'data': anchors_data(data, A), 'nom': anchors_rec(recs['PRE'], 'nominal', A, 2, gamma)}
	for arm in MODEL_ARMS:
		sets[f'own_{arm}'] = anchors_rec(recs[arm], 'shift', A, 1, gamma)
	# determinism: restoring a logged state and replaying the logged action reproduces the next logged observation
	ra = recs['D2000']
	det = [float(np.abs(rs.rollout(ra['shift_state'][0, t], [ra['shift_act'][0, t]])[0] - ra['shift_obs'][0, t + 1]).max())
		   for t in range(50, 70)]
	t0 = time.time()
	truth = {name: true_rollouts(rs, an, noise) for name, (an, noise) in sets.items()}
	print('true rollouts', f'{time.time() - t0:.0f}s', flush=True)
	raw, res = {}, dict(det_max_abs=max(det), gamma=gamma, a1={}, a2={})
	E = {}
	for arm in MODEL_ARMS:
		load_model_arm(agent, cfg, a.cell, arm)
		for name, (obs_t, seqs, t1, t3) in truth.items():
			E[arm, name] = errs(m, obs_t, seqs, t1, t3)
			for h in ('1', '3'):
				for k, v in E[arm, name][h].items():
					raw[f'a1/{arm}/{name}/{h}/{k}'] = v
	for name in truth:
		res['a1'][name] = {}
		for arm in MODEL_ARMS:
			e, p = E[arm, name]['3'], E['PRE', name]['3']
			L, C = e['lev_num'].sum() / e['lev_den'].sum(), e['con_num'].sum() / e['con_den'].sum()
			kappa = L / (p['lev_num'].sum() / p['lev_den'].sum())
			ci = boot_ratio(e['lev_num'], e['lev_den'], p['lev_num'], p['lev_den']) if arm != 'PRE' else [1.0, 1.0]
			res['a1'][name][arm] = dict(L=float(L), C=float(C), kappa=float(kappa), kappa_ci=ci)
	d01k2 = json.load(open(f'{D01}/{a.cell}/probe_k2.json'))
	res['d01_check'] = {arm: dict(L_d01=d01k2[arm]['3']['L'], L_now=res['a1']['data'][arm]['L'],
									C_d01=d01k2[arm]['3']['C'], C_now=res['a1']['data'][arm]['C']) for arm in ('PRE', 'D2000')}
	# A2: each model arm plans at its own visited states
	rng = np.random.RandomState(3)
	for arm in MODEL_ARMS:
		load_model_arm(agent, cfg, a.cell, arm)
		torch.manual_seed(5)
		an = sets[f'own_{arm}'][0]
		noise = rng.randn(len(an), 8, 3, A)
		seqs, R3, O3 = [], [], []
		for i, x in enumerate(an):
			s0 = plan_seq(agent, x['obs'])
			S = np.concatenate([s0[None], np.clip(s0[None] + 0.3 * noise[i], -1, 1)], 0)
			r3, o3 = [], []
			for s in S:
				outs, rews = rs.rollout_r(x['state'], s)
				r3.append(disc_return(np.asarray(rews), gamma))
				o3.append(outs[2])
			seqs.append(S); R3.append(r3); O3.append(o3)
		seqs, R3, O3 = np.asarray(seqs), np.asarray(R3), np.asarray(O3)
		N, K = seqs.shape[:2]
		Vhat = plan_values(m, cfg, gamma, np.asarray([x['obs'] for x in an]), seqs)
		V3 = R3 + gamma ** 3 * qbar(m, cfg, enc(m, O3.reshape(N * K, -1))).view(N, K).cpu().numpy()
		G = np.asarray([x['G'] for x in an])
		gm, gv = Vhat - V3, V3[:, 0] - G
		sel = gm[:, 0] - gm[:, 1:].mean(1)
		raw.update({f'a2/{arm}/Vhat': Vhat, f'a2/{arm}/V3': V3, f'a2/{arm}/G': G})
		res['a2'][arm] = dict(gm=float(gm[:, 0].mean()), gv=float(gv.mean()), total=float((Vhat[:, 0] - G).mean()),
							  sel=float(sel.mean()), sel_ci=boot_diff(sel, np.zeros(1)), Vhat=float(Vhat[:, 0].mean()),
							  V3=float(V3[:, 0].mean()), G=float(G.mean()))
		print(a.cell, arm, 'A2', res['a2'][arm], flush=True)
	for arm in MODEL_ARMS:
		if arm != 'PRE':
			gm_a = raw[f'a2/{arm}/Vhat'][:, 0] - raw[f'a2/{arm}/V3'][:, 0]
			gm_p = raw['a2/PRE/Vhat'][:, 0] - raw['a2/PRE/V3'][:, 0]
			res['a2'][arm]['gm_minus_pre_ci'] = boot_diff(gm_a, gm_p)
	np.savez_compressed(f'{OUT}/{a.cell}/probe_raw.npz', **raw)
	dump(f'{OUT}/{a.cell}/probe.json', res)
	print(a.cell, 'A1 own_D2000', res['a1']['own_D2000']['D2000'], 'd01 check', res['d01_check'], 'det', res['det_max_abs'], flush=True)


# ----------------------------------------------------------------------------- summary (registered criteria, design 02 §5)

def _load(path):
	return json.load(open(path)) if os.path.exists(path) else None


def _ep(ev, c, k):
	return np.array([e[k] for e in ev[c]['episodes']])


def cmd_summarize(a):
	cells = a.cells.split(',') if a.cells else CELLS
	fail_cells = [c for c in cells if c in FAIL_CELLS]
	bs = np.random.RandomState(0)
	ev = {(c, arm): _load(f'{OUT}/{c}/eval_{arm}.json') for c in cells for arm in ALL_ARMS}
	pr = {c: _load(f'{OUT}/{c}/probe.json') for c in cells}
	rho, dfall = {}, {}
	L = ['# 설계 02 결과 요약 (자동 생성, `d02.py summarize`)', '',
		 '평가: 평가 시드 2, 조건마다 20 에피소드. ρ = (팔 − PRE) / (평소 PRE − PRE), 바뀐 조건 return 기준. [ ]는 에피소드 bootstrap 95 % 구간.', '']
	rows_ret, rows_rho, rows_nom, rows_q = [], [], [], []
	for c in cells:
		pre = ev[c, 'PRE']
		if pre is None:
			continue
		rp, rn = _ep(pre, 'shift', 'ret'), _ep(pre, 'nominal', 'ret')
		fp = _ep(pre, 'shift', 'falls').mean()
		cr, cq, cn, cro = [], [], [], []
		for arm in ALL_ARMS:
			e = ev[c, arm]
			if e is None:
				cr.append('-'); cro.append('-'); cn.append('-'); cq.append('-')
				continue
			rf = _ep(e, 'shift', 'ret')
			r = (rf.mean() - rp.mean()) / (rn.mean() - rp.mean())
			boots = []
			for _ in range(2000):
				i = bs.randint(0, len(rp), len(rp))
				den = rn[i].mean() - rp[i].mean()
				boots.append((rf[i].mean() - rp[i].mean()) / den if abs(den) > 1e-6 else np.nan)
			lo, hi = np.nanpercentile(boots, [2.5, 97.5])
			rho[c, arm], dfall[c, arm] = float(r), float(_ep(e, 'shift', 'falls').mean() - fp)
			ok = rho[c, arm] >= 0.25 and dfall[c, arm] < 0.5
			gain = rho[c, arm] >= 0.5 and dfall[c, arm] < 0.5
			tag = '이득' if gain else ('막음' if ok else '실패')
			cr.append(f"{rf.mean():.0f} ({_ep(e, 'shift', 'falls').mean():.1f})")
			cro.append(f'{r:.2f} [{lo:.2f}, {hi:.2f}] {tag if arm != "PRE" else ""}')
			cn.append(f"{_ep(e, 'nominal', 'ret').mean():.0f} ({_ep(e, 'nominal', 'falls').mean():.1f})")
			cq.append(f"{_ep(e, 'shift', 'smooth').mean():.3f} / {_ep(e, 'shift', 'effort').mean():.2f} / {_ep(e, 'shift', 'angvel_rms').mean():.2f}")
		rows_ret.append(f'| {c} | ' + ' | '.join(cr) + ' |')
		rows_rho.append(f'| {c} | ' + ' | '.join(cro[1:]) + ' |')
		rows_nom.append(f'| {c} | ' + ' | '.join(cn) + ' |')
		rows_q.append(f'| {c} | ' + ' | '.join(cq) + ' |')
	hdr = '| 셀 | ' + ' | '.join(ALL_ARMS) + ' |'
	sep = '|---' * (len(ALL_ARMS) + 1) + '|'
	L += ['## 팔별 바뀐 조건 return (넘어짐 사건/에피소드)', '', hdr, sep] + rows_ret
	L += ['', '## 팔별 회복 비율 ρ와 판정 (막음: ρ ≥ 0.25이고 넘어짐 증가 < 0.5, 이득: ρ ≥ 0.5이고 넘어짐 증가 < 0.5)', '',
		  '| 셀 | ' + ' | '.join(ALL_ARMS[1:]) + ' |', '|---' * len(ALL_ARMS) + '|'] + rows_rho
	L += ['', '## 팔별 평소 조건 return (넘어짐): 보고만', '', hdr, sep] + rows_nom
	L += ['', '## 팔별 동작 품질, 바뀐 조건 (부드러움 / 노력 / 흔들림, 모두 작을수록 부드럽고 적음)', '', hdr, sep] + rows_q
	out = {}
	# R
	r_flags = [bool(rho[c, 'D2000'] < 0.25 or dfall[c, 'D2000'] >= 0.5) for c in cells if (c, 'D2000') in rho]
	out['R'] = dict(n=sum(r_flags), of=len(r_flags), yes=bool(len(r_flags) >= 4 and sum(r_flags) >= 3))
	# B
	b_rows, b1_best, b2_best = [], 0, 0
	out['B'] = {}
	for arm in REMEDY_ARMS:
		ok = [c for c in cells if (c, arm) in rho and rho[c, arm] >= 0.25 and dfall[c, arm] < 0.5]
		gain = [c for c in cells if (c, arm) in rho and rho[c, arm] >= 0.5 and dfall[c, arm] < 0.5]
		n = sum((c, arm) in rho for c in cells)
		out['B'][arm] = dict(ok=len(ok), gain=len(gain), n=n)
		b1_best, b2_best = max(b1_best, len(ok)), max(b2_best, len(gain))
		b_rows.append(f'| {arm} | {len(ok)}/{n} | {len(gain)}/{n} |')
	out['B1'] = bool(b1_best >= 5)
	out['B2'] = bool(b2_best >= 4)
	L += ['', '## B: 처방 팔별 "막음"·"이득" 셀 수', '', '| 처방 팔 | 막음 | 이득 |', '|---|---|---|'] + b_rows
	# A1
	a1_rows, a1_flags, own_rows = [], [], []
	for c in cells:
		p = pr[c]
		if not p:
			continue
		k = {s: p['a1'][s]['D2000'] for s in ('data', 'own_D2000', 'nom')}
		a1_rows.append(f"| {c}{' (실패 셀)' if c in fail_cells else ''} | " + ' | '.join(
			f"{k[s]['kappa']:.2f} [{k[s]['kappa_ci'][0]:.2f}, {k[s]['kappa_ci'][1]:.2f}]" for s in ('data', 'own_D2000', 'nom')) + ' |')
		if c in fail_cells:
			a1_flags.append(bool(k['own_D2000']['kappa'] >= 1.0))
		own_rows.append(f'| {c} | ' + ' | '.join(
			f"{p['a1'][f'own_{arm}'][arm]['kappa']:.2f}" for arm in MODEL_ARMS[1:]) + ' |')
	out['A1'] = dict(n=sum(a1_flags), of=len(a1_flags), yes=bool(len(a1_flags) >= 3 and sum(a1_flags) >= 3))
	L += ['', '## A1: D2000 모델의 3스텝 수준 오차 / 사전학습 모델의 것 (κ, 1 이상이면 그 상태에서 적응 이득 없음)', '',
		  '| 셀 | S_data (PRE가 모은 held-out) | S_own(D2000) (D2000 planner가 간 상태) | S_nom (평소 조건 걸음 상태) |', '|---|---|---|---|'] + a1_rows
	L += ['', '## 보고: 각 팔 모델의 κ, 그 팔 planner가 간 상태에서', '', '| 셀 | ' + ' | '.join(MODEL_ARMS[1:]) + ' |', '|---' * len(MODEL_ARMS) + '|'] + own_rows
	# A2
	a2_rows, a2_flags, p9 = [], [], []
	for c in cells:
		p = pr[c]
		if not p:
			continue
		q, d = p['a2']['PRE'], p['a2']['D2000']
		ci = d['gm_minus_pre_ci']
		a2_rows.append(f"| {c}{' (실패 셀)' if c in fail_cells else ''} | {q['gm']:.2f} | {d['gm']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {q['gv']:.2f} | {d['gv']:.2f} | {q['sel']:.2f} | {d['sel']:.2f} |")
		if c in fail_cells:
			a2_flags.append(bool(d['gm'] - q['gm'] > 0 and ci[0] > 0))
		p9.append(bool(abs(q['gv']) > abs(q['gm'])))
	out['A2'] = dict(n=sum(a2_flags), of=len(a2_flags), yes=bool(len(a2_flags) >= 3 and sum(a2_flags) >= 3))
	L += ['', '## A2: 계획 가치 과대평가 (보상 단위). 모델 부분 g_m = 예측 − 3스텝 실제, 가치 부분 g_v = 3스텝 실제 − 실현 return', '',
		  '| 셀 | g_m PRE | g_m D2000 | D2000 − PRE 95 % 구간 | g_v PRE | g_v D2000 | 선택 지수 PRE | 선택 지수 D2000 |', '|---|---|---|---|---|---|---|---|'] + a2_rows
	# predictions
	es_steps = [(_load(f'{OUT}/{c}/finetune_ES.json') or {}).get('best_step') for c in cells]
	lam = [(_load(f'{OUT}/{c}/anc.json') or {}).get('lam') for c in cells]
	p7 = [rho[c, 'MIX'] > rho[c, 'D2000'] for c in cells if (c, 'MIX') in rho and (c, 'D2000') in rho]
	p8 = [rho[c, 'PEN1'] > rho[c, 'D2000'] for c in cells if (c, 'PEN1') in rho and (c, 'D2000') in rho]
	yn = lambda b: '그렇다' if b else '아니다'
	L += ['', '## 판정', '',
		  f"- R 실패 재현: D2000 실패 {out['R']['n']}/{out['R']['of']}셀 (기준 3) → {yn(out['R']['yes'])}",
		  f"- A1 자기 상태에서 이득이 사라진다: {out['A1']['n']}/{out['A1']['of']} 실패 셀 (기준 3) → {yn(out['A1']['yes'])}",
		  f"- A2 FT planner가 모델 오차를 더 부풀린다: {out['A2']['n']}/{out['A2']['of']} 실패 셀 (기준 3) → {yn(out['A2']['yes'])}",
		  f"- B1 알려진 처방이 실패를 막는다: 가장 많은 처방 팔의 막음 {b1_best}/{len(cells)} (기준 5) → {yn(out['B1'])}",
		  f"- B2 알려진 처방이 이득까지 낸다: 가장 많은 처방 팔의 이득 {b2_best}/{len(cells)} (기준 4) → {yn(out['B2'])}",
		  '', '## 예측 대조', '',
		  f"- P6 ES 단계 ≥ 2000: {sum(s is not None and s >= 2000 for s in es_steps)}/{len(cells)} (단계: {es_steps})",
		  f'- P7 MIX ρ > D2000 ρ: {sum(p7)}/{len(p7)}', f'- P8 PEN1 ρ > D2000 ρ: {sum(p8)}/{len(p8)}',
		  f'- P9 PRE에서 |g_v| > |g_m|: {sum(p9)}/{len(p9)}', '', f'ANC λ_c: {lam}']
	if any(pr.values()):
		L += ['', '## 확인: S_data에서 설계 01 K2b 값 재현 (3스텝 L, C)', '']
		for c in cells:
			if pr[c]:
				L.append(f"- {c}: " + ', '.join(f"{arm} L {v['L_d01']:.3f}→{v['L_now']:.3f}, C {v['C_d01']:.3f}→{v['C_now']:.3f}" for arm, v in pr[c]['d01_check'].items()) + f", 복원 결정성 최대 차이 {pr[c]['det_max_abs']:.1e}")
	out.update(rho={f'{c}|{arm}': v for (c, arm), v in rho.items()}, dfall={f'{c}|{arm}': v for (c, arm), v in dfall.items()},
			   es_steps=es_steps, anc_lam=lam)
	dst = os.path.join(d01.REPO, 'research', 'results', 'd02') if OUT == OFFICIAL_OUT else os.path.join(OUT, 'summary')
	os.makedirs(dst, exist_ok=True)
	open(os.path.join(dst, 'summary.md'), 'w').write('\n'.join(L) + '\n')
	dump(os.path.join(dst, 'summary.json'), out)
	print('\n'.join(L))


# ----------------------------------------------------------------------------- main

def main():
	global D01, OUT
	ap = argparse.ArgumentParser()
	ap.add_argument('cmd')
	ap.add_argument('--task')
	ap.add_argument('--cell')
	ap.add_argument('--cells', default=None)
	ap.add_argument('--mode', default='es')
	ap.add_argument('--arm', default='PRE')
	ap.add_argument('--eps', type=int, default=EVAL_EPS)
	ap.add_argument('--sim_eps', type=int, default=N_SIM)
	ap.add_argument('--steps', type=int, default=None, help='smoke tests only')
	ap.add_argument('--out_root', default=None, help='smoke tests only')
	ap.add_argument('--d01_root', default=None, help='smoke tests only')
	a = ap.parse_args()
	if a.out_root:
		OUT = a.out_root
	if a.d01_root:
		D01 = a.d01_root
	dict(calib_anc=cmd_calib_anc, collect_sim=cmd_collect_sim, finetune=cmd_finetune, eval=cmd_eval, probe=cmd_probe,
		 summarize=cmd_summarize)[a.cmd](a)


if __name__ == '__main__':
	main()
