"""Design 01 (research/docs/design01_adapt_failure_recurrence_ko.md).

Commands (run from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d01.py screen   --task walker-run
  GPU=N research/scripts/td.sh research/scripts/d01.py select
  GPU=N research/scripts/td.sh research/scripts/d01.py collect  --cell walker-run__friction_0.3
  GPU=N research/scripts/td.sh research/scripts/d01.py finetune --cell ... --mode dyn|all
  GPU=N research/scripts/td.sh research/scripts/d01.py eval     --cell ... --arm PRE|D500|D2000|D8000|A2000 --conds shift,nominal [--iters 16] [--dither 0.1]
  GPU=N research/scripts/td.sh research/scripts/d01.py probe_k2 --cell ...
  GPU=N research/scripts/td.sh research/scripts/d01.py probe_k1 --cell ...
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'disable')
import numpy as np
import torch
import torch.nn.functional as F

import hydra.utils
hydra.utils.get_original_cwd = lambda: os.getcwd()  # parse_cfg outside a hydra app
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from common.buffer import Buffer
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CFG_PATH = os.path.join(REPO, 'tdmpc2', 'config.yaml')
CKPT = '/mnt/newssd/cjkim/tdmpc2_ckpt/{task}-1.pt'
OUT = os.environ.get('D01_OUT', '/mnt/newssd/cjkim/td/d01')  # smoke tests write elsewhere
TASKS = ['walker-run', 'quadruped-run', 'cheetah-run']
LADDER = {'friction': [0.5, 0.3, 0.15], 'motor': [0.7, 0.5, 0.35], 'mass': [1.5, 2.0, 3.0]}
EXTEND = {'friction': 0.08, 'motor': 0.25, 'mass': 4.0}
KIND_ORDER = ['friction', 'motor', 'mass']
N_TRAIN, N_HELD = 10, 2
DELTAS = [0.05, 0.1, 0.2]
torch.set_num_threads(2)


# ----------------------------------------------------------------------------- setup

def make(task, compile_=True, iterations=6, seed=1):
	cfg = OmegaConf.load(CFG_PATH)
	cfg.task, cfg.model_size, cfg.checkpoint = task, 5, CKPT.format(task=task)
	cfg.compile, cfg.iterations, cfg.seed = compile_, iterations, seed
	cfg.save_video, cfg.enable_wandb = False, False
	cfg = parse_cfg(cfg)
	set_seed(seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	return cfg, env, agent


def dmw_of(env):
	e = env
	while not hasattr(e, '_obs_to_array'):
		e = e.env
	return e


def phys_of(env):
	return dmw_of(env).env.physics


class Shifter:
	"""Sets physics-model multipliers from stored originals (so changes never compound)."""

	def __init__(self, physics):
		m, n = physics.model, physics.named.model
		self.p = physics
		self.o = dict(fric=m.geom_friction[:, 0].copy(), gain=m.actuator_gainprm[:, 0].copy(),
					  bias=m.actuator_biasprm[:, 0:3].copy(), mass=float(n.body_mass['torso']),
					  inertia=n.body_inertia['torso'].copy())

	def set(self, cond):
		kind, k = parse_cond(cond)
		m, n, o = self.p.model, self.p.named.model, self.o
		m.geom_friction[:, 0] = o['fric']
		m.actuator_gainprm[:, 0] = o['gain']
		m.actuator_biasprm[:, 0:3] = o['bias']
		n.body_mass['torso'] = o['mass']
		n.body_inertia['torso'] = o['inertia']
		if kind == 'friction':
			m.geom_friction[:, 0] = o['fric'] * k
		elif kind == 'motor':
			m.actuator_gainprm[:, 0] = o['gain'] * k
			m.actuator_biasprm[:, 0:3] = o['bias'] * k
		elif kind == 'mass':
			n.body_mass['torso'] = o['mass'] * k
			n.body_inertia['torso'] = o['inertia'] * k


def parse_cond(cond):
	if cond in (None, 'nominal'):
		return None, 1.0
	kind, k = cond.split('_')
	return kind, float(k)


def cell_parts(cell):
	task, cond = cell.split('__')
	return task, cond


def domain_of(task):
	return task.split('-')[0]


def fallen(domain, p):
	if domain == 'walker':
		return p.torso_height() < 0.6
	if domain == 'quadruped':
		return p.torso_upright() < 0
	return np.cos(p.named.data.qpos['rooty']) < 0  # cheetah: back towards the floor


def angvel(domain, p):
	if domain == 'quadruped':
		return float(np.linalg.norm(p.data.qvel[3:6]))
	return abs(float(p.named.data.qvel['rooty']))


def reseed_env(env, seed):
	dmw = dmw_of(env)
	dmw.env._env.task._random = np.random.RandomState(seed)


# ----------------------------------------------------------------------------- episodes

def run_episode(agent, env, domain, eval_mode, dither=0.0, rng=None, record=False):
	p = phys_of(env)
	obs, done, t, R = env.reset(), False, 0, 0.0
	acts, av = [], []
	falls, fallen_steps, prev = 0, 0, False
	rec = dict(obs=[obs.numpy()], state=[p.get_state().copy()], act=[], rew=[]) if record else None
	while not done:
		a = agent.act(obs, t0=t == 0, eval_mode=eval_mode)
		if dither > 0:
			u = rng.choice([-1.0, 1.0], size=a.shape[0]).astype(np.float32)
			a = (a + dither * torch.from_numpy(u)).clamp(-1, 1)
		obs, r, done, _ = env.step(a)
		R += float(r)
		t += 1
		f = bool(fallen(domain, p))
		falls += int(f and not prev)
		fallen_steps += int(f)
		prev = f
		av.append(angvel(domain, p))
		acts.append(a.numpy())
		if record:
			rec['obs'].append(obs.numpy())
			rec['state'].append(p.get_state().copy())
			rec['act'].append(a.numpy())
			rec['rew'].append(float(r))
	A = np.asarray(acts)
	d2 = A[2:] - 2 * A[1:-1] + A[:-2]
	m = dict(ret=R, falls=falls, fallen_frac=fallen_steps / t, smooth=float((d2 ** 2).sum(1).mean()),
			 effort=float((A ** 2).sum(1).mean()), angvel_rms=float(np.sqrt(np.mean(np.square(av)))))
	return m, rec


def run_cond(agent, env, shifter, domain, cond, n, eval_mode=True, dither=0.0, seed=1, record=False):
	shifter.set(cond)
	reseed_env(env, seed)
	torch.manual_seed(seed)
	rng = np.random.RandomState(seed + 7)
	eps, recs = [], []
	for _ in range(n):
		m, rec = run_episode(agent, env, domain, eval_mode, dither, rng, record)
		eps.append(m)
		recs.append(rec)
	return eps, recs


def summary(eps):
	return {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}


def dump(path, obj):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------- K0

def cmd_screen(a):
	cfg, env, agent = make(a.task)
	shifter, domain = Shifter(phys_of(env)), domain_of(a.task)
	conds = ['nominal'] + [f'{k}_{v}' for k in KIND_ORDER for v in LADDER[k]]
	if a.extend:
		conds = [f'{k}_{EXTEND[k]}' for k in KIND_ORDER]
	res = {}
	for c in conds:
		t0 = time.time()
		eps, _ = run_cond(agent, env, shifter, domain, c, a.eps)
		res[c] = dict(summary=summary(eps), episodes=eps)
		print(f'{a.task} {c}: R {res[c]["summary"]["ret"]:.1f} falls {res[c]["summary"]["falls"]:.2f} ({time.time()-t0:.0f}s)', flush=True)
	dump(f'{OUT}/screen/{a.task}{"_ext" if a.extend else ""}.json', res)


def pick_cells(screens):
	"""Registered rule (design 01 §3 K0)."""
	cells = []
	for task in TASKS:
		res = screens.get(task, {})
		if 'nominal' not in res:
			continue
		r_nom = res['nominal']['summary']['ret']
		best = {}
		for c, v in res.items():
			kind, k = parse_cond(c)
			if kind is None:
				continue
			q = v['summary']['ret'] / r_nom
			if 0.3 <= q <= 0.85 and (kind not in best or abs(q - 0.6) < abs(best[kind][1] - 0.6)):
				best[kind] = (c, q)
		ranked = sorted(best.items(), key=lambda kv: (abs(kv[1][1] - 0.6), KIND_ORDER.index(kv[0])))
		for kind, (c, q) in ranked[:2]:
			cells.append(dict(cell=f'{task}__{c}', task=task, cond=c, q=q, r_nom=r_nom))
	return cells


def cmd_select(a):
	screens = {}
	for task in TASKS:
		p = f'{OUT}/screen/{task}.json'
		if os.path.exists(p):
			screens[task] = json.load(open(p))
		pe = f'{OUT}/screen/{task}_ext.json'
		if os.path.exists(pe):
			screens.setdefault(task, {}).update(json.load(open(pe)))
	cells = pick_cells(screens)
	dump(f'{OUT}/cells.json', cells)
	for c in cells:
		print(f"{c['cell']}: q {c['q']:.2f}")
	print('n_cells', len(cells))


# ----------------------------------------------------------------------------- data + fine-tuning

def cmd_collect(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	eps, recs = run_cond(agent, env, shifter, domain, cond, N_TRAIN + N_HELD, eval_mode=False, seed=101, record=True)
	data = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in recs[0]}
	os.makedirs(f'{OUT}/{a.cell}', exist_ok=True)
	np.savez_compressed(f'{OUT}/{a.cell}/data.npz', **data)
	dump(f'{OUT}/{a.cell}/collect.json', dict(summary=summary(eps), episodes=eps))
	print('collected', {k: v.shape for k, v in data.items()}, summary(eps), flush=True)


def load_data(cell):
	d = np.load(f'{OUT}/{cell}/data.npz')
	return {k: d[k] for k in d.files}


def make_buffer(cfg, data):
	cfg.buffer_size = 20_000
	buf = Buffer(cfg)
	obs = torch.from_numpy(data['obs'][:N_TRAIN]).float()           # E x (T+1) x obs
	act = torch.from_numpy(data['act'][:N_TRAIN]).float()           # E x T x act
	rew = torch.from_numpy(data['rew'][:N_TRAIN]).float()           # E x T
	E, T1 = obs.shape[:2]
	nan_a = torch.full((E, 1, act.shape[-1]), float('nan'))
	nan_r = torch.full((E, 1), float('nan'))
	td = TensorDict(obs=obs, action=torch.cat([nan_a, act], 1), reward=torch.cat([nan_r, rew], 1),
					terminated=torch.zeros(E, T1), batch_size=(E, T1))
	buf.load(td)
	return buf


def cmd_finetune(a):
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	torch.manual_seed(11)
	buf = make_buffer(cfg, load_data(a.cell))
	steps = [500, 2000, 8000] if a.mode == 'dyn' else [2000]
	log = []
	if a.mode == 'dyn':
		model = agent.model
		opt = torch.optim.Adam(model._dynamics.parameters(), lr=cfg.lr)
		for step in range(1, steps[-1] + 1):
			obs, action, _, _, _ = buf.sample()
			with torch.no_grad():
				next_z = model.encode(obs[1:], None)
				z = model.encode(obs[0], None)
			loss = 0
			for t in range(cfg.horizon):
				z = model.next(z, action[t], None)
				loss = loss + F.mse_loss(z, next_z[t]) * cfg.rho ** t
			loss = loss / cfg.horizon
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model._dynamics.parameters(), cfg.grad_clip_norm)
			opt.step()
			opt.zero_grad(set_to_none=True)
			if step % 100 == 0:
				log.append(dict(step=step, consistency=float(loss)))
			if step in steps:
				agent.save(f'{OUT}/{a.cell}/D{step}.pt')
				print(a.cell, 'D', step, float(loss), flush=True)
	else:
		for step in range(1, steps[-1] + 1):
			info = agent.update(buf)
			if step % 100 == 0:
				log.append({k: float(v) for k, v in info.items()} | dict(step=step))
		agent.save(f'{OUT}/{a.cell}/A2000.pt')
		print(a.cell, 'A2000', log[-1], flush=True)
	dump(f'{OUT}/{a.cell}/finetune_{a.mode}.json', log)


def load_arm(agent, cell, arm):
	if arm != 'PRE':
		agent.load(f'{OUT}/{cell}/{arm}.pt')


# ----------------------------------------------------------------------------- evaluation

def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, iterations=a.iters)
	load_arm(agent, a.cell, a.arm)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	res = {}
	for c in a.conds.split(','):
		real = cond if c == 'shift' else 'nominal'
		eps, _ = run_cond(agent, env, shifter, domain, real, a.eps, dither=a.dither)
		res[c] = dict(cond=real, summary=summary(eps), episodes=eps)
		print(a.cell, a.arm, c, f'iters{a.iters}', f'dither{a.dither}', res[c]['summary'], flush=True)
	tag = f'{a.arm}_it{a.iters}' + (f'_d{a.dither}' if a.dither > 0 else '')
	dump(f'{OUT}/{a.cell}/eval_{tag}.json', res)


# ----------------------------------------------------------------------------- restore-based probes

class Restorer:
	"""Measurement only: restores the simulator state and steps the official wrapper (2 physics steps per action)."""

	def __init__(self, env):
		self.dmw = dmw_of(env)
		self.p = self.dmw.env.physics
		self.denv = self.dmw.env._env

	def rollout(self, state, seq):
		with self.p.reset_context():
			self.p.set_state(state)
		self.denv._step_count = 0
		self.denv._reset_next_step = False
		outs = []
		for act in seq:
			act = np.clip(act, -1, 1).astype(self.dmw.action_spec_dtype)
			for _ in range(2):
				ts = self.dmw.env.step(act)
			outs.append(self.dmw._obs_to_array(ts.observation).numpy())
		return outs


def _cuda(x):
	if isinstance(x, torch.Tensor):
		return x.to('cuda', torch.float32)
	return torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cuda')


@torch.no_grad()
def enc(model, obs):
	return model.encode(_cuda(obs), None)


@torch.no_grad()
def nxt(model, z, act):
	return model.next(z, _cuda(act), None)


def cmd_probe_k2(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	rs = Restorer(env)
	data = load_data(a.cell)
	rng = np.random.RandomState(0)
	anchors = [(e, t) for e in range(N_TRAIN, N_TRAIN + N_HELD) for t in rng.choice(np.arange(50, 497), 100, replace=False)]
	K, A = 8, cfg.action_dim
	noise = rng.randn(len(anchors), K, 3, A)
	# determinism check: restore s_t, apply logged a_t, compare with logged obs_{t+1}
	det = [np.abs(rs.rollout(data['state'][e, t], [data['act'][e, t]])[0] - data['obs'][e, t + 1]).max() for e, t in anchors[:20]]
	true1, true3, obs_t, seqs = [], [], [], []
	for i, (e, t) in enumerate(anchors):
		seq0 = data['act'][e, t:t + 3]
		S = np.concatenate([seq0[None], np.clip(seq0[None] + 0.3 * noise[i], -1, 1)], 0)  # (K+1) x 3 x A
		o1, o3 = [], []
		for s in S:
			outs = rs.rollout(data['state'][e, t], s)
			o1.append(outs[0])
			o3.append(outs[2])
		true1.append(o1); true3.append(o3); obs_t.append(data['obs'][e, t]); seqs.append(S)
	true1, true3, obs_t, seqs = map(np.asarray, (true1, true3, obs_t, seqs))  # N x (K+1) x obs ; N x (K+1) x 3 x A
	N = len(anchors)
	res = dict(det_max_abs=float(np.max(det)), n=N)
	for arm in a.arms.split(','):
		agent.load(cfg.checkpoint)
		load_arm(agent, a.cell, arm)
		m = agent.model
		zt = enc(m, obs_t)                                                    # N x L
		z1 = enc(m, true1.reshape(N * (K + 1), -1)).view(N, K + 1, -1)
		z3 = enc(m, true3.reshape(N * (K + 1), -1)).view(N, K + 1, -1)
		z = zt[:, None].expand(N, K + 1, zt.shape[-1]).reshape(N * (K + 1), -1)
		S = torch.as_tensor(seqs, dtype=torch.float32, device='cuda').reshape(N * (K + 1), 3, A)
		preds = []
		for h in range(3):
			z = nxt(m, z, S[:, h])
			preds.append(z.view(N, K + 1, -1))
		out = {}
		for name, pred, true in (('1', preds[0], z1), ('3', preds[2], z3)):
			lev_num = ((pred - true) ** 2).sum(-1)                           # N x (K+1)
			lev_den = ((true - zt[:, None]) ** 2).sum(-1)
			dp, dt = pred[:, 1:] - pred[:, :1], true[:, 1:] - true[:, :1]
			con_num = ((dp - dt) ** 2).sum(-1)                               # N x K
			con_den = (dt ** 2).sum(-1)
			bs = np.random.RandomState(1)
			def ratio(num, den, idx=None):
				return float(num[idx].sum() / den[idx].sum()) if idx is not None else float(num.sum() / den.sum())
			L, C = ratio(lev_num, lev_den), ratio(con_num, con_den)
			Lb, Cb = [], []
			for _ in range(1000):
				idx = torch.as_tensor(bs.randint(0, N, N), device='cuda')
				Lb.append(ratio(lev_num, lev_den, idx)); Cb.append(ratio(con_num, con_den, idx))
			out[name] = dict(L=L, C=C, L_ci=np.percentile(Lb, [2.5, 97.5]).tolist(), C_ci=np.percentile(Cb, [2.5, 97.5]).tolist())
		res[arm] = out
		print(a.cell, arm, out, flush=True)
	dump(f'{OUT}/{a.cell}/probe_k2.json', res)


def cmd_probe_k1(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	rs = Restorer(env)
	m = agent.model
	data = load_data(a.cell)
	E, T1 = data['obs'].shape[:2]
	T = T1 - 1
	X = data['obs'][:, :T].reshape(E * T, -1)
	mu, sd = X.mean(0), X.std(0) + 1e-6
	Xs = torch.as_tensor((X - mu) / sd, device='cuda')
	ep_idx = np.repeat(np.arange(E), T)
	t_idx = np.tile(np.arange(T), E)
	rng = np.random.RandomState(0)
	pool = np.where(t_idx >= 50)[0]
	anchors = rng.choice(pool, 300, replace=False)
	d = torch.cdist(Xs[anchors], Xs)                                     # 300 x E*T
	same = torch.as_tensor((ep_idx[anchors][:, None] == ep_idx[None]) & (np.abs(t_idx[anchors][:, None] - t_idx[None]) < 20), device='cuda')
	d[same] = float('inf')
	match = d.argmin(1).cpu().numpy()
	match_dist = d.min(1).values.cpu().numpy()
	allowed = (~same).cpu().numpy()
	rand = np.array([rng.choice(np.where(allowed[i])[0]) for i in range(len(anchors))])
	step_dist = np.linalg.norm(((data['obs'][:, 1:T + 1] - data['obs'][:, :T]) / sd).reshape(E * T, -1), axis=1)
	U = rng.choice([-1.0, 1.0], size=(len(anchors), cfg.action_dim))

	def st(i):
		return data['state'][ep_idx[i], t_idx[i]]

	def ac(i):
		return data['act'][ep_idx[i], t_idx[i]]

	res = dict(match_dist_median=float(np.median(match_dist)), step_dist_median=float(np.median(step_dist)),
			   rand_dist_median=float(np.median(np.linalg.norm((X[anchors] - X[rand]) / sd, axis=1))))
	for delta in DELTAS:
		zp, zm, zpm, zr, at_m, am_m, ar_m = [], [], [], [], [], [], []
		for j, i in enumerate(anchors):
			u = U[j]
			ap, am = np.clip(ac(i) + delta * u, -1, 1), np.clip(ac(i) - delta * u, -1, 1)
			amm = np.clip(ac(match[j]) - delta * u, -1, 1)
			amr = np.clip(ac(rand[j]) - delta * u, -1, 1)
			zp.append(rs.rollout(st(i), [ap])[0])
			zm.append(rs.rollout(st(i), [am])[0])
			zpm.append(rs.rollout(st(match[j]), [amm])[0])
			zr.append(rs.rollout(st(rand[j]), [amr])[0])
			at_m.append(am); am_m.append(amm); ar_m.append(amr)
		zp, zm, zpm, zr = (enc(m, np.asarray(x)) for x in (zp, zm, zpm, zr))
		o_t = data['obs'][ep_idx[anchors], t_idx[anchors]]
		o_m = data['obs'][ep_idx[match], t_idx[match]]
		o_r = data['obs'][ep_idx[rand], t_idx[rand]]
		f_t = nxt(m, enc(m, o_t), np.asarray(at_m))
		f_m = nxt(m, enc(m, o_m), np.asarray(am_m))
		f_r = nxt(m, enc(m, o_r), np.asarray(ar_m))
		y = zp - zm
		e = zm - zpm
		ec = e - (f_t - f_m)
		er = zm - zr
		erc = er - (f_t - f_r)

		def snr(sig, noise):
			return float(torch.sqrt((sig ** 2).sum(-1).mean() / (noise ** 2).sum(-1).mean()))
		res[str(delta)] = dict(R=snr(y, e), R_c=snr(y, ec), R_rand=snr(y, er), R_rand_c=snr(y, erc),
							   signal_rms=float(torch.sqrt((y ** 2).sum(-1).mean())))
		print(a.cell, 'delta', delta, res[str(delta)], flush=True)
	dump(f'{OUT}/{a.cell}/probe_k1.json', res)


# ----------------------------------------------------------------------------- summary (registered criteria, design 01 §5)

def _load(path):
	return json.load(open(path)) if os.path.exists(path) else None


def _rets(ev, c):
	return np.array([e['ret'] for e in ev[c]['episodes']])


def _mean(ev, c, k):
	return float(np.mean([e[k] for e in ev[c]['episodes']]))


def cmd_summarize(a):
	cells = _load(f'{OUT}/cells.json') or []
	bs = np.random.RandomState(0)
	L = ['# 설계 01 결과 요약 (자동 생성, `d01.py summarize`)', '']
	# K0
	L += ['## K0: 사전학습 planner의 return 비율 q (바뀐 조건 / 평소 조건, 10 에피소드)', '', '| 과제 | 조건 | q | 넘어짐/에피소드 |', '|---|---|---|---|']
	for task in TASKS:
		for suffix in ('', '_ext'):
			s = _load(f'{OUT}/screen/{task}{suffix}.json')
			if not s:
				continue
			r_nom = s['nominal']['summary']['ret'] if 'nominal' in s else _load(f'{OUT}/screen/{task}.json')['nominal']['summary']['ret']
			for c, v in s.items():
				L.append(f"| {task} | {c} | {v['summary']['ret'] / r_nom:.2f} | {v['summary']['falls']:.1f} |")
	L += ['', '고른 셀: ' + ', '.join(f"{c['cell']} (q {c['q']:.2f})" for c in cells), '']
	out = dict(cells=[c['cell'] for c in cells])
	# K2a
	k2a_rows, k2a_flags = [], []
	arm_rows, nom_rows, it_rows, q_rows, k2b_rows, k1_rows = [], [], [], [], [], []
	k2b_flags, k1_flags, p2, p4, p5, p7, p8 = [], [], [], [], [], [], []
	for c in cells:
		cell = c['cell']
		pre = _load(f'{OUT}/{cell}/eval_PRE_it6.json')
		d2 = _load(f'{OUT}/{cell}/eval_D2000_it6.json')
		if pre and d2:
			rp, rf, rn = _rets(pre, 'shift'), _rets(d2, 'shift'), _rets(pre, 'nominal')
			rho = (rf.mean() - rp.mean()) / (rn.mean() - rp.mean())
			boots = []
			for _ in range(2000):
				i = bs.randint(0, len(rp), len(rp))
				den = rn[i].mean() - rp[i].mean()
				boots.append((rf[i].mean() - rp[i].mean()) / den if abs(den) > 1e-6 else np.nan)
			lo, hi = np.nanpercentile(boots, [2.5, 97.5])
			fp, ff = _mean(pre, 'shift', 'falls'), _mean(d2, 'shift', 'falls')
			fail = bool(rho < 0.25 or ff - fp >= 0.5)
			k2a_flags.append(fail)
			k2a_rows.append(f'| {cell} | {rn.mean():.0f} | {rp.mean():.0f} | {rf.mean():.0f} | {rho:.2f} [{lo:.2f}, {hi:.2f}] | {fp:.1f} → {ff:.1f} | {"실패" if fail else "-"} |')
		evs = {arm: _load(f'{OUT}/{cell}/eval_{arm}_it6.json') for arm in ('PRE', 'D500', 'D2000', 'D8000', 'A2000')}
		if all(evs.values()):
			arm_rows.append(f'| {cell} | ' + ' | '.join(f"{_rets(evs[k], 'shift').mean():.0f} ({_mean(evs[k], 'shift', 'falls'):.1f})" for k in evs) + ' |')
			nom_rows.append(f'| {cell} | ' + ' | '.join(f"{_rets(evs[k], 'nominal').mean():.0f} ({_mean(evs[k], 'nominal', 'falls'):.1f})" for k in evs) + ' |')
			q_rows.append(f'| {cell} | ' + ' | '.join(f"{_mean(evs[k], 'shift', 'smooth'):.3f} / {_mean(evs[k], 'shift', 'effort'):.2f} / {_mean(evs[k], 'shift', 'angvel_rms'):.2f}" for k in evs) + ' |')
			p4.append(_rets(evs['D8000'], 'shift').mean() < _rets(evs['D2000'], 'shift').mean())
		p16, d16 = _load(f'{OUT}/{cell}/eval_PRE_it16.json'), _load(f'{OUT}/{cell}/eval_D2000_it16.json')
		if pre and d2 and p16 and d16:
			dp = _rets(pre, 'shift').mean() - _rets(p16, 'shift').mean()
			dd = _rets(d2, 'shift').mean() - _rets(d16, 'shift').mean()
			p5.append(dd > dp)
			it_rows.append(f'| {cell} | {dp:.0f} | {dd:.0f} |')
		k2 = _load(f'{OUT}/{cell}/probe_k2.json')
		if k2 and 'PRE' in k2 and 'D2000' in k2:
			Lp, Cp, Lf, Cf = k2['PRE']['3']['L'], k2['PRE']['3']['C'], k2['D2000']['3']['L'], k2['D2000']['3']['C']
			rL, rC = 1 - Lf / Lp, 1 - Cf / Cp
			flag = bool(rL >= 0.3 and rC <= rL / 2)
			k2b_flags.append(flag)
			p2.append(rL >= 0.3)
			k2b_rows.append(f"| {cell} | {Lp:.3f} → {Lf:.3f} | {Cp:.3f} → {Cf:.3f} | {rL:.2f} | {rC:.2f} | {k2['PRE']['1']['C']:.3f} → {k2['D2000']['1']['C']:.3f} | {'그렇다' if flag else '-'} |")
		k1 = _load(f'{OUT}/{cell}/probe_k1.json')
		if k1 and pre:
			ok = False
			r0, f0 = _rets(pre, 'shift').mean(), _mean(pre, 'shift', 'falls')
			for dlt in DELTAS:
				ev = _load(f'{OUT}/{cell}/eval_PRE_it6_d{dlt}.json')
				if ev is None:
					continue
				drop = 1 - _rets(ev, 'shift').mean() / r0
				dfall = _mean(ev, 'shift', 'falls') - f0
				r = k1[str(dlt)]
				good = drop <= 0.05 and dfall <= 0.2 and max(r['R'], r['R_c']) >= 1.0
				ok = ok or good
				p7.append(r['R_rand'] < r['R'])
				if dlt == 0.1:
					p8.append(drop <= 0.05 and dfall <= 0.2)
				k1_rows.append(f"| {cell} | {dlt} | {r['R']:.2f} | {r['R_c']:.2f} | {r['R_rand']:.2f} | {100 * drop:.1f} % | {dfall:+.1f} | {'예' if good else '-'} |")
			k1_flags.append(ok)
	n = len(cells)

	def verdict(flags, need):
		return f'{sum(flags)}/{len(flags)} (기준 {need}) → ' + ('충족' if len(flags) >= 4 and sum(flags) >= need else '미충족')
	need_half, need_23 = int(np.ceil(n / 2)), int(np.ceil(2 * n / 3))
	L += ['## K2a: 적은 데이터 dynamics fine-tuning(FT-D2000)이 잃은 return을 되찾나', '',
		  'ρ = (FT − PRE) / (평소 − PRE), 바뀐 조건 return 기준. [ ]는 에피소드 짝 bootstrap 95 % 구간. 넘어짐은 에피소드당 사건 수.', '',
		  '| 셀 | 평소 PRE | 바뀐 PRE | 바뀐 FT-D2000 | ρ | 넘어짐 PRE → FT | 판정 |', '|---|---|---|---|---|---|---|'] + k2a_rows
	L += ['', f'K2a "적응 실패가 있다": {verdict(k2a_flags, need_half)}', '']
	L += ['## 팔별 바뀐 조건 return (넘어짐/에피소드)', '', '| 셀 | PRE | D500 | D2000 | D8000 | A2000 |', '|---|---|---|---|---|---|'] + arm_rows
	L += ['', '## 팔별 평소 조건 return (넘어짐/에피소드): 보존', '', '| 셀 | PRE | D500 | D2000 | D8000 | A2000 |', '|---|---|---|---|---|---|'] + nom_rows
	L += ['', '## 팔별 동작 품질, 바뀐 조건 (부드러움 / 노력 / 흔들림, 모두 작을수록 부드럽고 적음)', '', '| 셀 | PRE | D500 | D2000 | D8000 | A2000 |', '|---|---|---|---|---|---|'] + q_rows
	L += ['', '## MPPI 반복 6 → 16의 return 하락 (바뀐 조건)', '', '| 셀 | PRE | FT-D2000 |', '|---|---|---|'] + it_rows
	L += ['', '## K2b: 3스텝 수준 오차 L과 행동 효과 오차 C (held-out 상태 200개, 사전학습 → FT-D2000)', '',
		  '| 셀 | L | C | r_L | r_C | 1스텝 C | 판정 |', '|---|---|---|---|---|---|---|'] + k2b_rows
	L += ['', f'K2b "수준만 고쳐진다": {verdict(k2b_flags, need_23)}', '']
	L += ['## K1: 보폭 반복 fork의 신호 대 잡음비와 흔들기 비용', '',
		  'R = 실측 fork 대비 짝 불일치 잡음, R_c = 사전학습 모델로 보정, R_rand = 무작위 짝. 비용 = PRE planner에 흔들기를 넣었을 때 return 감소와 넘어짐 증가.', '',
		  '| 셀 | δ | R | R_c | R_rand | return 감소 | 넘어짐 증가 | 기준 충족 |', '|---|---|---|---|---|---|---|---|'] + k1_rows
	L += ['', f'K1 "쓸 만하다": {verdict(k1_flags, need_23)}', '']
	L += ['## 예측 대조', '', f'- P0 셀 ≥ 4: {n}', f'- P2 r_L ≥ 0.3: {sum(p2)}/{len(p2)}', f'- P4 D8000 < D2000: {sum(p4)}/{len(p4)}',
		  f'- P5 반복 16 하락이 FT에서 더 큼: {sum(p5)}/{len(p5)}', f'- P7 R_rand < R (셀×δ): {sum(p7)}/{len(p7)}', f'- P8 δ=0.1 비용 기준 안: {sum(p8)}/{len(p8)}']
	out.update(k2a=k2a_flags, k2b=k2b_flags, k1=k1_flags)
	official = OUT == '/mnt/newssd/cjkim/td/d01'
	dst = os.path.join(REPO, 'research', 'results', 'd01') if official else os.path.join(OUT, 'summary')
	os.makedirs(dst, exist_ok=True)
	open(os.path.join(dst, 'summary.md'), 'w').write('\n'.join(L) + '\n')
	dump(os.path.join(dst, 'summary.json'), out)
	print('\n'.join(L))


# ----------------------------------------------------------------------------- main

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('cmd')
	ap.add_argument('--task')
	ap.add_argument('--cell')
	ap.add_argument('--mode', default='dyn')
	ap.add_argument('--arm', default='PRE')
	ap.add_argument('--arms', default='PRE,D500,D2000,D8000,A2000')
	ap.add_argument('--conds', default='shift,nominal')
	ap.add_argument('--iters', type=int, default=6)
	ap.add_argument('--dither', type=float, default=0.0)
	ap.add_argument('--eps', type=int, default=10)
	ap.add_argument('--extend', action='store_true')
	ap.add_argument('--out_root', default=None, help='smoke tests only')
	a = ap.parse_args()
	if a.out_root:
		global OUT
		OUT = a.out_root
	dict(screen=cmd_screen, select=cmd_select, collect=cmd_collect, finetune=cmd_finetune, eval=cmd_eval,
		 probe_k2=cmd_probe_k2, probe_k1=cmd_probe_k1, summarize=cmd_summarize)[a.cmd](a)


if __name__ == '__main__':
	main()
