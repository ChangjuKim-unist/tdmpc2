"""Design 06, unregistered analysis with a realised-return yardstick (등록 안 함; research/docs/prior_consensus_survey_ko.md §8).

d06_attr.py scored plans with exact 3-step dynamics + the pretrained Q. That yardstick cannot see value staleness.
Here a plan is scored by what actually happens: restore a recorded state under the shifted physics, execute the plan's
3 actions, then let one fixed reference planner (the un-adapted planner, eval mode) run for --hc steps, and use the
realised discounted reward sum. No training, no new data for fine-tuning.

Stage 1  plans    (one process, CPU pool for exact scoring): at each sampled state run four planners from the same warm
                  start and keep K sampled elites of each:  S un-adapted, L learned consensus (pretrained vs MIX),
                  O oracle consensus (pretrained vs exact), E exact dynamics only (design 04 planner).
                  Model scores J_s, J_x, J_o of every kept plan are stored.
Stage 2  roll     (many processes, one shard each): realised rollouts, R repeats with common random numbers across plans.
Stage 3  report

How the result will be read (written before running)
  1. quadruped motor, fallen + tilted states: D = V(L) - V(S). State-bootstrap CI below 0 -> the closed-loop harm (-48) is
     real and comes from single decisions in these postures. CI covering 0 -> single decisions do not explain it
     (noise or compounding remains).
  2. For each pair (L-S, O-S, O-L, O-E): if the mean difference in V and in J_o have opposite signs, the pretrained-Q
     yardstick is wrong there (value side), and section 7 conclusions about that pair are withdrawn.
  3. cheetah friction: V(O) - V(L) with CI above 0 -> the closed-loop gap (572 vs 383) is a model-accuracy gap at the
     decision level. V(O) - V(E) with CI above 0 -> the benefit of consensus over exact-only (572 vs 276) is real at the
     decision level.
  4. Rank correlation of each scorer with V over the kept plans of a state replaces the numbers of section 7.5;
     the repeat-to-repeat correlation of V is reported as the noise ceiling.
  5. J_o - (V + gamma^n Q_end) by posture / phase: how far the pretrained Q is from realised return.
Limit: V follows the un-adapted planner after the plan, so it is the value of one deviation, not of a policy.

Addendum (2026-11-27, 등록 안 함; survey section 9): states visited by the exact-only planner itself
  Why: section 8.5 found that at states visited by learned consensus one exact-only plan is BETTER than an un-adapted plan
  (+1.42), while the exact-only planner in closed loop is 177 below the un-adapted planner. The performance difference
  lemma says the closed-loop difference J(E) - J(S) equals the sum of decision-level advantages measured at the states
  that E visits, followed by S. Section 8 sampled the wrong states for this pair.
  What: cheetah friction, --src ORA (design 04 exact-only records), 96 states with t < 150 (start-up) and 64 with t >= 150,
  warm start from the exact-only planner, same four planners, same rollout. Two continuations: the un-adapted planner
  (--cont planner, the one the lemma is exact for) and the deterministic policy prior mean (--cont pi, no sampling noise,
  but its reference policy is the prior, so the lemma is only approximate there).
  How it will be read (written before running)
  A1. Start-up states (t < 150), planner continuation: V(E) - V(S) with CI below 0 -> the start-up delay hypothesis is
      confirmed: the closed-loop loss of exact-only comes from decisions at its own start-up states. CI covering 0 or
      above 0 -> not confirmed.
  A2. Lemma total: (500 / 3) x weighted mean of the undiscounted window-reward difference E - S over all sampled states,
      with CI, against the closed-loop -177. The 150-step window and t <= 343 sampling truncate late effects (bias to 0).
      CI containing -177 or below -> decision-level account is consistent in size.
  A3. Policy-prior continuation is secondary. Same sign as A1 with CI excluding 0 -> supports A1. Opposite sign ->
      planner continuation governs, and the disagreement is reported.
  A4. V(O) - V(E) at start-up states with CI above 0 -> at the states where exact-only loses, agreement with the sim
      model corrects it at decision level (572 vs 276).
  Prediction: A1 negative with CI below 0 (0.65); A2 CI contains -177 (0.5); A4 above 0 (0.6).

  GPU=N research/scripts/td.sh research/scripts/d06_cont.py plans --cell quadruped-run__motor_0.35
  GPU=N research/scripts/td.sh research/scripts/d06_cont.py roll --cell quadruped-run__motor_0.35 --shard 0 --shards 16
  GPU=N research/scripts/td.sh research/scripts/d06_cont.py report --cell quadruped-run__motor_0.35 --shards 16
"""
import argparse
import copy
import glob
import multiprocessing as mp
import time

import numpy as np
import torch

from d01 import Shifter, cell_parts, dmw_of, domain_of, fallen, make, phys_of
from d02 import model_path
from d04 import Oracle, _winit
from d06 import consensus, plan_value
from d06_attr import BINS, UP, mppi, spearman

D06 = '/mnt/newssd/cjkim/td/d06'
D04 = '/mnt/newssd/cjkim/td/d04'
PL = ('S', 'L', 'O', 'E')
GAMMA = 0.99


def cmd_plans(a):
	task, cond = cell_parts(a.cell)
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, cond))  # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	ora = Oracle(agent, env, pool, a.workers)
	dyn0 = copy.deepcopy(agent.model._dynamics).eval()
	agent.load(model_path(a.cell, 'MIX'))
	dynx = copy.deepcopy(agent.model._dynamics).eval()
	m = agent.model
	if a.src == 'ORA':                                                   # states visited by the exact-only planner (design 04)
		rec = np.load(f'{D04}/{a.cell}/rec_ORA.npz')
		obs, st = rec['obs'], rec['state']
	else:
		rec = np.load(f'{a.out_root}/{a.cell}/rec_CONMIX.npz')
		obs, st = rec['shift_obs'], rec['shift_state']
	E, T = obs.shape[0], obs.shape[1] - 1
	rs = np.random.RandomState(a.seed)
	ee, tt = np.meshgrid(np.arange(E), np.arange(2, T - a.hc - 5), indexing='ij')
	ee, tt = ee.ravel(), tt.ravel()
	if domain_of(task) == 'quadruped':                                   # mostly fallen / tilted postures
		key = obs[ee, tt, UP]
		strata = [(key < 0, 0.4), ((key >= 0) & (key < 0.8), 0.4), (key >= 0.8, 0.2)]
	else:                                                                # start-up phase and the rest
		strata = [(tt < a.t_split, a.share0), (tt >= a.t_split, 1 - a.share0)]
	pick, binid, wt = [], [], []
	for b, (mask, share) in enumerate(strata):
		idx = np.where(mask)[0]
		k = min(len(idx), int(round(a.n * share)))
		pick += list(rs.choice(idx, k, replace=False))
		binid += [b] * k
		wt += [mask.mean() / max(k, 1)] * k
	pick, binid, wt = np.array(pick), np.array(binid), np.array(wt)
	N, H, A, S, K = len(pick), cfg.horizon, cfg.action_dim, cfg.num_samples, a.k
	js_fn = lambda zs, act: plan_value(agent, zs, act, dyn0).nan_to_num(0)
	jx_fn = lambda zs, act: plan_value(agent, zs, act, dynx).nan_to_num(0)
	plans = np.zeros((N, len(PL), K, H, A), np.float32)
	score = np.zeros((N, len(PL), K, 3), np.float32)                       # J_s, J_x, J_o of each kept plan
	t0 = time.time()
	for i, j in enumerate(pick):
		e, t = int(ee[j]), int(tt[j])
		torch.manual_seed(a.seed * 100003 + i)
		z_of = lambda tau: m.encode(torch.as_tensor(obs[e, tau], dtype=torch.float32, device='cuda').unsqueeze(0), None)
		con_x = lambda zs: (lambda act: (lambda js, jx: (consensus(js, jx)[0], (js, jx)))(js_fn(zs, act), jx_fn(zs, act)))
		mean0 = torch.zeros(H, A, device='cuda')
		for tau in (t - 2, t - 1):                                         # warm start as in deployment
			zz = z_of(tau)
			if a.src == 'ORA':                                             # the deployed planner was exact-only
				r = mppi(agent, zz, mean0, lambda act: (ora.values(st[e, tau], act).nan_to_num(0), None), dyn0)
			else:
				r = mppi(agent, zz, mean0, con_x(zz.repeat(S, 1)), dynx)
			mean0 = torch.zeros(H, A, device='cuda')
			mean0[:-1] = r['mean'][1:]
		z, state = z_of(t), st[e, t]
		zs = z.repeat(S, 1)
		exact = lambda act: ora.values(state, act).nan_to_num(0)
		def con_o(act):
			js, jo = js_fn(zs, act), exact(act)
			return consensus(js, jo)[0], (js, jo)
		res = dict(S=mppi(agent, z, mean0, lambda act: (js_fn(zs, act), None), dyn0),
				   L=mppi(agent, z, mean0, con_x(zs), dynx),
				   O=mppi(agent, z, mean0, con_o, dyn0),
				   E=mppi(agent, z, mean0, lambda act: (lambda jo: (jo, (None, jo)))(exact(act)), dyn0))
		for pi_, name in enumerate(PL):
			pl = res[name]
			sel = pl['elite'][torch.multinomial(pl['w'], K, replacement=False)]
			act = pl['actions'][:, sel]                                    # H x K x A
			zk = z.repeat(K, 1)
			jo = pl['parts'][1][sel] if name in ('O', 'E') else exact(act)
			plans[i, pi_] = act.permute(1, 0, 2).cpu().numpy()
			score[i, pi_, :, 0] = js_fn(zk, act).squeeze(1).cpu().numpy()
			score[i, pi_, :, 1] = jx_fn(zk, act).squeeze(1).cpu().numpy()
			score[i, pi_, :, 2] = jo.squeeze(1).cpu().numpy()
		if i % 10 == 0:
			print(f'{a.cell} plans {i}/{N} {time.time() - t0:.0f}s', flush=True)
	np.savez_compressed(f'{a.out_root}/{a.cell}/cont{a.tag}_plans.npz', plans=plans, score=score, state=st[ee[pick], tt[pick]],
						obs=obs[ee[pick], tt[pick]], ep=ee[pick], t=tt[pick], bin=binid, wt=wt)
	pool.close()
	print(a.cell, 'plans done', N, flush=True)


def cmd_roll(a):
	task, cond = cell_parts(a.cell)
	d = np.load(f'{a.out_root}/{a.cell}/cont{a.tag}_plans.npz')
	plans, states = d['plans'], d['state']
	ctag = '' if a.cont == 'planner' else '_' + a.cont
	N, P, K, H, A = plans.shape
	cfg, env, agent = make(task, compile_=False)
	domain = domain_of(task)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	dmw = dmw_of(env)
	p, denv, m = dmw.env.physics, dmw.env._env, agent.model

	def restore(state):
		with p.reset_context():
			p.set_state(state)
		denv._step_count = 0
		denv._reset_next_step = False

	def step(act):
		act = np.clip(np.asarray(act), -1, 1).astype(dmw.action_spec_dtype)
		r = 0.0
		for _ in range(2):
			ts = dmw.env.step(act)
			r += float(ts.reward or 0.0)
		return dmw._obs_to_array(ts.observation), r

	mine = [i for i in range(N) if i % a.shards == a.shard]
	L = H + a.hc
	rew = np.zeros((len(mine), a.reps, P, K, L), np.float32)
	fal = np.zeros((len(mine), a.reps, P, K), np.float32)
	qend = np.zeros((len(mine), a.reps, P, K), np.float32)
	t0 = time.time()
	for n, i in enumerate(mine):
		for r in range(a.reps):
			for pi_ in range(P):
				for k in range(K):
					torch.manual_seed(900001 * (r + 1) + i)                # common random numbers across plans
					restore(states[i])
					f = 0
					for h in range(H):
						obs, rew[n, r, pi_, k, h] = step(plans[i, pi_, k, h])
						f += int(bool(fallen(domain, p)))
					for c in range(a.hc):
						if a.cont == 'pi':                                         # deterministic policy prior mean
							with torch.no_grad():
								act = m.pi(m.encode(obs.to('cuda').unsqueeze(0), None), None)[1]['mean'][0].cpu()
						else:
							act = agent.act(obs, t0=c == 0, eval_mode=True)
						obs, rew[n, r, pi_, k, H + c] = step(act.numpy())
						f += int(bool(fallen(domain, p)))
					fal[n, r, pi_, k] = f / L
					with torch.no_grad():
						z = m.encode(obs.to('cuda').unsqueeze(0), None)
						qend[n, r, pi_, k] = float(m.Q(z, m.pi(z, None)[0], None, return_type='avg'))
		print(f'{a.cell} shard {a.shard} {n + 1}/{len(mine)} {time.time() - t0:.0f}s', flush=True)
	np.savez_compressed(f'{a.out_root}/{a.cell}/cont{a.tag}_roll{ctag}_{a.shard:02d}.npz', idx=np.array(mine), rew=rew, fal=fal, qend=qend)
	print(a.cell, 'shard', a.shard, 'done', flush=True)


def boot(x, w, n=4000, seed=0):
	"""Weighted mean over states with a state-bootstrap 95 % interval."""
	rs = np.random.RandomState(seed)
	w = w / w.sum()
	est = float((x * w).sum())
	idx = rs.randint(0, len(x), (n, len(x)))
	bs = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
	lo, hi = np.percentile(bs, [2.5, 97.5])
	return est, float(lo), float(hi)


def cmd_report(a):
	d = np.load(f'{a.out_root}/{a.cell}/cont{a.tag}_plans.npz')
	score, binid, wt = d['score'], d['bin'], d['wt']
	ctag = '' if a.cont == 'planner' else '_' + a.cont
	N, P, K = score.shape[:3]
	files = sorted(glob.glob(f'{a.out_root}/{a.cell}/cont{a.tag}_roll{ctag}_[0-9]*.npz'))
	rew = fal = qend = None
	for f in files:
		s = np.load(f)
		if rew is None:
			rew = np.full((N,) + s['rew'].shape[1:], np.nan, np.float32)
			fal = np.full((N,) + s['fal'].shape[1:], np.nan, np.float32)
			qend = np.full((N,) + s['qend'].shape[1:], np.nan, np.float32)
		rew[s['idx']], fal[s['idx']], qend[s['idx']] = s['rew'], s['fal'], s['qend']
	assert not np.isnan(rew).any(), 'missing shards'
	L = rew.shape[-1]
	disc = GAMMA ** np.arange(L)
	V = (rew * disc).sum(-1)                                               # N x R x P x K
	Vb = V + GAMMA ** L * qend
	U = rew.sum(-1)                                                        # undiscounted reward over the window
	Vm, Um, Fm = V.mean((1, 3)), U.mean((1, 3)), fal.mean((1, 3))          # N x P
	J = score.mean(2)                                                      # N x P x 3 (J_s, J_x, J_o)
	quad = a.cell.startswith('quadruped')
	ix = {n: i for i, n in enumerate(PL)}
	names = ('fallen', 'tilted', 'upright') if quad else (f't<{a.t_split}', f't>={a.t_split}')
	groups = [('all', np.ones(N, bool))] + [(nm, binid == b) for b, nm in enumerate(names)]
	if quad:
		groups.insert(1, ('fallen+tilted', binid < 2))
	print(f'\n== source {a.src}, continuation {a.cont}')
	for u in ('L', 'O', 'E'):                                              # performance difference lemma total (3-action plans)
		e = boot(Um[:, ix[u]] - Um[:, ix['S']], wt)
		print(f'   lemma total {u}-S: {500 / 3 * e[0]:+.0f} [{500 / 3 * e[1]:+.0f}, {500 / 3 * e[2]:+.0f}] (exact only when the states are those visited by {u})')
	print(f'\n== {a.cell}: {N} states, {rew.shape[1]} repeats, {K} elites per planner, window {L} steps')
	for gname, g in groups:
		if not g.any():
			continue
		print(f'-- {gname} (n {int(g.sum())}, share of visited states {wt[g].sum() / wt.sum():.3f})')
		print('   mean window reward (undiscounted): ' + '  '.join(f'{n} {boot(Um[g, ix[n]], wt[g])[0]:.1f}' for n in PL)
			  + ' | fallen time: ' + '  '.join(f'{n} {boot(Fm[g, ix[n]], wt[g])[0]:.2f}' for n in PL))
		for u, v in (('L', 'S'), ('O', 'S'), ('O', 'L'), ('O', 'E'), ('E', 'S')):
			dv = boot(Vm[g, ix[u]] - Vm[g, ix[v]], wt[g])
			du = boot(Um[g, ix[u]] - Um[g, ix[v]], wt[g])
			dj = boot(J[g, ix[u], 2] - J[g, ix[v], 2], wt[g])
			flag = 'SIGN DIFFERS' if dv[0] * dj[0] < 0 else ''
			print(f'   {u}-{v}: realised V {dv[0]:+.2f} [{dv[1]:+.2f}, {dv[2]:+.2f}]  window reward {du[0]:+.1f} [{du[1]:+.1f}, {du[2]:+.1f}]'
				  f'  exact-3-step score {dj[0]:+.2f} [{dj[1]:+.2f}, {dj[2]:+.2f}] {flag}')
		Vp = V.mean(1).reshape(N, P * K)                                   # plans of a state, mean over repeats
		sc = score.reshape(N, P * K, 3)
		rho = np.array([[spearman(torch.as_tensor(sc[i, :, c]), torch.as_tensor(Vp[i])) for c in range(3)] for i in np.where(g)[0]])
		rel = np.array([spearman(torch.as_tensor(V[i, 0].reshape(-1)), torch.as_tensor(V[i, 1].reshape(-1))) for i in np.where(g)[0]]) if V.shape[1] > 1 else np.zeros(int(g.sum()))
		ww = wt[g] / wt[g].sum()
		print(f'   rank corr with realised V over the {P * K} plans of a state: sim {(rho[:, 0] * ww).sum():.2f}  MIX {(rho[:, 1] * ww).sum():.2f}  exact-3-step {(rho[:, 2] * ww).sum():.2f}'
			  f'  | repeat-to-repeat {(rel * ww).sum():.2f}')
		gapq = (score[..., 2][:, None] - Vb).mean((1, 2, 3))
		print(f'   exact-3-step score minus realised (V + tail Q): {boot(gapq[g], wt[g])[0]:+.1f}  (score level {boot(J[g, :, 2].mean(1), wt[g])[0]:.1f})')


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('cmd', choices=['plans', 'roll', 'report'])
	ap.add_argument('--cell', required=True)
	ap.add_argument('--n', type=int, default=100)
	ap.add_argument('--k', type=int, default=3)
	ap.add_argument('--hc', type=int, default=150)
	ap.add_argument('--reps', type=int, default=2)
	ap.add_argument('--shard', type=int, default=0)
	ap.add_argument('--shards', type=int, default=16)
	ap.add_argument('--workers', type=int, default=40)
	ap.add_argument('--seed', type=int, default=11)
	ap.add_argument('--out_root', default=D06)
	ap.add_argument('--src', choices=['CONMIX', 'ORA'], default='CONMIX')   # whose visited states
	ap.add_argument('--tag', default='')                                   # file name suffix, e.g. _ora
	ap.add_argument('--cont', choices=['planner', 'pi'], default='planner')
	ap.add_argument('--t_split', type=int, default=100)
	ap.add_argument('--share0', type=float, default=0.4)
	a = ap.parse_args()
	dict(plans=cmd_plans, roll=cmd_roll, report=cmd_report)[a.cmd](a)
