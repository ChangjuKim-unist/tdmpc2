"""Design 06, unregistered error-attribution analysis (등록 안 함; research/docs/prior_consensus_survey_ko.md §7).

No training, no new data collection, no closed-loop evaluation. At states that the learned-model consensus planner
(CON-MIX) actually visited under the shifted physics (design 06 records), three planners are run once each from the same
warm start, and their plans are re-scored with the exact simulator (state restore, design 04 scorer):
  S = un-adapted planner (pretrained dynamics only)
  L = learned consensus (pretrained vs MIX dynamics), run twice with different noise (L, L2) to get the planner-noise scale
  O = oracle consensus (pretrained vs exact dynamics)
Scorers: J_s (pretrained dynamics), J_x (MIX dynamics), J_o (exact 3 steps); all use the same pretrained reward/Q.
For each planner the executed plan is a random elite, so every quantity is the elite-weighted mean E[.] over its final elites.

Quantities per state
  harm = E_o(L) - E_o(S)       learned consensus vs un-adapted planner, in exact terms (negative = harm)
  gap  = E_o(O) - E_o(L)       what the learned model loses against exact dynamics inside the same consensus rule
  b    = mean over a broad reference set R (first-iteration candidates of S) of (J_x - J_o): level bias of the MIX model
  over_L  = E_x(L) - E_o(L) - b     MIX over-rates the plan that learned consensus picked
  under_S = b - (E_x(S) - E_o(S))   MIX under-rates the plan that the un-adapted planner picked (false veto)
  under_O = b - (E_x(O) - E_o(O))   MIX under-rates the plan that oracle consensus picked
  identity: [E_x(L) - E_x(S)] - harm = over_L + under_S
  scale: std over candidates of the centred scores of each side (last iteration of L, and R), who binds among L's elites
  rank: Spearman correlation over candidates between each model's score and the exact score

How the result will be read (written before running)
  1. harm states (harm < -2 x planner-noise scale): if under_S is the larger term -> the missing data is "the un-adapted
     planner's plan executed at newly visited states" (add a consensus <-> un-adapted alternation arm);
     if over_L is the larger term -> data collected by the consensus planner itself tests it (consensus-only arm suffices).
  2. if the std ratio (MIX side / sim side) is outside [0.5, 2] in most states -> scale mismatch decides the min;
     consider scale normalisation before any data device.
  3. the cell with the largest mean gap is the candidate second cell for arm comparison in design 07.
Limit: "exact" means exact 3-step dynamics with the same pretrained Q, so value staleness is not measured here.

  GPU=N research/scripts/td.sh research/scripts/d06_attr.py --cell quadruped-run__motor_0.35 [--n 300] [--workers 40]
"""
import argparse
import copy
import multiprocessing as mp
import time

import numpy as np
import torch

from d01 import cell_parts, domain_of, make
from d02 import model_path
from d04 import Oracle, _winit
from d06 import consensus, plan_value

D06 = '/mnt/newssd/cjkim/td/d06'
UP = 47
BINS = (('fallen', -2.0, 0.0), ('tilted', 0.0, 0.8), ('upright', 0.8, 2.0))


def rank(x):
	return x.argsort().argsort().float()


def spearman(x, y):
	a, b = rank(x.flatten()), rank(y.flatten())
	a, b = a - a.mean(), b - b.mean()
	return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


@torch.no_grad()
def mppi(agent, z, mean0, score_fn, dyn_pi):
	"""Official MPPI loop (same as design 04/06 planners) with an explicit score function; returns elites and candidate sets."""
	cfg, m = agent.cfg, agent.model
	pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device='cuda')
	_z = z.repeat(cfg.num_pi_trajs, 1)
	for t in range(cfg.horizon - 1):
		pi_actions[t], _ = m.pi(_z, None)
		_z = dyn_pi(torch.cat([_z, pi_actions[t]], dim=-1))
	pi_actions[-1], _ = m.pi(_z, None)
	mean = mean0.clone()
	std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device='cuda')
	actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda')
	actions[:, :cfg.num_pi_trajs] = pi_actions
	first = None
	for it in range(cfg.iterations):
		r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device='cuda')
		actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
		value, parts = score_fn(actions)
		if it == 0:
			first = actions.clone()
		elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
		score = torch.exp(cfg.temperature * (elite_value - elite_value.max(0).values))
		score = score / score.sum(0)
		mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
		std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
		std = std.clamp(cfg.min_std, cfg.max_std)
	return dict(mean=mean, first=first, actions=actions.clone(), elite=elite_idxs, w=score.squeeze(1), parts=parts)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--cell', required=True)
	p.add_argument('--n', type=int, default=300)
	p.add_argument('--workers', type=int, default=40)
	p.add_argument('--seed', type=int, default=7)
	p.add_argument('--out_root', default=D06)
	a = p.parse_args()
	task, cond = cell_parts(a.cell)
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, cond))  # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	m = agent.model
	ora = Oracle(agent, env, pool, a.workers)
	dyn0 = copy.deepcopy(m._dynamics).eval()
	agent.load(model_path(a.cell, 'MIX'))
	dynx = copy.deepcopy(agent.model._dynamics).eval()
	m = agent.model

	rec = np.load(f'{a.out_root}/{a.cell}/rec_CONMIX.npz')
	obs, st = rec['shift_obs'], rec['shift_state']
	E, T = obs.shape[0], obs.shape[1] - 1
	rs = np.random.RandomState(a.seed)
	ee, tt = np.meshgrid(np.arange(E), np.arange(2, T), indexing='ij')
	ee, tt = ee.ravel(), tt.ravel()
	if domain_of(task) == 'quadruped':                                   # stratify by posture (harm sits in rare postures)
		up = obs[ee, tt, UP]
		pick, binid, binfrac = [], [], []
		for b, (name, lo, hi) in enumerate(BINS):
			idx = np.where((up >= lo) & (up < hi))[0]
			k = min(len(idx), a.n // 3)
			pick += list(rs.choice(idx, k, replace=False))
			binid += [b] * k
			binfrac.append(len(idx) / len(up))
		pick, binid = np.array(pick), np.array(binid)
		wt = np.array([binfrac[b] / max((binid == b).sum(), 1) for b in binid])
	else:
		pick = rs.choice(len(ee), a.n, replace=False)
		binid, wt = np.zeros(a.n, int), np.full(a.n, 1.0 / a.n)
	N = len(pick)

	H, A, S = cfg.horizon, cfg.action_dim, cfg.num_samples
	js_fn = lambda zs, act: plan_value(agent, zs, act, dyn0).nan_to_num(0)
	jx_fn = lambda zs, act: plan_value(agent, zs, act, dynx).nan_to_num(0)
	keys = ['E_s', 'E_x', 'E_o']
	out = {f'{k}_{pl}': np.zeros(N, np.float32) for k in keys for pl in ('S', 'L', 'L2', 'O')}
	for k in ('b_x', 'b_s', 'sdR_s', 'sdR_x', 'sdR_o', 'sdL_s', 'sdL_x', 'sdL_o', 'rhoR_xo', 'rhoR_so', 'rhoR_sx',
			  'rhoL_xo', 'rhoL_so', 'simbind_L', 'simbind_O', 'da_LS', 'da_LO', 'da_LL2', 'da_OS'):
		out[k] = np.zeros(N, np.float32)
	t_start = time.time()
	for i, j in enumerate(pick):
		e, t = int(ee[j]), int(tt[j])
		torch.manual_seed(a.seed * 100003 + i)
		z_of = lambda tau: m.encode(torch.as_tensor(obs[e, tau], dtype=torch.float32, device='cuda').unsqueeze(0), None)
		con_x = lambda zs: (lambda act: (lambda js, jx: (consensus(js, jx)[0], (js, jx)))(js_fn(zs, act), jx_fn(zs, act)))
		# warm start: replay learned consensus on the two preceding recorded states (as in deployment)
		mean0 = torch.zeros(H, A, device='cuda')
		for tau in (t - 2, t - 1):
			zz = z_of(tau)
			r = mppi(agent, zz, mean0, con_x(zz.repeat(S, 1)), dynx)
			mean0 = torch.zeros(H, A, device='cuda')
			mean0[:-1] = r['mean'][1:]
		z, state = z_of(t), st[e, t]
		zs = z.repeat(S, 1)
		pS = mppi(agent, z, mean0, lambda act: (js_fn(zs, act), None), dyn0)
		pL = mppi(agent, z, mean0, con_x(zs), dynx)
		pL2 = mppi(agent, z, mean0, con_x(zs), dynx)
		def con_o(act):
			js, jo = js_fn(zs, act), ora.values(state, act).nan_to_num(0)
			return consensus(js, jo)[0], (js, jo)
		pO = mppi(agent, z, mean0, con_o, dyn0)
		R = pS['first']
		sc = {}
		for name, pl in (('S', pS), ('L', pL), ('L2', pL2), ('O', pO)):
			act = pl['actions']
			jo = pl['parts'][1] if name == 'O' else ora.values(state, act).nan_to_num(0)
			sc[name] = (js_fn(zs, act), jx_fn(zs, act), jo)
			for k, v in zip(keys, sc[name]):
				out[f'{k}_{name}'][i] = float((pl['w'] * v[pl['elite']].squeeze(1)).sum())
		rs_, rx_, ro_ = js_fn(zs, R), jx_fn(zs, R), ora.values(state, R).nan_to_num(0)
		out['b_x'][i], out['b_s'][i] = float((rx_ - ro_).mean()), float((rs_ - ro_).mean())
		out['sdR_s'][i], out['sdR_x'][i], out['sdR_o'][i] = float(rs_.std()), float(rx_.std()), float(ro_.std())
		out['rhoR_xo'][i], out['rhoR_so'][i], out['rhoR_sx'][i] = spearman(rx_, ro_), spearman(rs_, ro_), spearman(rs_, rx_)
		ls, lx, lo = sc['L']
		out['sdL_s'][i], out['sdL_x'][i], out['sdL_o'][i] = float(ls.std()), float(lx.std()), float(lo.std())
		out['rhoL_xo'][i], out['rhoL_so'][i] = spearman(lx, lo), spearman(ls, lo)
		el = pL['elite']
		out['simbind_L'][i] = float(((ls - ls.mean())[el] < (lx - lx.mean())[el]).float().mean())
		os_, _, oo = sc['O']
		out['simbind_O'][i] = float(((os_ - os_.mean())[pO['elite']] < (oo - oo.mean())[pO['elite']]).float().mean())
		a0 = {k: v['mean'][0] for k, v in (('S', pS), ('L', pL), ('L2', pL2), ('O', pO))}
		for k, (u, v) in (('da_LS', ('L', 'S')), ('da_LO', ('L', 'O')), ('da_LL2', ('L', 'L2')), ('da_OS', ('O', 'S'))):
			out[k][i] = float((a0[u] - a0[v]).abs().mean())
		if i % 20 == 0:
			print(f'{a.cell} {i}/{N} {time.time() - t_start:.0f}s', flush=True)
	out.update(ep=ee[pick], t=tt[pick], bin=binid, wt=wt)
	np.savez_compressed(f'{a.out_root}/{a.cell}/attr.npz', **out)
	pool.close()
	report(a.cell, out)


def report(cell, o):
	w = o['wt'] / o['wt'].sum()
	avg = lambda x, s=None: float((x * w).sum()) if s is None else (float((x[s] * w[s]).sum() / w[s].sum()) if s.any() else float('nan'))
	frac = lambda s: float(w[s].sum())
	harm = o['E_o_L'] - o['E_o_S']
	gap = o['E_o_O'] - o['E_o_L']
	noise = float(np.median(np.abs(o['E_o_L'] - o['E_o_L2'])))
	over_L = o['E_x_L'] - o['E_o_L'] - o['b_x']
	under_S = o['b_x'] - (o['E_x_S'] - o['E_o_S'])
	under_O = o['b_x'] - (o['E_x_O'] - o['E_o_O'])
	hs, gs = harm < -2 * noise, gap > 2 * noise
	ratio_L, ratio_R = o['sdL_x'] / (o['sdL_s'] + 1e-9), o['sdR_x'] / (o['sdR_s'] + 1e-9)
	print(f'\n== {cell}: {len(w)} states, planner-noise scale (median |E_o(L)-E_o(L2)|) {noise:.3f}')
	print(f'harm = E_o(L)-E_o(S): mean {avg(harm):+.3f}, states with harm < -2 noise: {frac(hs):.2f}, states with harm > +2 noise: {frac(harm > 2 * noise):.2f}')
	print(f'gap  = E_o(O)-E_o(L): mean {avg(gap):+.3f}, states with gap > 2 noise: {frac(gs):.2f};  E_o(O)-E_o(S) mean {avg(o["E_o_O"] - o["E_o_S"]):+.3f}')
	print(f'in harm states: over_L {avg(over_L, hs):+.3f}  under_S {avg(under_S, hs):+.3f}  (harm {avg(harm, hs):+.3f}); share of states where under_S > over_L: {avg((under_S > over_L).astype(float), hs):.2f}')
	print(f'in gap states:  over_L {avg(over_L, gs):+.3f}  under_O {avg(under_O, gs):+.3f}  (gap {avg(gap, gs):+.3f}); share of states where under_O > over_L: {avg((under_O > over_L).astype(float), gs):.2f}')
	print(f'all states:     over_L {avg(over_L):+.3f}  under_S {avg(under_S):+.3f}  under_O {avg(under_O):+.3f}')
	print(f'scale: std ratio MIX/sim, last iteration of L: median {np.median(ratio_L):.2f}, outside [0.5,2]: {frac((ratio_L < 0.5) | (ratio_L > 2)):.2f}; reference set: median {np.median(ratio_R):.2f}')
	print(f'       std of scores on reference set: sim {avg(o["sdR_s"]):.2f} MIX {avg(o["sdR_x"]):.2f} exact {avg(o["sdR_o"]):.2f}; last iteration of L: sim {avg(o["sdL_s"]):.2f} MIX {avg(o["sdL_x"]):.2f} exact {avg(o["sdL_o"]):.2f}')
	print(f'who binds among elites (share where sim side is the min): L {avg(o["simbind_L"]):.2f}  O {avg(o["simbind_O"]):.2f}')
	print(f'rank corr with exact: reference set MIX {avg(o["rhoR_xo"]):.2f} sim {avg(o["rhoR_so"]):.2f}; last iteration of L: MIX {avg(o["rhoL_xo"]):.2f} sim {avg(o["rhoL_so"]):.2f}')
	print(f'first-action distance (mean abs): L-L2 {avg(o["da_LL2"]):.3f}  L-S {avg(o["da_LS"]):.3f}  L-O {avg(o["da_LO"]):.3f}  O-S {avg(o["da_OS"]):.3f}')
	if o['bin'].max() > 0:
		for b, (name, _, _) in enumerate(BINS):
			s = o['bin'] == b
			if s.any():
				print(f'  {name:8s} share {frac(s):.3f}: harm {avg(harm, s):+.3f} gap {avg(gap, s):+.3f} over_L {avg(over_L, s):+.3f} under_S {avg(under_S, s):+.3f} '
					  f'rank MIX {avg(o["rhoL_xo"], s):.2f} sim {avg(o["rhoL_so"], s):.2f} ratio {np.median(ratio_L[s]):.2f}', flush=True)


if __name__ == '__main__':
	import sys
	if len(sys.argv) > 2 and sys.argv[1] == 'report':
		for c in sys.argv[2:]:
			report(c, dict(np.load(f'{D06}/{c}/attr.npz')))
	else:
		main()
