"""Design 06, unregistered record analysis (등록 안 함; research/docs/design06_consensus_planner_ko.md §11).

Quadruped motor 0.35: where does the fine-tuned (MIX) dynamics differ from the pretrained one, by posture?
No environment step, no planning, no training: forward passes of saved models on saved trajectories.

  GPU=N research/scripts/td.sh research/scripts/d06_diag.py --cell quadruped-run__motor_0.35

Posture is read from the observation (torso_upright, index 47 of the quadruped observation):
fallen < 0, tilted 0-0.8, upright >= 0.8, at the first step of each 3-step window.
Error = squared distance between the 3-step latent rollout and the encoded real observation, mean over windows.
"""
import argparse
import json

import numpy as np
import torch

from d01 import N_HELD, N_TRAIN, cell_parts, dump, make
from d02 import load_d01, load_sim, model_path

UP = 47
BINS = (('fallen', -2.0, 0.0), ('tilted', 0.0, 0.8), ('upright', 0.8, 2.0))
D04 = '/mnt/newssd/cjkim/td/d04'
D06 = '/mnt/newssd/cjkim/td/d06'


def wins(obs, act, H=3):
	"""obs E x (T+1) x D, act E x T x A -> windows obs (H+1) x N x D, act H x N x A, posture at start N."""
	T = act.shape[1]
	idx = np.arange(T - H + 1)
	O = np.stack([obs[:, idx + h] for h in range(H + 1)], 0).reshape(H + 1, -1, obs.shape[-1])
	A = np.stack([act[:, idx + h] for h in range(H)], 0).reshape(H, -1, act.shape[-1])
	return O, A, O[0][:, UP]


@torch.no_grad()
def err3(m, dyn, O, A, bs=20000):
	out = []
	for i in range(0, O.shape[1], bs):
		o = torch.as_tensor(O[:, i:i + bs], dtype=torch.float32, device='cuda')
		a = torch.as_tensor(A[:, i:i + bs], dtype=torch.float32, device='cuda')
		z = m.encode(o[0], None)
		for t in range(A.shape[0]):
			z = dyn(torch.cat([z, a[t]], dim=-1))
		out.append(((z - m.encode(o[-1], None)) ** 2).sum(-1).cpu().numpy())
	return np.concatenate(out)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--cell', default='quadruped-run__motor_0.35')
	p.add_argument('--out_root', default=D06)
	a = p.parse_args()
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	m = agent.model
	import copy
	dyn = {'PRE': copy.deepcopy(m._dynamics).eval()}
	for arm in ('MIX', 'D2000'):
		agent.load(model_path(a.cell, arm))
		dyn[arm] = copy.deepcopy(agent.model._dynamics).eval()
	real, sim = load_d01(a.cell), load_sim(task)
	ora = np.load(f'{a.out_root}/{a.cell}/rec_CONORA.npz')
	mix = np.load(f'{a.out_root}/{a.cell}/rec_CONMIX.npz')
	sets = {
		'real_train': (real['obs'][:N_TRAIN], real['act'][:N_TRAIN]),
		'sim_train': (sim['obs'], sim['act']),
		'real_heldout': (real['obs'][N_TRAIN:N_TRAIN + N_HELD], real['act'][N_TRAIN:N_TRAIN + N_HELD]),
		'conora_traj': (ora['obs'], ora['act']),
		'conmix_traj': (mix['shift_obs'], mix['shift_act']),
	}
	res = {}
	for name, (obs, act) in sets.items():
		O, A, up = wins(obs, act)
		e = {k: err3(m, d, O, A) for k, d in dyn.items()}
		res[name] = {}
		for b, lo, hi in BINS:
			s = (up >= lo) & (up < hi)
			res[name][b] = dict(n=int(s.sum()), frac=float(s.mean()), **{k: (float(v[s].mean()) if s.any() else None) for k, v in e.items()})
		f = lambda x: '   -  ' if x is None else f'{x:6.3f}'
		for b, _, _ in BINS:
			r = res[name][b]
			print(f'{name:13s} {b:8s} n {r["n"]:6d} ({r["frac"]:.3f})  err PRE {f(r["PRE"])}  MIX {f(r["MIX"])}  D2000 {f(r["D2000"])}', flush=True)
	dump(f'{a.out_root}/{a.cell}/diag_posture.json', res)


if __name__ == '__main__':
	main()
