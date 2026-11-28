"""Design 18 §(rule simulation): offline simulation of the handover / revert rules on design 13's training-episode returns.

No GPU, no design-18 result is used. Noise sources (per condition, from design 13's RAW runs, seeds 1-3, training episodes with
exploration noise, the same collection mode as design 18's frozen-agent and probe episodes):
  frozen   the first 10 episodes of each run (pretrained planner, no update yet), 30 per condition
  learner  residuals of update-phase episodes around a centred 9-episode moving average;
           'all' (every update-phase episode, used for wrong handover, conservative) and
           'near' (episodes whose moving average is >= 0.9 x the frozen mean, used for wrong revert)
A learner at relative level L returns L x (frozen mean) + a resampled residual.

Rules (one probe every 5 episodes, 200 episodes per run, the first 10 are the frozen agent's seed phase):
  handover  after each probe: mean of the last k probes >= (1 + m) x the cumulative mean of all frozen-agent episodes so far
  revert    after each learner episode (handed over): mean of the last w learner episodes < r x the frozen cumulative mean
Usage: python3 research/scripts/d18_rule_sim.py <folder with design 13 log.json copies named {cell}_RAW_s{n}.json>
"""
import glob
import itertools
import json
import sys

import numpy as np

EPS, SEED_EPS, EVERY = 200, 10, 5
N_SIM = 4000


def pools(folder, cell):
	fr, all_, near = [], [], []
	for f in sorted(glob.glob(f'{folder}/{cell}_RAW_s*.json')):
		t = np.array([x['ret'] for x in json.load(open(f))['train']])
		fr += list(t[:SEED_EPS])
		u = t[SEED_EPS:]
		ma = np.convolve(u, np.ones(9) / 9, 'valid')
		r = u[4:-4] - ma
		all_ += list(r)
		near += list(r[ma >= 0.9 * t[:SEED_EPS].mean()])
	return np.array(fr), np.array(all_), np.array(near)


def handover_prob(rng, fr, res, L, k, m, by_ep=EPS):
	"""Share of simulated runs whose learner at level L is handed over before episode by_ep."""
	F = fr.mean()
	hits = 0
	for _ in range(N_SIM):
		fsum, fn, probes = float(rng.choice(fr, SEED_EPS).sum()), SEED_EPS, []
		for ep in range(SEED_EPS, by_ep):
			if (ep - SEED_EPS) % EVERY == EVERY - 1:
				probes.append(L * F + rng.choice(res))
				if len(probes) >= k and np.mean(probes[-k:]) >= (1 + m) * fsum / fn:
					hits += 1
					break
			else:
				fsum += rng.choice(fr)
				fn += 1
	return hits / N_SIM


def revert_prob(rng, fr, res, L, w, r, n_eps=EPS - SEED_EPS, n_frozen=60):
	"""Share of runs where a learner at level L (already handed over) is reverted within n_eps episodes; the frozen mean is fixed
	at its value after n_frozen frozen episodes."""
	F = fr.mean()
	hits, first = 0, []
	for _ in range(N_SIM):
		fm = rng.choice(fr, n_frozen).mean()
		x = L * F + rng.choice(res, n_eps)
		c = np.convolve(x, np.ones(w) / w, 'valid')
		idx = np.nonzero(c < r * fm)[0]
		if len(idx):
			hits += 1
			first.append(idx[0] + w)
	return hits / N_SIM, (float(np.median(first)) if first else float('nan'))


def main(folder):
	rng = np.random.default_rng(0)
	out = {}
	for cell in ('quadruped-run__motor_0.35', 'walker-run__friction_0.15'):
		fr, all_, near = pools(folder, cell)
		print(f'\n## {cell}: frozen mean {fr.mean():.1f} sd {fr.std():.1f} (n {len(fr)}), learner residual sd all {all_.std():.1f} (n {len(all_)}), near {near.std():.1f} (n {len(near)})')
		H = {}
		print('handover: k m | wrong (L=0.85, whole run) | wrong (L=0.95) | power L=1.1 by episode 110 (~55k steps) | power L=1.25 by 110')
		for k, m in itertools.product((2, 3, 4, 5, 6, 8), (0.0, 0.05, 0.1, 0.15, 0.2)):
			h = dict(wrong85=handover_prob(rng, fr, all_, 0.85, k, m), wrong95=handover_prob(rng, fr, all_, 0.95, k, m),
					 pow110=handover_prob(rng, fr, all_, 1.10, k, m, 110), pow125=handover_prob(rng, fr, all_, 1.25, k, m, 110))
			H[f'{k}_{m}'] = h
			print(f'  {k} {m:.2f} | {h["wrong85"]:.3f} | {h["wrong95"]:.3f} | {h["pow110"]:.3f} | {h["pow125"]:.3f}')
		R = {}
		print('revert: w r | wrong (L=1.0, 190 episodes) | power L=0.5 within 190 (median delay, episodes) | power L=0.7')
		for w, r in itertools.product((3, 5, 8, 10, 15), (0.75, 0.7, 0.6)):
			fa, _ = revert_prob(rng, fr, near, 1.0, w, r)
			p5, d5 = revert_prob(rng, fr, all_, 0.5, w, r)
			p7, d7 = revert_prob(rng, fr, all_, 0.7, w, r)
			R[f'{w}_{r}'] = dict(wrong100=fa, pow50=p5, delay50=d5, pow70=p7, delay70=d7)
			print(f'  {w} {r:.2f} | {fa:.3f} | {p5:.3f} ({d5:.0f}) | {p7:.3f} ({d7:.0f})')
		out[cell] = dict(frozen_mean=float(fr.mean()), frozen_sd=float(fr.std()), resid_sd_all=float(all_.std()), resid_sd_near=float(near.std()), handover=H, revert=R)
	json.dump(out, open('research/results/d18/rule_sim.json', 'w'), indent=1)


if __name__ == '__main__':
	main(sys.argv[1])
