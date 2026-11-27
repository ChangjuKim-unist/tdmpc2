"""Design 12 (research/docs/design12_veto_on_ceiling_agent_ko.md): gate 1 of the "adapt everything under the frozen agent's veto" direction.

Does the consensus of the frozen pretrained agent and a fully fine-tuned agent keep what the fine-tuned agent gained?
The fine-tuned agent is the ceiling checkpoint of design 07 (official online training continued for 200k steps under the shifted
physics; encoder, dynamics, reward, Q and policy all changed). No training here, evaluation only.

Arms (no exploration noise, seed 3000 + s; same initial states across arms):
  PRE    frozen agent, official planner
  CEIL   fine-tuned agent, official planner
  CON    official MPPI loop; every candidate is scored by both agents (each with its own encoder, dynamics, reward and Q),
         score = min of the two relative scores (design 06's rule)
  MEAN   the same with the average of the two relative scores
  ZCON   like CON, with each relative score first divided by its standard deviation over the candidates
  WMEAN  the same with a weighted average; weights are inversely proportional to each agent's mean absolute 100-step value
         residual on real data collected by the un-adapted planner (no tuned constant)
Half of the policy-prior candidates come from each agent's policy.

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d12.py run --cell quadruped-run__motor_0.35 --arm CON [--proto deploy|nominal --seed_from 1 --seed_to 16 --eps 2]
  GPU=N research/scripts/td.sh research/scripts/d12.py summarize --res_dir /mnt/newssd/cjkim/td/d12/res
Smoke tests must pass --out_root (another folder); `run --check` only prints the code check and the WMEAN weights.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from common import math as tdmath

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d07 import fresh

D07, D10 = '/mnt/newssd/cjkim/td/d07', '/mnt/newssd/cjkim/td/d10'
OUT = '/mnt/newssd/cjkim/td/d12'
CEIL = D07 + '/ceiling/{cell}/final.pt'
CELLS = ['quadruped-run__motor_0.35', 'walker-run__friction_0.15']
SHORT = {'quadruped-run__motor_0.35': 'quadruped 모터', 'walker-run__friction_0.15': 'walker 마찰'}
ARMS = ['PRE', 'CEIL', 'CON', 'MEAN', 'WMEAN', 'ZCON']
ANAME = {'PRE': '적응 없음', 'CEIL': '전부 고친 agent', 'CON': '합의 (min)', 'MEAN': '평균', 'WMEAN': '검증 무게 평균', 'ZCON': '표준화 합의 (min)'}
# dynamics-only reference (exact dynamics + sim consensus): design 08 re-measurement for quadruped motor (same protocol), design 06 for walker friction
DYN_REF = {'quadruped-run__motor_0.35': 571.0, 'walker-run__friction_0.15': 624.0}
MARGIN, PASS_R = -25.0, 0.5
TQ = {15: 2.1314}


class Dual:
	"""Official MPPI loop (tdmpc2._plan, single task, non-episodic) with every candidate scored by two whole agents."""

	def __init__(self, a0, a1, rule, w=(0.5, 0.5)):
		self.a0, self.a1, self.cfg, self.rule, self.w = a0, a1, a0.cfg, rule, w
		self.prev_mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device='cuda')
		self.binds, self.j0_sel, self.j1_sel = [], [], []

	def score(self, j0, j1):
		r0, r1 = j0 - j0.mean(0, keepdim=True), j1 - j1.mean(0, keepdim=True)
		if self.rule == 'CON':
			return torch.minimum(r0, r1), r0, r1
		if self.rule == 'ZCON':                                               # each relative score divided by its spread over the candidates
			r0, r1 = r0 / (r0.std(0, keepdim=True) + 1e-6), r1 / (r1.std(0, keepdim=True) + 1e-6)
			return torch.minimum(r0, r1), r0, r1
		w0, w1 = (0.5, 0.5) if self.rule == 'MEAN' else self.w
		return w0 * r0 + w1 * r1, r0, r1

	def pi_trajs(self, ag, z, n):
		cfg, m = self.cfg, ag.model
		out = torch.empty(cfg.horizon, n, cfg.action_dim, device='cuda')
		_z = z.repeat(n, 1)
		for t in range(cfg.horizon - 1):
			out[t], _ = m.pi(_z, None)
			_z = m.next(_z, out[t], None)
		out[-1], _ = m.pi(_z, None)
		return out

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		cfg = self.cfg
		o = obs.to('cuda').unsqueeze(0)
		z0, z1 = self.a0.model.encode(o, None), self.a1.model.encode(o, None)
		n0 = cfg.num_pi_trajs // 2
		pi_actions = torch.cat([self.pi_trajs(self.a0, z0, n0), self.pi_trajs(self.a1, z1, cfg.num_pi_trajs - n0)], 1)
		Z0, Z1 = z0.repeat(cfg.num_samples, 1), z1.repeat(cfg.num_samples, 1)
		mean = torch.zeros(cfg.horizon, cfg.action_dim, device='cuda')
		std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device='cuda')
		if not t0:
			mean[:-1] = self.prev_mean[1:]
		actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda')
		actions[:, :cfg.num_pi_trajs] = pi_actions
		for _ in range(cfg.iterations):
			r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device='cuda')
			actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
			j0 = self.a0._estimate_value(Z0, actions, None).nan_to_num(0)
			j1 = self.a1._estimate_value(Z1, actions, None).nan_to_num(0)
			value, r0, r1 = self.score(j0, j1)
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
		sel = elite_idxs[idx]
		self.binds.append(float((r0[sel] < r1[sel]).squeeze()))               # the frozen agent's relative score is the lower one
		self.j0_sel.append(float(j0[sel].squeeze()))
		self.j1_sel.append(float(j1[sel].squeeze()))
		return a.clamp(-1, 1).cpu()


def real_data(cell, per_seed=5):
	"""Real data collected by the un-adapted planner under the shifted physics (seeds 1-16, first 5 episodes each): design 07's first round, else design 10's data."""
	ds = []
	for s in range(1, 17):
		for p in (f'{D07}/{cell}/s{s:02d}/data1.npz', f'{D10}/{cell}/s{s:02d}/data.npz'):
			if os.path.exists(p):
				d = np.load(p)
				ds.append({k: d[k][:per_seed] for k in ('obs', 'act', 'rew')})
				break
	return ds


def weights(a0, a1, cell):
	from d11 import q_stats
	ds = real_data(cell)
	e = [float(torch.cat([q_stats(ag, d)['resid'] for d in ds]).abs().mean()) for ag in (a0, a1)]   # mean over episodes of |episode-mean 100-step value residual|, one seed's data at a time
	inv = [1.0 / max(x, 1e-6) for x in e]
	return (inv[0] / sum(inv), inv[1] / sum(inv)), e, len(ds)


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	real = 'nominal' if a.proto == 'nominal' else cond
	folder = f'{a.out_root}/{a.cell}/{a.proto}_{a.arm}'
	os.makedirs(folder, exist_ok=True)
	cfg, env, a0 = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	a1, extra = None, {}
	if a.arm != 'PRE' or a.check:
		a1 = fresh(cfg)
		a1.load(CEIL.format(cell=a.cell))
	if a.check:
		shifter.set(cond)
		obs = env.reset()
		d = Dual(a0, a1, 'CON')
		o = obs.to('cuda').unsqueeze(0)
		acts = torch.rand(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda') * 2 - 1
		val = lambda ag: (torch.manual_seed(0), ag._estimate_value(ag.model.encode(o, None).repeat(cfg.num_samples, 1), acts, None))[1]
		j0, j1 = val(a0), val(a1)
		same = Dual(a0, a0, 'CON').score(j0, j0)[0]
		print(f'check: consensus of an agent with itself − its own relative score max abs {float((same - (j0 - j0.mean())).abs().max()):.6f} | '
			  f'min ≤ mean everywhere: {bool((d.score(j0, j1)[0] <= Dual(a0, a1, "MEAN").score(j0, j1)[0] + 1e-6).all())} | '
			  f'sd frozen {float(j0.std()):.3f} fine-tuned {float(j1.std()):.3f} | rank correlation of the two scores {float(torch.corrcoef(torch.stack([j0.squeeze(1).argsort().argsort().float(), j1.squeeze(1).argsort().argsort().float()]))[0, 1]):.3f}', flush=True)
		w, e, n = weights(a0, a1, a.cell)
		print(f'WMEAN weights (frozen, fine-tuned) {w[0]:.3f} {w[1]:.3f} from residuals {e[0]:.2f} {e[1]:.2f} on real data of {n} seeds', flush=True)
		tic = time.time()
		for i in range(10):
			d.act(obs, t0=i == 0, eval_mode=True)
		print(f'{(time.time() - tic) / 10:.3f} s/plan', flush=True)
		return
	if a.arm == 'PRE':
		actor = a0
	elif a.arm == 'CEIL':
		actor = a1
	else:
		w = (0.5, 0.5)
		if a.arm == 'WMEAN':
			w, e, n = weights(a0, a1, a.cell)
			extra = dict(weights=w, residuals=e, weight_seeds=n)
			print(a.cell, 'WMEAN weights', w, 'residuals', e, flush=True)
		actor = Dual(a0, a1, a.arm, w)
	for s in range(a.seed_from, a.seed_to + 1):
		if os.path.exists(f'{folder}/s{s:02d}.json'):
			continue
		t0 = time.time()
		n0 = len(actor.binds) if isinstance(actor, Dual) else 0
		eps, recs = run_cond(actor, env, shifter, domain, real, a.eps, eval_mode=True, seed=3000 + s, record=True)
		np.savez_compressed(f'{folder}/rec_s{s:02d}.npz', rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
		out = dict(cell=a.cell, seed=s, args=vars(a), summary=summary(eps), episodes=eps, seconds=time.time() - t0, **extra)
		if isinstance(actor, Dual):
			out['frozen_binds'] = float(np.mean(actor.binds[n0:]))
		dump(f'{folder}/s{s:02d}.json', out)
		print(a.cell, a.arm, a.proto, s, {k: round(v, 2) for k, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 12 §4)

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


def fmt(iv, d=1, sign=True):
	p = '+' if sign else ''
	if not np.isfinite(iv['t'][0]):
		return f'{iv["mean"]:{p}.{d}f} (seed {iv["n"]}개)'
	return f'{iv["mean"]:{p}.{d}f} [{iv["t"][0]:{p}.{d}f}, {iv["t"][1]:{p}.{d}f}]'


def cmd_summarize(a):
	L, S = ['# 설계 12 결과 요약 (자동 생성, `d12.py summarize`)\n'], {}
	L.append('회복률 = (팔의 return − dynamics만 고친 기준) / (전부 고친 agent의 return − 같은 기준). 구간은 seed 단위(seed마다 2 에피소드 평균) t 구간 95 %다. '
			 f'판정: 회복률 구간 전체가 {PASS_R} 위면 "통과", 아래면 "실패", 그 밖은 "미결". "해 없음"은 (팔 − 적응 없음)의 구간 아래 끝이 {MARGIN:.0f} 위일 때 통과다.\n')
	K = ('ret', 'falls', 'fallen_frac', 'smooth', 'effort', 'angvel_rms')
	verd = {}
	for cell in CELLS:
		R = {}
		for arm in ARMS:
			for proto in ('deploy', 'nominal'):
				for f in sorted(glob.glob(f'{a.out_root}/{cell}/{proto}_{arm}/s*.json')):
					R.setdefault((arm, proto), {})[int(os.path.basename(f)[1:3])] = json.load(open(f))
		if ('PRE', 'deploy') not in R or ('CEIL', 'deploy') not in R:
			continue
		sm = lambda arm, proto, k='ret': {s: float(np.mean([e[k] for e in r['episodes']])) for s, r in R[arm, proto].items()}
		pre, ceil, D = sm('PRE', 'deploy'), sm('CEIL', 'deploy'), DYN_REF[cell]
		gap = float(np.mean(list(ceil.values()))) - D
		S[cell] = dict(dyn_ref=D, ceil=float(np.mean(list(ceil.values()))), gap=gap, valid=bool(gap >= 25), arms={})
		L.append(f'\n## {SHORT[cell]}\n')
		L.append(f'- dynamics만 고친 기준(정확한 dynamics + sim 합의): {D:.0f}. 전부 고친 agent: {S[cell]["ceil"]:.0f}. 격차 {gap:+.0f}' + ('' if gap >= 25 else ' → **판정 불가(격차 25 미만)**') + '\n')
		L.append('| 팔 | seed 수 | return | − 적응 없음 | 해 없음 | − 전부 고친 agent | 회복률 | 판정 | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 | 평소 조건 return | 얼린 쪽이 정한 비율 |')
		L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
		for arm in ARMS:
			if (arm, 'deploy') not in R:
				continue
			v = sm(arm, 'deploy')
			seeds = sorted(set(v) & set(pre) & set(ceil))
			m = {k: float(np.mean([sm(arm, 'deploy', k)[s] for s in seeds])) for k in K}
			g, c = interval([v[s] - pre[s] for s in seeds]), interval([v[s] - ceil[s] for s in seeds])
			r = interval([(v[s] - D) / gap for s in seeds]) if gap >= 25 else None
			vd = '–' if r is None or arm in ('PRE', 'CEIL') else ('통과' if r['t'][0] > PASS_R else ('실패' if r['t'][1] < PASS_R else '미결'))
			hm = '통과' if g['t'][0] > MARGIN else ('실패' if g['t'][1] < MARGIN else '미결')
			nom = float(np.mean(list(sm(arm, 'nominal').values()))) if (arm, 'nominal') in R else None
			fb = [R[arm, 'deploy'][s].get('frozen_binds') for s in seeds]
			fb = [x for x in fb if x is not None]
			S[cell]['arms'][arm] = dict(n=len(seeds), **m, vs_pre=g, harm=hm, vs_ceil=c, recovery=r, verdict=vd, nominal=nom, frozen_binds=float(np.mean(fb)) if fb else None,
										weights=R[arm, 'deploy'][seeds[0]].get('weights'))
			verd[cell, arm] = vd
			L.append(f'| {ANAME[arm]} | {len(seeds)} | {m["ret"]:.0f} | {"–" if arm == "PRE" else fmt(g)} | {"–" if arm == "PRE" else hm} | {"–" if arm == "CEIL" else fmt(c)} | '
					 f'{"–" if r is None else fmt(r, 2, False)} | {vd} | {m["falls"]:.1f} / {m["fallen_frac"]:.2f} | {m["smooth"]:.2f} / {m["effort"]:.2f} / {m["angvel_rms"]:.2f} | '
					 f'{"–" if nom is None else f"{nom:.0f}"} | {"–" if not fb else f"{np.mean(fb):.2f}"} |')
		w = S[cell]['arms'].get('WMEAN', {}).get('weights')
		if w:
			L.append(f'\n- 검증 무게(얼린 agent, 전부 고친 agent): {w[0]:.2f}, {w[1]:.2f}')
	L.append('\n## 관문 1\n')
	for arm in ('CON', 'MEAN', 'WMEAN', 'ZCON'):
		vs = [verd.get((c, arm), '(없음)') for c in CELLS]
		S[f'gate1_{arm}'] = '통과' if all(v == '통과' for v in vs) else ('실패' if any(v == '실패' for v in vs) else '미결')
		L.append(f'- {ANAME[arm]}: ' + ', '.join(f'{SHORT[c]} {v}' for c, v in zip(CELLS, vs)) + f' → **{S[f"gate1_{arm}"]}**')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--arm', choices=ARMS, required=True)
	p.add_argument('--proto', choices=['deploy', 'nominal'], default='deploy')
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=16)
	p.add_argument('--eps', type=int, default=2)
	p.add_argument('--check', action='store_true')
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)
