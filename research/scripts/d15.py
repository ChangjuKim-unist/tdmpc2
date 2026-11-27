"""Design 15 (research/docs/design15_q_spread_direction_ko.md): in which direction does Q spread right after full fine-tuning starts?

Same start as design 13's RAW arm (design 07 section 9): the pretrained planner collects --seed_steps steps under the shifted physics
with exploration noise and no update, then official online training (all components, one update per env step) continues.
Snapshots of the model are kept every --snap_every updates up to --steps updates. Afterwards every snapshot scores a FIXED set of states
(the seed episodes, steps --t_min..--t_max-1):
  Q(z_t, a_t)       the executed action        (mean over the 5 Q members, two-hot inverted)
  Q(z_t, pi(z_t))   the policy's mean action   (what the planner's terminal value and the policy loss use)
against a fixed reference return G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k + gamma^(T-t) Q_pre(z_T, pi_pre(z_T)) (pretrained bootstrap, the same
for every snapshot). Also logged per online episode: return, and Q(z_t, a_t) of the snapshot nearest in time vs that episode's own G_t.

  GPU=N research/scripts/td.sh research/scripts/d15.py run --cell quadruped-run__motor_0.35 --seed 1
  GPU=N research/scripts/td.sh research/scripts/d15.py summarize --res_dir /mnt/newssd/cjkim/td/d15/res
The official TD-MPC2 code is imported, not modified.
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
from common.buffer import Buffer

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, reseed_env, run_episode
from d13 import to_td

OUT = '/mnt/newssd/cjkim/td/d15'
CELLS = ['quadruped-run__motor_0.35', 'walker-run__friction_0.15']
SHORT = {'quadruped-run__motor_0.35': 'quadruped 모터', 'walker-run__friction_0.15': 'walker 마찰'}
WINDOW, DOM, WIDEN = 1500, 3.0, 1.5


@torch.no_grad()
def q_on(agent, obs, act):
	"""obs (N, D), act (N, A) -> Q(z, a) and Q(z, pi_mean(z)), each (N,)."""
	m, cfg = agent.model, agent.cfg
	z = m.encode(obs, None)
	q_a = tdmath.two_hot_inv(m.Q(z, act, None, return_type='all'), cfg).mean(0).squeeze(-1)
	_, info = m.pi(z, None)
	q_pi = tdmath.two_hot_inv(m.Q(z, info['mean'], None, return_type='all'), cfg).mean(0).squeeze(-1)
	return q_a, q_pi


@torch.no_grad()
def ref_return(agent0, ep, gamma):
	"""Discounted return to the end of the episode plus the pretrained bootstrap at the last observation. ep: dict of obs (T+1, D), rew (T,)."""
	rew = torch.as_tensor(ep['rew'], dtype=torch.float32, device='cuda')
	T = rew.shape[0]
	_, boot = q_on(agent0, torch.as_tensor(ep['obs'][-1:], dtype=torch.float32, device='cuda'), torch.zeros(1, agent0.cfg.action_dim, device='cuda'))
	G = torch.empty(T, device='cuda')
	acc = boot[0]
	for t in range(T - 1, -1, -1):
		acc = rew[t] + gamma * acc
		G[t] = acc
	return G


def snap_sd(agent):
	return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in agent.model.state_dict().items()}


def cmd_run(a):
	from d07 import fresh
	task, cond = cell_parts(a.cell)
	folder = f'{a.out_root}/{a.cell}/s{a.seed}'
	if os.path.exists(f'{folder}/done.json'):
		return
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent = make(task, compile_=True, seed=a.seed)
	cfg.steps = a.steps + a.seed_steps
	cfg.buffer_size = cfg.steps + 1000
	buffer = Buffer(cfg)
	domain = domain_of(task)
	Shifter(phys_of(env)).set(cond)
	reseed_env(env, 6000 + a.seed)
	gamma = float(agent.discount)
	snaps, episodes, step, n_upd, t0 = [], [], 0, 0, time.time()
	snaps.append(dict(updates=0, sd=snap_sd(agent), scale=float(agent.scale.value)))
	while step < cfg.steps:
		obs, done = env.reset(), False
		tds, O, A, R = [to_td(env, obs)], [obs.numpy()], [], []
		while not done:
			action = agent.act(obs, t0=len(tds) == 1, eval_mode=False)
			obs, reward, done, info = env.step(action)
			tds.append(to_td(env, obs, action, reward))
			O.append(obs.numpy()); A.append(action.numpy()); R.append(float(reward))
			if step >= a.seed_steps and buffer.num_eps > 0:
				agent.update(buffer)
				n_upd += 1
				if n_upd % a.snap_every == 0 and n_upd <= a.steps:
					snaps.append(dict(updates=n_upd, sd=snap_sd(agent), scale=float(agent.scale.value)))
			step += 1
		buffer.add(torch.cat(tds))
		episodes.append(dict(obs=np.asarray(O, np.float32), act=np.asarray(A, np.float32), rew=np.asarray(R, np.float32), updates_end=n_upd, ret=float(sum(R))))
		print(a.cell, a.seed, 'episode', len(episodes), 'return', round(sum(R), 1), 'updates', n_upd, f'{time.time() - t0:.0f}s', flush=True)
	# ---- measurement with every snapshot on the fixed seed episodes and on each online episode
	n_seed = a.seed_steps // 500
	agent0 = fresh(cfg)
	seed_eps = episodes[:n_seed]
	Gs = [ref_return(agent0, ep, gamma)[a.t_min:a.t_max] for ep in seed_eps]
	G = torch.cat(Gs)
	obs_f = torch.as_tensor(np.concatenate([ep['obs'][a.t_min:a.t_max] for ep in seed_eps]), dtype=torch.float32, device='cuda')
	act_f = torch.as_tensor(np.concatenate([ep['act'][a.t_min:a.t_max] for ep in seed_eps]), dtype=torch.float32, device='cuda')
	probe = fresh(cfg)
	rows = []
	pct = lambda x: [float(v) for v in torch.quantile(x, torch.tensor([0.05, 0.5, 0.95], device=x.device))]
	for sn in snaps:
		probe.model.load_state_dict(sn['sd'])
		q_a, q_pi = q_on(probe, obs_f, act_f)
		row = dict(updates=sn['updates'], scale=sn['scale'], q_a=pct(q_a), q_pi=pct(q_pi), err_a=pct(q_a - G), err_pi=pct(q_pi - G),
				   bias_a=float((q_a - G).mean()), bias_pi=float((q_pi - G).mean()), G=pct(G))
		rows.append(row)
	online = []
	for ep in episodes[n_seed:]:
		sn = min(snaps, key=lambda s: abs(s['updates'] - ep['updates_end']))
		probe.model.load_state_dict(sn['sd'])
		Ge = ref_return(agent0, ep, gamma)[a.t_min:a.t_max]
		q_a, _ = q_on(probe, torch.as_tensor(ep['obs'][a.t_min:a.t_max], device='cuda'), torch.as_tensor(ep['act'][a.t_min:a.t_max], device='cuda'))
		online.append(dict(updates_end=ep['updates_end'], snap=sn['updates'], ret=ep['ret'], bias=float((q_a - Ge).mean()), q=pct(q_a), G=pct(Ge)))
	dump(f'{folder}/done.json', dict(cell=a.cell, seed=a.seed, args=vars(a), gamma=gamma, seed_returns=[ep['ret'] for ep in seed_eps], snapshots=rows, online=online,
									 seconds=time.time() - t0))
	print(a.cell, a.seed, 'done', flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 15 section 4)

def direction(r0, rk, key):
	"""Did the 5-95 width grow by WIDEN, and if so, which tail moved (relative to the pre-update snapshot)?"""
	d_lo, d_hi = rk[key][0] - r0[key][0], rk[key][2] - r0[key][2]
	w = (rk[key][2] - rk[key][0]) / max(r0[key][2] - r0[key][0], 1e-6)
	if w < WIDEN:
		v = '벌어지지 않음'
	elif d_hi > 0 and d_hi >= DOM * max(-d_lo, 0):
		v = '위로'
	elif d_lo < 0 and -d_lo >= DOM * max(d_hi, 0):
		v = '아래로'
	else:
		v = '양쪽으로'
	return v, d_lo, d_hi, w


def cmd_summarize(a):
	L, S = ['# 설계 15 결과 요약 (자동 생성, `d15.py summarize`)\n'], {}
	L.append(f'고정한 상태 집합(그 seed의 수집 에피소드, 각 50–349 스텝)에서 잰 Q의 5 / 50 / 95 백분위가 update 전 snapshot에서 얼마나 움직였는가. '
			 f'비교 대상은 처음 {WINDOW} update 안에서 Q(z, π(z))의 5–95 폭이 가장 큰 snapshot이다. 폭이 {WIDEN:g}배 미만이면 "벌어지지 않음". 벌어졌으면 한쪽 끝의 이동이 다른 쪽의 {DOM:g}배 이상일 때 그쪽 방향, 아니면 "양쪽으로".\n')
	for cell in CELLS:
		R = [json.load(open(f)) for f in sorted(glob.glob(f'{a.out_root}/{cell}/s*/done.json'))]
		if not R:
			continue
		L.append(f'\n## {SHORT[cell]}\n')
		L.append('| seed | 수집 return | 비교 snapshot (update) | Q(z, π(z)): 5 / 50 / 95 백분위, 전 → 후 | 폭 배율 | 방향 (5 %, 95 % 이동) | Q(z, a)의 방향 (폭 배율) | 편향 Q(z, a) − G, 전 → 후 | 그때까지 학습 에피소드 return |')
		L.append('|---|---|---|---|---|---|---|---|---|')
		S[cell] = []
		for r in R:
			sn = r['snapshots']
			r0 = sn[0]
			win = [s for s in sn[1:] if s['updates'] <= WINDOW] or sn[1:]
			rk = max(win, key=lambda s: s['q_pi'][2] - s['q_pi'][0])
			v_pi, lo_pi, hi_pi, w_pi = direction(r0, rk, 'q_pi')
			v_a, lo_a, hi_a, w_a = direction(r0, rk, 'q_a')
			rets = [round(o['ret']) for o in r['online'] if o['updates_end'] <= rk['updates'] + 500]
			S[cell].append(dict(seed=r['seed'], snap=rk['updates'], dir_pi=v_pi, dir_a=v_a, width_pi=w_pi, width_a=w_a, d_pi=[lo_pi, hi_pi], d_a=[lo_a, hi_a], bias=[r0['bias_a'], rk['bias_a']]))
			f3 = lambda q: ' / '.join(f'{x:.1f}' for x in q)
			L.append(f'| {r["seed"]} | {np.mean(r["seed_returns"]):.0f} | {rk["updates"]} | {f3(r0["q_pi"])} → {f3(rk["q_pi"])} | {w_pi:.2f} | **{v_pi}** ({lo_pi:+.1f}, {hi_pi:+.1f}) | {v_a} ({w_a:.2f}) | '
					 f'{r0["bias_a"]:+.1f} → {rk["bias_a"]:+.1f} | {rets} |')
		vs = [x['dir_pi'] for x in S[cell]]
		top = max(set(vs), key=vs.count)
		S[f'verdict/{cell}'] = top if vs.count(top) >= 2 else '미결'
		L.append(f'\n- **판정 ({SHORT[cell]}, seed 3개 중 2개 이상이 같은 방향): {S[f"verdict/{cell}"]}**')
		L.append('\n| seed | update 수별 Q(z, π(z)) 5 / 95 백분위와 편향 Q(z, a) − G |')
		L.append('|---|---|')
		for r in R:
			L.append(f'| {r["seed"]} | ' + '; '.join(f'{s["updates"]}: {s["q_pi"][0]:.1f} / {s["q_pi"][2]:.1f}, {s["bias_a"]:+.1f}' for s in r['snapshots']) + ' |')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, default=1)
	p.add_argument('--steps', type=int, default=3000)
	p.add_argument('--seed_steps', type=int, default=5000)
	p.add_argument('--snap_every', type=int, default=250)
	p.add_argument('--t_min', type=int, default=50)
	p.add_argument('--t_max', type=int, default=350)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)
