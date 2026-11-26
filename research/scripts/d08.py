"""Design 08 (research/docs/design08_dose_and_next_round_ko.md): what the adapted-model collection buys, per episode.

No new arm is collected for the dose question: everything is re-fit from the design-07 data of the same seed.
One process = one (cell, seed):
  dose      k in (1, 2): M2[k] = the design-07 fine-tuning (torch seed s) on round 1 + the first k episodes of the
            adapted-collector round 2 + the last 5-k episodes of the control round 2 (same initial states by index);
            deployment with the consensus planner (sim model vs M2[k]), 20 shifted episodes, no noise (seed 3000 + s)
  round 3   5 episodes with exploration noise (seed 5000 + s) by the ORDINARY planner with each model:
            PRE (un-adapted), M1, M2[CTRL] (k=0), M2[k=1], M2[k=2], M2[CON], M2[ADP] (k=5)
  error     official consistency loss of every model on the round-3 episodes collected with M1 (held out for all models)
Oracle consensus (exact dynamics as the adapted side, design 06) on the design-07 evaluation seeds:
  GPU=N research/scripts/td.sh research/scripts/d08.py oracle --cell quadruped-run__motor_0.35 --workers 28
Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d08.py run --cell cheetah-run__friction_0.15 --seed 1
  GPU=N research/scripts/td.sh research/scripts/d08.py summarize
Smoke tests must pass --out_root (another folder) and may shorten with --r_eps / --eval_eps / --ft_steps / --eps.
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
import torch.nn.functional as F

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import load_sim
from d07 import CELLS, FT_STEPS, R_EPS, EVAL_EPS, finetune_mix, fresh, stack

D07 = '/mnt/newssd/cjkim/td/d07'
OUT = '/mnt/newssd/cjkim/td/d08'
KS = (1, 2)
LEVEL = 0.95
TQ = {15: 2.1314, 31: 2.0395}                                               # t quantile 0.975 (two-sided 95 %)
ACTORS = ['PRE', 'M1', 'CTRL', 'K1', 'K2', 'CON', 'ADP']                    # round-3 collectors (ordinary planner)
ANAME = {'PRE': '적응 없음', 'M1': '첫 라운드 모델', 'CTRL': 'k=0 (대조)', 'K1': 'k=1', 'K2': 'k=2', 'CON': '합의 수집', 'ADP': 'k=5 (적응 수집)'}
ORA_SEEDS, ORA_EPS = 16, 2


@torch.no_grad()
def model_error(m, cfg, obs, act):
	"""Official consistency loss over every window of the given episodes (obs E x (T+1) x D, act E x T x A), and its 1-step part."""
	H = cfg.horizon
	obs = torch.as_tensor(obs, dtype=torch.float32, device='cuda')
	act = torch.as_tensor(act, dtype=torch.float32, device='cuda')
	E, T = act.shape[:2]
	zs = m.encode(obs.reshape(E * (T + 1), -1), None).reshape(E, T + 1, -1)
	n = T - H + 1
	z = zs[:, :n].reshape(E * n, -1)
	total, one = 0.0, None
	for h in range(H):
		z = m.next(z, act[:, h:h + n].reshape(E * n, -1), None)
		e = F.mse_loss(z, zs[:, h + 1:h + 1 + n].reshape(E * n, -1))
		one = float(e) if h == 0 else one
		total = total + float(e) * cfg.rho ** h
	return dict(loss=total / H, one_step=one)


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	src, folder = f'{a.d07_root}/{a.cell}/s{s:02d}', f'{a.out_root}/{a.cell}/s{s:02d}'
	if os.path.exists(f'{folder}/done.json'):
		print('exists, skip', folder, flush=True)
		return
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent0 = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	dyn0 = copy.deepcopy(agent0.model._dynamics).eval()
	for p in dyn0.parameters():
		p.requires_grad_(False)
	sim = load_sim(task)
	load = lambda f: {k: v for k, v in np.load(f'{src}/{f}.npz').items()}
	d1, dC, dA = load('data1'), load('data2_CTRL'), load('data2_ADP')
	assert np.allclose(dC['state'][:, 0], dA['state'][:, 0]), 'round-2 initial states differ between arms'
	res, t_all = dict(cell=a.cell, seed=s, args=vars(a)), time.time()

	def with_dyn(sd):
		ag = fresh(cfg)
		ag.model._dynamics.load_state_dict(sd)
		return ag

	dyn = {'M1': torch.load(f'{src}/M1_dyn.pt')}
	for arm in ('CTRL', 'CON', 'ADP'):
		dyn[arm] = torch.load(f'{src}/M2_{arm}_dyn.pt')

	# dose: re-fit with k adapted-collector episodes, deploy with the consensus planner
	from d06 import ConAgent
	res['dose'] = {}
	for k in ((R_EPS,) if a.check else ()) + KS:
		data = {key: np.concatenate([d1[key], dA[key][:k], dC[key][k:]], 0) for key in ('obs', 'act', 'rew')}
		m2, log = finetune_mix(cfg, data, sim, s, a.ft_steps)
		sd = {n: v.detach().clone() for n, v in m2.model._dynamics.state_dict().items()}
		if k == R_EPS:                                                      # pipeline check: k=5 must reproduce the design-07 model
			diff = max(float((sd[n] - dyn['ADP'][n].to(sd[n].device)).abs().max()) for n in sd)
			res['check_k5_max_abs_diff'] = diff
			print(a.cell, s, 'check: k=5 re-fit vs stored M2_ADP, max abs weight difference', diff, flush=True)
			continue
		dyn[f'K{k}'] = sd
		torch.save(sd, f'{folder}/M2_k{k}_dyn.pt')
		con = ConAgent(m2, dyn0)
		t0 = time.time()
		eps, recs = run_cond(con, env, shifter, domain, cond, a.eval_eps, eval_mode=True, seed=3000 + s, record=True)
		d = stack(recs)
		np.savez_compressed(f'{folder}/rec_k{k}.npz', obs=d['obs'].astype(np.float32), state=d['state'], rew=d['rew'].astype(np.float32))
		res['dose'][f'K{k}'] = dict(M2=log[-1] if log else None, summary=summary(eps), episodes=eps,
									sim_binds=float(np.nanmean(con.sim_binds)) if con.sim_binds else None, seconds=time.time() - t0)
		print(a.cell, s, f'k={k}', {key: round(v, 2) for key, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)
		dump(f'{folder}/partial.json', res)

	# round 3: the ordinary planner with each model, exploration noise on
	res['round3'], data3 = {}, None
	for name in ACTORS:
		actor = agent0 if name == 'PRE' else with_dyn(dyn[name])
		t0 = time.time()
		eps, recs = run_cond(actor, env, shifter, domain, cond, a.r_eps, eval_mode=False, seed=5000 + s, record=True)
		d = stack(recs)
		np.savez_compressed(f'{folder}/data3_{name}.npz', **d)
		if name == 'M1':
			data3 = d
		res['round3'][name] = dict(summary=summary(eps), episodes=eps, seconds=time.time() - t0)
		print(a.cell, s, 'round3', name, {key: round(v, 2) for key, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)
		dump(f'{folder}/partial.json', res)

	# model error on the M1-exploit episodes of round 3 (held out for every model)
	res['error'] = {}
	for name in ACTORS:
		ag = agent0 if name == 'PRE' else with_dyn(dyn[name])
		res['error'][name] = model_error(ag.model, cfg, data3['obs'], data3['act'])
	print(a.cell, s, 'error', {k: round(v['loss'], 5) for k, v in res['error'].items()}, flush=True)
	res['seconds'] = time.time() - t_all
	dump(f'{folder}/done.json', res)
	print(a.cell, s, 'done', f'{res["seconds"]:.0f}s', flush=True)


def cmd_oracle(a):
	from d04 import _winit
	from d06 import OracleCon
	task, cond = cell_parts(a.cell)
	folder = f'{a.out_root}/oracle/{a.cell}'
	os.makedirs(folder, exist_ok=True)
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, cond))   # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	ora = OracleCon(agent, env, pool, a.workers)
	for s in range(a.seed_from, a.seed_to + 1):
		if os.path.exists(f'{folder}/s{s:02d}.json'):
			continue
		t0 = time.time()
		eps, recs = run_cond(ora, env, shifter, domain, cond, a.eps, eval_mode=True, seed=3000 + s, record=True)
		np.savez_compressed(f'{folder}/rec_s{s:02d}.npz', state0=np.stack([np.asarray(r['state'][0]) for r in recs]),
							rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
		dump(f'{folder}/s{s:02d}.json', dict(cell=a.cell, seed=s, summary=summary(eps), episodes=eps, seconds=time.time() - t0))
		print(a.cell, 'oracle consensus', s, {k: round(v, 2) for k, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- summary (registered analysis, design 08 §5)

def interval(x, seed=0):
	x = np.asarray(x, float)
	n = len(x)
	if n < 2:
		return dict(mean=float(x.mean()) if n else float('nan'), t=[float('nan')] * 2, boot=[float('nan')] * 2, n=n)
	if n - 1 in TQ:
		q = TQ[n - 1]
	else:
		from scipy import stats
		q = float(stats.t.ppf(1 - (1 - LEVEL) / 2, n - 1))
	h = q * x.std(ddof=1) / np.sqrt(n)
	bs = x[np.random.RandomState(seed).randint(0, n, (10000, n))].mean(1)
	p = 100 * (1 - LEVEL) / 2
	return dict(mean=float(x.mean()), t=[float(x.mean() - h), float(x.mean() + h)], boot=[float(v) for v in np.percentile(bs, [p, 100 - p])],
				n=n, sd_seed=float(x.std(ddof=1)))


def verdict(iv, margin=0.0):
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return '통과' if lo > margin else ('실패' if hi < margin else '미결')


def cmd_summarize(a):
	L, S = [], {}
	fmt = lambda iv, d=1: f'{iv["mean"]:+.{d}f} [{iv["t"][0]:+.{d}f}, {iv["t"][1]:+.{d}f}]'
	L.append('# 설계 08 결과 요약 (자동 생성, `d08.py summarize`)\n')
	L.append('k는 둘째 라운드 5 에피소드 가운데 적응 모델만 쓰는 planner가 모은 에피소드 수다(나머지는 적응 없는 planner). k=0과 k=5는 설계 07의 대조와 적응 수집이다. '
			 '배치는 모두 합의 planner다. 구간은 seed 단위 t 구간 95 %다.\n')
	for cell in CELLS:
		pairs = []
		for f in sorted(glob.glob(f'{a.out_root}/{cell}/s*/done.json')):
			g = f.replace(a.out_root, a.d07_root)
			if os.path.exists(g):
				pairs.append((json.load(open(g)), json.load(open(f))))
		if not pairs:
			continue
		n = len(pairs)
		C = S[cell] = dict(n_seeds=n, seeds=[p[1]['seed'] for p in pairs])
		L.append(f'\n## {cell} (seed {n}개)\n')
		dep = lambda r7, r8, k: (r7['CTRL']['eval']['shift'] if k == 0 else r7['ADP']['eval']['shift'] if k == 5 else r8['dose'][f'K{k}'])['episodes']
		q = lambda k, key: np.array([np.mean([e[key] for e in dep(r7, r8, k)]) for r7, r8 in pairs])
		pre = np.array([np.mean([e['ret'] for e in r7['PRE']['shift']['episodes']]) for r7, _ in pairs])
		col = lambda k: np.array([np.mean([e['ret'] for e in r7['ADP']['round2']['episodes'][:k] + r7['CTRL']['round2']['episodes'][k:]]) for r7, _ in pairs])
		colf = lambda k: np.array([np.mean([e['fallen_frac'] for e in r7['ADP']['round2']['episodes'][:k] + r7['CTRL']['round2']['episodes'][k:]]) for r7, _ in pairs])
		ret = {k: q(k, 'ret') for k in (0, 1, 2, 5)}

		L.append('### 용량-반응 (배치)\n')
		L.append('| k | 배치 return | 적응 없음 대비 | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 | 둘째 라운드 수집 return (넘어져 있는 시간) |')
		L.append('|---|---|---|---|---|---|')
		for k in (0, 1, 2, 5):
			L.append(f'| {k} | {ret[k].mean():.0f} | {fmt(interval(ret[k] - pre))} | {q(k, "falls").mean():.1f} / {q(k, "fallen_frac").mean():.2f} | '
					 f'{q(k, "smooth").mean():.2f} / {q(k, "effort").mean():.2f} / {q(k, "angvel_rms").mean():.2f} | {col(k).mean():.0f} ({colf(k).mean():.2f}) |')
		T = C['dose'] = dict(ret={k: float(ret[k].mean()) for k in ret}, col={k: float(col(k).mean()) for k in ret})
		T['gain5'] = interval(ret[5] - ret[0])
		L.append('\n| 판정 | 비교 | 평균 [t 구간 95 %] | 결과 |')
		L.append('|---|---|---|---|')
		L.append(f'| (기준) | k=5 − k=0 | {fmt(T["gain5"])} | – |')
		for k in (1, 2):
			T[f'gain{k}'] = interval(ret[k] - ret[0])
			T[f'half{k}'] = interval(ret[k] - 0.5 * (ret[0] + ret[5]))
			T[f'cost{k}'] = interval(col(k) - col(0))
			L.append(f'| 이득 | k={k} − k=0 | {fmt(T[f"gain{k}"])} | {verdict(T[f"gain{k}"])} |')
			L.append(f'| D{k} 절반 넘음 | k={k} − (k=0 + k=5)/2 | {fmt(T[f"half{k}"])} | {verdict(T[f"half{k}"])} |')
		T['cost5'] = interval(col(5) - col(0))
		L.append('\n수집 return의 대조 대비 차이: ' + ', '.join(f'k={k} {fmt(T[f"cost{k}"])}' for k in (1, 2, 5)) + '.')

		L.append('\n### 다음 라운드의 수집 (보통 planner, 탐색 잡음 켬, 5 에피소드)\n')
		c3 = {m: np.array([r8['round3'][m]['summary']['ret'] for _, r8 in pairs]) for m in ACTORS}
		f3 = {m: np.array([r8['round3'][m]['summary']['fallen_frac'] for _, r8 in pairs]) for m in ACTORS}
		n3 = {m: np.array([r8['round3'][m]['summary']['falls'] for _, r8 in pairs]) for m in ACTORS}
		er = {m: np.array([r8['error'][m]['loss'] for _, r8 in pairs]) for m in ACTORS}
		e1 = {m: np.array([r8['error'][m]['one_step'] for _, r8 in pairs]) for m in ACTORS}
		L.append('| planner가 쓰는 모델 | 수집 return | 적응 없음 대비 | 넘어짐 사건 / 넘어져 있는 시간 | 첫 라운드 모델 수집 데이터에서의 모델 오차 (첫 라운드 모델 = 1) |')
		L.append('|---|---|---|---|---|')
		for m in ACTORS:
			L.append(f'| {ANAME[m]} | {c3[m].mean():.0f} | {fmt(interval(c3[m] - c3["PRE"]))} | {n3[m].mean():.1f} / {f3[m].mean():.2f} | {np.mean(er[m] / er["M1"]):.2f} |')
		R = C['round3'] = dict(col={m: float(c3[m].mean()) for m in ACTORS}, fallen={m: float(f3[m].mean()) for m in ACTORS},
							   err_ratio={m: float(np.mean(er[m] / er['M1'])) for m in ACTORS}, err1_ratio={m: float(np.mean(e1[m] / e1['M1'])) for m in ACTORS})
		L.append('\n| 판정 | 비교 | 평균 [t 구간 95 %] | 결과 |')
		L.append('|---|---|---|---|')
		for m, tag in (('ADP', 'R5'), ('K1', 'R1'), ('K2', 'R2'), ('CON', '(보고)'), ('CTRL', '(보고)')):
			R[f'rec_{m}'] = interval(c3[m] - 0.5 * (c3['M1'] + c3['PRE']))
			L.append(f'| {tag} 비용 절반 넘게 회복 | {ANAME[m]} − (첫 라운드 모델 + 적응 없음)/2 | {fmt(R[f"rec_{m}"])} | {verdict(R[f"rec_{m}"])} |')
		R['dd'] = interval(((er['CTRL'] - er['K1']) - (er['K1'] - er['K2'])) / er['M1'])
		R['e01'] = interval((er['CTRL'] - er['K1']) / er['M1'])
		R['e12'] = interval((er['K1'] - er['K2']) / er['M1'])
		L.append(f'| E 첫 에피소드가 더 많이 고친다 | 오차 감소 (k=0→1) − (k=1→2), 첫 라운드 모델 오차 = 1 | {fmt(R["dd"], 3)} | {verdict(R["dd"])} |')
		L.append(f'\n오차 감소(첫 라운드 모델 오차 = 1): k=0→1 {fmt(R["e01"], 3)}, k=1→2 {fmt(R["e12"], 3)}.')

		# descriptive: where in the episode the adapted collector loses reward (design-07 round-2 data)
		blocks = {}
		for arm in ('CTRL', 'ADP'):
			rews = [np.load(f'{a.d07_root}/{cell}/s{s:02d}/data2_{arm}.npz')['rew'] for s in C['seeds']]
			blocks[arm] = np.concatenate(rews).reshape(-1, 5, 100).sum(2).mean(0)
		C['blocks'] = {k: [float(x) for x in v] for k, v in blocks.items()}
		L.append('\n둘째 라운드 수집의 100 step 구간별 reward 합(설계 07 데이터): 대조 ' + ' / '.join(f'{x:.0f}' for x in blocks['CTRL'])
				 + ', 적응 수집 ' + ' / '.join(f'{x:.0f}' for x in blocks['ADP']) + '.')

		# oracle consensus on the evaluation seeds (first ORA_EPS episodes of each seed)
		ora = {}
		for f in sorted(glob.glob(f'{a.out_root}/oracle/{cell}/s*.json')):
			d = json.load(open(f))
			ora[d['seed']] = d
		if ora:
			seeds = [s for s in C['seeds'] if s in ora]
			by = {p[1]['seed']: p for p in pairs}
			o = np.array([ora[s]['summary']['ret'] for s in seeds])
			ne = len(ora[seeds[0]]['episodes'])
			same = lambda eps: float(np.mean([e['ret'] for e in eps[:ne]]))
			O = C['oracle'] = dict(n_seeds=len(seeds), eps=ne, ret=interval(o),
								   falls=float(np.mean([ora[s]['summary']['falls'] for s in seeds])), fallen=float(np.mean([ora[s]['summary']['fallen_frac'] for s in seeds])))
			L.append(f'\n### 정확한 dynamics 합의 재측정 (seed {len(seeds)}개 × 에피소드 {ne}개, 판정 없음)\n')
			L.append(f'배치 return {O["ret"]["mean"]:.0f} [{O["ret"]["t"][0]:.0f}, {O["ret"]["t"][1]:.0f}], 넘어짐 사건 {O["falls"]:.1f}, 넘어져 있는 시간 {O["fallen"]:.2f}.\n')
			L.append('| 비교 (같은 에피소드끼리 짝) | 평균 [t 구간 95 %] | (참고) 팔의 전체 에피소드 평균과의 차이 |')
			L.append('|---|---|---|')
			arms = [('적응 없음', lambda r7, r8: r7['PRE']['shift']['episodes']), ('k=0 (대조)', lambda r7, r8: dep(r7, r8, 0)),
					('합의 수집', lambda r7, r8: r7['CON']['eval']['shift']['episodes']), ('k=1', lambda r7, r8: dep(r7, r8, 1)),
					('k=2', lambda r7, r8: dep(r7, r8, 2)), ('k=5 (적응 수집)', lambda r7, r8: dep(r7, r8, 5))]
			for name, get in arms:
				x = np.array([same(get(*by[s])) for s in seeds])
				full = float(np.mean([np.mean([e['ret'] for e in get(*p)]) for p in pairs]))
				O[name] = dict(paired=interval(o - x), unpaired=float(o.mean() - full))
				L.append(f'| 정확한 dynamics 합의 − {name} | {fmt(O[name]["paired"])} | {o.mean() - full:+.1f} |')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed', type=int, required=True)
	p.add_argument('--r_eps', type=int, default=R_EPS)
	p.add_argument('--eval_eps', type=int, default=EVAL_EPS)
	p.add_argument('--ft_steps', type=int, default=FT_STEPS)
	p.add_argument('--check', action='store_true', help='also re-fit k=5 and compare with the stored design-07 model')
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('oracle')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=ORA_SEEDS)
	p.add_argument('--eps', type=int, default=ORA_EPS)
	p.add_argument('--workers', type=int, default=28)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'd08'))
	a = ap.parse_args()
	dict(run=cmd_run, oracle=cmd_oracle, summarize=cmd_summarize)[a.cmd](a)
