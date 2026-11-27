"""Design 09 (research/docs/design09_mismatch_scope_ko.md): scope of the exact-dynamics / stale-value mismatch (measurement, no training).

One process = one (cell, pretrained checkpoint, horizon, protocol, planner), looping over seeds.
  planner  PRE     official planner, pretrained (sim) model
           ORA     design-04 planner: candidates rolled out in the true simulator, pretrained Q at the true state
           CONORA  design-06 consensus of the sim model and the exact rollout
           ORAPEN  ORA with the FOWM penalty of design 02 (lam x std of the Q ensemble) at the true states (--lam 1|3)
  proto    deploy  shifted physics, no exploration noise, seed 3000 + s   (design 07 evaluation protocol)
           collect shifted physics, exploration noise on, seed 5000 + s   (design 08 round-3 protocol)
           nominal nominal physics, no exploration noise, seed 3000 + s

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d09.py run --cell cheetah-run__friction_0.15 --planner ORA [--ckpt 1 --horizon 3 --proto deploy --seed_from 1 --seed_to 16 --eps 2 --workers 40]
  GPU=N research/scripts/td.sh research/scripts/d09.py selferr --cell cheetah-run__friction_0.15 --seed_to 32
  GPU=N research/scripts/td.sh research/scripts/d09.py summarize --res_dir /mnt/newssd/cjkim/td/d09/res
Smoke tests must pass --out_root (another folder); `run --timing 20` only times 20 planning calls.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from common import math as tdmath
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

from d01 import CFG_PATH, Shifter, cell_parts, domain_of, dump, phys_of, run_cond, summary

D07 = '/mnt/newssd/cjkim/td/d07'
D08 = '/mnt/newssd/cjkim/td/d08'
OUT = '/mnt/newssd/cjkim/td/d09'
CKPT = '/mnt/newssd/cjkim/tdmpc2_ckpt/{task}-{k}.pt'
CF, QM = 'cheetah-run__friction_0.15', 'quadruped-run__motor_0.35'
OTHER = ['walker-run__mass_1.5', 'walker-run__friction_0.15', 'quadruped-run__friction_0.15', 'cheetah-run__mass_2.0', QM]
SHORT = {'walker-run__mass_1.5': 'walker 질량', 'walker-run__friction_0.15': 'walker 마찰', 'quadruped-run__friction_0.15': 'quadruped 마찰',
		 'cheetah-run__mass_2.0': 'cheetah 질량', QM: 'quadruped 모터', CF: 'cheetah 마찰'}
HS = (10, 25)
V_RATIO = 0.9
TQ = {15: 2.1314, 7: 2.3646, 31: 2.0395, 2: 4.3027}                         # t quantile 0.975 (two-sided 95 %)
ACTORS = ['PRE', 'M1', 'CTRL', 'K1', 'K2', 'CON', 'ADP']
ANAME = {'PRE': '적응 없음', 'M1': '첫 라운드 모델', 'CTRL': 'k=0 (대조)', 'K1': 'k=1', 'K2': 'k=2', 'CON': '합의 수집', 'ADP': 'k=5 (적응 수집)'}


def make_h(task, ckpt=1, horizon=3, seed=1):
	"""d01.make with the pretrained checkpoint index and the planning horizon given explicitly."""
	cfg = OmegaConf.load(CFG_PATH)
	cfg.task, cfg.model_size, cfg.checkpoint = task, 5, CKPT.format(task=task, k=ckpt)
	cfg.compile, cfg.iterations, cfg.seed, cfg.horizon = False, 6, seed, horizon
	cfg.save_video, cfg.enable_wandb = False, False
	cfg = parse_cfg(cfg)
	set_seed(seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	return cfg, env, agent


def _wroll_all(args):
	"""Worker: like d04._wroll but returns the observation after every step (N x H x D)."""
	import d04
	state, seqs = args
	R = np.empty((len(seqs), seqs.shape[1]), np.float32)
	O = []
	for i, s in enumerate(seqs):
		outs, rews = d04._W.rollout_r(state, s)
		R[i] = rews
		O.append(outs)
	return R, np.asarray(O, np.float32)


def oracle_pen(agent, env, pool, workers, lam):
	from d04 import Oracle

	class OraclePen(Oracle):
		"""ORA with design 02's FOWM penalty: every step of the exact rollout and the terminal value lose lam x std of the Q ensemble."""

		@torch.no_grad()
		def values(self, state, actions):
			cfg, m = self.cfg, self.m
			seqs = actions.permute(1, 0, 2).cpu().numpy().astype(np.float32)
			chunks = [c for c in np.array_split(seqs, self.W) if len(c)]
			outs = self.pool.map(_wroll_all, [(state, c) for c in chunks], chunksize=1)
			R = torch.as_tensor(np.concatenate([o[0] for o in outs]), dtype=torch.float32, device='cuda')
			O = torch.as_tensor(np.concatenate([o[1] for o in outs]), dtype=torch.float32, device='cuda')
			z = self.z0.repeat(R.shape[0], 1)
			G, disc = 0, 1
			for t in range(cfg.horizon):
				u = tdmath.two_hot_inv(m.Q(z, actions[t], None, return_type='all'), cfg).std(0)
				G = G + disc * (R[:, t:t + 1] - lam * u)
				disc = disc * self.gamma
				z = m.encode(O[:, t], None)
			action, _ = m.pi(z, None)
			u = tdmath.two_hot_inv(m.Q(z, action, None, return_type='all'), cfg).std(0)
			return G + disc * (m.Q(z, action, None, return_type='avg') - lam * u)

		@torch.no_grad()
		def act(self, obs, t0=False, eval_mode=False):
			self.z0 = self.m.encode(obs.to('cuda').unsqueeze(0), None)
			return super().act(obs, t0=t0, eval_mode=eval_mode)

	return OraclePen(agent, env, pool, workers)


def pname(a):
	return f'ORAPEN{a.lam:g}' if a.planner == 'ORAPEN' else a.planner


def run_dir(root, cell, ckpt, horizon, proto, planner):
	base = f'{root}/nominal/{cell_parts(cell)[0]}' if proto == 'nominal' else f'{root}/{cell}'
	return f'{base}/k{ckpt}_h{horizon}_{proto}_{planner}'


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	real = 'nominal' if a.proto == 'nominal' else cond
	folder = run_dir(a.out_root, a.cell, a.ckpt, a.horizon, a.proto, pname(a))
	os.makedirs(folder, exist_ok=True)
	pool = None
	if a.planner != 'PRE':
		from d04 import _winit
		pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, real))  # before any CUDA use
	cfg, env, agent = make_h(task, a.ckpt, a.horizon)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	if a.planner == 'PRE':
		actor = agent
	else:
		from d04 import Oracle
		from d06 import OracleCon
		actor = oracle_pen(agent, env, pool, a.workers, a.lam) if a.planner == 'ORAPEN' else (Oracle if a.planner == 'ORA' else OracleCon)(agent, env, pool, a.workers)
	if a.timing:
		shifter.set(real)
		obs = env.reset()
		tic = time.time()
		for i in range(a.timing):
			actor.act(obs, t0=i == 0, eval_mode=True)
		print(a.cell, pname(a), f'k{a.ckpt} h{a.horizon}', f'{(time.time() - tic) / a.timing:.3f} s/plan', flush=True)
		if a.planner == 'ORAPEN':                                           # code check: the same candidates scored by ORA and by ORAPEN
			ref = Oracle(agent, env, pool, a.workers)
			state = phys_of(env).get_state().copy()
			acts = torch.rand(cfg.horizon, cfg.num_samples, cfg.action_dim, device='cuda') * 2 - 1
			actor.z0 = agent.model.encode(obs.to('cuda').unsqueeze(0), None)
			torch.manual_seed(0)
			v0 = ref.values(state, acts)
			torch.manual_seed(0)
			v1 = actor.values(state, acts)
			print(f'check: ORA − ORAPEN (lam {a.lam:g}) over {len(v0)} candidates: mean {float((v0 - v1).mean()):.5f}, max abs {float((v0 - v1).abs().max()):.5f}, ORA value sd {float(v0.std()):.3f}', flush=True)
		return
	base = 5000 if a.proto == 'collect' else 3000
	for s in range(a.seed_from, a.seed_to + 1):
		if os.path.exists(f'{folder}/s{s:02d}.json'):
			continue
		t0 = time.time()
		eps, recs = run_cond(actor, env, shifter, domain, real, a.eps, eval_mode=a.proto != 'collect', seed=base + s, record=True)
		np.savez_compressed(f'{folder}/rec_s{s:02d}.npz', state0=np.stack([np.asarray(r['state'][0]) for r in recs]),
							rew=np.stack([np.asarray(r['rew'], np.float32) for r in recs]))
		dump(f'{folder}/s{s:02d}.json', dict(cell=a.cell, seed=s, args=vars(a), summary=summary(eps), episodes=eps, seconds=time.time() - t0))
		print(a.cell, pname(a), f'k{a.ckpt} h{a.horizon} {a.proto}', s, {k: round(v, 2) for k, v in summary(eps).items()}, f'{time.time() - t0:.0f}s', flush=True)
	if pool is not None:
		pool.close()


# ----------------------------------------------------------------------------- check 1: model error at each planner's own states (design 08 data)

@torch.no_grad()
def self_stats(m, cfg, d, gamma):
	from d08 import model_error
	e = model_error(m, cfg, d['obs'], d['act'])
	obs = torch.as_tensor(d['obs'], dtype=torch.float32, device='cuda')
	act = torch.as_tensor(d['act'], dtype=torch.float32, device='cuda')
	rew = torch.as_tensor(d['rew'], dtype=torch.float32, device='cuda')
	E, T = act.shape[:2]
	H = cfg.horizon
	n = T - H + 1
	z = m.encode(obs.reshape(E * (T + 1), -1), None).reshape(E, T + 1, -1)[:, :n].reshape(E * n, -1)
	pred, true = 0, 0
	for h in range(H):
		ah = act[:, h:h + n].reshape(E * n, -1)
		pred = pred + gamma ** h * tdmath.two_hot_inv(m.reward(z, ah, None), cfg).squeeze(-1)
		true = true + gamma ** h * rew[:, h:h + n].reshape(E * n)
		z = m.next(z, ah, None)
	e.update(rew3_pred=float(pred.mean()), rew3_true=float(true.mean()))
	return e


def cmd_selferr(a):
	from d07 import fresh
	task, _ = cell_parts(a.cell)
	cfg, env, agent0 = make_h(task)
	gamma = float(agent0.discount)
	for s in range(a.seed_from, a.seed_to + 1):
		src7, src8 = f'{a.d07_root}/{a.cell}/s{s:02d}', f'{a.d08_root}/{a.cell}/s{s:02d}'
		out = f'{a.out_root}/selferr/{a.cell}/s{s:02d}.json'
		if os.path.exists(out) or not os.path.exists(f'{src8}/done.json'):
			continue
		dyn = {'M1': f'{src7}/M1_dyn.pt', 'CTRL': f'{src7}/M2_CTRL_dyn.pt', 'CON': f'{src7}/M2_CON_dyn.pt', 'ADP': f'{src7}/M2_ADP_dyn.pt',
			   'K1': f'{src8}/M2_k1_dyn.pt', 'K2': f'{src8}/M2_k2_dyn.pt'}
		data = {n: {k: v for k, v in np.load(f'{src8}/data3_{n}.npz').items()} for n in ACTORS}
		res = dict(cell=a.cell, seed=s, own={}, on_pre={}, on_m1={})
		for n in ACTORS:
			ag = agent0
			if n != 'PRE':
				ag = fresh(cfg)
				ag.model._dynamics.load_state_dict(torch.load(dyn[n]))
			res['own'][n] = self_stats(ag.model, cfg, data[n], gamma)        # the model at the states its own planner visited
			res['on_pre'][n] = self_stats(ag.model, cfg, data['PRE'], gamma)  # the same model at the un-adapted planner's states
			res['on_m1'][n] = self_stats(ag.model, cfg, data['M1'], gamma)    # design 08's held-out exploit states (cross-check)
		dump(out, res)
		print(a.cell, s, 'own-state loss', {n: round(res['own'][n]['loss'], 5) for n in ACTORS}, flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 09 §5)

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


def fmt(iv, d=1):
	if not np.isfinite(iv['t'][0]):
		return f'{iv["mean"]:+.{d}f} (seed {iv["n"]}개)'
	return f'{iv["mean"]:+.{d}f} [{iv["t"][0]:+.{d}f}, {iv["t"][1]:+.{d}f}]'


def three(iv):
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return '아래' if hi < 0 else ('위' if lo > 0 else '미결')


def _stack(eps_by_seed, n):
	"""list (seeds) of episode-dict lists -> dict of arrays seeds x n."""
	return {k: np.array([[e[k] for e in eps[:n]] for eps in eps_by_seed]) for k in ('ret', 'falls', 'fallen_frac', 'smooth', 'effort', 'angvel_rms')}


def get(a, cell, ckpt, horizon, proto, planner, seeds, n):
	"""Episodes of one run as arrays seeds x n, or None if any seed is missing. Reuses design 07 / 08 results where they are the same run."""
	eps = []
	for s in seeds:
		f = f'{run_dir(a.out_root, cell, ckpt, horizon, proto, planner)}/s{s:02d}.json'
		if os.path.exists(f):
			e = json.load(open(f))['episodes']
		elif ckpt == 1 and horizon == 3 and proto == 'deploy' and planner == 'CONORA' and os.path.exists(f'{a.d08_root}/oracle/{cell}/s{s:02d}.json'):
			e = json.load(open(f'{a.d08_root}/oracle/{cell}/s{s:02d}.json'))['episodes']
		elif ckpt == 1 and horizon == 3 and proto == 'deploy' and planner == 'PRE' and os.path.exists(f'{a.d07_root}/{cell}/s{s:02d}/done.json'):
			e = json.load(open(f'{a.d07_root}/{cell}/s{s:02d}/done.json'))['PRE']['shift']['episodes']
		else:
			return None
		if len(e) < n:
			return None
		eps.append(e)
	return _stack(eps, n)


def get_r3(a, cell, name, seeds, n):
	eps = []
	for s in seeds:
		f = f'{a.d08_root}/{cell}/s{s:02d}/done.json'
		if not os.path.exists(f):
			return None
		eps.append(json.load(open(f))['round3'][name]['episodes'])
	return _stack(eps, n)


def blocks(a, cell, ckpt, horizon, proto, planner, seeds, n):
	out = []
	for s in seeds:
		f = f'{run_dir(a.out_root, cell, ckpt, horizon, proto, planner)}/rec_s{s:02d}.npz'
		if not os.path.exists(f):
			return None
		out.append(np.load(f)['rew'][:n])
	r = np.concatenate(out)
	return [float(r[:, i:i + 100].sum(1).mean()) for i in range(0, r.shape[1], 100)]


def row(name, d):
	return (f'| {name} | {d["ret"].mean():.0f} | {d["falls"].mean():.1f} / {d["fallen_frac"].mean():.2f} | '
			f'{d["smooth"].mean():.2f} / {d["effort"].mean():.2f} / {d["angvel_rms"].mean():.2f} |')


def trace(a, cell, k, planner):
	"""Not registered (design 09 section F): episodes x metrics and per-step rewards of one planner from wherever the run is stored."""
	E, R = [], []
	d = run_dir(a.out_root, cell, k, 3, 'deploy', planner)
	for f in sorted(glob.glob(f'{d}/s*.json')):
		r = f'{d}/rec_{os.path.basename(f)[:-5]}.npz'
		if os.path.exists(r):
			E += json.load(open(f))['episodes']
			R.append(np.load(r)['rew'])
	if not E and k == 1:
		if planner == 'PRE':
			for s in range(1, 17):
				f7, f10 = f'{a.d07_root}/{cell}/s{s:02d}', f'{a.d10_root}/{cell}/s{s:02d}'
				if os.path.exists(f'{f7}/done.json') and os.path.exists(f'{f7}/rec_PRE.npz'):
					E += json.load(open(f'{f7}/done.json'))['PRE']['shift']['episodes'][:2]
					R.append(np.load(f'{f7}/rec_PRE.npz')['rew'][:2])
				elif os.path.exists(f'{f10}/eval_PRE.json') and os.path.exists(f'{f10}/rec_PRE.npz'):
					E += json.load(open(f'{f10}/eval_PRE.json'))['shift']['episodes']
					R.append(np.load(f'{f10}/rec_PRE.npz')['rew'])
		else:
			root = a.d04_root if planner == 'ORA' else a.d06_root
			f = f'{root}/{cell}/eval_{planner}.json'
			if os.path.exists(f):
				E += json.load(open(f))['episodes']
				R.append(np.load(f'{root}/{cell}/rec_{planner}.npz')['rew'])
	if not E:
		return None
	R = np.concatenate(R)
	if len(R) != len(E):
		return None
	return dict(n=len(E), ret=float(np.mean([e['ret'] for e in E])), first100=float(R[:, :100].sum(1).mean()),
				effort=float(np.mean([e['effort'] for e in E])), smooth=float(np.mean([e['smooth'] for e in E])))


HEAD = ['| planner | return | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 |', '|---|---|---|---|']


def cmd_summarize(a):
	L, S = [], {}
	S16, S8, S3 = list(range(1, 17)), list(range(1, 9)), list(range(1, 4))
	sm = lambda d: d['ret'].mean(1)                                          # seed-level value
	L.append('# 설계 09 결과 요약 (자동 생성, `d09.py summarize`)\n')
	L.append('PRE = 적응 없는 planner, ORA = 정확한 dynamics만 쓰는 planner(사전학습 가치), CON-ORA = sim 모델과 정확한 dynamics의 합의. 구간은 seed 단위 t 구간 95 %다. "아래/위/미결"은 구간이 전부 0 아래/전부 0 위/0을 포함한다는 뜻이다.\n')

	def triple(cell, ckpt, seeds, n, title, key):
		pre, ora, con = (get(a, cell, ckpt, 3, 'deploy', p, seeds, n) for p in ('PRE', 'ORA', 'CONORA'))
		L.append(f'\n### {title} (seed {len(seeds)}개 × 에피소드 {n}개, 배치 규약)\n')
		if pre is None or ora is None:
			L.append('(결과 없음)')
			return
		L.extend(HEAD)
		L.append(row('PRE', pre))
		L.append(row('ORA', ora))
		if con is not None:
			L.append(row('CON-ORA', con))
		L.append('\n| 비교 (같은 에피소드끼리 짝) | 평균 [t 구간 95 %] | 판정 |')
		L.append('|---|---|---|')
		S[key] = {}
		pairs = [('ORA − PRE', ora, pre)] + ([('CON-ORA − ORA', con, ora), ('CON-ORA − PRE', con, pre)] if con is not None else [])
		for nm, x, y in pairs:
			iv = interval(sm(x) - sm(y))
			S[key][nm] = iv
			L.append(f'| {nm} | {fmt(iv)} | {three(iv)} |')
		S[key]['ret'] = dict(PRE=float(pre['ret'].mean()), ORA=float(ora['ret'].mean()), CONORA=None if con is None else float(con['ret'].mean()))

	# ---- A: check 2
	L.append('\n## A. 확인 2: cheetah 마찰, 체크포인트 1, 호라이즌 3\n')
	triple(CF, 1, S16, 2, 'A1 배치 규약', 'A1')
	if 'A1' in S:
		v = three(S['A1']['ORA − PRE'])
		S['M1'] = {'아래': '유지', '위': '반증', '미결': '미결'}.get(v, v)
		L.append(f'\n- **주 판정 M1 (ORA − PRE):** {S["M1"]}')
		b = blocks(a, CF, 1, 3, 'deploy', 'ORA', S16, 2)
		if b:
			L.append('- ORA의 100 step 구간별 reward 합: ' + ' / '.join(f'{v:.0f}' for v in b))
	L.append('\n### A3 정확한 dynamics + 가치 불확실성 벌점 (배치 규약, seed 16개 × 에피소드 2개)\n')
	pre, ora, con = (get(a, CF, 1, 3, 'deploy', p, S16, 2) for p in ('PRE', 'ORA', 'CONORA'))
	S['A3'] = {}
	for lam in (1, 3):
		pen = get(a, CF, 1, 3, 'deploy', f'ORAPEN{lam}', S16, 2)
		if pen is None or pre is None or ora is None:
			L.append(f'- 벌점 계수 {lam}: (결과 없음)')
			continue
		if not S['A3']:
			L.extend(HEAD)
		L.append(row(f'ORA + 벌점 (계수 {lam})', pen))
		S['A3'][lam] = {nm: interval(sm(x) - sm(y)) for nm, x, y in (('− ORA', pen, ora), ('− PRE', pen, pre)) + ((('CON-ORA −', con, pen),) if con is not None else ())}
	if S['A3']:
		L.append('\n| 비교 (같은 에피소드끼리 짝) | 평균 [t 구간 95 %] | 판정 |')
		L.append('|---|---|---|')
		for lam, d in S['A3'].items():
			for nm, iv in d.items():
				lab = f'CON-ORA − (ORA + 벌점 {lam})' if nm.startswith('CON') else f'(ORA + 벌점 {lam}) {nm}'
				L.append(f'| {lab} | {fmt(iv)} | {three(iv)} |')
	ora_c = get(a, CF, 1, 3, 'collect', 'ORA', S16, 2)
	L.append('\n### A2 수집 규약 (탐색 잡음 켬, seed 16개 × 에피소드 2개)\n')
	if ora_c is None:
		L.append('(결과 없음)')
	else:
		r3 = {n: get_r3(a, CF, n, S16, 2) for n in ('PRE', 'M1', 'ADP')}
		L.extend(HEAD)
		L.append(row('PRE (설계 08 셋째 라운드)', r3['PRE']))
		L.append(row('ORA', ora_c))
		L.append(row('첫 라운드 모델만 (설계 08)', r3['M1']))
		L.append(row('k=5 모델만 (설계 08)', r3['ADP']))
		L.append('\n| 수집 손실의 분해 | 평균 [t 구간 95 %] | 판정 |')
		L.append('|---|---|---|')
		S['A2'] = {}
		for nm, x, y in (('가치 쪽 몫: ORA − PRE', ora_c, r3['PRE']), ('모델 쪽 몫: 첫 라운드 모델만 − ORA', r3['M1'], ora_c), ('모델 쪽 몫: k=5 모델만 − ORA', r3['ADP'], ora_c)):
			iv = interval(sm(x) - sm(y))
			S['A2'][nm] = iv
			L.append(f'| {nm} | {fmt(iv)} | {three(iv)} |')
		b = blocks(a, CF, 1, 3, 'collect', 'ORA', S16, 2)
		if b:
			L.append('\n- ORA의 100 step 구간별 reward 합: ' + ' / '.join(f'{v:.0f}' for v in b))

	# ---- B: horizon
	L.append('\n## B. 호라이즌: cheetah 마찰, 체크포인트 1\n')
	pre3, ora3 = get(a, CF, 1, 3, 'deploy', 'PRE', S16, 1), get(a, CF, 1, 3, 'deploy', 'ORA', S16, 1)
	nom3 = get(a, CF, 1, 3, 'nominal', 'PRE', S3, 1)
	L.append('| 호라이즌 | 평소 조건 ORA / PRE(3) | 유효 | ORA return | ORA(H) − PRE(3) | 판정 | ORA(H) − ORA(3) | PRE(H) return | PRE(H) − PRE(3) |')
	L.append('|---|---|---|---|---|---|---|---|---|')
	S['B'] = {}
	if pre3 is not None and ora3 is not None:
		L.append(f'| 3 | – | – | {ora3["ret"].mean():.0f} | {fmt(interval(sm(ora3) - sm(pre3)))} | {three(interval(sm(ora3) - sm(pre3)))} | – | {pre3["ret"].mean():.0f} | – |')
	for H in HS:
		oh, ph = get(a, CF, 1, H, 'deploy', 'ORA', S16, 1), get(a, CF, 1, H, 'deploy', 'PRE', S16, 1)
		on = get(a, CF, 1, H, 'nominal', 'ORA', S3, 1)
		ratio = None if on is None or nom3 is None else float(on['ret'].mean() / nom3['ret'].mean())
		valid = None if ratio is None else bool(ratio >= V_RATIO)
		c = [f'{H}', '(없음)' if ratio is None else f'{ratio:.2f}', '(없음)' if valid is None else ('예' if valid else '아니오')]
		S['B'][H] = dict(nominal_ratio=ratio, valid=valid)
		if oh is not None and pre3 is not None and ora3 is not None:
			iv, iv3 = interval(sm(oh) - sm(pre3)), interval(sm(oh) - sm(ora3))
			v = '판정 불가' if valid is False else {'아래': '유지', '위': '사라짐', '미결': '미결'}[three(iv)]
			S['B'][H].update(ora=float(oh['ret'].mean()), vs_pre3=iv, vs_ora3=iv3, verdict=v)
			c += [f'{oh["ret"].mean():.0f}', fmt(iv), v, fmt(iv3)]
		else:
			c += ['(없음)'] * 4
		if ph is not None and pre3 is not None:
			ivp = interval(sm(ph) - sm(pre3))
			S['B'][H]['pre_h'] = ivp
			c += [f'{ph["ret"].mean():.0f}', fmt(ivp)]
		else:
			c += ['(없음)'] * 2
		L.append('| ' + ' | '.join(c) + ' |')
	for H in HS:
		b = blocks(a, CF, 1, H, 'deploy', 'ORA', S16, 1)
		if b:
			L.append(f'\n- ORA(호라이즌 {H})의 100 step 구간별 reward 합: ' + ' / '.join(f'{v:.0f}' for v in b))

	# ---- C: checkpoints
	L.append('\n## C. 다른 사전학습 체크포인트\n')
	for k in (2, 3):
		triple(CF, k, S16, 2, f'C1 cheetah 마찰, 체크포인트 {k}', f'C1_k{k}')
	rep = [k for k in (1, 2, 3) if three(S.get('A1' if k == 1 else f'C1_k{k}', {}).get('ORA − PRE', dict(t=[float('nan')] * 2))) == '아래']
	S['K_reproduced'] = rep
	L.append(f'\n- **cheetah 마찰에서 ORA − PRE가 "아래"인 체크포인트:** {rep if rep else "없음"} (3개 중)')
	L.append('\n### C2 나머지 5개 조건 (seed 8개 × 에피소드 1개)\n')
	L.append('| 조건 | 체크포인트 | PRE | ORA | ORA − PRE | 판정 | CON-ORA − PRE (돌린 경우) |')
	L.append('|---|---|---|---|---|---|---|')
	S['C2'] = {}
	for cell in OTHER:
		for k in (2, 3):
			pre, ora, con = (get(a, cell, k, 3, 'deploy', p, S8, 1) for p in ('PRE', 'ORA', 'CONORA'))
			if pre is None or ora is None:
				L.append(f'| {SHORT[cell]} | {k} | (없음) | | | | |')
				continue
			iv = interval(sm(ora) - sm(pre))
			S['C2'][f'{cell}/k{k}'] = iv
			cc = '–' if con is None else fmt(interval(sm(con) - sm(pre)))
			L.append(f'| {SHORT[cell]} | {k} | {pre["ret"].mean():.0f} | {ora["ret"].mean():.0f} | {fmt(iv)} | {three(iv)} | {cc} |')
	L.append('\n### 평소 조건 PRE (체크포인트 확인, seed 3개 × 에피소드 1개)\n')
	for task in ('cheetah-run', 'walker-run', 'quadruped-run'):
		cell = [c for c in [CF] + OTHER if cell_parts(c)[0] == task][0]
		vals = []
		for k in (1, 2, 3):
			d = get(a, cell, k, 3, 'nominal', 'PRE', S3, 1)
			vals.append('(없음)' if d is None else f'{d["ret"].mean():.0f}')
		L.append(f'- {task}: 체크포인트 1, 2, 3 = ' + ', '.join(vals))

	# ---- D: quadruped motor
	L.append('\n## D. quadruped 모터, 체크포인트 1 (보고만)\n')
	triple(QM, 1, S16, 2, 'D 배치 규약', 'D')

	# ---- E: check 1
	L.append('\n## E. 확인 1: 각 planner가 스스로 간 상태에서의 모델 오차 (설계 08의 셋째 라운드 데이터)\n')
	for cell in (CF, QM):
		fs = sorted(glob.glob(f'{a.out_root}/selferr/{cell}/s*.json'))
		if not fs:
			continue
		R = [json.load(open(f)) for f in fs]
		ref = np.array([r['own']['M1']['loss'] for r in R])
		L.append(f'\n### {SHORT[cell]} (seed {len(R)}개)\n')
		L.append('| planner가 쓰는 모델 | 자기 상태 오차 (첫 라운드 모델의 자기 상태 오차 = 1) | 적응 없는 planner의 상태에서 (같은 단위) | 자기 상태의 3스텝 보상: 예측 − 실제 (실제) |')
		L.append('|---|---|---|---|')
		S[f'E/{cell}'] = {}
		for n in ACTORS:
			own = interval(np.array([r['own'][n]['loss'] for r in R]) / ref)
			onp = interval(np.array([r['on_pre'][n]['loss'] for r in R]) / ref)
			bias = interval(np.array([r['own'][n]['rew3_pred'] - r['own'][n]['rew3_true'] for r in R]))
			true = float(np.mean([r['own'][n]['rew3_true'] for r in R]))
			S[f'E/{cell}'][n] = dict(own=own, on_pre=onp, bias=bias, true=true)
			L.append(f'| {ANAME[n]} | {own["mean"]:.2f} [{own["t"][0]:.2f}, {own["t"][1]:.2f}] | {onp["mean"]:.2f} | {fmt(bias, 2)} ({true:.2f}) |')
	# ---- F: mechanism traces (not registered)
	L.append('\n## F. 기전의 흔적: 여섯 조건 × 체크포인트 (등록 안 함)\n')
	L.append('값은 "적응 없음 / 정확한 dynamics만 / 정확한 dynamics + sim 합의"다. 첫 100 스텝 보상은 에피소드 첫 100 결정 스텝의 보상 합이다. 체크포인트 1의 일부는 설계 04, 06, 07, 10의 기록이라 planner 사이에 에피소드가 짝이 아니다(에피소드 수를 적는다).\n')
	L.append('| 조건 | 체크포인트 | 에피소드 수 | return | 첫 100 스텝 보상 | 노력 | 부드러움 |')
	L.append('|---|---|---|---|---|---|---|')
	S['F'] = {}
	for cell in [CF] + OTHER:
		for k in (1, 2, 3):
			T = [trace(a, cell, k, p) for p in ('PRE', 'ORA', 'CONORA')]
			if T[0] is None or T[1] is None:
				continue
			S['F'][f'{cell}/k{k}'] = dict(PRE=T[0], ORA=T[1], CONORA=T[2])
			col = lambda key, d: ' / '.join('–' if t is None else f'{t[key]:.{d}f}' for t in T)
			L.append(f'| {SHORT[cell]} | {k} | {col("n", 0)} | {col("ret", 0)} | {col("first100", 0)} | {col("effort", 2)} | {col("smooth", 2)} |')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--planner', choices=['PRE', 'ORA', 'CONORA', 'ORAPEN'], required=True)
	p.add_argument('--lam', type=float, default=1.0)
	p.add_argument('--ckpt', type=int, default=1)
	p.add_argument('--horizon', type=int, default=3)
	p.add_argument('--proto', choices=['deploy', 'collect', 'nominal'], default='deploy')
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=16)
	p.add_argument('--eps', type=int, default=2)
	p.add_argument('--workers', type=int, default=40)
	p.add_argument('--timing', type=int, default=0)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('selferr')
	p.add_argument('--cell', required=True)
	p.add_argument('--seed_from', type=int, default=1)
	p.add_argument('--seed_to', type=int, default=32)
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--d08_root', default=D08)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--d07_root', default=D07)
	p.add_argument('--d08_root', default=D08)
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	p.add_argument('--d04_root', default='/mnt/newssd/cjkim/td/d04')
	p.add_argument('--d06_root', default='/mnt/newssd/cjkim/td/d06')
	p.add_argument('--d10_root', default='/mnt/newssd/cjkim/td/d10')
	a = ap.parse_args()
	dict(run=cmd_run, selferr=cmd_selferr, summarize=cmd_summarize)[a.cmd](a)
