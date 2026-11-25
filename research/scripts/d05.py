"""Design 05 stage 1 (research/docs/design05_real_return_value_ko.md): terminal value refit to real returns, tested with the oracle-dynamics planner.

Commands (run from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d05.py fit --cell cheetah-run__friction_0.15
  GPU=N research/scripts/td.sh research/scripts/d05.py f0 --cell cheetah-run__friction_0.15        (exit code 1 if F0 fails)
  GPU=N research/scripts/td.sh research/scripts/d05.py calib --cell cheetah-run__friction_0.15      (after fit; a + b*Q0, design 05 §4a)
  GPU=N research/scripts/td.sh research/scripts/d05.py eval --cell cheetah-run__friction_0.15 --arm VR|VRP|VC [--eps 20] [--workers 40]
  GPU=N research/scripts/td.sh research/scripts/d05.py summarize
All commands take --out_root (default /mnt/newssd/cjkim/td/d05); smoke tests must pass another folder.
fit: only the Q ensemble is trained, as a state value V(z) = Q(z, pi(z)), on the real returns of the data-collecting planner
(discounted sum to the episode end + bootstrapped tail from the target ensemble mean). No max, no min, no imagination.
eval: the design-04 oracle planner with the refit Q (VR), optionally minus a kNN-distance penalty (VRP),
or with the pretrained Q rescaled as a + b*Q0 (VC).
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from common import math as tdmath

from d01 import N_HELD, N_TRAIN, Shifter, _cuda, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import EVAL_EPS, EVAL_SEED, FT_SEED
from d04 import GAMMA, H_CELLS, Oracle, _ep, _winit, _wroll, paired_ci, speed

D01 = '/mnt/newssd/cjkim/td/d01'
D02 = '/mnt/newssd/cjkim/td/d02'
D04 = '/mnt/newssd/cjkim/td/d04'
OUT = '/mnt/newssd/cjkim/td/d05'
FIT_STEPS, TAU = 20_000, 0.02
KNN, D0_PCT, BETA, PEN_CLIP = 5, 95, 1.0, 3.0
F0_CELL, F0_TMAX, F0_CORR, F0_SLOPE = 'cheetah-run__friction_0.15', 400, 0.7, (0.5, 2.0)
G1_CELL, G2_CELL = 'cheetah-run__friction_0.15', 'quadruped-run__motor_0.35'
ARMS = ['VR', 'VRP', 'VC']
PREFER = ['VC', 'VRP', 'VR']  # design 05 §4a: first passing arm in this order unless a later one is clearly higher


# ----------------------------------------------------------------------------- value fit (design 05 §3)

@torch.no_grad()
def vmean(m, cfg, z, target=False):
	"""Mean of the 5 heads at the policy action: B x 1."""
	act, _ = m.pi(z, None)
	return tdmath.two_hot_inv(m.Q(z, act, None, return_type='all', target=target), cfg).mean(0)


@torch.no_grad()
def vmean_chunked(m, cfg, z, n=4096):
	return torch.cat([vmean(m, cfg, z[i:i + n]) for i in range(0, len(z), n)])


def knn_dist(z, Zs, k=KNN):
	return torch.topk(torch.cdist(z, Zs), k, dim=1, largest=False).values.mean(1, keepdim=True)


def fit_stats(v, g):
	v, g = np.asarray(v, np.float64).ravel(), np.asarray(g, np.float64).ravel()
	return dict(corr=float(np.corrcoef(v, g)[0, 1]), slope=float(np.polyfit(v, g, 1)[0]), v_mean=float(v.mean()),
				g_mean=float(g.mean()), v_sd=float(v.std()), g_sd=float(g.std()))


def cmd_fit(a):
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	torch.manual_seed(FT_SEED)
	m, gamma = agent.model, float(agent.discount)
	assert abs(gamma - GAMMA) < 1e-6
	d = np.load(f'{a.d01_root}/{a.cell}/data.npz')
	obs, rew = d['obs'], d['rew']
	E, T = rew.shape
	ntr = N_TRAIN if E >= N_TRAIN + N_HELD else E - 1                   # smoke data may have fewer episodes
	with torch.no_grad():
		Z = m.encode(_cuda(obs.reshape(-1, obs.shape[-1])), None).reshape(E, T + 1, -1)
	R = _cuda(rew)
	S = torch.zeros(E, T, device='cuda')                                # discounted reward sum from t to the episode end
	acc = torch.zeros(E, device='cuda')
	for t in range(T - 1, -1, -1):
		acc = R[:, t] + gamma * acc
		S[:, t] = acc
	w = gamma ** (T - torch.arange(T, device='cuda', dtype=torch.float32))  # weight of the bootstrapped tail

	def targets(eps, online):
		tail = vmean_chunked(m, cfg, Z[eps, T]) if online else vmean(m, cfg, Z[eps, T], target=True)
		return S[eps] + w.unsqueeze(0) * tail                            # len(eps) x T

	tr, ho = torch.arange(ntr, device='cuda'), torch.arange(ntr, E, device='cuda')
	tmax = min(F0_TMAX, T)

	def held_out():
		v = vmean_chunked(m, cfg, Z[ho, :tmax].reshape(-1, Z.shape[-1])).reshape(len(ho), tmax)
		return fit_stats(v.cpu().numpy(), targets(ho, True)[:, :tmax].cpu().numpy())

	chk = Z[tr, :T].reshape(-1, Z.shape[-1])[torch.randperm(ntr * T, device='cuda')[:2000]]
	pre = dict(held_out=held_out(), v_train=float(vmean_chunked(m, cfg, chk).mean()))
	print(a.cell, 'before fit', pre, flush=True)
	opt = torch.optim.Adam(m._Qs.parameters(), lr=cfg.lr)
	steps = a.steps or FIT_STEPS
	B, log, t0 = cfg.batch_size, [], time.time()
	for step in range(1, steps + 1):
		e = torch.randint(0, ntr, (B,), device='cuda')
		t = torch.randint(0, T, (B,), device='cuda')
		with torch.no_grad():
			z = Z[e, t]
			act, _ = m.pi(z, None)
			G = (S[e, t] + w[t] * vmean(m, cfg, Z[e, T], target=True).squeeze(1)).unsqueeze(1)
		m.train()
		qs = m.Q(z, act, None, return_type='all')
		loss = sum(tdmath.soft_ce(qs[i], G, cfg).mean() for i in range(cfg.num_q)) / cfg.num_q
		loss.backward()
		torch.nn.utils.clip_grad_norm_(m._Qs.parameters(), cfg.grad_clip_norm)
		opt.step()
		opt.zero_grad(set_to_none=True)
		m._target_Qs_params.lerp_(m._detach_Qs_params, TAU)
		m.eval()
		if step % 500 == 0 or step == 1 or step == steps:
			log.append(dict(step=step, loss=float(loss), v_train=float(vmean_chunked(m, cfg, chk).mean()), g_batch=float(G.mean())))
			if step % 2500 == 0 or step == 1 or step == steps:
				print(a.cell, log[-1], f'{time.time() - t0:.0f}s', flush=True)
	with torch.no_grad():
		Gtr = targets(tr, True)
		sigma = float(Gtr.std())
		Zs = Z[tr, :T].reshape(-1, Z.shape[-1]).contiguous()
		D = torch.cdist(Zs, Zs)
		ep = torch.arange(ntr, device='cuda').repeat_interleave(T)
		D[ep.unsqueeze(0) == ep.unsqueeze(1)] = float('inf')             # other episodes only
		d_loo = torch.topk(D, KNN, dim=1, largest=False).values.mean(1)
		d0 = float(torch.quantile(d_loo, D0_PCT / 100))
		d_ho = knn_dist(Z[ho, :T].reshape(-1, Z.shape[-1]), Zs)
		post = held_out()
		vtr = vmean_chunked(m, cfg, Zs).reshape(ntr, T)
	train = fit_stats(vtr.cpu().numpy(), Gtr.cpu().numpy())
	f0 = bool(post['corr'] >= F0_CORR and F0_SLOPE[0] <= post['slope'] <= F0_SLOPE[1])
	v_at = {s: [l['v_train'] for l in log if l['step'] == s] for s in (steps // 2, steps * 3 // 4, steps)}
	res = dict(steps=steps, ntr=ntr, n_held=int(len(ho)), before=pre, held_out=post, train=train, sigma_G=sigma, d0=d0,
			   d_loo_median=float(d_loo.median()), held_out_d_over_d0=float((d_ho / d0).mean()),
			   held_out_pen_frac=float((d_ho > d0).float().mean()), v_train_at=v_at, f0=f0, log=log, seconds=time.time() - t0)
	folder = f'{a.out_root}/{a.cell}'
	os.makedirs(folder, exist_ok=True)
	agent.save(f'{folder}/VR.pt')
	torch.save(dict(Z=Zs.cpu(), d0=d0, sigma=sigma), f'{folder}/support.pt')
	dump(f'{folder}/fit.json', res)
	print(a.cell, 'after fit', dict(held_out=post, train=train, sigma_G=sigma, d0=d0, v_train_at=v_at, f0=f0), flush=True)


def cmd_calib(a):
	"""Design 05 §4a: least-squares (a, b) of the fit targets on the pretrained value, training episodes."""
	task, _ = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	torch.manual_seed(FT_SEED)
	m, gamma = agent.model, float(agent.discount)
	d = np.load(f'{a.d01_root}/{a.cell}/data.npz')
	obs, rew = d['obs'], d['rew']
	E, T = rew.shape
	ntr = N_TRAIN if E >= N_TRAIN + N_HELD else E - 1
	tmax = min(F0_TMAX, T)
	with torch.no_grad():
		Z = m.encode(_cuda(obs.reshape(-1, obs.shape[-1])), None).reshape(E, T + 1, -1)
		V0 = vmean_chunked(m, cfg, Z[:, :T].reshape(-1, Z.shape[-1])).reshape(E, T).cpu().numpy().astype(np.float64)  # pretrained
		R = _cuda(rew)
		S = torch.zeros(E, T, device='cuda')
		acc = torch.zeros(E, device='cuda')
		for t in range(T - 1, -1, -1):
			acc = R[:, t] + gamma * acc
			S[:, t] = acc
		w = gamma ** (T - torch.arange(T, device='cuda', dtype=torch.float32))
		agent.load(f'{a.out_root}/{a.cell}/VR.pt')                         # same targets as the fit: tail from the fitted value
		G = (S + w.unsqueeze(0) * vmean_chunked(m, cfg, Z[:, T])).cpu().numpy().astype(np.float64)
	b, a0 = np.polyfit(V0[:ntr].ravel(), G[:ntr].ravel(), 1)
	res = dict(a=float(a0), b=float(b), ntr=ntr, train=fit_stats(a0 + b * V0[:ntr], G[:ntr]),
			   held_out=fit_stats(a0 + b * V0[ntr:, :tmax], G[ntr:, :tmax]))
	dump(f'{a.out_root}/{a.cell}/calib.json', res)
	print(a.cell, 'calib', res, flush=True)


def cmd_f0(a):
	f = json.load(open(f'{a.out_root}/{a.cell}/fit.json'))
	print(a.cell, 'F0', f['f0'], f['held_out'], flush=True)
	sys.exit(0 if f['f0'] else 1)


# ----------------------------------------------------------------------------- oracle planner with the refit value (design 05 §4)

class OracleV(Oracle):
	"""Design-04 oracle planner; terminal value optionally minus the kNN-distance penalty."""

	def __init__(self, agent, env, pool, workers, support=None, calib=None):
		super().__init__(agent, env, pool, workers)
		self.sup, self.calib = support, calib
		self.dcur = []

	def penalty(self, z):
		d = knn_dist(z, self.sup['Z'])
		return BETA * self.sup['sigma'] * (d / self.sup['d0'] - 1).clamp(0, PEN_CLIP), d

	@torch.no_grad()
	def values(self, state, actions):
		seqs = actions.permute(1, 0, 2).cpu().numpy().astype(np.float32)
		chunks = [c for c in np.array_split(seqs, self.W) if len(c)]
		outs = self.pool.map(_wroll, [(state, c) for c in chunks], chunksize=1)
		R = np.concatenate([o[0] for o in outs])
		O = np.concatenate([o[1] for o in outs])
		disc = self.gamma ** np.arange(self.cfg.horizon)
		G = torch.as_tensor(R @ disc, dtype=torch.float32, device='cuda').unsqueeze(1)
		z = self.m.encode(_cuda(O), None)
		action, _ = self.m.pi(z, None)
		v = self.m.Q(z, action, None, return_type='avg')
		if self.sup is not None:
			v = v - self.penalty(z)[0]
		if self.calib is not None:
			v = self.calib[0] + self.calib[1] * v
		return G + self.gamma ** self.cfg.horizon * v

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		if self.sup is not None:
			z = self.m.encode(obs.to('cuda').unsqueeze(0), None)
			self.dcur.append(float(self.penalty(z)[1]) / self.sup['d0'])
		return super().act(obs, t0=t0, eval_mode=eval_mode)


def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	pool = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(task, cond))  # before any CUDA use
	cfg, env, agent = make(task, compile_=False)
	folder = f'{a.out_root}/{a.cell}'
	sup, calib = None, None
	if a.arm == 'VC':                                                      # pretrained Q, rescaled
		c = json.load(open(f'{folder}/calib.json'))
		calib = (float(c['a']), float(c['b']))
	else:
		agent.load(f'{folder}/VR.pt')
	if a.arm == 'VRP':
		s = torch.load(f'{folder}/support.pt', weights_only=False)
		sup = dict(Z=s['Z'].to('cuda'), d0=float(s['d0']), sigma=float(s['sigma']))
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	ora = OracleV(agent, env, pool, a.workers, sup, calib)
	t0 = time.time()
	eps, recs = run_cond(ora, env, shifter, domain, cond, a.eps, seed=EVAL_SEED, record=True)
	res = dict(cond=cond, summary=summary(eps), episodes=eps, plan_time=float(np.mean(ora.plan_time)), seconds=time.time() - t0)
	if calib is not None:
		res['calib'] = calib
	rec = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in ('obs', 'state', 'act', 'rew')}
	rec['vsel'] = np.asarray(ora.vsel, np.float32).reshape(len(recs), -1)
	if sup is not None:
		rec['d_over_d0'] = np.asarray(ora.dcur, np.float32).reshape(len(recs), -1)
		res['d_over_d0_mean'] = float(rec['d_over_d0'].mean())
		res['pen_frac'] = float((rec['d_over_d0'] > 1).mean())
	dump(f'{folder}/eval_{a.arm}.json', res)
	np.savez_compressed(f'{folder}/rec_{a.arm}.npz', **rec)
	print(a.cell, a.arm, res['summary'], f'plan {res["plan_time"]:.3f}s/step total {res["seconds"]:.0f}s', flush=True)
	pool.close()


# ----------------------------------------------------------------------------- summary (registered criteria, design 05 §6)

def first_reach(obs, v=2.0):
	"""cheetah: median over episodes of the first step with forward speed > v (episode length if never)."""
	s = obs[:, :, 8]
	return float(np.median([int(np.argmax(x > v)) if (x > v).any() else s.shape[1] for x in s]))


def gap(rec):
	rew, T = rec['rew'], rec['rew'].shape[1]
	G, acc = np.zeros_like(rew), np.zeros(len(rew))
	for t in range(T - 1, -1, -1):
		acc = rew[:, t] + GAMMA * acc
		G[:, t] = acc
	return float((rec['vsel'][:, 50:251] - G[:, 50:251]).mean())


def cmd_summarize(a):
	cells = a.cells.split(',') if a.cells else H_CELLS
	L, S = [], dict(cells={}, fit={})
	L.append('# 설계 05 1단계 결과 요약 (자동 생성, `d05.py summarize`)\n')
	L.append('모든 팔은 후보의 3스텝을 실제 simulator로 실행하는 planner다. ORA = 사전학습 가치(설계 04), ORA-VR = 실제 return에 맞춘 가치, ORA-VRP = 그 가치 − 데이터에서 먼 상태 벌점, ORA-VC = a + b × 사전학습 가치(눈금만 고침, §4a). PRE = 적응 없는 planner(설계 02). 평가 시드 2.\n')
	L.append('## 가치 맞추기 (확인용 에피소드, t ≤ 400)\n')
	L.append('| 셀 | 맞추기 전 상관 / 기울기 | 맞춘 뒤 상관 / 기울기 | 평균 V 전 → 뒤 (목표 평균) | 학습 상태 평균 V: 절반 / 3/4 / 끝 | σ_G | d₀ | F0 | 눈금 고침 a, b (확인용 상관 / 기울기) |')
	L.append('|---|---|---|---|---|---|---|---|---|')
	for c in cells:
		f = json.load(open(f'{a.out_root}/{c}/fit.json'))
		S['fit'][c] = {k: f[k] for k in ('before', 'held_out', 'train', 'sigma_G', 'd0', 'v_train_at', 'f0', 'steps', 'ntr', 'held_out_d_over_d0', 'held_out_pen_frac')}
		b, h = f['before']['held_out'], f['held_out']
		va = ' / '.join(f'{v[0]:.1f}' if v else '–' for v in f['v_train_at'].values())
		cs = '–'
		if os.path.exists(f'{a.out_root}/{c}/calib.json'):
			cj = S['fit'][c]['calib'] = json.load(open(f'{a.out_root}/{c}/calib.json'))
			cs = f'{cj["a"]:.1f}, {cj["b"]:.2f} ({cj["held_out"]["corr"]:.2f} / {cj["held_out"]["slope"]:.2f})'
		L.append(f'| {c} | {b["corr"]:.2f} / {b["slope"]:.2f} | {h["corr"]:.2f} / {h["slope"]:.2f} | {b["v_mean"]:.1f} → {h["v_mean"]:.1f} ({h["g_mean"]:.1f}) | {va} | {f["sigma_G"]:.1f} | {f["d0"]:.2f} | {("통과" if f["f0"] else "실패") if c == F0_CELL else "(보고만)"} | {cs} |')
	f0 = S['fit'].get(F0_CELL, {}).get('f0')
	ev, rec = {}, {}
	for c in cells:
		ev[c, 'PRE'] = json.load(open(f'{a.d02_root}/{c}/eval_PRE.json'))['shift']
		ev[c, 'ORA'] = json.load(open(f'{a.d04_root}/{c}/eval_ORA.json'))
		rec[c, 'ORA'] = np.load(f'{a.d04_root}/{c}/rec_ORA.npz')
		for arm in ARMS:
			if os.path.exists(f'{a.out_root}/{c}/eval_{arm}.json'):
				ev[c, arm] = json.load(open(f'{a.out_root}/{c}/eval_{arm}.json'))
				rec[c, arm] = np.load(f'{a.out_root}/{c}/rec_{arm}.npz')
	L.append('\n## 바뀐 조건 return (넘어짐 사건 / 넘어져 있는 시간 비율)\n')
	L.append('| 셀 | PRE | ORA | ' + ' | '.join('ORA-' + arm for arm in ARMS) + ' |')
	L.append('|---|---|---|' + '---|' * len(ARMS))
	for c in cells:
		row = []
		for arm in ['PRE', 'ORA'] + ARMS:
			s = ev[c, arm]['summary'] if (c, arm) in ev else None
			row.append(f'{s["ret"]:.0f} ({s["falls"]:.1f} / {s["fallen_frac"]:.2f})' if s else '(없음)')
		L.append(f'| {c} | ' + ' | '.join(row) + ' |')
	L.append('\n## 에피소드 짝 차이 (평균 [bootstrap 95 % 구간])\n')
	L.append('| 셀 | 팔 | 팔 − ORA | 팔 − PRE | 초기 관측 일치 |')
	L.append('|---|---|---|---|---|')
	for c in cells:
		S['cells'][c] = {}
		for arm in ARMS:
			if (c, arm) not in ev:
				continue
			x = _ep(ev[c, arm], 'ret')
			n = len(x)
			e = S['cells'][c][arm] = dict(summary=ev[c, arm]['summary'], n=n)
			row = []
			for ref in ('ORA', 'PRE'):
				mdiff, ci = paired_ci(x, _ep(ev[c, ref], 'ret')[:n])
				e[ref] = dict(diff=mdiff, ci=ci)
				row.append(f'{mdiff:+.0f} [{ci[0]:+.0f}, {ci[1]:+.0f}]')
			same = float(np.abs(rec[c, arm]['obs'][:, 0] - rec[c, 'ORA']['obs'][:n, 0]).max())
			e['init_obs_maxdiff'] = same
			L.append(f'| {c} | ORA-{arm} | ' + ' | '.join(row) + f' | 최대 차이 {same:.1e} |')
		S['cells'][c]['between'] = {}
		for i, x in enumerate(PREFER):
			for y in PREFER[i + 1:]:
				if (c, x) in ev and (c, y) in ev:
					n = min(len(_ep(ev[c, x], 'ret')), len(_ep(ev[c, y], 'ret')))
					mdiff, ci = paired_ci(_ep(ev[c, y], 'ret')[:n], _ep(ev[c, x], 'ret')[:n])
					S['cells'][c]['between'][f'{y}-{x}'] = dict(diff=mdiff, ci=ci)
					L.append(f'| {c} | ORA-{y} − ORA-{x} | {mdiff:+.0f} [{ci[0]:+.0f}, {ci[1]:+.0f}] | – | – |')
	L.append('\n## 동작 (모든 에피소드 기록)\n')
	L.append('| 셀 | 팔 | 속도 (m/s) | 2 m/s에 처음 닿는 스텝(중앙값, cheetah만) | 부드러움 / 노력 / 흔들림 | 계획 가치 − 실현 return | d/d₀ 평균 (벌점 > 0인 스텝 비율) | 계획 시간(초/스텝) |')
	L.append('|---|---|---|---|---|---|---|---|')
	for c in cells:
		dom = domain_of(cell_parts(c)[0])
		for arm in ['ORA'] + ARMS:
			if (c, arm) not in ev:
				continue
			r, s = rec[c, arm], ev[c, arm]['summary']
			sp = speed(dom, r['state'], r['rew'])
			fr = first_reach(r['obs']) if dom == 'cheetah' else None
			g = gap(r)
			dd = f'{ev[c, arm]["d_over_d0_mean"]:.2f} ({ev[c, arm]["pen_frac"]:.2f})' if 'pen_frac' in ev[c, arm] else '–'
			if arm in ARMS:
				S['cells'][c][arm].update(speed=sp, first_reach=fr, gap=g)
			else:
				S['cells'][c]['ORA'] = dict(speed=sp, first_reach=fr, gap=g, summary=s)
			L.append(f'| {c} | {"ORA" if arm == "ORA" else "ORA-" + arm} | {sp:.2f} | {"–" if fr is None else f"{fr:.0f}"} | {s["smooth"]:.2f} / {s["effort"]:.2f} / {s["angvel_rms"]:.2f} | {g:.1f} | {dd} | {ev[c, arm]["plan_time"]:.2f} |')
	# verdicts
	L.append('\n## 판정\n')
	ci = lambda c, arm, ref, i: S['cells'].get(c, {}).get(arm, {}).get(ref, {}).get('ci', [None, None])[i]
	g1 = {arm: (None if ci(G1_CELL, arm, 'ORA', 0) is None else bool(ci(G1_CELL, arm, 'ORA', 0) > 0 and ci(G1_CELL, arm, 'PRE', 1) >= 0)) for arm in ARMS}
	g2 = {arm: (None if ci(G2_CELL, arm, 'PRE', 1) is None else bool(ci(G2_CELL, arm, 'PRE', 1) >= 0)) for arm in ARMS}
	gain = {arm: (None if ci(G1_CELL, arm, 'PRE', 0) is None else bool(ci(G1_CELL, arm, 'PRE', 0) > 0)) for arm in ARMS}
	S.update(F0=f0, G1=g1, G2=g2, gain=gain)
	yn = lambda v: '(없음)' if v is None else ('예' if v else '아니오')
	L.append(f'- F0 맞추기 타당성 (cheetah 마찰): {yn(f0)}')
	for arm in ARMS:
		L.append(f'- ORA-{arm}: G1 {yn(g1[arm])}, G2 {yn(g2[arm])}, cheetah 마찰에서 PRE보다 확실히 높음(보고만) {yn(gain[arm])}')
	if not f0:
		gate, use = None, None
		L.append('- 관문: 판정 불가 (F0 실패 또는 없음)')
	elif any(v is None for v in list(g1.values()) + list(g2.values())):
		gate, use = None, None
		L.append('- 관문: 아직 판정할 수 없음 (결과 없는 팔이 있음)')
	else:
		ok = [arm for arm in PREFER if g1[arm] and g2[arm]]
		gate = bool(ok)
		use = ok[0] if ok else None
		for y in ok[1:]:                                                  # a later arm replaces the current choice only if clearly higher in the G1 cell
			lo = paired_ci(_ep(ev[G1_CELL, y], 'ret'), _ep(ev[G1_CELL, use], 'ret'))[1][0]
			if lo > 0:
				use = y
		L.append(f'- 관문 통과: {yn(gate)}' + (f' → 쓸 팔: ORA-{use}의 가치' if use else ''))
		if not gate:
			L.append('- → G1을 만족하는 팔: ' + (', '.join(arm for arm in ARMS if g1[arm]) or '없음'))
	S.update(gate=gate, use=use)
	L.append('\n## 예측 대조\n')
	fr = lambda arm: S['cells'].get(G1_CELL, {}).get(arm, {}).get('first_reach')
	c0 = lambda arm: ci(G1_CELL, arm, 'ORA', 0)
	chk = {
		'P1 F0 통과': f0,
		'P2 cheetah 마찰 ORA-VR − ORA > 0': None if c0('VR') is None else c0('VR') > 0,
		'P3 cheetah 마찰 ORA-VRP − ORA > 0': None if c0('VRP') is None else c0('VRP') > 0,
		'P4 G1 만족하는 팔 있음': None if None in g1.values() else any(g1.values()),
		'P5 cheetah 마찰에서 PRE보다 확실히 높은 팔 있음': None if None in gain.values() else any(gain.values()),
		'P6 quadruped 모터 ORA-VRP G2': g2['VRP'],
		'P7 관문 통과': gate,
		'P8 cheetah 마찰 ORA-VR이 2 m/s에 닿는 스텝 < ORA': None if fr('VR') is None else fr('VR') < fr('ORA'),
		'P9 cheetah 마찰 ORA-VC − ORA > 0': None if c0('VC') is None else c0('VC') > 0,
		'P10 quadruped 모터 ORA-VC G2': g2['VC'],
		'P11 quadruped 모터 ORA-VR 넘어져 있는 시간 > ORA': None if (G2_CELL, 'VR') not in ev else
			ev[G2_CELL, 'VR']['summary']['fallen_frac'] > ev[G2_CELL, 'ORA']['summary']['fallen_frac'],
	}
	for k, v in chk.items():
		L.append(f'- {k}: {yn(v)}')
	S['pred'] = {k: (None if v is None else bool(v)) for k, v in chk.items()}
	os.makedirs(f'{a.out_root}/summary', exist_ok=True)
	open(f'{a.out_root}/summary/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.out_root}/summary/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	for name in ('fit', 'calib', 'f0', 'eval', 'summarize'):
		p = sub.add_parser(name)
		p.add_argument('--out_root', default=OUT)
		if name != 'summarize':
			p.add_argument('--cell', required=True)
		if name == 'fit':
			p.add_argument('--steps', type=int, default=None)
		if name in ('fit', 'calib'):
			p.add_argument('--d01_root', default=D01)
		if name == 'eval':
			p.add_argument('--arm', choices=ARMS, required=True)
			p.add_argument('--eps', type=int, default=EVAL_EPS)
			p.add_argument('--workers', type=int, default=40)
		if name == 'summarize':
			p.add_argument('--cells', default=None)
			p.add_argument('--d02_root', default=D02)
			p.add_argument('--d04_root', default=D04)
	a = ap.parse_args()
	dict(fit=cmd_fit, calib=cmd_calib, f0=cmd_f0, eval=cmd_eval, summarize=cmd_summarize)[a.cmd](a)
