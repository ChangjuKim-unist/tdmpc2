"""Design 03 (research/docs/design03_value_side_remedies_ko.md).

Commands (run from ~/tdmpc2 on gpu1: GPU=N research/scripts/td.sh research/scripts/d03.py <cmd> ...):
  refit     --cell C --arm D2000IMG|MIXIMG|D2000IMG15|D2000IMGM     Q + policy refit by imagined rollouts
  eval      --cell C --arm <new arm> [--pi]                          --pi: policy mean only, no planner (shift only)
  probe     --cell C                                                 value gaps and model error on each new arm's own states
  summarize
Design 01/02 data, checkpoints and evals are read only. The official TD-MPC2 code is imported, not modified.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import d01
import d02
from d01 import make, phys_of, Shifter, cell_parts, domain_of, run_cond, summary, dump, _cuda, enc, N_TRAIN
from d02 import CELLS, EVAL_SEED, EVAL_EPS, FT_SEED
from common import math as tdmath

OUT = '/mnt/newssd/cjkim/td/d03'
U_CELLS = ['walker-run__friction_0.15', 'quadruped-run__motor_0.35', 'cheetah-run__friction_0.15']  # unsolved in design 02
# arm -> (base dynamics arm, refit spec or None, planner terminal value)
IMG = {'D2000IMG': ('D2000', 3, False), 'MIXIMG': ('MIX', 3, False), 'D2000IMG15': ('D2000', 15, False),
	   'D2000IMGM': ('D2000', 3, True)}     # (base, K, min over all target Qs)
ARMS = {'D2000MIN': ('D2000', 'min'), 'MIXMIN': ('MIX', 'min'), 'D2000IMG': ('D2000', 'avg'), 'MIXIMG': ('MIX', 'avg'),
		'D2000IMG15': ('D2000', 'avg'), 'D2000IMGM': ('D2000', 'min')}
NEW = list(ARMS)
REF = ['PRE', 'D2000', 'MIX', 'MIXPEN1']    # design 02 evals, reused
PI_ARMS = ['PRE'] + list(IMG)
REFIT_STEPS = 10_000


def load_arm(agent, cfg, cell, arm):
	"""Load the model of a new arm (or a design 01/02 model arm)."""
	agent.load(cfg.checkpoint)
	if arm == 'PRE':
		return
	if arm in IMG:
		agent.load(f'{OUT}/{cell}/{arm}.pt')
	else:
		agent.load(d02.model_path(cell, ARMS[arm][0] if arm in ARMS else arm))


def min_value(agent):
	"""MRS.Q-style: terminal value is the minimum over the whole Q ensemble."""
	cfg, m = agent.cfg, agent.model

	@torch.no_grad()
	def _est(z, actions, task):
		G, disc = 0, 1
		for t in range(cfg.horizon):
			reward = tdmath.two_hot_inv(m.reward(z, actions[t], task), cfg)
			z = m.next(z, actions[t], task)
			G = G + disc * reward
			disc = disc * agent.discount
		action, _ = m.pi(z, task)
		q = tdmath.two_hot_inv(m.Q(z, action, task, return_type='all'), cfg).min(0).values
		return G + disc * q
	return _est


def set_planner(agent, arm):
	if ARMS.get(arm, (None, 'avg'))[1] == 'min':
		agent._estimate_value = min_value(agent)


# ----------------------------------------------------------------------------- refit (ReDRAW-style, TD-MPC2 parts)

def cmd_refit(a):
	task, _ = cell_parts(a.cell)
	base, K, tmin = IMG[a.arm]
	cfg, env, agent = make(task, compile_=False)
	agent.load(d02.model_path(a.cell, base))
	torch.manual_seed(FT_SEED)
	m = agent.model
	data = d02.load_d01(a.cell)
	obs0 = data['obs'][:N_TRAIN, :-1]
	with torch.no_grad():
		Z0 = m.encode(_cuda(obs0.reshape(-1, obs0.shape[-1])), None)
		chk = Z0[:512]
		act, _ = m.pi(chk, None)
		q_on = tdmath.two_hot_inv(m.Q(chk, act, None, return_type='all'), cfg)
		q_tg = tdmath.two_hot_inv(m.Q(chk, act, None, return_type='all', target=True), cfg)
	print('target-vs-online Q max abs diff at load', float((q_on - q_tg).abs().max()), 'mean Q', float(q_on.mean()), flush=True)
	q_opt = torch.optim.Adam(m._Qs.parameters(), lr=cfg.lr)
	steps = a.steps or REFIT_STEPS
	log, t0 = [], time.time()
	B, L = cfg.batch_size, Z0.shape[-1]
	for step in range(1, steps + 1):
		idx = torch.randint(0, Z0.shape[0], (B,), device='cuda')
		with torch.no_grad():
			z = Z0[idx]
			zs, acts, rews = [z], [], []
			for _ in range(K):
				act, _ = m.pi(z, None)
				rews.append(tdmath.two_hot_inv(m.reward(z, act, None), cfg))
				z = m.next(z, act, None)
				zs.append(z)
				acts.append(act)
			zs = torch.stack(zs)                                          # (K+1) x B x L
			z_t, a_t = zs[:-1].reshape(K * B, L), torch.stack(acts).reshape(K * B, -1)
			nz, r = zs[1:].reshape(K * B, L), torch.stack(rews).reshape(K * B, 1)
			na, _ = m.pi(nz, None)
			if tmin:
				tq = tdmath.two_hot_inv(m.Q(nz, na, None, return_type='all', target=True), cfg).min(0).values
			else:
				tq = m.Q(nz, na, None, return_type='min', target=True)
			td = r + agent.discount * tq
		m.train()
		qs = m.Q(z_t, a_t, None, return_type='all')                      # num_q x KB x bins
		q_loss = sum(tdmath.soft_ce(qs[i], td, cfg).mean() for i in range(cfg.num_q)) / cfg.num_q
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(m._Qs.parameters(), cfg.grad_clip_norm)
		q_opt.step()
		q_opt.zero_grad(set_to_none=True)
		pi_info = agent.update_pi(zs.reshape(1, (K + 1) * B, L), None)
		m.soft_update_target_Q()
		m.eval()
		if step % 500 == 0 or step == 1:
			with torch.no_grad():
				act, _ = m.pi(chk, None)
				qm = float(tdmath.two_hot_inv(m.Q(chk, act, None, return_type='all'), cfg).mean())
			log.append(dict(step=step, q_loss=float(q_loss), pi_loss=float(pi_info['pi_loss']), q_mean=qm,
							td_mean=float(td.mean()), r_mean=float(r.mean())))
			print(a.cell, a.arm, log[-1], f'{time.time() - t0:.0f}s', flush=True)
	agent.save(f'{OUT}/{a.cell}/{a.arm}.pt')
	dump(f'{OUT}/{a.cell}/refit_{a.arm}.json', dict(log=log, seconds=time.time() - t0, q_mean_at_load=float(q_on.mean())))


# ----------------------------------------------------------------------------- evaluation

def cmd_eval(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=not a.pi)
	load_arm(agent, cfg, a.cell, a.arm)
	if a.pi:
		agent.cfg.mpc = False
	else:
		set_planner(agent, a.arm)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	res, rec = {}, {}
	for c in (('shift',) if a.pi else ('shift', 'nominal')):
		real = cond if c == 'shift' else 'nominal'
		t0 = time.time()
		eps, recs = run_cond(agent, env, shifter, domain, real, a.eps, seed=EVAL_SEED, record=True)
		res[c] = dict(cond=real, summary=summary(eps), episodes=eps)
		for k in ('obs', 'state', 'act', 'rew'):
			rec[f'{c}_{k}'] = np.stack([np.asarray(r[k]) for r in recs[:2]])
		print(a.cell, a.arm, 'pi' if a.pi else 'mpc', c, res[c]['summary'], f'{time.time() - t0:.0f}s', flush=True)
	tag = a.arm + ('_pi' if a.pi else '')
	dump(f'{OUT}/{a.cell}/eval_{tag}.json', res)
	if not a.pi:
		np.savez_compressed(f'{OUT}/{a.cell}/rec_{a.arm}.npz', **rec)


# ----------------------------------------------------------------------------- probe (restore is measurement only)

def cmd_probe(a):
	task, cond = cell_parts(a.cell)
	cfg, env, agent = make(task, compile_=False)
	Shifter(phys_of(env)).set(cond)
	env.reset()
	rs = d02.Restorer2(env)
	gamma, A, m = float(agent.discount), cfg.action_dim, agent.model
	official_est = agent._estimate_value
	res, raw = {}, {}
	rng = np.random.RandomState(3)
	for arm in NEW:
		rec = dict(np.load(f'{OUT}/{a.cell}/rec_{arm}.npz'))
		an, noise1 = d02.anchors_rec(rec, 'shift', A, 1, gamma)
		# model error of the base dynamics vs pretrained on this arm's own states
		obs_t, seqs1, t1, t3 = d02.true_rollouts(rs, an, noise1)
		agent.load(cfg.checkpoint)
		e_pre = d02.errs(m, obs_t, seqs1, t1, t3)['3']
		load_arm(agent, cfg, a.cell, arm)
		e_arm = d02.errs(m, obs_t, seqs1, t1, t3)['3']
		kappa = (e_arm['lev_num'].sum() / e_arm['lev_den'].sum()) / (e_pre['lev_num'].sum() / e_pre['lev_den'].sum())
		# plan value gaps with this arm's planner
		agent._estimate_value = official_est
		set_planner(agent, arm)
		torch.manual_seed(5)
		noise = rng.randn(len(an), 8, 3, A)
		seqs, R3, O3 = [], [], []
		for i, x in enumerate(an):
			s0 = d02.plan_seq(agent, x['obs'])
			S = np.concatenate([s0[None], np.clip(s0[None] + 0.3 * noise[i], -1, 1)], 0)
			r3, o3 = [], []
			for s in S:
				outs, rews = rs.rollout_r(x['state'], s)
				r3.append(d02.disc_return(np.asarray(rews), gamma))
				o3.append(outs[2])
			seqs.append(S); R3.append(r3); O3.append(o3)
		seqs, R3, O3 = np.asarray(seqs), np.asarray(R3), np.asarray(O3)
		N, K = seqs.shape[:2]
		Vhat = d02.plan_values(m, cfg, gamma, obs_t, seqs)
		V3 = R3 + gamma ** 3 * d02.qbar(m, cfg, enc(m, O3.reshape(N * K, -1))).view(N, K).cpu().numpy()
		G = np.asarray([x['G'] for x in an])
		gm, gv = Vhat - V3, V3[:, 0] - G
		raw.update({f'{arm}/Vhat': Vhat, f'{arm}/V3': V3, f'{arm}/G': G})
		res[arm] = dict(kappa=float(kappa), gm=float(gm[:, 0].mean()), gv=float(gv.mean()), Vhat=float(Vhat[:, 0].mean()),
						V3=float(V3[:, 0].mean()), G=float(G.mean()), sel=float((gm[:, 0] - gm[:, 1:].mean(1)).mean()))
		print(a.cell, arm, res[arm], flush=True)
	# G criterion: D2000IMG vs D2000 (design 02 measurement), unpaired state bootstrap
	r2 = np.load(f'{d02.OUT}/{a.cell}/probe_raw.npz')
	gv_d = r2['a2/D2000/V3'][:, 0] - r2['a2/D2000/G']
	for arm in NEW:
		gv_a = raw[f'{arm}/V3'][:, 0] - raw[f'{arm}/G']
		res[arm]['gv_minus_d2000'] = float(gv_a.mean() - gv_d.mean())
		res[arm]['gv_minus_d2000_ci'] = d02.boot_diff(gv_a, gv_d)
	res['D2000_d02'] = dict(gv=float(gv_d.mean()))
	np.savez_compressed(f'{OUT}/{a.cell}/probe_raw.npz', **raw)
	dump(f'{OUT}/{a.cell}/probe.json', res)


# ----------------------------------------------------------------------------- summary (registered criteria, design 03 §5)

def _load(path):
	return json.load(open(path)) if os.path.exists(path) else None


def _ep(ev, c, k):
	return np.array([e[k] for e in ev[c]['episodes']])


def cmd_summarize(a):
	cells = a.cells.split(',') if a.cells else CELLS
	u_cells = [c for c in cells if c in U_CELLS]
	bs = np.random.RandomState(0)
	arms = REF + NEW
	ev = {(c, arm): _load(f'{d02.OUT}/{c}/eval_{arm}.json') for c in cells for arm in REF}
	ev.update({(c, arm): _load(f'{OUT}/{c}/eval_{arm}.json') for c in cells for arm in NEW})
	pr = {c: _load(f'{OUT}/{c}/probe.json') for c in cells}
	rho, dfall, ok, gain = {}, {}, {}, {}
	rows_ret, rows_rho, rows_nom, rows_q = [], [], [], []
	for c in cells:
		pre = ev[c, 'PRE']
		if pre is None:
			continue
		rp, rn, fp = _ep(pre, 'shift', 'ret'), _ep(pre, 'nominal', 'ret'), _ep(pre, 'shift', 'falls').mean()
		cr, cro, cn, cq = [], [], [], []
		for arm in arms:
			e = ev[c, arm]
			if e is None:
				cr.append('-'); cro.append('-'); cn.append('-'); cq.append('-')
				continue
			rf = _ep(e, 'shift', 'ret')
			n = min(len(rf), len(rp))
			r = (rf.mean() - rp.mean()) / (rn.mean() - rp.mean())
			boots = []
			for _ in range(2000):
				i = bs.randint(0, n, n)
				den = rn[i].mean() - rp[i].mean()
				boots.append((rf[i].mean() - rp[i].mean()) / den if abs(den) > 1e-6 else np.nan)
			lo, hi = np.nanpercentile(boots, [2.5, 97.5])
			rho[c, arm], dfall[c, arm] = float(r), float(_ep(e, 'shift', 'falls').mean() - fp)
			ok[c, arm] = bool(r >= 0.25 and dfall[c, arm] < 0.5)
			gain[c, arm] = bool(r >= 0.5 and dfall[c, arm] < 0.5)
			tag = '' if arm == 'PRE' else ('이득' if gain[c, arm] else ('막음' if ok[c, arm] else '실패'))
			cr.append(f"{rf.mean():.0f} ({_ep(e, 'shift', 'falls').mean():.1f})")
			cro.append(f'{r:.2f} [{lo:.2f}, {hi:.2f}] {tag}')
			cn.append(f"{_ep(e, 'nominal', 'ret').mean():.0f} ({_ep(e, 'nominal', 'falls').mean():.1f})")
			cq.append(f"{_ep(e, 'shift', 'smooth').mean():.3f} / {_ep(e, 'shift', 'effort').mean():.2f} / {_ep(e, 'shift', 'angvel_rms').mean():.2f}")
		u = ' (U)' if c in u_cells else ''
		rows_ret.append(f'| {c}{u} | ' + ' | '.join(cr) + ' |')
		rows_rho.append(f'| {c}{u} | ' + ' | '.join(cro[1:]) + ' |')
		rows_nom.append(f'| {c}{u} | ' + ' | '.join(cn) + ' |')
		rows_q.append(f'| {c}{u} | ' + ' | '.join(cq) + ' |')
	hdr, sep = '| 셀 | ' + ' | '.join(arms) + ' |', '|---' * (len(arms) + 1) + '|'
	L = ['# 설계 03 결과 요약 (자동 생성, `d03.py summarize`)', '',
		 '평가: 평가 시드 2, 조건마다 20 에피소드. PRE, D2000, MIX, MIXPEN1은 설계 02의 평가다. (U) = 설계 02에서 안 풀린 셀.',
		 'ρ = (팔 − PRE) / (평소 PRE − PRE), 바뀐 조건 return 기준. [ ]는 에피소드 bootstrap 95 % 구간.', '',
		 '## 팔별 바뀐 조건 return (넘어짐 사건/에피소드)', '', hdr, sep] + rows_ret
	L += ['', '## 팔별 회복 비율 ρ와 판정 (막음: ρ ≥ 0.25이고 넘어짐 증가 < 0.5, 이득: ρ ≥ 0.5이고 넘어짐 증가 < 0.5)', '',
		  '| 셀 | ' + ' | '.join(arms[1:]) + ' |', '|---' * len(arms) + '|'] + rows_rho
	L += ['', '## 팔별 평소 조건 return (넘어짐): 보고만', '', hdr, sep] + rows_nom
	L += ['', '## 팔별 동작 품질, 바뀐 조건 (부드러움 / 노력 / 흔들림, 모두 작을수록 부드럽고 적음)', '', hdr, sep] + rows_q
	# counts
	out, cnt_rows = {}, []
	v1_best = v2_best = 0
	for arm in arms[1:]:
		have = [c for c in cells if (c, arm) in rho]
		uo, ug = sum(ok[c, arm] for c in have if c in u_cells), sum(gain[c, arm] for c in have if c in u_cells)
		ao, ag = sum(ok[c, arm] for c in have), sum(gain[c, arm] for c in have)
		out[arm] = dict(u_ok=uo, u_gain=ug, ok=ao, gain=ag, n=len(have))
		if arm in NEW:
			v1_best, v2_best = max(v1_best, uo), max(v2_best, ug)
		cnt_rows.append(f"| {arm}{'' if arm in NEW else ' (설계 02)'} | {uo}/{len(u_cells)} | {ug}/{len(u_cells)} | {ao}/{len(have)} | {ag}/{len(have)} |")
	L += ['', '## 팔별 "막음"·"이득" 셀 수', '', '| 팔 | U에서 막음 | U에서 이득 | 6셀 막음 | 6셀 이득 |', '|---|---|---|---|---|'] + cnt_rows
	out['V1'], out['V2'] = bool(v1_best >= 2), bool(v2_best >= 2)
	# H
	h_cells = [c for c in cells if (c, 'D2000IMG') in rho and (c, 'D2000') in rho]
	h_flags = [bool(rho[c, 'D2000IMG'] <= rho[c, 'D2000'] - 0.1 or dfall[c, 'D2000IMG'] >= dfall[c, 'D2000'] + 0.5) for c in h_cells]
	out['H'] = dict(n=sum(h_flags), of=len(h_flags), yes=bool(sum(h_flags) >= 3), cells=[c for c, f in zip(h_cells, h_flags) if f])
	# G + probe table
	g_rows, g_flags = [], []
	for c in cells:
		p = pr[c]
		if not p:
			continue
		g_rows.append(f"| {c}{' (U)' if c in u_cells else ''} | {p['D2000_d02']['gv']:.1f} | " + ' | '.join(
			f"{p[arm]['gv']:.1f} / {p[arm]['gm']:.2f} / {p[arm]['kappa']:.2f}" for arm in NEW) + ' |')
		if c in u_cells:
			d = p['D2000IMG']
			g_flags.append(bool(d['gv_minus_d2000'] < 0 and d['gv_minus_d2000_ci'][1] < 0))
	out['G'] = dict(n=sum(g_flags), of=len(g_flags), yes=bool(len(g_flags) >= 2 and sum(g_flags) >= 2))
	L += ['', '## 가치 차이와 모델 오차, 그 팔 planner가 간 상태에서 (g_v / g_m / κ)', '',
		  'g_v = 3스텝 실제 가치 − 실현 return(보상 단위, 클수록 Q가 현실보다 낙관적). g_m = 예측 − 3스텝 실제. κ = 바탕 dynamics의 3스텝 오차 / 사전학습의 것.', '',
		  '| 셀 | D2000의 g_v (설계 02) | ' + ' | '.join(NEW) + ' |', '|---' * (len(NEW) + 2) + '|'] + g_rows
	# policy only
	pi_rows, p6 = [], []
	for c in cells:
		row, vals = [], {}
		for arm in PI_ARMS:
			e = _load(f'{OUT}/{c}/eval_{arm}_pi.json')
			if e:
				vals[arm] = _ep(e, 'shift', 'ret').mean()
				row.append(f"{vals[arm]:.0f} ({_ep(e, 'shift', 'falls').mean():.1f})")
			else:
				row.append('-')
		if 'PRE' in vals and 'D2000IMG' in vals:
			p6.append(bool(vals['D2000IMG'] > vals['PRE']))
		pi_rows.append(f'| {c} | ' + ' | '.join(row) + ' |')
	L += ['', '## 정책만 실행(planner 없음), 바뀐 조건 return (넘어짐): 보고만', '', '| 셀 | ' + ' | '.join(PI_ARMS) + ' |', '|---' * (len(PI_ARMS) + 1) + '|'] + pi_rows
	# refit logs
	rf_rows = []
	for c in cells:
		row = []
		for arm in IMG:
			r = _load(f'{OUT}/{c}/refit_{arm}.json')
			row.append(f"{r['q_mean_at_load']:.0f} → {r['log'][-1]['q_mean']:.0f}" if r else '-')
		rf_rows.append(f'| {c} | ' + ' | '.join(row) + ' |')
	L += ['', '## 다시 맞추기 전 → 뒤의 평균 Q (데이터 상태 512개, 정책 행동)', '', '| 셀 | ' + ' | '.join(IMG) + ' |', '|---' * (len(IMG) + 1) + '|'] + rf_rows
	p3 = max(out[arm]['u_ok'] for arm in ('D2000MIN', 'MIXMIN')) if 'D2000MIN' in out else 0
	p7 = [rho[c, 'D2000IMG15'] < rho[c, 'D2000IMG'] for c in cells if (c, 'D2000IMG15') in rho and (c, 'D2000IMG') in rho]
	yn = lambda b: '그렇다' if b else '아니다'
	L += ['', '## 판정', '',
		  f"- V1 가치 쪽 처방이 막는다: 새 팔 중 U 막음 최대 {v1_best}/{len(u_cells)} (기준 2) → {yn(out['V1'])}",
		  f"- V2 가치 쪽 처방이 이득까지 낸다: 새 팔 중 U 이득 최대 {v2_best}/{len(u_cells)} (기준 2) → {yn(out['V2'])}",
		  f"- H 가치를 다시 맞추면 나빠진다: D2000+IMG가 D2000보다 나쁜 셀 {out['H']['n']}/{out['H']['of']} (기준 3) → {yn(out['H']['yes'])}",
		  f"- G Q의 약속이 현실에 가까워진다: U 중 {out['G']['n']}/{out['G']['of']} (기준 2) → {yn(out['G']['yes'])}",
		  '', '## 예측 대조', '', f'- P3 MIN 팔의 U 막음 최대: {p3}/{len(u_cells)}',
		  f'- P6 정책만 실행에서 D2000+IMG > PRE: {sum(p6)}/{len(p6)}', f'- P7 IMG15의 ρ < IMG의 ρ: {sum(p7)}/{len(p7)}']
	out.update(rho={f'{c}|{arm}': v for (c, arm), v in rho.items()}, dfall={f'{c}|{arm}': v for (c, arm), v in dfall.items()})
	dst = os.path.join(OUT, 'summary')
	os.makedirs(dst, exist_ok=True)
	open(os.path.join(dst, 'summary.md'), 'w').write('\n'.join(L) + '\n')
	dump(os.path.join(dst, 'summary.json'), out)
	print('\n'.join(L))


def main():
	global OUT
	ap = argparse.ArgumentParser()
	ap.add_argument('cmd')
	ap.add_argument('--cell')
	ap.add_argument('--cells', default=None)
	ap.add_argument('--arm')
	ap.add_argument('--pi', action='store_true')
	ap.add_argument('--eps', type=int, default=EVAL_EPS)
	ap.add_argument('--steps', type=int, default=None, help='smoke tests only')
	ap.add_argument('--out_root', default=None, help='smoke tests only')
	ap.add_argument('--d02_root', default=None, help='smoke tests only')
	ap.add_argument('--d01_root', default=None, help='smoke tests only')
	a = ap.parse_args()
	if a.out_root:
		OUT = a.out_root
	if a.d02_root:
		d02.OUT = a.d02_root
	if a.d01_root:
		d02.D01 = a.d01_root
	dict(refit=cmd_refit, eval=cmd_eval, probe=cmd_probe, summarize=cmd_summarize)[a.cmd](a)


if __name__ == '__main__':
	main()
