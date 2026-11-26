"""Design 07 (research/docs/design07_second_round_collector_ko.md): who collects the second half of the real data.

One process = one (cell, seed). Within a seed the three arms share the round-1 data, the round-1 model M1, the initial
states of round 2 and the initial states of evaluation; they differ only in the round-2 collector.
  round 1   5 episodes, un-adapted planner, exploration noise on            (seed 1000 + s)
  M1        sim-mix fine-tuning of design 02 on the 5 episodes               (torch seed s)
  round 2   5 episodes per arm, exploration noise on                         (seed 2000 + s)
              CTRL un-adapted planner | CON consensus planner (sim model vs M1) | ADP ordinary planner with M1
  M2[arm]   the same fine-tuning from the pretrained checkpoint on the arm's 10 episodes (torch seed s)
  eval      consensus planner (sim model vs M2[arm]), no noise: 20 shifted + 5 nominal episodes (seed 3000 + s)
  PRE       un-adapted planner on the same evaluation seed: 20 shifted + 5 nominal episodes

Commands (from ~/tdmpc2 on gpu1):
  GPU=N research/scripts/td.sh research/scripts/d07.py run --cell quadruped-run__motor_0.35 --seed 1
  GPU=N research/scripts/td.sh research/scripts/d07.py summarize
Smoke tests must pass --out_root (another folder) and may shorten with --r_eps / --eval_eps / --nom_eps / --ft_steps.
The official TD-MPC2 code is imported, not modified. Reward, Q, policy and encoder stay pretrained.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch

from tdmpc2 import TDMPC2

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, run_cond, summary
from d02 import N_SIM, consistency, load_sim, make_buffer
from d06 import ConAgent

OUT = '/mnt/newssd/cjkim/td/d07'
CELLS = ['quadruped-run__motor_0.35', 'cheetah-run__friction_0.15']
ARMS = ['CTRL', 'CON', 'ADP']
NAME = {'CTRL': '대조', 'CON': '합의 수집', 'ADP': '적응 수집', 'PRE': '적응 없음'}
R_EPS, EVAL_EPS, NOM_EPS, FT_STEPS = 5, 20, 5, 2000
MARGIN, LEVEL, V0_RATIO = -25.0, 0.975, 0.9
TQ = {15: 2.4899, 31: 2.3556}                                               # t quantile 0.9875 (two-sided 97.5 %)


# ----------------------------------------------------------------------------- pieces

def fresh(cfg):
	agent = TDMPC2(cfg)
	agent.load(cfg.checkpoint)
	return agent


def stack(recs):
	return {k: np.stack([np.asarray(r[k]) for r in recs]) for k in ('obs', 'state', 'act', 'rew')}


def finetune_mix(cfg, data, sim, seed, steps):
	"""Design 02 'mix' fine-tuning (dynamics only, half of each update from nominal sim data), from the pretrained checkpoint."""
	agent = fresh(cfg)
	torch.manual_seed(seed)
	n = data['obs'].shape[0]
	buf, sbuf = make_buffer(cfg, data, n), make_buffer(cfg, sim, N_SIM)
	m = agent.model
	params = list(m._dynamics.parameters())
	opt = torch.optim.Adam(params, lr=cfg.lr)
	log = []
	for step in range(1, steps + 1):
		obs, action, _, _, _ = buf.sample()
		loss = consistency(m, obs, action, cfg)
		sobs, saction, _, _, _ = sbuf.sample()
		loss_s = consistency(m, sobs, saction, cfg)
		(0.5 * (loss + loss_s)).backward()
		torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
		opt.step()
		opt.zero_grad(set_to_none=True)
		if step % 100 == 0:
			log.append(dict(step=step, consistency=float(loss), sim=float(loss_s)))
	return agent, log


def con_of(agent, dyn0):
	return ConAgent(agent, dyn0)


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	s = a.seed
	folder = f'{a.out_root}/{a.cell}/s{s:02d}'
	if os.path.exists(f'{folder}/done.json'):
		print('exists, skip', folder, flush=True)
		return
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent0 = make(task, compile_=False)
	shifter, domain = Shifter(phys_of(env)), domain_of(task)
	dyn0 = copy.deepcopy(agent0.model._dynamics).eval()                    # pretrained (sim) dynamics, kept aside
	for p in dyn0.parameters():
		p.requires_grad_(False)
	sim = load_sim(task)
	res, t_all = dict(cell=a.cell, seed=s, args=vars(a)), time.time()

	def collect(actor, seed):
		eps, recs = run_cond(actor, env, shifter, domain, cond, a.r_eps, eval_mode=False, seed=seed, record=True)
		return eps, stack(recs)

	def evaluate(actor, tag):
		out = {}
		for c, n in (('shift', a.eval_eps), ('nominal', a.nom_eps)):
			real = cond if c == 'shift' else 'nominal'
			t0 = time.time()
			eps, recs = run_cond(actor, env, shifter, domain, real, n, eval_mode=True, seed=3000 + s, record=(c == 'shift'))
			out[c] = dict(cond=real, summary=summary(eps), episodes=eps, seconds=time.time() - t0)
			if c == 'shift':
				d = stack(recs)
				np.savez_compressed(f'{folder}/rec_{tag}.npz', obs=d['obs'].astype(np.float32), state=d['state'], rew=d['rew'].astype(np.float32))
			print(a.cell, s, tag, c, {k: round(v, 2) for k, v in out[c]['summary'].items()}, f'{out[c]["seconds"]:.0f}s', flush=True)
		return out

	# round 1 and M1 (shared by the three arms)
	eps1, d1 = collect(agent0, 1000 + s)
	res['round1'] = dict(summary=summary(eps1), episodes=eps1)
	np.savez_compressed(f'{folder}/data1.npz', **d1)
	m1, log1 = finetune_mix(cfg, d1, sim, s, a.ft_steps)
	torch.save(m1.model._dynamics.state_dict(), f'{folder}/M1_dyn.pt')          # only the dynamics network changes
	res['M1'] = log1[-1] if log1 else None
	print(a.cell, s, 'round1', round(res['round1']['summary']['ret'], 1), 'M1', res['M1'], flush=True)

	# reference: un-adapted planner on the evaluation seed
	res['PRE'] = evaluate(agent0, 'PRE')

	for arm in ARMS:
		if arm == 'CTRL':
			actor = agent0
		elif arm == 'CON':
			ag = fresh(cfg)
			ag.model._dynamics.load_state_dict(m1.model._dynamics.state_dict())
			actor = con_of(ag, dyn0)
		else:
			actor = fresh(cfg)
			actor.model._dynamics.load_state_dict(m1.model._dynamics.state_dict())
		eps2, d2 = collect(actor, 2000 + s)
		np.savez_compressed(f'{folder}/data2_{arm}.npz', **d2)
		data = {k: np.concatenate([d1[k], d2[k]], 0) for k in ('obs', 'act', 'rew')}
		m2, log2 = finetune_mix(cfg, data, sim, s, a.ft_steps)
		torch.save(m2.model._dynamics.state_dict(), f'{folder}/M2_{arm}_dyn.pt')
		con = con_of(m2, dyn0)
		ev = evaluate(con, arm)
		res[arm] = dict(round2=dict(summary=summary(eps2), episodes=eps2), M2=log2[-1] if log2 else None, eval=ev,
						sim_binds=float(np.nanmean(con.sim_binds)) if con.sim_binds else None)
		print(a.cell, s, arm, 'round2', round(res[arm]['round2']['summary']['ret'], 1), 'M2', res[arm]['M2'], flush=True)
		dump(f'{folder}/partial.json', res)
	res['seconds'] = time.time() - t_all
	dump(f'{folder}/done.json', res)
	print(a.cell, s, 'done', f'{res["seconds"]:.0f}s', flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 07 §5)

def tq(n):
	if n - 1 in TQ:
		return TQ[n - 1]
	from scipy import stats
	return float(stats.t.ppf(1 - (1 - LEVEL) / 2, n - 1))


def interval(x, seed=0):
	"""Seed-level mean with the t interval (primary) and the seed-bootstrap percentile interval (reported alongside)."""
	x = np.asarray(x, float)
	n = len(x)
	if n < 2:
		return dict(mean=float(x.mean()), t=[float('nan')] * 2, boot=[float('nan')] * 2, n=n)
	h = tq(n) * x.std(ddof=1) / np.sqrt(n)
	rs = np.random.RandomState(seed)
	bs = x[rs.randint(0, n, (10000, n))].mean(1)
	q = 100 * (1 - LEVEL) / 2
	return dict(mean=float(x.mean()), t=[float(x.mean() - h), float(x.mean() + h)], boot=[float(v) for v in np.percentile(bs, [q, 100 - q])],
				n=n, sd_seed=float(x.std(ddof=1)))


def verdict(iv, margin):
	lo, hi = iv['t']
	if not np.isfinite(lo):
		return '(없음)'
	return '통과' if lo > margin else ('실패' if hi < margin else '미결')


def cmd_summarize(a):
	L, S = [], {}
	fmt = lambda iv: f'{iv["mean"]:+.1f} [{iv["t"][0]:+.1f}, {iv["t"][1]:+.1f}] (bootstrap [{iv["boot"][0]:+.1f}, {iv["boot"][1]:+.1f}])'
	L.append('# 설계 07 결과 요약 (자동 생성, `d07.py summarize`)\n')
	L.append('팔은 둘째 수집 라운드를 누가 모았는가다(대조 = 적응 없는 planner, 합의 수집 = 합의 planner, 적응 수집 = 첫 라운드 모델만 쓰는 planner). '
			 '배치는 모두 합의 planner다. 구간은 seed 단위 t 구간 97.5 %(주)와 seed bootstrap 97.5 %(함께 보고)다. 판정은 t 구간을 따른다.\n')
	for cell in CELLS:
		runs = [json.load(open(f)) for f in sorted(glob.glob(f'{a.out_root}/{cell}/s*/done.json'))]
		if not runs:
			continue
		n = len(runs)
		ep = lambda r, arm, c, k: np.array([e[k] for e in (r['PRE'] if arm == 'PRE' else r[arm]['eval'])[c]['episodes']])
		ret = {arm: np.stack([ep(r, arm, 'shift', 'ret') for r in runs]) for arm in ['PRE'] + ARMS}        # seeds x episodes
		nom = {arm: np.array([ep(r, arm, 'nominal', 'ret').mean() for r in runs]) for arm in ['PRE'] + ARMS}
		col = {arm: np.array([r[arm]['round2']['summary']['ret'] for r in runs]) for arm in ARMS}
		sm = {arm: ret[arm].mean(1) for arm in ret}
		C = S[cell] = dict(n_seeds=n, seeds=[r['seed'] for r in runs])
		L.append(f'\n## {cell} (seed {n}개)\n')
		L.append('| 팔 | 배치 return | 넘어짐 사건 / 넘어져 있는 시간 | 부드러움 / 노력 / 흔들림 | 평소 조건 return | 둘째 라운드 수집 return (넘어짐 / 넘어져 있는 시간) |')
		L.append('|---|---|---|---|---|---|')
		for arm in ['PRE'] + ARMS:
			q = lambda k: float(np.mean([ep(r, arm, 'shift', k).mean() for r in runs]))
			c2 = '–' if arm == 'PRE' else (f'{col[arm].mean():.0f} ({np.mean([r[arm]["round2"]["summary"]["falls"] for r in runs]):.1f} / '
										   f'{np.mean([r[arm]["round2"]["summary"]["fallen_frac"] for r in runs]):.2f})')
			L.append(f'| {NAME[arm]} | {sm[arm].mean():.0f} | {q("falls"):.1f} / {q("fallen_frac"):.2f} | {q("smooth"):.2f} / {q("effort"):.2f} / {q("angvel_rms"):.2f} | {nom[arm].mean():.0f} | {c2} |')
		C['return'] = {arm: float(sm[arm].mean()) for arm in sm}
		C['round1'] = float(np.mean([r['round1']['summary']['ret'] for r in runs]))
		L.append(f'\n첫 라운드 수집 return(세 팔 공통) {C["round1"]:.0f}.\n')
		T = C['tests'] = {}
		T['harm'] = {arm: interval((ret[arm] - ret['PRE']).mean(1)) for arm in ARMS}                       # paired by evaluation episode
		T['gain'] = interval(sm['CON'] - sm['CTRL'])
		T['preserve'] = interval(sm['CON'] - 0.5 * (sm['ADP'] + sm['CTRL']))
		T['adp_gain'] = interval(sm['ADP'] - sm['CTRL'])
		T['cost'] = interval(col['CON'] - col['CTRL'])
		T['cost_vs_adp'] = interval(col['CON'] - col['ADP'])
		pre_nom = nom['PRE'].mean()
		T['keep'] = {arm: dict(ratio=float(nom[arm].mean() / pre_nom), ok=bool(nom[arm].mean() >= V0_RATIO * pre_nom)) for arm in ARMS}
		g_ok = verdict(T['gain'], 0.0) == '통과'
		p_ok = g_ok and verdict(T['preserve'], 0.0) == '통과'
		L.append('| 판정 | 비교 | 한계 | seed 단위 평균 [t 구간 97.5 %] | 결과 |')
		L.append('|---|---|---|---|---|')
		for arm in ('CON', 'CTRL', 'ADP'):
			L.append(f'| 해 없음 | {NAME[arm]} − 적응 없음 | {MARGIN:.0f} | {fmt(T["harm"][arm])} | {verdict(T["harm"][arm], MARGIN)} |')
		L.append(f'| ① 이득 | 합의 수집 − 대조 | 0 | {fmt(T["gain"])} | {verdict(T["gain"], 0.0)} |')
		note = '' if g_ok else ' (앞이 서지 않아 보고만)'
		if T['adp_gain']['mean'] < 0:
			note += ' (적응 수집이 대조보다 낮음: 의미 없음)'
		L.append(f'| ② 효과 보존 | 합의 수집 − (적응 수집 + 대조)/2 | 0 | {fmt(T["preserve"])} | {verdict(T["preserve"], 0.0)}{note} |')
		L.append(f'| (참고) | 적응 수집 − 대조 | 0 | {fmt(T["adp_gain"])} | {verdict(T["adp_gain"], 0.0)} |')
		note = '' if p_ok else ' (앞이 서지 않아 보고만)'
		L.append(f'| ③ 수집 비용 | 수집 return: 합의 수집 − 대조 | {MARGIN:.0f} | {fmt(T["cost"])} | {verdict(T["cost"], MARGIN)}{note} |')
		L.append(f'| (함께 보고) | 수집 return: 합의 수집 − 적응 수집 | 0 | {fmt(T["cost_vs_adp"])} | {verdict(T["cost_vs_adp"], 0.0)} |')
		L.append('\n보존(평소 조건 return / 적응 없음의 평소 조건 return, 0.9 이상이면 통과): '
				 + ', '.join(f'{NAME[arm]} {T["keep"][arm]["ratio"]:.2f} {"통과" if T["keep"][arm]["ok"] else "실패"}' for arm in ARMS) + '.')
		differ = [k for k, iv, m in (('해 없음(합의 수집)', T['harm']['CON'], MARGIN), ('① 이득', T['gain'], 0.0))
				  if np.isfinite(iv['t'][0]) and ((iv['t'][0] > m) != (iv['boot'][0] > m) or (iv['t'][1] < m) != (iv['boot'][1] < m))]
		if differ:
			L.append(f'\nt 구간과 bootstrap 구간의 판정이 다른 곳: {", ".join(differ)}. 등록한 대로 t 구간을 따른다.')
	q, c = S.get(CELLS[0]), S.get(CELLS[1])
	if q and c:
		s1, g1 = verdict(q['tests']['harm']['CON'], MARGIN), verdict(c['tests']['gain'], 0.0)
		S['primary'] = dict(S1=s1, G1=g1, claim=bool(s1 == '통과' and g1 == '통과'))
		L.append(f'\n## 주 판정\n\n- S1 해 없음(quadruped 모터, 합의 수집 − 적응 없음, 한계 {MARGIN:.0f}): **{s1}** (seed {q["n_seeds"]}개)')
		L.append(f'- G1 이득(cheetah 마찰, 합의 수집 − 대조, 한계 0): **{g1}** (seed {c["n_seeds"]}개)')
		L.append(f'- 주장(둘 다 통과해야 성립): **{"성립" if S["primary"]["claim"] else "성립하지 않음"}**')
		ext = [n for n, v, d in (('quadruped 모터', s1, q), ('cheetah 마찰', g1, c)) if v == '미결' and d['n_seeds'] <= 16]
		if ext:
			L.append(f'- 연장 규칙(§6): {", ".join(ext)}에 seed 17–32를 한 번 추가한다.')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('cmd', choices=['run', 'summarize'])
	ap.add_argument('--cell')
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--r_eps', type=int, default=R_EPS, help='smoke tests only')
	ap.add_argument('--eval_eps', type=int, default=EVAL_EPS, help='smoke tests only')
	ap.add_argument('--nom_eps', type=int, default=NOM_EPS, help='smoke tests only')
	ap.add_argument('--ft_steps', type=int, default=FT_STEPS, help='smoke tests only')
	ap.add_argument('--out_root', default=OUT)
	ap.add_argument('--res_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'd07'))
	a = ap.parse_args()
	dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)
