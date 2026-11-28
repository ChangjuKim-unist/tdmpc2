"""Design 18 (research/docs/design18_probe_handover_ko.md): hand the robot to the learner only when its own probe episodes beat the
frozen agent, and take it back when it falls behind.

Online fine-tuning of every component (encoder, dynamics, reward, Q, policy) under the shifted physics, one update per environment step
after a 5,000-step seed phase, as design 13's RAW (d13.py). What changes is who acts in each training episode:
  RAW    the learner always (design 13's RAW, rerun to log falls in every training episode)
  TWO    the frozen pretrained agent acts; every 5th episode is a probe (the learner acts). Handover after a probe when the mean of the
         last K probes >= (1 + M) x the cumulative mean of every frozen-agent episode so far. After handover the learner acts; revert when
         the mean of its last W episodes < R x the frozen cumulative mean (the probe history restarts). The learner keeps learning.
  ONE    TWO without revert
  TIME   the same probes (used for learning only); handover at a fixed step (--handover_step, the mean first-handover step of TWO in
         that condition), never reverts
K, M, W, R were chosen by research/scripts/d18_rule_sim.py before any design-18 run (§3 of the design).
Frozen-agent and probe episodes use exploration noise (training collection), like design 13's training episodes.

Commands (from ~/tdmpc2 on gpu1, or with td4.sh on gpu4):
  GPU=N research/scripts/td.sh research/scripts/d18.py run --cell quadruped-run__motor_0.35 --arm TWO --seed 1
  GPU=N research/scripts/td.sh research/scripts/d18.py time_step --cell quadruped-run__motor_0.35
  GPU=N research/scripts/td.sh research/scripts/d18.py summarize --res_dir /mnt/newssd/cjkim/td/d18/res
Smoke tests must pass --out_root (another folder) and may shorten with --steps / --seed_steps / --eval_freq / --eval_eps.
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

from common.buffer import Buffer

from d01 import Shifter, angvel, cell_parts, domain_of, dump, fallen, make, phys_of, reseed_env, run_episode, summary
from d07 import fresh
from d11 import q_stats
from d13 import to_td

OUT = '/mnt/newssd/cjkim/td/d18'
CELLS = ['quadruped-run__motor_0.35', 'walker-run__friction_0.15']
SHORT = {'quadruped-run__motor_0.35': 'quadruped 모터', 'walker-run__friction_0.15': 'walker 마찰'}
ARMS = ['RAW', 'TWO', 'ONE', 'TIME']
ANAME = {'RAW': 'RAW (늘 학습자)', 'TWO': '양방향', 'ONE': '한 방향', 'TIME': '묶은 시간'}
EVERY, K, M, W, R = 5, 8, 0.10, 10, 0.7
POST = 20                                                                   # episodes (10k steps) after a handover for the post-handover minimum


def episode(actor, learner, buffer, env, domain, step, seed_steps, eval_mode=False):
	"""One training episode by `actor`; the learner updates once per step after the seed phase. Returns metrics, tds and the new step."""
	p = phys_of(env)
	obs, done, t = env.reset(), False, 0
	tds, acts, av = [to_td(env, obs)], [], []
	R_, falls, fsteps, prev, last = 0.0, 0, 0, False, {}
	while not done:
		a = actor.act(obs, t0=t == 0, eval_mode=eval_mode)
		obs, r, done, _ = env.step(a)
		tds.append(to_td(env, obs, a, r))
		R_ += float(r)
		t += 1
		f = bool(fallen(domain, p))
		falls += int(f and not prev)
		fsteps += int(f)
		prev = f
		av.append(angvel(domain, p))
		acts.append(a.numpy())
		if step >= seed_steps and buffer.num_eps > 0:
			last = learner.update(buffer)
		step += 1
	A = np.asarray(acts)
	d2 = A[2:] - 2 * A[1:-1] + A[:-2]
	m = dict(ret=R_, falls=falls, fallen_frac=fsteps / t, fallen_steps=fsteps, smooth=float((d2 ** 2).sum(1).mean()),
			 effort=float((A ** 2).sum(1).mean()), angvel_rms=float(np.sqrt(np.mean(np.square(av)))))
	return m, torch.cat(tds), step, last


@torch.no_grad()
def probe_signal(learner, frozen, td):
	"""Design 17's S2 and S3 on one probe episode (learner / frozen)."""
	d = dict(obs=td['obs'].numpy()[None], act=td['action'][1:].numpy()[None], rew=td['reward'][1:].numpy()[None])
	ql, qf = q_stats(learner, d), q_stats(frozen, d)
	return dict(S2=float(ql['resid'].abs().mean() / max(float(qf['resid'].abs().mean()), 1e-6)), S3=float(ql['u_vis'].mean() / max(float(qf['u_vis'].mean()), 1e-6)))


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	folder = f'{a.out_root}/{a.cell}/{a.arm}_s{a.seed}'
	if os.path.exists(f'{folder}/final.pt'):
		return
	if a.arm == 'TIME' and a.handover_step is None:
		raise SystemExit('TIME needs --handover_step')
	os.makedirs(folder, exist_ok=True)
	cfg, env, learner = make(task, compile_=True, seed=a.seed)
	fcfg = copy.copy(cfg)
	fcfg.compile = False
	frozen = fresh(fcfg)                                                    # the pretrained agent, never updated
	cfg.steps = a.steps + a.seed_steps
	cfg.buffer_size = cfg.steps + 1000
	buffer = Buffer(cfg)
	domain = domain_of(task)
	Shifter(phys_of(env)).set(cond)
	K_, M_, W_, R_ = a.K, a.M, a.W, a.R
	log = dict(cell=a.cell, arm=a.arm, args=vars(a), rule=dict(EVERY=EVERY, K=K_, M=M_, W=W_, R=R_), evals=[], train=[], handovers=[], reverts=[])
	step, t0 = 0, time.time()
	mode = 'frozen'                                                         # who acts outside probes
	fsum, fn, probes, since, learner_eps = 0.0, 0, [], 0, []

	def evaluate():
		reseed_env(env, 4000 + len(log['evals']))
		eps = [run_episode(learner, env, domain, True)[0] for _ in range(a.eval_eps)]
		e = dict(step=step, seconds=time.time() - t0, **summary(eps))
		log['evals'].append(e)
		dump(f'{folder}/log.json', log)
		print(a.cell, a.arm, a.seed, 'eval', {k: round(v, 2) for k, v in e.items()}, flush=True)

	next_eval = 0
	while step < cfg.steps:
		if step >= next_eval:
			evaluate()
			next_eval += a.eval_freq
		seed_phase = step < a.seed_steps
		if not seed_phase and a.arm == 'RAW':
			mode = 'learner'
		if not seed_phase and a.arm == 'TIME' and mode == 'frozen' and step >= a.handover_step:
			mode = 'learner'
			log['handovers'].append(step)
		if seed_phase:
			role = 'seed'
		elif mode == 'learner':
			role = 'learner'
		else:
			role = 'probe' if since % EVERY == EVERY - 1 else 'frozen'
		actor = learner if role in ('learner', 'probe') else frozen
		start = step
		m, td, step, last = episode(actor, learner, buffer, env, domain, step, a.seed_steps)
		buffer.add(td)
		rec = dict(start=start, step=step, role=role, **m, **{k: float(v) for k, v in last.items()})
		if role in ('seed', 'frozen'):
			fsum, fn = fsum + m['ret'], fn + 1
		if role in ('frozen', 'probe'):
			since += 1
		if role == 'probe':
			probes.append(m['ret'])
			rec.update(probe_signal(learner, frozen, td))
			if a.arm in ('TWO', 'ONE') and len(probes) >= K_ and np.mean(probes[-K_:]) >= (1 + M_) * fsum / fn:
				mode, learner_eps = 'learner', []
				log['handovers'].append(step)
		elif role == 'learner' and a.arm != 'RAW':
			learner_eps.append(m['ret'])
			if a.arm == 'TWO' and len(learner_eps) >= W_ and np.mean(learner_eps[-W_:]) < R_ * fsum / fn:
				mode, probes, since = 'frozen', [], 0
				log['reverts'].append(step)
		rec['frozen_mean'] = fsum / max(fn, 1)
		log['train'].append(rec)
		if len(log['train']) % 20 == 0:
			print(a.cell, a.arm, a.seed, 'train', step, role, round(m['ret'], 1), f'frozen mean {fsum / max(fn, 1):.1f}', f'{step / (time.time() - t0):.1f} steps/s', flush=True)
	evaluate()
	learner.save(f'{folder}/final.pt')
	log['seconds'] = time.time() - t0
	dump(f'{folder}/log.json', log)
	print(a.cell, a.arm, a.seed, 'done', 'handovers', log['handovers'], 'reverts', log['reverts'], flush=True)


def cmd_time_step(a):
	"""Mean first-handover step of the TWO runs in a condition, rounded up to an episode boundary (500 steps)."""
	firsts = [json.load(open(f))['handovers'][0] for f in sorted(glob.glob(f'{a.out_root}/{a.cell}/TWO_s*/log.json'))
			  if os.path.exists(os.path.dirname(f) + '/final.pt') and json.load(open(f))['handovers']]
	print(int(np.ceil(np.mean(firsts) / 500) * 500) if firsts else 'none', firsts)


# ----------------------------------------------------------------------------- summary (registered analysis, design 18 §5)

def run_stats(g):
	tr = g['train']
	cum_fallen = int(sum(t['fallen_steps'] for t in tr))
	ev = g['evals']
	end = float(np.mean([e['ret'] for e in ev[-3:]]))
	post = []
	for h in g['handovers']:
		after = [t['ret'] for t in tr if t['start'] >= h][:POST]
		if after:
			post.append(min(after))
	roles = {r: sum(t['role'] == r for t in tr) for r in ('seed', 'frozen', 'probe', 'learner')}
	probe_s2 = [(t['start'], t['S2']) for t in tr if t['role'] == 'probe' and 'S2' in t]
	return dict(cum_fallen=cum_fallen, end=end, post_min=post, handovers=g['handovers'], reverts=g['reverts'], roles=roles,
				train_ret=float(np.mean([t['ret'] for t in tr])), probe_s2=probe_s2)


def cmd_summarize(a):
	L, S = ['# 설계 18 결과 요약 (자동 생성, `d18.py summarize`)\n'], {}
	L.append('넘어진 시간 누적 = 학습 에피소드 전체(처음 5,000 스텝, 얼린 agent, 탐침, 학습자 모두)에서 넘어져 있던 스텝 수의 합. 끝점 = 학습자 단독 평가(10 에피소드)의 마지막 세 번 평균. '
			 f'넘겨받은 직후 최저 = 넘긴 뒤 {POST} 에피소드(1만 스텝) 안의 가장 낮은 학습 에피소드 return.\n')
	R_ = {}
	for cell in CELLS:
		for arm in ARMS:
			for f in sorted(glob.glob(f'{a.out_root}/{cell}/{arm}_s*/log.json')):
				if os.path.exists(os.path.dirname(f) + '/final.pt'):
					g = json.load(open(f))
					R_[cell, arm, g['args']['seed']] = run_stats(g)
	for cell in CELLS:
		L.append(f'\n## {SHORT[cell]}\n')
		L.append('| 팔 | seed | 넘어진 시간 누적 (스텝) | 끝점 | 넘긴 스텝 | 되돌린 스텝 | 넘겨받은 직후 최저 | 학습 에피소드 평균 return | 에피소드 수 (처음 / 얼린 / 탐침 / 학습자) |')
		L.append('|---|---|---|---|---|---|---|---|---|')
		for arm in ARMS:
			for s in (1, 2, 3):
				r = R_.get((cell, arm, s))
				if r is None:
					continue
				ro = r['roles']
				L.append(f'| {ANAME[arm]} | {s} | {r["cum_fallen"]:,} | {r["end"]:.0f} | {", ".join(f"{h:,}" for h in r["handovers"]) or "없음"} | {", ".join(f"{h:,}" for h in r["reverts"]) or "–"} | '
						 f'{", ".join(f"{x:.0f}" for x in r["post_min"]) or "–"} | {r["train_ret"]:.0f} | {ro["seed"]} / {ro["frozen"]} / {ro["probe"]} / {ro["learner"]} |')
	pairs = [(c, s) for c in CELLS for s in (1, 2, 3) if (c, 'TWO', s) in R_ and (c, 'RAW', s) in R_]
	lower = [R_[c, 'TWO', s]['cum_fallen'] < R_[c, 'RAW', s]['cum_fallen'] for c, s in pairs]
	handed = [bool(R_[c, 'TWO', s]['handovers']) for c, s in pairs]
	if len(pairs) < 6:
		verdict = f'미완 (짝 {len(pairs)} / 6)'
	elif not all(handed):
		verdict = '판정 불가: 넘기지 못한 실행이 있다'
	elif all(lower):
		verdict = '통과'
	else:
		verdict = '아님'
	S['primary'] = dict(pairs=len(pairs), lower=int(sum(lower)), all_handed=all(handed), verdict=verdict)
	L.append(f'\n## 주 판정: 양방향의 넘어진 시간 누적이 RAW보다 낮은가 (같은 조건·seed 짝 6개)\n')
	L.append(f'- 낮은 짝 {sum(lower)} / {len(pairs)}, 모든 양방향 실행이 한 번 이상 넘김: {"예" if all(handed) else "아니오"} → **{verdict}**')
	L.append('\n## 보조 비교 (보고만, 같은 조건·seed 짝의 차이)\n')
	L.append('| 비교 | 답하는 질문 | 넘어진 시간 누적 차이 (짝별) | 끝점 차이 (짝별) |')
	L.append('|---|---|---|---|')
	for x, y, q in (('ONE', 'TIME', '실제 return으로 시점을 정하는 것이 필요한가'), ('TWO', 'ONE', '되돌리기가 무엇을 더하는가'), ('TWO', 'RAW', '방법 전체의 효과')):
		ps = [(c, s) for c in CELLS for s in (1, 2, 3) if (c, x, s) in R_ and (c, y, s) in R_]
		dfall = ', '.join(f'{R_[c, x, s]["cum_fallen"] - R_[c, y, s]["cum_fallen"]:+,}' for c, s in ps)
		dend = ', '.join(f'{R_[c, x, s]["end"] - R_[c, y, s]["end"]:+.0f}' for c, s in ps)
		S[f'{x}-{y}'] = dict(fallen=[R_[c, x, s]['cum_fallen'] - R_[c, y, s]['cum_fallen'] for c, s in ps], end=[R_[c, x, s]['end'] - R_[c, y, s]['end'] for c, s in ps])
		L.append(f'| {ANAME[x]} − {ANAME[y]} | {q} | {dfall or "–"} | {dend or "–"} |')
	S['runs'] = {f'{c}|{arm}|{s}': v for (c, arm, s), v in R_.items()}
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', choices=CELLS, required=True)
	p.add_argument('--arm', choices=ARMS, required=True)
	p.add_argument('--seed', type=int, default=1)
	p.add_argument('--handover_step', type=int, default=None)
	p.add_argument('--steps', type=int, default=100_000)
	p.add_argument('--seed_steps', type=int, default=5_000)
	p.add_argument('--eval_freq', type=int, default=10_000)
	p.add_argument('--eval_eps', type=int, default=10)
	p.add_argument('--K', type=int, default=K)                              # the registered rule; other values only in smoke tests
	p.add_argument('--M', type=float, default=M)
	p.add_argument('--W', type=int, default=W)
	p.add_argument('--R', type=float, default=R)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('time_step')
	p.add_argument('--cell', choices=CELLS, required=True)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, time_step=cmd_time_step, summarize=cmd_summarize)[a.cmd](a)
