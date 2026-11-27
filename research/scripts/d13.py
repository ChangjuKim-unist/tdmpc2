"""Design 13 (research/docs/design13_finetune_dip_ko.md): gate 0. Does the early collapse of full online fine-tuning reproduce over seeds, and does it survive sim-mixed updates.

Based on d07_ceiling.py (design 07 §9). Arms:
  RAW    the procedure of design 07 §9 unchanged
  MIX    every update batch is half real (online buffer) and half nominal sim data (design 02's sim episodes, with rewards)
  SCALE  RAW, but before the first update the running Q scale of the policy loss (common/scale.py; not stored in the checkpoint, so it
         restarts at 1) is set to the pretrained agent's own value: the 5-95 percentile range of Q(z, pi(z)) over 10 batches of the
         seed buffer (registered amendment, design 13 section 3a)
Checkpoints are saved every --ckpt_freq steps up to --ckpt_until (for the later gates).

Original header of d07_ceiling.py:

Continue the official online TD-MPC2 training (encoder, dynamics, reward, Q, policy) from the pretrained checkpoint under the
shifted physics. The result is a LOWER bound of the return that can be reached under the shifted condition.

Differences from the official trainer (trainer/online_trainer.py), both to protect the pretrained weights:
  - the first --seed_steps environment steps are collected by the pretrained planner (not by random actions), with no update;
  - there is no burst of seed_steps updates at the end of seeding; after seeding it is one update per environment step.
Not restored from the checkpoint (the checkpoint stores model weights only): optimiser states and the running Q scale.

  GPU=N research/scripts/td.sh research/scripts/d13.py run --cell quadruped-run__motor_0.35 --arm RAW --seed 1
  GPU=N research/scripts/td.sh research/scripts/d13.py summarize --res_dir /mnt/newssd/cjkim/td/d13/res
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
from tensordict.tensordict import TensorDict

from common.buffer import Buffer

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, reseed_env, run_episode, summary

OUT = '/mnt/newssd/cjkim/td/d13'
CELLS = ['quadruped-run__motor_0.35', 'walker-run__friction_0.15']
SHORT = {'quadruped-run__motor_0.35': 'quadruped 모터', 'walker-run__friction_0.15': 'walker 마찰'}
DYN_REF = {'quadruped-run__motor_0.35': 571.0, 'walker-run__friction_0.15': 624.0}
DIP, WINDOW = 0.25, 60_000


class MixBuffer:
	"""sample() returns half a batch from the online (real) buffer and half from the nominal sim buffer."""

	def __init__(self, real, sim):
		self.real, self.sim, self.calls = real, sim, 0

	@property
	def num_eps(self):
		return self.real.num_eps

	def add(self, td):
		return self.real.add(td)

	def sample(self):
		a, b = self.real.sample(), self.sim.sample()
		if self.calls < 200:                                                # guard: both halves must be free of NaN
			self.calls += 1
			for name, t in (('real', a), ('sim', b)):
				assert not any(torch.isnan(x).any() for x in t[:4] if x is not None), f'NaN in the {name} half of the batch'
		return tuple(None if x is None else torch.cat([x, y], 1) for x, y in zip(a[:4], b[:4])) + (None,)


def to_td(env, obs, action=None, reward=None):
	"""Same layout as OnlineTrainer.to_td (non-episodic: terminated is always 0 after the first step)."""
	if action is None:
		action = torch.full_like(env.rand_act(), float('nan'))
	if reward is None:
		reward = torch.tensor(float('nan'))
		terminated = torch.tensor(float('nan'))
	else:
		terminated = torch.tensor(0.0)
	return TensorDict(obs=obs.unsqueeze(0).cpu(), action=action.unsqueeze(0), reward=torch.as_tensor(reward, dtype=torch.float32).unsqueeze(0),
					  terminated=terminated.unsqueeze(0), batch_size=(1,))


@torch.no_grad()
def init_scale(agent, buffer, n=10):
	"""Set agent.scale (the running Q scale of the policy loss) to the pretrained Q range, as update_pi would measure it, before any update."""
	vals = []
	for _ in range(n):
		obs = buffer.sample()[0]                                            # (horizon + 1, batch, obs)
		z = agent.model.encode(obs, None)
		action, _ = agent.model.pi(z, None)
		qs = agent.model.Q(z, action, None, return_type='avg', detach=True)
		p = agent.scale._percentile(qs[0])
		vals.append(float(torch.clamp(p[1] - p[0], min=1.0)))
	agent.scale.value.fill_(float(np.mean(vals)))
	return dict(value=float(np.mean(vals)), batches=vals)


def cmd_run(a):
	task, cond = cell_parts(a.cell)
	folder = f'{a.out_root}/{a.cell}/{a.arm}_s{a.seed}'
	if os.path.exists(f'{folder}/final.pt'):
		return
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent = make(task, compile_=True, seed=a.seed)
	cfg.steps = a.steps + a.seed_steps
	cfg.buffer_size = cfg.steps + 1000
	if a.arm == 'MIX':
		from d02 import N_SIM, load_sim, make_buffer
		half = copy.copy(cfg)
		half.batch_size = cfg.batch_size // 2
		simcfg = copy.copy(half)
		simcfg.steps = 10 ** 6                                              # Buffer capacity is min(buffer_size, steps); keep all sim steps even in short smoke runs
		sim = make_buffer(simcfg, load_sim(task), N_SIM)                    # make_buffer sets buffer_size (20,000) on the copy it is given
		buffer = MixBuffer(Buffer(half), sim)
	else:
		buffer = Buffer(cfg)
	domain = domain_of(task)
	Shifter(phys_of(env)).set(cond)
	log = dict(cell=a.cell, args=vars(a), evals=[], train=[])
	step, t0, last, saved = 0, time.time(), {}, set()

	def evaluate():
		reseed_env(env, 4000 + len(log['evals']))
		eps = [run_episode(agent, env, domain, True)[0] for _ in range(a.eval_eps)]
		e = dict(step=step, updates=max(0, step - a.seed_steps), seconds=time.time() - t0, **summary(eps))
		log['evals'].append(e)
		dump(f'{folder}/log.json', log)
		print(a.cell, 'eval', {k: round(v, 2) for k, v in e.items()}, flush=True)

	while step <= cfg.steps:
		if step % a.eval_freq == 0:
			evaluate()
		k = (step // a.ckpt_freq) * a.ckpt_freq
		if 0 < k <= a.ckpt_until and k not in saved and step % a.ckpt_freq == 0:
			agent.save(f'{folder}/step{k}.pt')
			saved.add(k)
		obs, done, tds, R = env.reset(), False, None, 0.0
		tds = [to_td(env, obs)]
		while not done:
			action = agent.act(obs, t0=len(tds) == 1, eval_mode=False)
			obs, reward, done, info = env.step(action)
			tds.append(to_td(env, obs, action, reward))
			R += float(reward)
			if a.arm == 'SCALE' and not log.get('scale_init') and step >= a.seed_steps and buffer.num_eps > 0:
				log['scale_init'] = init_scale(agent, buffer)
				print(a.cell, 'Q scale initialised to', log['scale_init'], flush=True)
			if step >= a.seed_steps and buffer.num_eps > 0:
				last = agent.update(buffer)
			step += 1
		buffer.add(torch.cat(tds))
		log['train'].append(dict(step=step, ret=R, **{k: float(v) for k, v in last.items()}))
		if len(log['train']) % 20 == 0:
			print(a.cell, 'train', step, round(R, 1), f'{step / (time.time() - t0):.1f} steps/s', flush=True)
	agent.save(f'{folder}/final.pt')
	ev = [e['ret'] for e in log['evals']]
	log['value'] = float(np.mean(ev[-3:]))
	log['rising'] = bool(len(ev) >= 6 and np.mean(ev[-3:]) - np.mean(ev[-6:-3]) > 20)
	log['seconds'] = time.time() - t0
	dump(f'{folder}/log.json', log)
	print(a.cell, 'done: value (mean of last 3 evaluations)', round(log['value'], 1), 'rising', log['rising'], flush=True)


# ----------------------------------------------------------------------------- summary (registered analysis, design 13 §4)

def cmd_summarize(a):
	L, S = ['# 설계 13 결과 요약 (자동 생성, `d13.py summarize`)\n'], {}
	L.append(f'꺼짐 깊이 = (학습 전 평가 − 처음 {WINDOW:,} 스텝 안의 가장 낮은 평가) / 학습 전 평가. {DIP} 이상이면 그 실행은 "꺼졌다". 평가는 실행마다 10 에피소드다.\n')
	for cell in CELLS:
		L.append(f'\n## {SHORT[cell]}\n')
		L.append('| 팔 | seed | 학습 전 | 가장 낮은 평가 (스텝) | 꺼짐 깊이 | 꺼졌나 | 그때 넘어져 있는 시간 | 학습 에피소드 10개 평균의 최저 | 마지막 세 평가의 평균 | dynamics만 고친 기준 |')
		L.append('|---|---|---|---|---|---|---|---|---|---|')
		S[cell] = {}
		for arm in ('RAW', 'MIX', 'SCALE'):
			rows = []
			for f in sorted(glob.glob(f'{a.out_root}/{cell}/{arm}_s*/log.json')):
				g = json.load(open(f))
				ev = g['evals']
				if len(ev) < 2:
					continue
				first = [e for e in ev if 0 < e['step'] <= WINDOW]
				if not first:
					continue
				lo = min(first, key=lambda e: e['ret'])
				depth = (ev[0]['ret'] - lo['ret']) / ev[0]['ret']
				tr = [t['ret'] for t in g['train'] if t['step'] <= WINDOW + g['args']['seed_steps']][g['args']['seed_steps'] // 500:]
				roll = min(float(np.mean(tr[i:i + 10])) for i in range(0, max(1, len(tr) - 9))) if tr else float('nan')
				r = dict(seed=g['args']['seed'], start=ev[0]['ret'], low=lo['ret'], low_step=lo['step'], depth=depth, dipped=bool(depth >= DIP), fallen=lo['fallen_frac'],
						 train_low=roll, final=float(np.mean([e['ret'] for e in ev[-3:]])), last_step=ev[-1]['step'], done=os.path.exists(os.path.dirname(f) + '/final.pt'))
				rows.append(r)
				L.append(f'| {arm} | {r["seed"]} | {r["start"]:.0f} | {r["low"]:.0f} ({r["low_step"]:,}) | {depth:.2f} | {"예" if r["dipped"] else "아니오"} | {r["fallen"]:.2f} | {roll:.0f} | '
						 f'{r["final"]:.0f}{"" if r["done"] else f" (진행 중, {r_last(r)})"} | {DYN_REF[cell]:.0f} |')
			S[cell][arm] = dict(runs=rows, dipped=int(sum(r['dipped'] for r in rows)), n=len(rows))
		for arm in ('RAW', 'MIX', 'SCALE'):
			d = S[cell][arm]
			L.append(f'\n- {arm}: 꺼진 실행 {d["dipped"]} / {d["n"]}')
	os.makedirs(a.res_dir, exist_ok=True)
	open(f'{a.res_dir}/summary.md', 'w').write('\n'.join(L) + '\n')
	dump(f'{a.res_dir}/summary.json', S)
	print('\n'.join(L))


def r_last(r):
	return f'{r["last_step"]:,} 스텝까지'


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest='cmd', required=True)
	p = sub.add_parser('run')
	p.add_argument('--cell', required=True)
	p.add_argument('--arm', choices=['RAW', 'MIX', 'SCALE'], required=True)
	p.add_argument('--steps', type=int, default=100_000)
	p.add_argument('--seed_steps', type=int, default=5_000)
	p.add_argument('--eval_freq', type=int, default=10_000)
	p.add_argument('--eval_eps', type=int, default=10)
	p.add_argument('--ckpt_freq', type=int, default=10_000)
	p.add_argument('--ckpt_until', type=int, default=60_000)
	p.add_argument('--seed', type=int, default=1)
	p.add_argument('--out_root', default=OUT)
	p = sub.add_parser('summarize')
	p.add_argument('--out_root', default=OUT)
	p.add_argument('--res_dir', default=f'{OUT}/res')
	a = ap.parse_args()
	dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)
