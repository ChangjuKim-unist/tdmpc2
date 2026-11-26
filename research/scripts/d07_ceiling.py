"""Design 07 §9 (research/docs/design07_second_round_collector_ko.md): ceiling measurement, no verdict.

Continue the official online TD-MPC2 training (encoder, dynamics, reward, Q, policy) from the pretrained checkpoint under the
shifted physics. The result is a LOWER bound of the return that can be reached under the shifted condition.

Differences from the official trainer (trainer/online_trainer.py), both to protect the pretrained weights:
  - the first --seed_steps environment steps are collected by the pretrained planner (not by random actions), with no update;
  - there is no burst of seed_steps updates at the end of seeding; after seeding it is one update per environment step.
Not restored from the checkpoint (the checkpoint stores model weights only): optimiser states and the running Q scale.

  GPU=N research/scripts/td.sh research/scripts/d07_ceiling.py --cell quadruped-run__motor_0.35
Smoke tests must pass --out_root (another folder) and may shorten with --steps / --seed_steps / --eval_freq / --eval_eps.
The official TD-MPC2 code is imported, not modified.
"""
import argparse
import os
import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from common.buffer import Buffer

from d01 import Shifter, cell_parts, domain_of, dump, make, phys_of, reseed_env, run_episode, summary

OUT = '/mnt/newssd/cjkim/td/d07/ceiling'


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


def main(a):
	task, cond = cell_parts(a.cell)
	folder = f'{a.out_root}/{a.cell}'
	os.makedirs(folder, exist_ok=True)
	cfg, env, agent = make(task, compile_=True, seed=a.seed)
	cfg.steps = a.steps + a.seed_steps
	cfg.buffer_size = cfg.steps + 1000
	buffer = Buffer(cfg)
	domain = domain_of(task)
	Shifter(phys_of(env)).set(cond)
	log = dict(cell=a.cell, args=vars(a), evals=[], train=[])
	step, t0, last = 0, time.time(), {}

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
		obs, done, tds, R = env.reset(), False, None, 0.0
		tds = [to_td(env, obs)]
		while not done:
			action = agent.act(obs, t0=len(tds) == 1, eval_mode=False)
			obs, reward, done, info = env.step(action)
			tds.append(to_td(env, obs, action, reward))
			R += float(reward)
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


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--cell', required=True)
	ap.add_argument('--steps', type=int, default=200_000)
	ap.add_argument('--seed_steps', type=int, default=5_000)
	ap.add_argument('--eval_freq', type=int, default=20_000)
	ap.add_argument('--eval_eps', type=int, default=10)
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--out_root', default=OUT)
	main(ap.parse_args())
