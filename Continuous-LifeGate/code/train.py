"""PPO in Continuous LifeGate under three reward designs.

PPO (Stable-Baselines3): learning rate 3e-4, 64 environments, rollouts of 128 steps, batches of 512, 5 epochs,
GAE lambda 0.95, clip range 0.2, entropy coefficient 0.01, two hidden layers of 64 units for the policy and the
value function, 20,000,000 transitions.  Training seeds 31000-31004 and test seeds 94000-94004 (the same for every
design, so test patients are paired across designs); 5,000 deterministic test episodes per seed.
Without the entropy bonus (--ent 0 --tag _no_entropy), the survival policy does not learn the detour around the wall.

Designs:  survival  +1 exit, 0 death
          death      0 exit, -1 death
          stepcost  +1 exit, 0 death, -0.04 per non-terminal step (the step cost of the LifeGate study)
Observation: (x / 10, y / 10, remaining time).  Initial positions: uniform on the free space.
Output: ../results<tag>/<design>/fold<k>/{model.zip, evaluation.npz, result.json}

Usage:  python train.py --design survival --fold 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

import lifegate_continuous as lifegate

DESIGNS = {'survival': (1., 0., 0.), 'death': (0., -1., 0.), 'stepcost': (1., 0., -0.04)}  # (exit, death, step)
SEEDS = list(range(31000, 31005))            # training seed of each fold
TEST_SEEDS = list(range(94000, 94005))       # test seed of each fold
BUDGET = 20_000_000                          # transitions
ENT = 0.01                                   # entropy coefficient
N_ENVS = 64
N_TEST = 5000                                # test episodes per seed
N_PATHS = 20                                 # test episodes whose paths are saved
ROOT = Path(__file__).resolve().parents[1]


def observation(xy, ages, cfg):
    """(x / 10, y / 10, remaining fraction of the horizon), float32."""
    return np.c_[xy / 10., 1 - np.asarray(ages) / cfg.horizon].astype(np.float32)


class LifeGateVecEnv(VecEnv):
    """`n` Continuous LifeGate episodes stepped together; finished episodes restart from uniform free positions."""

    render_mode = None          # read through get_attr by VecEnv.__init__

    def __init__(self, n, design, cfg=lifegate.Config(), seed=0):
        self.cfg = cfg
        self.r_exit, self.r_death, self.step_reward = DESIGNS[design]
        self.rng = np.random.default_rng(seed)
        self.xy, self.ages = np.zeros((n, 2)), np.zeros(n, dtype=int)
        self.returns = np.zeros(n)
        self.actions = None
        super().__init__(n, spaces.Box(np.array([0, 0, 0], np.float32), np.array([1, 1, 1], np.float32)),
                         spaces.Box(-1., 1., (2,), dtype=np.float32))

    def reset(self):
        if self._seeds[0] is not None:
            self.rng = np.random.default_rng(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        self.xy = lifegate.initial_states(self.num_envs, self.rng)
        self.ages[:] = 0
        self.returns[:] = 0
        return observation(self.xy, self.ages, self.cfg)

    def step_async(self, actions):
        self.actions = np.asarray(actions).copy()

    def step_wait(self):
        cfg = self.cfg
        self.xy = lifegate.transition(self.xy, self.actions, self.rng.standard_normal((self.num_envs, 2)), cfg)
        self.ages += 1
        ev = lifegate.terminal(self.xy[:, 0], self.xy[:, 1])
        rewards = self.r_exit * (ev == 1) + self.r_death * (ev == -1) + self.step_reward * (ev == 0)
        self.returns += rewards
        done = (ev != 0) | (self.ages >= cfg.horizon)
        obs = observation(self.xy, self.ages, cfg)
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(done):
            # the horizon is part of the task: no bootstrapping beyond it
            infos[i] = {'terminal_observation': obs[i].copy(), 'TimeLimit.truncated': False,
                        'episode': {'r': float(self.returns[i]), 'l': int(self.ages[i])}}
        k = int(done.sum())
        if k:
            self.xy[done] = lifegate.initial_states(k, self.rng)
            self.ages[done] = 0
            self.returns[done] = 0
            obs[done] = observation(self.xy[done], self.ages[done], cfg)
        return obs, rewards.astype(np.float32), done, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def test_starts(n, seed):
    """The test patients of a test seed (the same for every design)."""
    start_seed, _ = np.random.SeedSequence(seed).spawn(2)
    return lifegate.initial_states(n, np.random.default_rng(start_seed))


def evaluate(model, cfg, n, seed):
    """`n` test episodes with deterministic actions; returns outcomes, end steps and the first N_PATHS paths."""
    _, noise_seed = np.random.SeedSequence(seed).spawn(2)
    xy = test_starts(n, seed)
    rng = np.random.default_rng(noise_seed)
    outcome, end = np.zeros(n, np.int8), np.full(n, cfg.horizon, np.int32)
    paths = np.empty((N_PATHS, cfg.horizon + 1, 2), np.float32)
    paths[:, 0] = xy[:N_PATHS]
    with torch.random.fork_rng(devices=[]):          # leaves the global torch generator untouched
        torch.manual_seed(seed + 123456)
        for step in range(cfg.horizon):
            a, _ = model.predict(observation(xy, np.full(n, step), cfg), deterministic=True)
            nxt = lifegate.transition(xy, a, rng.standard_normal((n, 2)), cfg)
            live = outcome == 0
            xy = np.where(live[:, None], nxt, xy)
            ev = lifegate.terminal(xy[:, 0], xy[:, 1])
            hit = live & (ev != 0)
            outcome[hit], end[hit] = ev[hit], step + 1
            paths[:, step + 1] = xy[:N_PATHS]
    return outcome, end, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', choices=list(DESIGNS), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--steps', type=int, default=BUDGET)
    parser.add_argument('--ent', type=float, default=ENT, help='entropy coefficient')
    parser.add_argument('--tag', default='', help='suffix of the output folder')
    args = parser.parse_args()
    cfg, seed = lifegate.Config(), SEEDS[args.fold]
    folder = ROOT / f'results{args.tag}' / args.design / f'fold{args.fold}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    start = time.perf_counter()
    env = LifeGateVecEnv(N_ENVS, args.design, cfg, seed)
    model = PPO('MlpPolicy', env, learning_rate=3e-4, gamma=cfg.gamma, seed=seed, device='cpu', verbose=0,
                n_steps=128, batch_size=512, n_epochs=5, ent_coef=args.ent, gae_lambda=.95, clip_range=.2,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    model.learn(total_timesteps=args.steps)
    model.save(folder / 'model.zip')
    outcome, end, paths = evaluate(model, cfg, N_TEST, TEST_SEEDS[args.fold])
    np.savez_compressed(folder / 'evaluation.npz', outcome=outcome, end=end, paths=paths)
    result = dict(algorithm='ppo', steps=args.steps, ent_coef=args.ent, width=64, design=args.design,
                  reward=DESIGNS[args.design], fold=args.fold, training_seed=seed, test_seed=TEST_SEEDS[args.fold],
                  transitions=model.num_timesteps, elapsed_seconds=time.perf_counter() - start,
                  fractions=dict(exit=float(np.mean(outcome == 1)), death=float(np.mean(outcome == -1)),
                                 unresolved=float(np.mean(outcome == 0))))
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    print('FINISHED', args.design, args.fold, result['fractions'], round(result['elapsed_seconds'] / 60, 1), 'min',
          flush=True)


if __name__ == '__main__':
    main()
