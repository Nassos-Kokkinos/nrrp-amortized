"""Gymnasium view of the MDP M(phi), for MaskablePPO (sb3-contrib).

Needs the ``rl`` extra (``pip install -e ".[rl]"``).

One environment holds a *pool* of instances of the same size (n employees, k
shift types), so that the observation has one dimension, and draws an instance
uniformly at every reset: the policy is trained over a distribution of
contexts, which is what makes the optimisation amortized. A separate pool of
held-out instances measures generalisation.

* observation: the vector of :class:`nrrp.env.RosterEnv` (Table 2), float32;
* action: Discrete(n + 1), index n is the empty action;
* ``action_masks()``: the mask (32), as MaskablePPO expects;
* reward: ``reward_scale * (-c_t)``. With the default ``include_constants=False``
  (Remark 7) the context constant c^con is not charged; it cannot change which
  roster is best. The last step's ``info`` always reports the full J (and the
  evaluation under ``eval_profile``), never a scaled or shifted value.
"""
from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np

from .env import RosterEnv
from .instance import Context, Instance


class GymRosterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instances: Sequence[Instance], profile, *,
                 contexts: Callable[[np.random.Generator, Instance], Context] | None = None,
                 kappa: int = 1, eval_profile=None, reward_scale: float = 1.0,
                 include_constants: bool = False):
        """``contexts``: None for construction (d_xi = 0, no disruption), or a
        callable (rng, instance) -> Context drawing a re-rostering context."""
        super().__init__()
        instances = list(instances)
        if not instances:
            raise ValueError("the instance pool is empty")
        n, k = instances[0].n_emp, instances[0].n_shifts
        odd = [i.name for i in instances if (i.n_emp, i.n_shifts) != (n, k)]
        if odd:
            raise ValueError(f"all instances in a pool need {n} employees and {k} shift types; not: {odd}")
        if not reward_scale > 0:
            raise ValueError("reward_scale must be positive")
        self.instances, self.profile, self.eval_profile = instances, profile, eval_profile
        self.kappa, self.reward_scale = int(kappa), float(reward_scale)
        self.include_constants = bool(include_constants)
        self.contexts = contexts
        self.n_emp, self.n_shifts = n, k
        self._envs: dict = {}
        self.env: RosterEnv | None = None
        self.instance: Instance | None = None
        dim = self._roster_env(0).obs_dim
        # features are clipped to [-1, 1] (Table 2); see RosterEnv.FEATURES
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n + 1)

    def _roster_env(self, idx: int) -> RosterEnv:
        if idx not in self._envs:
            inst = self.instances[idx]
            ctx = None
            if self.contexts is not None:
                draw = self.contexts
                ctx = (lambda rng, inst=inst: draw(rng, inst))
            self._envs[idx] = RosterEnv(inst, self.profile, contexts=ctx, kappa=self.kappa,
                                        include_constants=self.include_constants,
                                        eval_profile=self.eval_profile)
        return self._envs[idx]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        idx = int(options["instance"]) if options and "instance" in options else \
            int(self.np_random.integers(len(self.instances)))
        self.env = self._roster_env(idx)
        self.instance = self.instances[idx]
        obs = self.env.reset(seed=int(self.np_random.integers(2**31 - 1)))
        return obs, {"instance": self.instance.name, "context": self.env.ctx.name}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        info = dict(info)
        if done:
            info["instance"] = self.instance.name
        return obs, self.reward_scale * float(reward), bool(done), False, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_mask()

    @property
    def roster(self) -> np.ndarray:
        return self.env.roster.copy()
