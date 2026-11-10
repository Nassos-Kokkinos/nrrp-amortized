"""Learned large-neighbourhood search for re-rostering (decision of 10/11/2026).

The constructive MDP of Section 3.5 asks a policy for some 300 decisions per
disruption, and trained with PPO it never improved on the published roster x°
(Instance 5 and 8 of Burke & Curtois, November 2026). Here the policy takes K
large decisions instead: which employees to release for CP-SAT.

Episode, for a context phi = (Inst, x°, xi):

* x_0 = x° without the assignments of the absent employees (the classical start);
* at step t = 0..K-1 the policy selects a subset S_t of the employees (one
  Bernoulli per employee, an action in {0, 1}^n); the rows of S_t, together with
  the rows of every employee whose row violates a hard constraint in x_t, are
  freed on the days d >= d_xi, the rest is fixed to x_t, and CP-SAT solves (20)
  on that neighbourhood within ``step_time`` seconds, hinted with x_t;
* x_{t+1} is the solution if it improves on x_t, else x_t (the incumbent is kept);
* reward r_t = Phi(x_t) - Phi(x_{t+1}) >= 0, with Phi = J + ``violation_weight``
  times the total hard violation, divided by |Phi(x_0)|, so that the return
  telescopes to (Phi(x_0) - Phi(x_K)) / |Phi(x_0)|: maximising it is minimising
  the final objective (the argument of Prop. 2).

Every x_t satisfies the history constraint and the absences; every subproblem
imposes all hard constraints, so the policy can only change the quality of the
roster, never break a constraint that x_t meets.

Observation: one row per employee (FEATURES) and a global vector (GLOBAL),
clipped to [-1, 1]; the policy (nrrp.policy.LNSPolicy) is the same shared
per-employee MLP as Eq. (39), so it is equivariant in the employees.
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from .cpsat import RosterModel
from .disruption import DisruptionModel, changes, without_absences
from .instance import OFF, Context, roster_to_binary
from .objective import evaluate
from .profile import BURKE_CURTOIS


class LNSEnv(gym.Env):
    metadata = {"render_modes": []}

    FEATURES = ("absent", "requests", "can_help", "can_leave", "changes", "slack", "shortfall",
                "violates", "chosen_last", "chosen_share")
    GLOBAL = ("progress", "day", "under_cover", "improvement", "absent_share")
    (ABSENT, REQUESTS, CAN_HELP, CAN_LEAVE, CHANGES, SLACK, SHORTFALL,
     VIOLATES, CHOSEN_LAST, CHOSEN_SHARE) = range(10)
    G_PROGRESS, G_DAY, G_UNDER, G_IMPROVEMENT, G_ABSENT = range(5)

    def __init__(self, contexts, profile=BURKE_CURTOIS, *, steps: int = 8, step_time: float = 0.05,
                 workers: int = 1, seed: int = 0, violation_weight: float = 1000.0,
                 deterministic_time: float | None = None):
        """``contexts``: a DisruptionModel (a fresh context every episode) or a list of
        Contexts of one instance (fixed; ``reset(options={"instance": j})`` picks one).
        ``step_time``: CP-SAT wall-clock seconds per step; ``deterministic_time`` replaces
        it by CP-SAT's deterministic time (reproducible, for tests)."""
        super().__init__()
        if isinstance(contexts, DisruptionModel):
            self.model_P, self.fixed, inst = contexts, None, contexts.instance
        else:
            self.model_P, self.fixed = None, list(contexts)
            if not self.fixed:
                raise ValueError("no contexts")
            inst = self.fixed[0].instance
            if any(c.instance is not inst for c in self.fixed):
                raise ValueError("all contexts must be of one instance")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.inst, self.profile = inst, profile
        self.K, self.step_time, self.workers, self.seed_cp = int(steps), float(step_time), int(workers), int(seed)
        self.violation_weight, self.det_time = float(violation_weight), deterministic_time
        n = inst.n_emp
        self.n_emp, self.FE, self.FG = n, len(self.FEATURES), len(self.GLOBAL)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(n * self.FE + self.FG,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(n)
        self.w_req = max(float(inst.on_req.max()), float(inst.off_req.max()), 1.0)
        self.w_cov = max(float(inst.w_under.max()), 1.0)
        self.ctx = None

    # ------------------------------------------------------------------ helpers
    def potential(self, roster: np.ndarray) -> tuple:
        """(Phi, J, feasible, total hard violation) of a roster in the current context."""
        ev = evaluate(roster, self.ctx, self.profile)
        v = float(sum(ev["violations"].values()))
        return ev["J"] + self.violation_weight * v, ev["J"], ev["feasible"], v

    def violators(self, roster: np.ndarray) -> np.ndarray:
        """(n,) employees whose row violates some hard row constraint."""
        bad = np.zeros(self.n_emp, bool)
        for c in self.profile.pre + self.profile.ter:
            if c.scope == "row":
                bad |= c.violation(roster, self.ctx) > 0
        return bad

    def _observe(self) -> np.ndarray:
        inst, ctx, x, n = self.inst, self.ctx, self.x, self.n_emp
        D, S, d0 = inst.n_days, inst.n_shifts, ctx.d_xi
        xb = roster_to_binary(x, S)
        after = np.zeros(D, bool)
        after[d0:] = True
        staff = xb.sum(axis=0)                                           # (m, k)
        under = np.maximum(inst.rho - staff, 0) * inst.w_under * after[:, None]
        over = np.maximum(staff - inst.rho, 0) * inst.w_over * after[:, None]
        works = (x != OFF)
        free_day = ~works & ~ctx.absent & after[None, :]
        E = np.zeros((n, self.FE), np.float64)
        E[:, self.ABSENT] = ctx.absent.sum(axis=1) / D
        req = (inst.on_req * ~xb + inst.off_req * xb)[:, d0:, :].sum(axis=(1, 2))
        E[:, self.REQUESTS] = req / (self.w_req * max(D - d0, 1))
        E[:, self.CAN_HELP] = (free_day.astype(float) @ under.sum(axis=1)) / (self.w_cov * max(D - d0, 1))
        E[:, self.CAN_LEAVE] = (xb * over[None]).sum(axis=(1, 2)) / (self.w_cov * max(D - d0, 1))
        E[:, self.CHANGES] = (xb != roster_to_binary(ctx.published, S)).sum(axis=(1, 2)) / max(D - d0, 1)
        minutes = np.where(works, inst.shift_len[np.where(works, x, 0)], 0).sum(axis=1)
        E[:, self.SLACK] = (inst.max_minutes - minutes) / np.maximum(inst.max_minutes, 1)
        E[:, self.SHORTFALL] = np.maximum(ctx.min_minutes - minutes, 0) / np.maximum(ctx.min_minutes, 1)
        E[:, self.VIOLATES] = self.violators(x)
        E[:, self.CHOSEN_LAST] = self.last
        E[:, self.CHOSEN_SHARE] = self.count / self.K
        G = np.zeros(self.FG, np.float64)
        G[self.G_PROGRESS] = self.t / self.K
        G[self.G_DAY] = d0 / D
        G[self.G_UNDER] = under.sum() / (self.w_cov * n)
        G[self.G_IMPROVEMENT] = (self.phi0 - self.phi) / max(abs(self.phi0), 1.0)
        G[self.G_ABSENT] = ctx.absent.any(axis=1).mean()
        return np.clip(np.concatenate([E.ravel(), G]), -1.0, 1.0).astype(np.float32)

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.fixed is not None:
            j = int(options["instance"]) if options and "instance" in options else \
                int(self.np_random.integers(len(self.fixed)))
            self.ctx = self.fixed[j]
        else:
            self.ctx = self.model_P.sample(self.np_random)
        self.x = without_absences(self.ctx)
        self.phi0, self.J0, self.feasible0, _ = self.potential(self.x)
        self.phi, self.J, self.feasible = self.phi0, self.J0, self.feasible0
        self.t, self.last, self.count = 0, np.zeros(self.n_emp), np.zeros(self.n_emp)
        self.cp_seconds, self.statuses, self.freed = 0.0, [], []
        return self._observe(), {"context": self.ctx.name, "J": self.J, "feasible": self.feasible}

    def step(self, action):
        a = np.asarray(action, dtype=bool).reshape(self.n_emp)
        rows = a | self.violators(self.x)
        phi_before = self.phi
        t0 = time.perf_counter()
        status = "SKIPPED"
        if rows.any():
            free = np.zeros((self.n_emp, self.inst.n_days), bool)
            free[rows, self.ctx.d_xi:] = True
            model = RosterModel(self.ctx, self.profile, free=free, base=self.x)
            left = self.step_time - (time.perf_counter() - t0)      # building counts against the step
            if left > 0 or self.det_time is not None:
                sol = model.solve(time_limit=None if self.det_time is not None else left,
                                  workers=self.workers, seed=self.seed_cp, hint=self.x,
                                  deterministic_limit=self.det_time)
                status = sol.status
                if sol.has_solution:
                    phi, J, feas, _ = self.potential(sol.roster)
                    if phi < self.phi - 1e-9:
                        self.x, self.phi, self.J, self.feasible = sol.roster, phi, J, feas
        self.cp_seconds += time.perf_counter() - t0
        self.statuses.append(status)
        self.freed.append(int(rows.sum()))
        self.last, self.count = a.astype(float), self.count + a
        self.t += 1
        done = self.t >= self.K
        reward = (phi_before - self.phi) / max(abs(self.phi0), 1.0)    # >= 0, telescopes to Phi(x_0) - Phi(x_K)
        info = {"J": self.J, "feasible": self.feasible, "status": status, "freed": int(rows.sum())}
        if done:
            info.update({"J0": self.J0, "feasible0": self.feasible0, "changes": changes(self.x, self.ctx),
                         "cp_seconds": self.cp_seconds, "statuses": list(self.statuses), "freed_per_step": list(self.freed)})
        return self._observe(), float(reward), done, False, info

    @property
    def roster(self) -> np.ndarray:
        return self.x.copy()


# ------------------------------------------------------------------ running an episode

def top_q(logits: np.ndarray) -> np.ndarray:
    """The deterministic release decision of a policy: the q employees with the highest
    log-odds, q = round(sum of the release probabilities), at least one."""
    logits = np.asarray(logits, dtype=float)
    q = max(1, int(round(float((1.0 / (1.0 + np.exp(-logits))).sum()))))
    a = np.zeros(len(logits), np.int8)
    a[np.argsort(-logits, kind="stable")[:q]] = 1
    return a


def random_q(n: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """q employees released uniformly at random (the random-LNS baseline)."""
    a = np.zeros(n, np.int8)
    a[rng.choice(n, size=min(max(q, 1), n), replace=False)] = 1
    return a


def run_episode(env: LNSEnv, idx: int, choose) -> tuple:
    """K steps on the fixed context ``idx`` with ``choose(obs, env) -> action``.
    Returns (roster, info); info["seconds"] is the wall-clock time of the whole episode,
    the policy's decisions included (T_infer)."""
    obs, _ = env.reset(seed=0, options={"instance": idx})
    t0 = time.perf_counter()
    done, info = False, {}
    while not done:
        obs, _, done, _, info = env.step(choose(obs, env))
    info["seconds"] = time.perf_counter() - t0
    return env.roster, info
