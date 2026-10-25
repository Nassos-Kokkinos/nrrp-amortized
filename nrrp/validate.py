"""Theory validator: random masked rollouts checked against from-scratch definitions.

At every decision point of an episode (``per_step=True``):

* state     the environment's incremental state equals RosterState(ctx, roster, d);
* mask      the mask equals its definition (32): employee free on day d and, for
            every constraint in H^pre, the row with (d, s) assigned still
            satisfies it (sound and complete, Lemma 1 / Prop. 3);
* features  the observation equals the one recomputed from the roster (Table 2);
* reward    -r_t = J(x^(t+1)) - J(x^(t)), with J from scratch, plus c^con on the
            last step when constants are included (Prop. 2, step by step).

At the end of every episode:

* Prop. 2   sum_t r_t + J(x^(T)) = 0 with c^con included, = c^con without it;
* Prop. 3   the final roster violates no constraint of H^pre;
* info      the J the environment reports equals J(x^(T)) from scratch.

Also reported, not checked: the share of employees whose final row violates a
row constraint of H^ter, i.e. work left for the repair operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import CONSTRAINTS, RosterState
from .env import RosterEnv
from .generator import random_context
from .instance import OFF
from .objective import J, hard_violations
from .profile import BURKE_CURTOIS, ConstraintProfile, burke_curtois_training

REL_TOL = 1e-9   # |gap| <= REL_TOL * max(1, |J|): exact with integer weights, rounding otherwise
FEATURE_TOL = 1e-5

# Profiles that exist to exercise the code. The weights are arbitrary and model nothing.
TEST_PROFILES = {
    "burke_curtois": BURKE_CURTOIS,
    "bc_training": burke_curtois_training(w_min_minutes=0.1, w_min_consecutive=20.0),
    "all_soft": BURKE_CURTOIS.with_penalised(
        {"shift_rotation": 10.0, "max_shifts_per_type": 5.0, "max_total_minutes": 0.1,
         "min_total_minutes": 0.1, "max_consecutive_shifts": 10.0, "min_consecutive_shifts": 10.0,
         "min_consecutive_days_off": 10.0, "max_weekends": 20.0, "days_off": 50.0, "skills": 50.0,
         "min_coverage": 30.0}, name="all_soft"),
    "cover_hard": ConstraintProfile("cover_hard", hard=tuple(CONSTRAINTS)),
}


@dataclass
class Report:
    label: str
    per_step: bool
    episodes: int = 0
    steps: int = 0
    total_gap: float = 0.0          # worst |sum r + J - target| / max(1, |J|)
    step_gap: float = 0.0           # worst per-step |-r - dJ| / max(1, |J|)
    info_gap: float = 0.0           # worst |info J - J| / max(1, |J|)
    pre_violation: float = 0.0      # Prop. 3: total violation of H^pre in final rosters
    mask_mismatches: int = 0
    state_mismatches: int = 0
    feature_error: float = 0.0
    ter_rows: int = 0
    rows: int = 0

    @property
    def ok(self) -> bool:
        return (self.total_gap <= REL_TOL and self.step_gap <= REL_TOL and self.info_gap <= REL_TOL
                and self.pre_violation == 0 and self.mask_mismatches == 0
                and self.state_mismatches == 0 and self.feature_error <= FEATURE_TOL)

    def line(self) -> str:
        ter = f"{self.ter_rows / self.rows:6.1%}" if self.rows else "     -"
        step = f"{self.step_gap:8.1e}" if self.per_step else "    (off)"
        return (f"{self.label:<52} {self.episodes:>3} {self.steps:>8,} {self.total_gap:8.1e} {step} "
                f"{self.pre_violation:>5g} {self.mask_mismatches:>5} {self.state_mismatches:>5} "
                f"{self.feature_error:8.1e} {ter}  {'PASS' if self.ok else 'FAIL'}")

    HEADER = (f"{'instance / profile / regime':<52} {'ep':>3} {'steps':>8} {'Prop2 Σ':>8} {'Prop2 t':>8} "
              f"{'Prop3':>5} {'mask':>5} {'state':>5} {'features':>8} {'H^ter':>6}  result")


# ------------------------------------------------------------- from-scratch versions

def mask_from_scratch(env: RosterEnv) -> np.ndarray:
    d, s = env.cd, env.cs
    m = np.zeros(env.I + 1, bool)
    m[env.I] = True
    for i in range(env.I):
        if env.roster[i, d] != OFF:
            continue
        row = env.roster[i].copy()
        row[d] = s
        m[i] = all(c.violation_row(i, row, env.ctx) == 0 for c in env.profile.pre)
    return m


def state_mismatches(env: RosterEnv) -> int:
    st = RosterState(env.ctx, env.roster, env.cd)
    bad = sum(int(not np.array_equal(getattr(env, f), getattr(st, f)))
              for f in ("prev", "count", "minutes", "run_work", "run_off", "weekends", "last_wk"))
    return bad + int(not np.array_equal(env.assigned, env.roster[:, env.cd] != OFF))


def features_from_scratch(env: RosterEnv) -> np.ndarray:
    """The observation of Table 2, recomputed from the roster and the slot position."""
    inst, ctx, roster = env.inst, env.ctx, env.roster
    I, D = env.I, env.D
    d, s, k, _f, t = env.position
    st = RosterState(ctx, roster, d)
    E = np.zeros_like(env.E, dtype=float)
    G = np.zeros_like(env.G, dtype=float)
    before = roster[:, :d]
    mins = np.where(before != OFF, inst.shift_len[np.where(before != OFF, before, 0)], 0).sum(axis=1)
    hmax, hmin = np.maximum(inst.max_minutes, 1), np.maximum(inst.min_minutes, 1)
    w_req = max(inst.on_req.max(), inst.off_req.max(), 1.0)
    mask = mask_from_scratch(env)
    E[:, env.P] = (inst.off_req[:, d, s] - inst.on_req[:, d, s]) / w_req
    E[:, env.X0] = ctx.published[:, d] == s
    E[:, env.MINUTES] = np.minimum(mins / hmax, 1.0)
    E[:, env.SHORTFALL] = np.maximum(inst.min_minutes - mins, 0) / hmin
    E[:, env.SLACK] = np.clip((inst.max_minutes - mins) / hmax, -1.0, 1.0)
    E[:, env.RUN_WORK] = np.minimum(st.run_work / np.maximum(inst.max_consec, 1), 1.0)
    E[:, env.REST] = np.minimum(st.run_off / np.maximum(inst.min_consec_off, 1), 1.0)
    E[:, env.COUNT_S] = np.minimum(st.count[:, s] / np.maximum(inst.max_shifts[:, s], 1), 1.0)
    E[:, env.WEEKENDS] = np.minimum(st.weekends / np.maximum(inst.max_weekends, 1), 1.0)
    E[:, env.ALLOWED] = mask[:I]
    if d > 0:
        worked = roster[:, d - 1] != OFF
        E[np.flatnonzero(worked), env.PREV + roster[worked, d - 1]] = 1.0
    rho_bar = np.minimum(inst.rho + env.kappa, I)
    T = int(rho_bar[ctx.d_xi:].sum())
    f = int(np.count_nonzero(roster[:, d] == s))
    G[env.G_PROGRESS] = t / T
    G[env.G_DAY] = (d + 1) / D
    G[env.G_FILL] = min(f / max(inst.rho[d, s], 1), 1.0)
    G[env.G_SLOTS_LEFT] = (rho_bar[d, s] - k) / rho_bar[d, s]
    G[env.G_ADMISSIBLE] = mask.sum() / (I + 1)
    G[env.G_OVERCOVER] = float(f >= inst.rho[d, s])
    G[env.G_SHIFT + s] = 1.0
    return np.concatenate([E.ravel(), G])


# ----------------------------------------------------------------------- episodes

def run_episode(env: RosterEnv, rng: np.random.Generator, report: Report) -> None:
    env.reset()
    ctx, profile = env.ctx, env.profile
    total, j_prev = 0.0, (J(env.roster, ctx, profile) if report.per_step else None)
    while True:
        if report.per_step:
            report.state_mismatches += state_mismatches(env)
            report.mask_mismatches += int(np.count_nonzero(env.action_mask() != mask_from_scratch(env)))
            report.feature_error = max(report.feature_error,
                                       float(np.abs(env.obs - features_from_scratch(env)).max()))
        allowed = np.flatnonzero(env.action_mask())
        _, r, done, info = env.step(int(allowed[rng.integers(len(allowed))]))
        total += r
        report.steps += 1
        if report.per_step:
            j_now = J(env.roster, ctx, profile)
            want = j_now - j_prev + (env.c_con if done and env.include_constants else 0.0)
            report.step_gap = max(report.step_gap, abs(-r - want) / max(1.0, abs(j_now)))
            j_prev = j_now
        if done:
            break
    j = J(env.roster, ctx, profile)
    scale = max(1.0, abs(j))
    target = 0.0 if env.include_constants else env.c_con
    report.total_gap = max(report.total_gap, abs(total + j - target) / scale)
    report.info_gap = max(report.info_gap, abs(info["J"] - j) / scale)
    report.pre_violation += float(sum(v.sum() for v in hard_violations(env.roster, ctx, profile, "pre").values()))
    rows_ter = [v for name, v in hard_violations(env.roster, ctx, profile, "ter").items()
                if CONSTRAINTS[name].scope == "row"]
    if rows_ter:
        report.ter_rows += int(np.count_nonzero(np.sum(rows_ter, axis=0)))
    report.rows += env.I
    report.episodes += 1


def smoke(instance, profile, *, regime: str = "construction", episodes: int = 10, kappa: int = 1,
          mu: float = 1.0, per_step: bool = True, seed: int = 0, label: str | None = None) -> Report:
    """Random masked rollouts; include_constants alternates between episodes."""
    if regime == "construction":
        contexts = None
    elif regime == "rerostering":
        def contexts(rng):
            return random_context(instance, profile, rng, mu=mu, kappa=kappa)
    else:
        raise ValueError("regime must be 'construction' or 'rerostering'")
    env = RosterEnv(instance, profile, contexts=contexts, kappa=kappa, seed=seed)
    rng = np.random.default_rng(seed + 1)
    report = Report(label or f"{instance.name} / {profile.name} / {regime}", per_step)
    for ep in range(episodes):
        env.include_constants = ep % 2 == 0
        run_episode(env, rng, report)
    return report
