"""The constructive MDP of Section 3.5.

Slots
    Day-major over the decidable days d >= d_xi; within a day, shift types in
    index order; for each (d, s), rho_bar_{ds} = min(rho_{ds} + kappa, I) slots.
    The order matters: prefix-monotonicity (Lemma 1) is defined with respect to
    it.

Actions
    a in 0..I-1 assigns employee a to (d, s); a = I leaves the slot empty.

Mask (Prop. 3)
    a < I is allowed iff the employee is not already assigned on day d and
    delta_c(a) = 0 for every hard prefix-monotone constraint c of the profile.
    The empty action is always allowed, so no state is a dead end.

Reward
    The negative change of J on partial rosters (see :mod:`nrrp.objective`):
    -(lin + cover increment + penalty increment) for an assignment, 0 for an
    empty slot. With ``include_constants=True`` the terminal step also charges
    K(phi) and the rewards sum to -J(x) (Prop. 2). With False, the training
    setting of section 1c of the brief, they sum to K(phi) - J(x); the info dict
    of the last step reports J either way.

State and features
    An assignment (i, d, s) changes row i only, so per step only row i of the
    feature matrix is rewritten. Columns that depend on (d, s) are refreshed when
    a new pair starts, and day-dependent ones when a new day starts. The
    incremental state carries the attributes of
    :class:`nrrp.constraints.RosterState`; the smoke test checks it against a
    from-scratch recomputation at every step.

The core is plain numpy. A Gymnasium wrapper comes with the PPO step.
"""
from __future__ import annotations

import numpy as np

from .constraints import extended_forbidden
from .instance import OFF, Context, Instance, roster_to_binary
from .objective import context_constant, cover_increments, hard_violations, initial_roster, linear_costs


class RosterEnv:
    # per-employee feature columns
    (AVAIL, ALLOWED, TODAY, ON, OFFREQ, X0, MINUTES, DEFICIT,
     RUN_WORK, RUN_OFF, WEEKENDS, COUNT_S, PEN) = range(13)
    PREV = 13            # one-hot of the shift worked yesterday, width S
    # global feature columns
    G_DAY, G_DXI, G_MU, G_RHO, G_SLOT, G_FILL, G_LEFT = range(7)
    G_SHIFT = 7          # one-hot of the current shift type, width S
    RUN_OFF_CAP = 7

    FEATURES = {
        AVAIL: "1 if no fixed day off and no absence on d",
        ALLOWED: "the mask for the current slot",
        TODAY: "1 if already assigned on d",
        ON: "on-request weight for (d, s) / largest request weight",
        OFFREQ: "off-request weight for (d, s) / largest request weight",
        X0: "x°_{i,d,s}, the published assignment",
        MINUTES: "minutes so far / max minutes",
        DEFICIT: "max(min minutes - minutes so far, 0) / max minutes",
        RUN_WORK: "working days ending at d-1 / max consecutive shifts",
        RUN_OFF: "days off ending at d-1, capped at 7, / 7",
        WEEKENDS: "worked weekends so far / max weekends",
        COUNT_S: "shifts of type s so far / max shifts of type s",
        PEN: "penalty increment of the assignment / largest cover weight (0 if assigned today)",
    }

    def __init__(self, instance: Instance, profile, contexts=None, kappa: int = 1,
                 include_constants: bool = False, seed=None, check_history: bool = True):
        """``contexts``: None (construction every episode), a fixed Context, or a
        callable rng -> Context drawing one per episode (the P of Eq. (23))."""
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        inst = instance
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        self.inst, self.profile, self.kappa = inst, profile, int(kappa)
        self.include_constants, self.check_history = bool(include_constants), bool(check_history)
        self.I, self.D, self.S = I, D, S
        self.n_actions = I + 1
        self.rng = np.random.default_rng(seed)
        if contexts is None:
            fixed = Context(inst)
            self._draw = lambda rng: fixed
        elif isinstance(contexts, Context):
            self._draw = lambda rng: contexts
        elif callable(contexts):
            self._draw = contexts
        else:
            raise TypeError("contexts must be None, a Context or a callable rng -> Context")
        self.masked = profile.masked
        self.penalties = profile.penalties

        self.rows = np.arange(I)
        self.forbidden_ext = extended_forbidden(inst.forbidden)
        self.eye_ext = np.vstack([np.eye(S, dtype=np.float32), np.zeros((1, S), np.float32)])
        w_req = max(float(inst.on_req.max()), float(inst.off_req.max()), 1.0)
        self.on_f = (inst.on_req / w_req).astype(np.float32)
        self.off_f = (inst.off_req / w_req).astype(np.float32)
        self.inv_max_minutes = 1.0 / np.maximum(inst.max_minutes, 1)
        self.inv_max_consec = 1.0 / np.maximum(inst.max_consec, 1)
        self.inv_max_weekends = 1.0 / np.maximum(inst.max_weekends, 1)
        self.inv_max_shifts = 1.0 / np.maximum(inst.max_shifts, 1)
        self.pen_scale = 1.0 / max(float(inst.w_under.max()), float(inst.w_over.max()), 1.0)

        FE, FG = self.PREV + S, self.G_SHIFT + S
        self.obs = np.zeros(I * FE + FG, np.float32)
        self.E = self.obs[: I * FE].reshape(I, FE)   # views into obs
        self.G = self.obs[I * FE:]
        self.obs_dim = self.obs.size
        self.ctx = None
        self.done = True

    # ------------------------------------------------------------------ context
    def _setup_context(self, ctx: Context) -> None:
        if ctx.instance is not self.inst:
            raise ValueError(f"context {ctx.name} belongs to another instance")
        I, S = self.I, self.S
        rho_bar = np.minimum(ctx.rho + self.kappa, I)
        dd, ss = np.nonzero(rho_bar[ctx.d_xi:] > 0)     # row-major: day-major, then shift
        self.pair_d = dd + ctx.d_xi
        self.pair_s = ss
        self.pair_rb = rho_bar[self.pair_d, self.pair_s]
        self.T = int(self.pair_rb.sum())
        if self.T == 0:
            raise ValueError(f"context {ctx.name} has no slots (rho = 0 everywhere and kappa = 0)")
        self.cov_inc = [cover_increments(ctx, d, s, rb)
                        for d, s, rb in zip(self.pair_d, self.pair_s, self.pair_rb)]
        self.lin = linear_costs(ctx)
        self.K = context_constant(ctx, self.profile)
        self.y_init = initial_roster(ctx)
        self.unavail = self.inst.day_off | ctx.absent
        self.x0_f = roster_to_binary(ctx.y0, S).astype(np.float32)
        self.inv_rho = 1.0 / np.maximum(ctx.rho, 1)
        self.ctx = ctx

    # -------------------------------------------------------------------- API
    def reset(self, seed=None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ctx = self._draw(self.rng)
        if ctx is not self.ctx:
            self._setup_context(ctx)
        self._init_state()
        return self.obs.copy()

    def action_mask(self) -> np.ndarray:
        return self.mask.copy()

    def step(self, a):
        if self.done:
            raise RuntimeError("the episode is over; call reset()")
        a = int(a)
        if not 0 <= a <= self.I or not self.mask[a]:
            raise ValueError(f"action {a} is masked")
        cost = 0.0
        if a < self.I:
            d, s, inst = self.cd, self.cs, self.inst
            cost = float(self.lin[a, d, s] + self.pen[a] + self.cov[self.f])
            self.y[a, d] = s
            self.minutes[a] += inst.shift_len[s]
            self.count[a, s] += 1
            wk = inst.weekend_of_day[d]
            if wk >= 0 and self.last_wk[a] != wk:
                self.weekends[a] += 1
                self.last_wk[a] = wk
            self.assigned[a] = True
            self.mask[a] = False
            self.f += 1
            E = self.E
            E[a, self.ALLOWED] = 0.0
            E[a, self.TODAY] = 1.0
            E[a, self.PEN] = 0.0
            E[a, self.MINUTES] = self.minutes[a] * self.inv_max_minutes[a]
            E[a, self.DEFICIT] = max(inst.min_minutes[a] - self.minutes[a], 0) * self.inv_max_minutes[a]
            E[a, self.WEEKENDS] = self.weekends[a] * self.inv_max_weekends[a]
            E[a, self.COUNT_S] = self.count[a, s] * self.inv_max_shifts[a, s]
        self.t += 1
        self.k += 1
        self.J_acc += cost
        info = {}
        if self.t == self.T:
            self.done = True
            info["J"] = self.K + self.J_acc
            if self.include_constants:
                cost += self.K
        elif self.k == self.rb:
            self._enter_pair(self.p + 1)
        else:
            self._slot_features()
        return self.obs.copy(), -cost, self.done, info

    def ter_violations(self) -> dict:
        """H^ter violations of the current roster (what the repair step must fix)."""
        return hard_violations(self.y, self.ctx, self.profile, "ter")

    @property
    def position(self) -> tuple:
        """(day, shift, slot within the pair, employees already on the pair, step)."""
        return self.cd, self.cs, self.k, self.f, self.t

    # --------------------------------------------------------------- internals
    def _init_state(self) -> None:
        inst, ctx, I, S = self.inst, self.ctx, self.I, self.S
        self.y = self.y_init.copy()
        self.prev = np.full(I, OFF, np.int64)
        self.count = np.zeros((I, S), np.int64)
        self.minutes = np.zeros(I, np.int64)
        self.run_work = np.zeros(I, np.int64)
        self.run_off = np.zeros(I, np.int64)
        self.weekends = np.zeros(I, np.int64)
        self.last_wk = np.full(I, -1, np.int64)
        self.assigned = np.zeros(I, bool)
        self.mask = np.ones(I + 1, bool)
        self.d = 0
        for d in range(ctx.d_xi):
            self._replay_day(d)
        E, G = self.E, self.G
        E[:] = 0.0
        G[:] = 0.0
        E[:, self.MINUTES] = self.minutes * self.inv_max_minutes
        E[:, self.DEFICIT] = np.maximum(inst.min_minutes - self.minutes, 0) * self.inv_max_minutes
        E[:, self.WEEKENDS] = self.weekends * self.inv_max_weekends
        G[self.G_DXI] = ctx.d_xi / self.D
        G[self.G_MU] = ctx.mu / (1.0 + ctx.mu)
        self.t, self.J_acc, self.done = 0, 0.0, False
        self._feat_day = -1
        self._enter_pair(0)

    def _replay_day(self, d: int) -> None:
        """Apply frozen day d of the published roster, checking it against the mask."""
        inst = self.inst
        col = self.y[:, d]
        w = col != OFF
        if w.any():
            s_arr = np.where(w, col, 0).astype(np.int64)
            if self.check_history:
                for c in self.masked:
                    bad = w & (c.delta(self, d, s_arr) != 0)
                    if bad.any():
                        who = [inst.employee_ids[i] for i in np.flatnonzero(bad)]
                        raise ValueError(f"{self.ctx.name}: the published roster violates the hard "
                                         f"constraint {c.name} on frozen day {d} (employees {who})")
            self.minutes += np.where(w, inst.shift_len[s_arr], 0)
            self.count[self.rows[w], s_arr[w]] += 1
            wk = inst.weekend_of_day[d]
            if wk >= 0:
                new = w & (self.last_wk != wk)
                self.weekends += new
                self.last_wk[new] = wk
        self._close_day()

    def _close_day(self) -> None:
        col = self.y[:, self.d]
        w = col != OFF
        self.run_work = np.where(w, self.run_work + 1, 0)
        self.run_off = np.where(w, 0, self.run_off + 1)
        self.prev = col.astype(np.int64)
        self.assigned[:] = False
        self.d += 1

    def _enter_pair(self, p: int) -> None:
        d, s = int(self.pair_d[p]), int(self.pair_s[p])
        while self.d < d:
            self._close_day()
        E, G, I = self.E, self.G, self.I
        if d != self._feat_day:
            self._feat_day = d
            E[:, self.AVAIL] = ~self.unavail[:, d]
            E[:, self.TODAY] = 0.0
            E[:, self.RUN_WORK] = self.run_work * self.inv_max_consec
            E[:, self.RUN_OFF] = np.minimum(self.run_off, self.RUN_OFF_CAP) / self.RUN_OFF_CAP
            E[:, self.PREV:] = self.eye_ext[self.prev]
            G[self.G_DAY] = d / self.D
        m = ~self.assigned
        for c in self.masked:
            m &= c.delta(self, d, s) == 0
        pen = np.zeros(I)
        for c, w in self.penalties:
            pen += w * c.delta(self, d, s)
        self.mask[:I] = m
        self.pen = pen
        E[:, self.ALLOWED] = m
        E[:, self.ON] = self.on_f[:, d, s]
        E[:, self.OFFREQ] = self.off_f[:, d, s]
        E[:, self.X0] = self.x0_f[:, d, s]
        E[:, self.COUNT_S] = self.count[:, s] * self.inv_max_shifts[:, s]
        E[:, self.PEN] = np.where(self.assigned, 0.0, pen * self.pen_scale)
        G[self.G_SHIFT:] = 0.0
        G[self.G_SHIFT + s] = 1.0
        G[self.G_RHO] = self.ctx.rho[d, s] / I
        self.p, self.cd, self.cs, self.k, self.f = p, d, s, 0, 0
        self.rb = int(self.pair_rb[p])
        self.cov = self.cov_inc[p]
        self._slot_features()

    def _slot_features(self) -> None:
        G = self.G
        G[self.G_SLOT] = self.k / self.rb
        G[self.G_FILL] = self.f * self.inv_rho[self.cd, self.cs]
        G[self.G_LEFT] = (self.T - self.t) / self.T
