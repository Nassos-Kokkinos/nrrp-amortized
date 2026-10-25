"""The constructive MDP M(phi) of Section 3.5.

Slots (Eq. (27))
    Day-major over the decidable days d >= d_xi; within a day, shift types in
    index order; for each (d, s), rho_bar_{ds} = min(rho_{ds} + kappa, n) slots
    (no more than n employees can work a pair). Prefix-monotonicity (Lemma 1)
    is defined with respect to this order.

Actions (Eq. (28))
    a in 0..n-1 assigns employee a to (d, s); a = n is the empty action.

Mask (Eq. (32), Prop. 3)
    a < n is allowed iff the employee is free on day d (constraint (3)) and
    delta_c(a) = 0 for every c in H^pre of the profile. The empty action is
    always allowed, so no state is a dead end.

Reward
    -c_t with c_t = J(x^(t+1)) - J(x^(t)) on partial rosters (see
    :mod:`nrrp.objective`): preference and instability cost of the assignment,
    the increment of the pair's coverage cost, and the increments of penalised
    constraints; 0 for the empty action. With ``include_constants=True`` the last
    step also charges c^con = J(x^(0)) and the rewards sum to -J(x^(T))
    (Prop. 2). With False, the training setting of Remark 7, they sum to
    c^con - J(x^(T)). The info dict of the last step reports J under the
    environment's profile and, if an evaluation profile is given, the result of
    :func:`nrrp.objective.evaluate` under it.

State encoding (Table 2)
    One row psi_i per employee and a global vector g; see ``FEATURES``. An
    assignment (i, d, s) changes row i only, so a step rewrites one row and a few
    scalars; columns that depend on (d, s) are refreshed when a pair starts and
    day columns when a day starts. The smoke test checks every entry against a
    from-scratch recomputation.

The core is plain numpy. A Gymnasium wrapper comes with the PPO step.
"""
from __future__ import annotations

import numpy as np

from .constraints import extended_forbidden
from .instance import OFF, Context, Instance, roster_to_binary
from .objective import context_constant, evaluate, fill_increments, hard_violations, initial_roster, linear_costs


class RosterEnv:
    # per-employee features psi_i (Table 2)
    P, X0, MINUTES, SHORTFALL, SLACK, RUN_WORK, REST, COUNT_S, WEEKENDS, ALLOWED = range(10)
    PREV = 10            # one-hot of the shift worked yesterday, width k
    # global features g (Table 2)
    G_PROGRESS, G_DAY, G_FILL, G_SLOTS_LEFT, G_ADMISSIBLE, G_OVERCOVER = range(6)
    G_SHIFT = 6          # one-hot of the current shift type, width k

    FEATURES = {
        "P": "p_{i,d,s} / largest request weight, in [-1, 1]",
        "X0": "x°_{i,d,s}: whether the published roster had this assignment",
        "MINUTES": "minutes assigned on days before d / H^max, clipped to [0, 1]",
        "SHORTFALL": "max(H^min - minutes before d, 0) / H^min",
        "SLACK": "(H^max - minutes before d) / H^max, clipped to [-1, 1]",
        "RUN_WORK": "working days ending at d-1 / C^max, clipped to 1",
        "REST": "days off ending at d-1 / R^min, clipped to 1",
        "COUNT_S": "assignments of shift type s so far / N^max_{i,s}, clipped to 1",
        "WEEKENDS": "worked weekends so far / weekend cap, clipped to 1",
        "ALLOWED": "1 if i is in the mask M(s_t)",
        "PREV": "one-hot of the shift worked on d-1 (width k)",
        "G_PROGRESS": "t / T",
        "G_DAY": "(d + 1) / m, the paper's d_t / m with 1-indexed days",
        "G_FILL": "employees already on (d, s) / max(rho_{d,s}, 1)",
        "G_SLOTS_LEFT": "(rho_bar_{d,s} - slots of the pair already decided) / rho_bar_{d,s}",
        "G_ADMISSIBLE": "|M(s_t)| / (n + 1)",
        "G_OVERCOVER": "1 if filling the slot over-covers (f >= rho_{d,s})",
        "G_SHIFT": "one-hot of s (width k)",
    }

    def __init__(self, instance: Instance, profile, contexts=None, kappa: int = 1,
                 include_constants: bool = False, seed=None, check_history: bool = True,
                 eval_profile=None):
        """``contexts``: None (construction every episode), a fixed Context, or a
        callable rng -> Context drawing one per episode (the P of Eq. (23)).
        ``profile`` defines the reward and the mask; ``eval_profile``, if given,
        is used only to evaluate the final roster."""
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        inst = instance
        I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
        self.inst, self.profile, self.eval_profile, self.kappa = inst, profile, eval_profile, int(kappa)
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
        self.pre = profile.pre
        self.row_penalties = tuple((c, w) for c, w in profile.penalties if c.scope == "row")

        self.rows = np.arange(I)
        self.forbidden_ext = extended_forbidden(inst.forbidden)
        self.eye_ext = np.vstack([np.eye(S, dtype=np.float32), np.zeros((1, S), np.float32)])
        w_req = max(float(inst.on_req.max()), float(inst.off_req.max()), 1.0)
        self.p_f = ((inst.off_req - inst.on_req) / w_req).astype(np.float32)
        self.inv_hmax = 1.0 / np.maximum(inst.max_minutes, 1)
        self.inv_hmin = 1.0 / np.maximum(inst.min_minutes, 1)
        self.inv_cmax = 1.0 / np.maximum(inst.max_consec, 1)
        self.inv_rmin = 1.0 / np.maximum(inst.min_consec_off, 1)
        self.inv_nmax = 1.0 / np.maximum(inst.max_shifts, 1)
        self.inv_wk = 1.0 / np.maximum(inst.max_weekends, 1)
        self.inv_rho = 1.0 / np.maximum(inst.rho, 1)

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
        inst, I, S = self.inst, self.I, self.S
        rho_bar = np.minimum(inst.rho + self.kappa, I)
        dd, ss = np.nonzero(rho_bar[ctx.d_xi:] > 0)     # row-major: day-major, then shift
        self.pair_d = dd + ctx.d_xi
        self.pair_s = ss
        self.pair_rb = rho_bar[self.pair_d, self.pair_s]
        self.T = int(self.pair_rb.sum())
        if self.T == 0:
            raise ValueError(f"context {ctx.name} has no slots (rho = 0 everywhere and kappa = 0)")
        self.fill_inc = [fill_increments(ctx, self.profile, d, s, rb)
                         for d, s, rb in zip(self.pair_d, self.pair_s, self.pair_rb)]
        self.lin = linear_costs(ctx)
        self.c_con = context_constant(ctx, self.profile)
        self.roster_init = initial_roster(ctx)
        self.x0_f = roster_to_binary(ctx.published, S).astype(np.float32)
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
            cost = float(self.lin[a, d, s] + self.pen[a] + self.inc[self.f])
            self.roster[a, d] = s
            self.minutes[a] += inst.shift_len[s]
            self.count[a, s] += 1
            wk = inst.weekend_of_day[d]
            if wk >= 0 and self.last_wk[a] != wk:
                self.weekends[a] += 1
                self.last_wk[a] = wk
            self.assigned[a] = True
            self.mask[a] = False
            self.n_allowed -= 1
            self.f += 1
            E = self.E
            E[a, self.ALLOWED] = 0.0
            E[a, self.COUNT_S] = min(self.count[a, s] * self.inv_nmax[a, s], 1.0)
            E[a, self.WEEKENDS] = min(self.weekends[a] * self.inv_wk[a], 1.0)
        self.t += 1
        self.k += 1
        self.J_acc += cost
        info = {}
        if self.t == self.T:
            self.done = True
            info["J"] = self.c_con + self.J_acc
            if self.eval_profile is not None:
                info["evaluation"] = evaluate(self.roster, self.ctx, self.eval_profile)
            if self.include_constants:
                cost += self.c_con
        elif self.k == self.rb:
            self._enter_pair(self.p + 1)
        else:
            self._slot_features()
        return self.obs.copy(), -cost, self.done, info

    def ter_violations(self) -> dict:
        """H^ter violations of the current roster under the environment's profile."""
        return hard_violations(self.roster, self.ctx, self.profile, "ter")

    @property
    def position(self) -> tuple:
        """(day, shift, slots of the pair already decided, employees already on the pair, step)."""
        return self.cd, self.cs, self.k, self.f, self.t

    # --------------------------------------------------------------- internals
    def _init_state(self) -> None:
        inst, ctx, I, S = self.inst, self.ctx, self.I, self.S
        self.roster = self.roster_init.copy()
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
        self.E[:] = 0.0
        self.G[:] = 0.0
        self.E[:, self.WEEKENDS] = np.minimum(self.weekends * self.inv_wk, 1.0)
        self.t, self.J_acc, self.done = 0, 0.0, False
        self._feat_day = -1
        self._enter_pair(0)

    def _replay_day(self, d: int) -> None:
        """Apply frozen day d of the published roster, checking it against the mask
        (the assumption x^(0) satisfies H^pre of Prop. 3)."""
        inst = self.inst
        col = self.roster[:, d]
        w = col != OFF
        if w.any():
            s_arr = np.where(w, col, 0).astype(np.int64)
            if self.check_history:
                for c in self.pre:
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
        col = self.roster[:, self.d]
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
        inst, E, G, I = self.inst, self.E, self.G, self.I
        if d != self._feat_day:   # day columns: nobody is assigned on d yet
            self._feat_day = d
            m = self.minutes
            E[:, self.MINUTES] = np.minimum(m * self.inv_hmax, 1.0)
            E[:, self.SHORTFALL] = np.maximum(inst.min_minutes - m, 0) * self.inv_hmin
            E[:, self.SLACK] = np.clip((inst.max_minutes - m) * self.inv_hmax, -1.0, 1.0)
            E[:, self.RUN_WORK] = np.minimum(self.run_work * self.inv_cmax, 1.0)
            E[:, self.REST] = np.minimum(self.run_off * self.inv_rmin, 1.0)
            E[:, self.PREV:] = self.eye_ext[self.prev]
            G[self.G_DAY] = (d + 1) / self.D
        mask = ~self.assigned
        for c in self.pre:
            mask &= c.delta(self, d, s) == 0
        pen = np.zeros(I)
        for c, w in self.row_penalties:
            pen += w * c.delta(self, d, s)
        self.mask[:I] = mask
        self.n_allowed = int(mask.sum()) + 1
        self.pen = pen
        E[:, self.P] = self.p_f[:, d, s]
        E[:, self.X0] = self.x0_f[:, d, s]
        E[:, self.COUNT_S] = np.minimum(self.count[:, s] * self.inv_nmax[:, s], 1.0)
        E[:, self.ALLOWED] = mask
        G[self.G_SHIFT:] = 0.0
        G[self.G_SHIFT + s] = 1.0
        self.p, self.cd, self.cs, self.k, self.f = p, d, s, 0, 0
        self.rb = int(self.pair_rb[p])
        self.inc = self.fill_inc[p]
        self.rho_ds = int(inst.rho[d, s])
        self._slot_features()

    def _slot_features(self) -> None:
        G = self.G
        G[self.G_PROGRESS] = self.t / self.T
        G[self.G_FILL] = self.f * self.inv_rho[self.cd, self.cs]
        G[self.G_SLOTS_LEFT] = (self.rb - self.k) / self.rb
        G[self.G_ADMISSIBLE] = self.n_allowed / (self.I + 1)
        G[self.G_OVERCOVER] = float(self.f >= self.rho_ds)
