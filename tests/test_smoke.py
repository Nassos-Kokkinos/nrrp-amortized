"""The smoke test (theory validator) and tests that it catches the errors it exists to catch."""
import numpy as np
import pytest

from nrrp.env import RosterEnv
from nrrp.generator import random_context, synthetic_instance
from nrrp.objective import cover_cost
from nrrp.profile import BURKE_CURTOIS
from nrrp.validate import TEST_PROFILES, Report, run_episode, smoke


@pytest.fixture(scope="module")
def inst():
    return synthetic_instance(6, 14, 3, seed=2)


@pytest.mark.parametrize("profile", list(TEST_PROFILES))
@pytest.mark.parametrize("regime", ["construction", "rerostering"])
@pytest.mark.parametrize("size, extras", [((5, 7, 3), False), ((8, 14, 2), True)])
def test_smoke(size, extras, regime, profile):
    """Prop. 2 per step and in total, Prop. 3, mask, state and features (nrrp.validate)."""
    inst = synthetic_instance(*size, seed=1, skills=extras, min_cover=extras)
    r = smoke(inst, TEST_PROFILES[profile], regime=regime, episodes=4)
    assert r.ok, Report.HEADER + "\n" + r.line()


@pytest.mark.parametrize("regime", ["construction", "rerostering"])
@pytest.mark.parametrize("profile", ["bc_training", "burke_curtois"])
def test_smoke_extended_features(regime, profile):
    """The extended encoding: every feature checked against its from-scratch value."""
    from nrrp.profile import burke_curtois_training
    prof = burke_curtois_training(w_min_minutes=0.2, w_min_consecutive=100.0) if profile == "bc_training" \
        else BURKE_CURTOIS
    inst = synthetic_instance(6, 14, 3, seed=3)
    r = smoke(inst, prof, regime=regime, episodes=4, features="extended")
    assert r.ok, Report.HEADER + "\n" + r.line()


def test_extended_features_by_hand():
    from nrrp.profile import burke_curtois_training
    inst = synthetic_instance(4, 7, 2, seed=0)
    inst.min_consec[:] = 3
    env = RosterEnv(inst, burke_curtois_training(w_min_minutes=0.2, w_min_consecutive=100.0),
                    features="extended", kappa=0)
    assert env.obs_dim == 4 * (RosterEnv.PREV + 2 + 3) + RosterEnv.G_SHIFT + 2 + 2
    env.reset()
    X, GX = env.EX, env.GX
    assert (env.E[:, X + env.RUN_DEFICIT] == 0).all()               # day 0: no run yet
    L = inst.shift_len[env.cs]
    np.testing.assert_allclose(env.E[:, X + env.URGENCY], np.minimum(inst.min_minutes / (L * 7), 1), rtol=1e-6)
    assert env.G[GX + env.G_SHIFT_LEN] == pytest.approx(min(L / 720, 1))
    total = (np.minimum(inst.rho, 4) * inst.shift_len[None, :]).sum()
    assert env.G[GX + env.G_TIGHTNESS] == pytest.approx(min(inst.min_minutes.sum() / (2 * total), 1), rel=1e-6)
    # the mask zeroes COST outside M(s_t)
    assert (env.E[~env.action_mask()[:-1], X + env.COST] == 0).all()


def test_smoke_with_kappa_zero_and_two(inst):
    for kappa in (0, 2):
        r = smoke(inst, BURKE_CURTOIS, episodes=3, kappa=kappa)
        assert r.ok, r.line()


# ---- mutation tests: each reintroduces one error; the validator must report it

class PerSlotCover(RosterEnv):
    """Fix 1a reverted: the first rho slots charge w^u when left empty, the rest w^o when filled."""

    def step(self, a):
        k, f, p, d, s = self.k, self.f, self.p, self.cd, self.cs
        obs, r, done, info = super().step(a)
        if a < self.I:
            r += self.fill_inc[p][f]
        rho = self.inst.rho[d, s]
        if a == self.I and k < rho:
            r -= self.inst.w_under[d, s]
        if a < self.I and k >= rho:
            r -= self.inst.w_over[d, s]
        return obs, r, done, info


class NoHistoryConstant(RosterEnv):
    """Fix 1b reverted: c^con holds c^req and the open pairs' C(0), but not the frozen history."""

    def _setup_context(self, ctx):
        super()._setup_context(ctx)
        inst = self.inst
        open_ = np.zeros_like(inst.rho, bool)
        open_[ctx.d_xi:] = True
        self.c_con = inst.on_req.sum() + cover_cost(0, inst.rho, inst.w_under, inst.w_over)[open_].sum()


class ForgetsMaxConsecutive(RosterEnv):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.pre = tuple(c for c in self.pre if c.name != "max_consecutive_shifts")


class StaleCountFeature(RosterEnv):
    """The per-step update of the assigned row is skipped for one feature."""

    def step(self, a):
        keep = self.E[:, self.COUNT_S].copy()
        out = super().step(a)
        if not self.done and self.k > 0:       # still inside the same pair
            self.E[:, self.COUNT_S] = keep
        return out


class StaleRunDeficit(RosterEnv):
    """The run-deficit feature of the extended encoding is computed one day late."""

    def _enter_pair(self, p):
        new_day = int(self.pair_d[p]) != self._feat_day
        keep = self.E[:, self.EX + self.RUN_DEFICIT].copy()
        super()._enter_pair(p)
        if new_day and self.cd > 0:
            self.E[:, self.EX + self.RUN_DEFICIT] = keep


def test_validator_catches_a_stale_extended_feature():
    inst = synthetic_instance(8, 21, 3, seed=5)
    inst.min_consec[:] = 3
    env = StaleRunDeficit(inst, BURKE_CURTOIS, seed=0, features="extended")
    report = Report("StaleRunDeficit", per_step=True)
    rng = np.random.default_rng(0)
    for _ in range(4):
        run_episode(env, rng, report)
    assert report.feature_error > 0 and not report.ok, report.line()


@pytest.mark.parametrize("mutant, regime", [
    (PerSlotCover, "construction"),
    (NoHistoryConstant, "rerostering"),
    (ForgetsMaxConsecutive, "construction"),
    (StaleCountFeature, "construction"),
])
def test_validator_catches(mutant, regime):
    inst = synthetic_instance(8, 21, 3, seed=5)
    inst.max_consec[:] = 2                       # make the forgotten constraint bind
    contexts = None if regime == "construction" else (
        lambda rng: random_context(inst, BURKE_CURTOIS, rng))
    env = mutant(inst, BURKE_CURTOIS, contexts=contexts, seed=0)
    rng = np.random.default_rng(0)
    report = Report(mutant.__name__, per_step=True)
    for ep in range(4):
        env.include_constants = True
        run_episode(env, rng, report)
    assert not report.ok, report.line()
