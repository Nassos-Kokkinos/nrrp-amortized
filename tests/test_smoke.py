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
@pytest.mark.parametrize("size", [(5, 7, 3), (8, 14, 2)])
def test_smoke(size, regime, profile):
    """Prop. 2 per step and in total, Prop. 3, mask, state and features (nrrp.validate)."""
    r = smoke(synthetic_instance(*size, seed=1), TEST_PROFILES[profile], regime=regime, episodes=4)
    assert r.ok, Report.HEADER + "\n" + r.line()


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
            r += self.cov_inc[p][f]
        rho = self.ctx.rho[d, s]
        if a == self.I and k < rho:
            r -= self.inst.w_under[d, s]
        if a < self.I and k >= rho:
            r -= self.inst.w_over[d, s]
        return obs, r, done, info


class NoHistoryConstant(RosterEnv):
    """Fix 1b reverted: K holds c^req and the open pairs' C(0), but not the frozen history."""

    def _setup_context(self, ctx):
        super()._setup_context(ctx)
        inst = self.inst
        open_ = np.zeros_like(ctx.rho, bool)
        open_[ctx.d_xi:] = True
        self.K = inst.on_req.sum() + cover_cost(0, ctx.rho, inst.w_under, inst.w_over)[open_].sum()


class ForgetsMaxConsecutive(RosterEnv):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.masked = tuple(c for c in self.masked if c.name != "max_consecutive_shifts")


class StaleMinutesFeature(RosterEnv):
    def step(self, a):
        keep = self.E[:, self.MINUTES].copy()
        out = super().step(a)
        self.E[:, self.MINUTES] = keep
        return out


@pytest.mark.parametrize("mutant, regime", [
    (PerSlotCover, "construction"),
    (NoHistoryConstant, "rerostering"),
    (ForgetsMaxConsecutive, "construction"),
    (StaleMinutesFeature, "construction"),
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
