import numpy as np
import pytest

from nrrp.env import RosterEnv
from nrrp.generator import random_roster, synthetic_instance
from nrrp.instance import OFF, Context
from nrrp.objective import J
from nrrp.profile import BURKE_CURTOIS, burke_curtois_training


@pytest.fixture(scope="module")
def inst():
    return synthetic_instance(6, 14, 3, seed=2)


def roll(env, rng):
    env.reset()
    done, total, n = False, 0.0, 0
    while not done:
        allowed = np.flatnonzero(env.action_mask())
        _, r, done, info = env.step(int(allowed[rng.integers(len(allowed))]))
        total, n = total + r, n + 1
    return total, n, info


def test_shapes_and_episode_length(inst):
    env = RosterEnv(inst, BURKE_CURTOIS)
    obs = env.reset()
    I, D, S = inst.n_emp, inst.n_days, inst.n_shifts
    assert obs.shape == (I * (10 + S) + 6 + S,) and obs.dtype == np.float32
    assert env.n_actions == I + 1
    assert env.T == np.minimum(inst.rho + 1, I).sum()
    _, n, _ = roll(env, np.random.default_rng(0))
    assert n == env.T


def test_empty_action_always_allowed_and_masked_actions_refused(inst):
    env = RosterEnv(inst, BURKE_CURTOIS)
    env.reset()
    rng = np.random.default_rng(1)
    refused = 0
    while not env.done:
        m = env.action_mask()
        assert m[-1]
        blocked = np.flatnonzero(~m)
        if len(blocked):
            with pytest.raises(ValueError, match="masked"):
                env.step(int(blocked[0]))
            refused += 1
        allowed = np.flatnonzero(m)
        env.step(int(allowed[rng.integers(len(allowed))]))
    assert refused > 0
    with pytest.raises(RuntimeError):
        env.step(env.I)


def test_constants_switch(inst):
    rng = np.random.default_rng(3)
    env = RosterEnv(inst, BURKE_CURTOIS, include_constants=True, seed=0)
    total, _, info = roll(env, rng)
    assert total == -info["J"]
    env.include_constants = False
    total, _, info = roll(env, rng)
    assert total == env.c_con - info["J"]


def test_frozen_history_that_violates_the_mask_is_refused(inst):
    y0 = random_roster(inst, BURKE_CURTOIS, np.random.default_rng(0))
    i = 0
    d = 3
    y0[i, d] = 0
    inst.day_off[i, d] = True          # the published roster now works on a fixed day off
    try:
        env = RosterEnv(inst, BURKE_CURTOIS, contexts=Context(inst, d_xi=5, published=y0, mu=1.0))
        with pytest.raises(ValueError, match="days_off on frozen day 3"):
            env.reset()
    finally:
        inst.day_off[i, d] = False


def test_contexts_from_a_sampler(inst):
    seen = []

    def sampler(rng):
        y0 = np.full((inst.n_emp, inst.n_days), OFF, np.int8)
        ctx = Context(inst, d_xi=int(rng.integers(1, inst.n_days)), published=y0, mu=0.5)
        seen.append(ctx.d_xi)
        return ctx

    env = RosterEnv(inst, BURKE_CURTOIS, contexts=sampler, seed=4)
    for _ in range(5):
        env.reset()
        assert env.cd >= env.ctx.d_xi
    assert len(seen) == 5


def test_training_reward_with_evaluation_under_the_published_profile(inst):
    train = burke_curtois_training(w_min_minutes=0.1, w_min_consecutive=20.0)
    env = RosterEnv(inst, train, eval_profile=BURKE_CURTOIS, include_constants=True, seed=1)
    total, _, info = roll(env, np.random.default_rng(5))
    ev = info["evaluation"]
    assert np.isclose(total, -info["J"], rtol=0, atol=1e-9)              # Prop. 2 for the training J
    assert ev["J"] == J(env.roster, env.ctx, BURKE_CURTOIS)                # evaluation J: no penalties
    assert info["J"] >= ev["J"]
    assert set(ev["violations"]) <= {"min_total_minutes", "min_consecutive_shifts"}   # only H^ter
