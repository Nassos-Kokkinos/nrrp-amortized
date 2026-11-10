"""The learned LNS of nrrp/lns_env.py: environment, policy and scripts."""
import numpy as np
import pytest

pytest.importorskip("ortools")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from nrrp.cpsat import solve  # noqa: E402
from nrrp.disruption import DisruptionModel  # noqa: E402
from nrrp.generator import synthetic_instance  # noqa: E402
from nrrp.instance import OFF, Context  # noqa: E402
from nrrp.lns_env import LNSEnv, decide, random_action, random_q, run_episode, top_q  # noqa: E402
from nrrp.profile import BURKE_CURTOIS  # noqa: E402

DET = dict(steps=4, workers=1, deterministic_time=2.0)          # reproducible CP-SAT steps


@pytest.fixture(scope="module")
def model_P():
    inst = synthetic_instance(8, 14, 2, seed=3)
    res = solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30)
    assert res.has_solution and res.feasible
    return DisruptionModel(inst, res.roster, n_absent=(1, 2), days=(2, 4))


def test_rewards_are_improvements_and_telescope(model_P):
    env = LNSEnv(model_P, **DET)
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        assert env.observation_space.contains(obs)
        ctx, phi0, total, done = env.ctx, env.phi0, 0.0, False
        while not done:
            obs, r, done, trunc, info = env.step(random_q(env.n_emp, 3, rng))
            assert r >= 0 and not trunc and env.observation_space.contains(obs)
            total += r
        assert total == pytest.approx((phi0 - env.phi) / max(abs(phi0), 1.0), abs=1e-9)
        x = env.roster
        np.testing.assert_array_equal(x[:, :ctx.d_xi], ctx.published[:, :ctx.d_xi])    # history kept
        assert (x[ctx.absent] == OFF).all()                                            # absences kept
        assert info["J"] <= info["J0"] + 1e-9 or not info["feasible0"]
        assert len(info["statuses"]) == DET["steps"]


def test_releasing_nobody_on_a_feasible_roster_changes_nothing(model_P):
    ctxs = [model_P.context(s) for s in range(20)]
    env = LNSEnv(ctxs, **DET)
    for j in range(len(ctxs)):
        env.reset(options={"instance": j})
        if env.feasible0 and not env.violators(env.x).any():
            _, r, _, _, info = env.step(np.zeros(env.n_emp, np.int8))
            assert r == 0 and info["status"] == "SKIPPED" and info["freed"] == 0
            return
    pytest.skip("no context with a feasible start")


def test_top_q_and_random_q():
    logits = np.array([3.0, -5.0, 0.5, -5.0, -5.0])                 # probabilities ~ 0.95, 0.62, ~0
    a = top_q(logits)
    assert a.tolist() == [1, 0, 1, 0, 0]
    assert top_q(np.full(6, -9.0)).sum() == 1                       # at least one
    r = random_q(10, 4, np.random.default_rng(1))
    assert r.sum() == 4 and set(np.unique(r)) <= {0, 1}


def test_the_policy_is_equivariant_and_starts_releasing_few(model_P):
    from stable_baselines3 import PPO

    from nrrp.policy import LNSPolicy
    env = LNSEnv(model_P, **DET)
    model = PPO(LNSPolicy, env, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs=dict(n_emp=env.n_emp),
                device="cpu", seed=0)
    obs, _ = env.reset(seed=1)
    ext = model.policy.mlp_extractor
    with torch.no_grad():
        p0 = torch.sigmoid(ext.forward_actor(torch.as_tensor(obs[None]))).numpy()[0]
    assert np.allclose(p0, 1 / (1 + np.exp(2.0)), atol=0.02)                     # init_bias -2
    torch.manual_seed(0)
    for p in model.policy.parameters():
        torch.nn.init.normal_(p, std=0.5)
    n, d = env.n_emp, env.FE
    perm = np.random.default_rng(2).permutation(n)
    rows = obs[: n * d].reshape(n, d)
    obs_p = np.concatenate([rows[perm].ravel(), obs[n * d:]]).astype(np.float32)
    with torch.no_grad():
        lo, v = ext(torch.as_tensor(obs[None]))
        lo_p, v_p = ext(torch.as_tensor(obs_p[None]))
    np.testing.assert_allclose(lo_p.numpy()[0], lo.numpy()[0][perm], atol=1e-5)
    with torch.no_grad():
        assert float(model.policy.value_net(v_p)) == pytest.approx(float(model.policy.value_net(v)), abs=1e-5)
    model.learn(total_timesteps=16)                                                # PPO runs on the MultiBinary action


def test_run_episode_reports_the_time(model_P):
    env = LNSEnv([model_P.context(7)], **DET)
    x, info = run_episode(env, 0, lambda obs, e: random_q(e.n_emp, 2, np.random.default_rng(0)))
    assert info["seconds"] > 0 and len(info["freed_per_step"]) == DET["steps"]


def test_releasing_days_frees_every_employee_on_them(model_P):
    """release="both": a released day d >= d_xi frees all employees on d; earlier days are ignored."""
    ctxs = [model_P.context(s) for s in range(10)]
    env = LNSEnv(ctxs, release="both", **DET)
    n, m = env.n_emp, env.inst.n_days
    assert env.action_space.n == n + m
    assert env.observation_space.shape == (n * env.FE + m * env.FD + env.FG,)
    j = next(k for k, c in enumerate(ctxs) if c.d_xi < m - 3)        # room for three days after d_xi
    obs, _ = env.reset(options={"instance": j})
    d0 = env.ctx.d_xi
    a = np.zeros(n + m, np.int8)
    a[n + d0 - 1] = 1                                              # a day of the history: ignored
    _, _, _, _, info = env.step(a)
    assert info["days_freed"] == 0
    a[n + d0] = 1                                                   # the first day that may change
    obs, r, _, _, info = env.step(a)
    assert info["days_freed"] == 1 and info["status"] != "SKIPPED" and r >= 0
    assert env.observation_space.contains(obs)
    logits = np.full(n + m, -9.0)
    logits[n: n + d0] = 9.0                                         # the policy prefers history days ...
    d = decide(logits, env)
    assert d[n: n + d0].sum() == 0 and d[n + d0:].sum() == 1        # ... decide never releases them
    r_a = random_action(env, 2, 3, np.random.default_rng(0))
    assert r_a[:n].sum() == 2 and r_a[n + d0:].sum() == 3 and r_a[n: n + d0].sum() == 0


def test_the_policy_scores_days_too(model_P):
    from stable_baselines3 import PPO

    from nrrp.policy import LNSPolicy
    env = LNSEnv(model_P, release="both", **DET)
    n, m = env.n_emp, env.inst.n_days
    model = PPO(LNSPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device="cpu", seed=0,
                policy_kwargs=dict(n_emp=n, n_days=m))
    torch.manual_seed(0)
    for p in model.policy.parameters():
        torch.nn.init.normal_(p, std=0.5)
    obs, _ = env.reset(seed=3)
    d = env.FE
    perm = np.random.default_rng(4).permutation(n)
    rows = obs[: n * d].reshape(n, d)
    obs_p = np.concatenate([rows[perm].ravel(), obs[n * d:]]).astype(np.float32)
    ext = model.policy.mlp_extractor
    with torch.no_grad():
        lo = ext.forward_actor(torch.as_tensor(obs[None])).numpy()[0]
        lo_p = ext.forward_actor(torch.as_tensor(obs_p[None])).numpy()[0]
    assert lo.shape == (n + m,)
    np.testing.assert_allclose(lo_p[:n], lo[:n][perm], atol=1e-5)         # employees permute
    np.testing.assert_allclose(lo_p[n:], lo[n:], atol=1e-5)               # days do not move
    model.learn(total_timesteps=16)


@pytest.mark.parametrize("release", ["employees", "both"])
def test_lns_scripts_end_to_end(tmp_path, model_P, release):
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path

    from nrrp.bc_format import write_bc, write_roster
    root = Path(__file__).resolve().parents[1]
    inst = model_P.instance
    (tmp_path / "solutions").mkdir()
    write_bc(inst, tmp_path / "MiniLNS.txt")
    J = float(solve(Context(inst), BURKE_CURTOIS, workers=1, deterministic_limit=30).J)
    write_roster(model_P.published, inst, tmp_path / "solutions" / f"MiniLNS.Solution.{J:g}.roster", penalty=J)

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=900)

    out = run("scripts/train_lns.py", "--name", "lns", "--reroster", str(tmp_path / "MiniLNS.txt"),
              "--n-absent", "1", "2", "--absence-days", "2", "4", "--eval-pool", "2", "--val-pool", "2",
              "--cp-time-limit", "5", "--steps", "2", "--step-time", "0.05", "--n-envs", "2",
              "--timesteps", "64", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "1", "--eval-every", "32",
              "--hidden", "16", "16", "--release", release,
              "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results"))
    assert out.returncode == 0, out.stdout + out.stderr
    for f in ("pools.json", "config.json", "latest.zip", "best.zip", "best.json", "progress.csv"):
        assert (tmp_path / "runs" / "lns" / f).exists(), f
    rows = list(csv.DictReader((tmp_path / "results" / "lns" / "results.csv").open()))
    assert len(rows) == 2 and all(float(r["J"]) <= float(r["J_start"]) + 1e-6 or r["feasible_start"] == "False"
                                  for r in rows)
    out = run("scripts/eval_lns.py", "--name", "lns", "--steps-list", "1", "2", "--workers", "1",
              "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "lns"))
    assert out.returncode == 0, out.stdout + out.stderr
    rows = list(csv.DictReader((tmp_path / "lns" / "lns" / "results.csv").open()))
    assert len(rows) == 2 * 2 * 5                                    # contexts x K x methods
    assert {r["method"] for r in rows} == {"learned_lns", "random_lns", "repair_x0", "cp_scratch", "cp_hint"}
    summary = json.loads((tmp_path / "lns" / "lns" / "run.json").read_text())["summary"]
    assert set(summary) == {"1", "2"} and sum(summary["2"]["vs_random_lns"]) == 2
