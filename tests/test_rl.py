"""The Gymnasium wrapper and the policy network of Eq. (39)."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

from nrrp.generator import random_context, synthetic_instance  # noqa: E402
from nrrp.gym_env import GymRosterEnv  # noqa: E402
from nrrp.objective import J  # noqa: E402
from nrrp.policy import EmployeePolicy  # noqa: E402
from nrrp.profile import BURKE_CURTOIS, burke_curtois_training  # noqa: E402

TRAIN = burke_curtois_training(w_min_minutes=0.2, w_min_consecutive=100.0)


def pool(n=5, m=7, k=3, seeds=range(4)):
    return [synthetic_instance(n, m, k, seed=s) for s in seeds]


def rollout(env, rng, seed=None, options=None):
    obs, info = env.reset(seed=seed, options=options)
    total, done = 0.0, False
    while not done:
        mask = env.action_masks()
        assert mask.dtype == bool and mask[-1]                     # the empty action is always allowed
        a = int(rng.choice(np.flatnonzero(mask)))
        obs, r, done, trunc, info = env.step(a)
        assert not trunc and env.observation_space.contains(obs)
        total += r
    return total, info


@pytest.mark.parametrize("features", ["table2", "extended"])
def test_gymnasium_checker(features):
    check_env(GymRosterEnv(pool(), TRAIN, eval_profile=BURKE_CURTOIS, features=features), skip_render_check=True)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_scaled_rewards_sum_to_the_context_constant_minus_J(scale):
    env = GymRosterEnv(pool(), TRAIN, eval_profile=BURKE_CURTOIS, reward_scale=scale)
    rng = np.random.default_rng(0)
    for ep in range(6):
        total, info = rollout(env, rng, seed=ep)
        inner = env.env
        assert info["J"] == pytest.approx(J(inner.roster, inner.ctx, TRAIN), abs=1e-9)
        assert total / scale == pytest.approx(inner.c_con - info["J"], abs=1e-7)   # Remark 7
        assert info["evaluation"]["J"] == J(inner.roster, inner.ctx, BURKE_CURTOIS)
        assert info["instance"] == inner.inst.name


def test_with_constants_rewards_sum_to_minus_J():
    env = GymRosterEnv(pool(), TRAIN, include_constants=True)
    total, info = rollout(env, np.random.default_rng(1), seed=3)
    assert total == pytest.approx(-info["J"], abs=1e-9)          # Prop. 2


def test_mask_matches_the_environment():
    env = GymRosterEnv(pool(), TRAIN)
    env.reset(seed=0)
    np.testing.assert_array_equal(env.action_masks(), env.env.action_mask())


def test_the_pool_is_sampled_and_can_be_chosen():
    insts = pool(seeds=range(5))
    env = GymRosterEnv(insts, TRAIN)
    seen = {env.reset(seed=s)[1]["instance"] for s in range(60)}
    assert seen == {i.name for i in insts}
    assert env.reset(options={"instance": 3})[1]["instance"] == insts[3].name


def test_a_pool_of_mixed_sizes_is_refused():
    with pytest.raises(ValueError, match="need 5 employees and 3 shift types"):
        GymRosterEnv([synthetic_instance(5, 7, 3), synthetic_instance(6, 7, 3)], TRAIN)


def test_rerostering_contexts():
    def draw(rng, inst):
        return random_context(inst, BURKE_CURTOIS, rng, mu=1.0)
    env = GymRosterEnv(pool(), TRAIN, contexts=draw, reward_scale=0.01)
    obs, info = env.reset(seed=4)
    assert env.env.ctx.d_xi > 0
    total, info = rollout(env, np.random.default_rng(2), seed=5)
    assert total / 0.01 == pytest.approx(env.env.c_con - info["J"], abs=1e-7)


def make_policy(env, **kw):
    return EmployeePolicy(env.observation_space, env.action_space, lambda _: 3e-4,
                          n_emp=env.n_emp, n_shifts=env.n_shifts, **kw)


@pytest.mark.parametrize("separate, features", [(False, "table2"), (True, "table2"), (True, "extended")])
def test_policy_is_equivariant_in_the_employees(separate, features):
    env = GymRosterEnv(pool(n=6, k=3), TRAIN, features=features)
    pol = make_policy(env, separate_critic=separate, features=features)
    torch.manual_seed(0)
    for p in pol.parameters():                      # random weights, not the near-uniform init
        torch.nn.init.normal_(p, std=0.5)
    obs, _ = env.reset(seed=1)
    for _ in range(5):                              # a few states into the episode
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    obs = env.env.obs.copy()
    n, d_psi = env.n_emp, pol._d_psi
    perm = np.random.default_rng(3).permutation(n)
    rows = obs[: n * d_psi].reshape(n, d_psi)
    obs_p = np.concatenate([rows[perm].ravel(), obs[n * d_psi:]]).astype(np.float32)
    ext = pol.mlp_extractor
    with torch.no_grad():
        lo, v = ext(torch.as_tensor(obs[None]))
        lo_p, v_p = ext(torch.as_tensor(obs_p[None]))
        v, v_p = pol.value_net(v), pol.value_net(v_p)
    lo, lo_p = lo.numpy()[0], lo_p.numpy()[0]
    np.testing.assert_allclose(lo_p[:n], lo[:n][perm], rtol=0, atol=1e-5)
    assert lo_p[n] == pytest.approx(lo[n])
    assert float(v_p) == pytest.approx(float(v), abs=1e-5)


def test_initial_policy_is_near_uniform_and_respects_the_mask():
    env = GymRosterEnv(pool(), TRAIN)
    pol = make_policy(env)
    obs, _ = env.reset(seed=2)
    mask = env.action_masks()
    with torch.no_grad():
        dist = pol.get_distribution(torch.as_tensor(obs[None]), action_masks=mask[None])
        probs = dist.distribution.probs.numpy()[0]
    assert probs[~mask].sum() == pytest.approx(0.0, abs=1e-7)
    allowed = probs[mask]
    assert allowed.max() / allowed.min() < 1.1
    for _ in range(20):
        a, _ = pol.predict(obs, action_masks=mask, deterministic=False)
        assert mask[int(a)]


def test_separate_critic_does_not_touch_the_actor():
    env = GymRosterEnv(pool(), TRAIN)
    pol = make_policy(env, separate_critic=True)
    ext = pol.mlp_extractor
    obs = torch.as_tensor(env.reset(seed=0)[0][None])
    logits, v = ext(obs)
    (v.sum()).backward()
    assert all(p.grad is None or not p.grad.abs().any() for p in ext.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().any() for p in ext.critic_encoder.parameters())
    torch.testing.assert_close(ext.forward_actor(obs), logits)
    torch.testing.assert_close(ext.forward_critic(obs), v)


@pytest.mark.parametrize("separate", [False, True])
def test_a_policy_transfers_to_another_number_of_employees(separate):
    from nrrp.policy import transfer_state
    small = GymRosterEnv(pool(n=5), TRAIN)
    big = GymRosterEnv(pool(n=8), TRAIN)
    src = make_policy(small)
    torch.manual_seed(1)
    for p in src.parameters():
        torch.nn.init.normal_(p, std=0.5)
    dst = make_policy(big, separate_critic=separate)
    copied = transfer_state(dst, src.state_dict())
    assert len(copied) == len(dst.state_dict())
    # an 8-employee state whose first 5 rows equal a 5-employee state: same scores for those 5
    obs8 = big.reset(seed=0)[0]
    d_psi = dst._d_psi
    obs5 = np.concatenate([obs8[: 5 * d_psi], obs8[8 * d_psi:]]).astype(np.float32)
    with torch.no_grad():
        lo5, _ = src.mlp_extractor(torch.as_tensor(obs5[None]))
        lo8, v8 = dst.mlp_extractor(torch.as_tensor(obs8[None]))
    np.testing.assert_allclose(lo8[0, :5].numpy(), lo5[0, :5].numpy(), atol=1e-6)
    assert float(lo8[0, -1]) == pytest.approx(float(lo5[0, -1]))
    if separate:                                     # the critic encoder starts as a copy of the encoder
        for a, b in zip(dst.mlp_extractor.encoder.parameters(), dst.mlp_extractor.critic_encoder.parameters()):
            torch.testing.assert_close(a, b)


def test_transfer_refuses_other_shift_types():
    from nrrp.policy import transfer_state
    src = make_policy(GymRosterEnv(pool(k=3), TRAIN))
    dst = make_policy(GymRosterEnv(pool(k=2), TRAIN))
    with pytest.raises(ValueError, match="different sizes"):
        transfer_state(dst, src.state_dict())


def test_policy_refuses_a_wrong_observation_size():
    env = GymRosterEnv(pool(n=5), TRAIN)
    with pytest.raises(ValueError, match="expected"):
        EmployeePolicy(env.observation_space, env.action_space, lambda _: 3e-4, n_emp=6, n_shifts=3)
    with pytest.raises(ValueError, match="extended features"):
        EmployeePolicy(env.observation_space, env.action_space, lambda _: 3e-4, n_emp=5, n_shifts=3,
                       features="extended")


def test_maskable_ppo_runs():
    env = GymRosterEnv(pool(), TRAIN, eval_profile=BURKE_CURTOIS, reward_scale=0.01)
    model = MaskablePPO(EmployeePolicy, env, n_steps=64, batch_size=32, n_epochs=2, gamma=1.0,
                        policy_kwargs=dict(n_emp=env.n_emp, n_shifts=env.n_shifts, hidden=(32, 32),
                                           value_hidden=(32,)), seed=0, verbose=0, device="cpu")
    model.learn(total_timesteps=256)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, r, done, _, info = env.step(int(a))
    assert np.isfinite(info["J"])


def test_train_script_end_to_end(tmp_path):
    """Pools, training, periodic evaluation, resume, final results and figures, at toy scale."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    pytest.importorskip("ortools")
    root = Path(__file__).resolve().parents[1]

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=600)

    common = ["--name", "t", "--size", "3", "7", "2", "--train-pool", "2", "--eval-pool", "2",
              "--n-steps", "64", "--batch-size", "32", "--n-epochs", "1", "--eval-every", "128",
              "--save-every", "128", "--k-samples", "2", "--hidden", "16", "16", "--cp-time-limit", "5",
              "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "results")]
    out = run("scripts/train.py", "--timesteps", "256", *common)
    assert out.returncode == 0, out.stdout + out.stderr
    run_dir, res_dir = tmp_path / "runs" / "t", tmp_path / "results" / "t"
    for f in ("config.json", "pools.json", "latest.zip", "progress.csv", "progress_instances.csv"):
        assert (run_dir / f).exists(), f
    inst_rows = (run_dir / "progress_instances.csv").read_text().splitlines()
    assert inst_rows[0].startswith("steps,instance,J_cp")
    assert len(inst_rows) - 1 == 2 * (len((run_dir / "progress.csv").read_text().splitlines()) - 1)
    pools = json.loads((run_dir / "pools.json").read_text())
    assert len(pools["train"]) == 2 and len(pools["eval"]) == 2
    assert all(r["J_cp"] is not None for r in pools["eval"])
    rows = (res_dir / "results.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[0].startswith("instance,J_cp")
    n_progress = len((run_dir / "progress.csv").read_text().splitlines())
    out = run("scripts/train.py", "--timesteps", "512", "--resume", *common)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "loaded" in out.stdout
    assert len((run_dir / "progress.csv").read_text().splitlines()) > n_progress
    pytest.importorskip("matplotlib")
    out = run("scripts/plot_results.py", "train", str(res_dir))
    assert out.returncode == 0, out.stdout + out.stderr
    for stem in ("learning_curve", "feasibility_curve", "final_gap"):
        assert (res_dir / "figures" / f"{stem}.pdf").stat().st_size > 1000
    assert (res_dir / "progress_instances.csv").exists()
    out = run("scripts/plot_results.py", "grid", str(res_dir), "--out", str(tmp_path / "grid"))
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "grid.pdf").stat().st_size > 1000

    # scaling options: same pools without CP-SAT, initialised from the 3-employee run on 4 employees
    out = run("scripts/train.py", "--name", "t4", "--size", "4", "7", "2", "--pools-from", "t", "--timesteps", "128",
              *common[6:])
    assert out.returncode != 0 and "is for size" in out.stdout + out.stderr      # pools of another size refused
    out = run("scripts/train.py", "--name", "t2", "--timesteps", "128", "--pools-from", "t", "--init-from", "t",
              "--separate-critic", "--gamma", "0.99", "--gae-lambda", "0.98", "--lr-schedule", "linear",
              "--target-kl", "0.02", "--ent-coef", "0.001", *common[2:])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "pools copied" in out.stdout and "policy initialised from" in out.stdout
    assert "building pools" not in out.stdout
    assert (tmp_path / "runs" / "t2" / "pools.json").read_text() == (run_dir / "pools.json").read_text()
    cfg = json.loads((tmp_path / "runs" / "t2" / "config.json").read_text())
    assert cfg["gamma"] == 0.99 and cfg["args"]["separate_critic"] and cfg["args"]["init_from"] == "t"
    out = run("scripts/train.py", "--name", "t2", "--timesteps", "256", "--resume", *common[2:])
    assert out.returncode == 0, out.stdout + out.stderr

    # extended state encoding: train, resume (the encoding is read back from config.json), repair
    out = run("scripts/train.py", "--name", "t3", "--timesteps", "128", "--pools-from", "t", "--features", "extended",
              *common[2:])
    assert out.returncode == 0, out.stdout + out.stderr
    out = run("scripts/train.py", "--name", "t3", "--timesteps", "256", "--resume", *common[2:])
    assert out.returncode == 0, out.stdout + out.stderr
    out = run("scripts/eval_repair.py", "--name", "t3", "--runs", str(tmp_path / "runs"),
              "--results", str(tmp_path / "repair"), "--budgets", "0", "1", "--workers", "1")
    assert out.returncode == 0, out.stdout + out.stderr
    out = run("scripts/train.py", "--name", "t5", "--timesteps", "128", "--pools-from", "t", "--features", "extended",
              "--init-from", "t", *common[2:])
    assert out.returncode != 0 and "cannot transfer" in out.stdout + out.stderr

    # policy alone against policy with repair, on the toy run (both forms of Algorithm 1)
    import csv
    for extra, folder in (([], "t-anytime"), (["--first-feasible"], "t-first"),
                          (["--source", "random"], "t-random-anytime")):
        out = run("scripts/eval_repair.py", "--name", "t", "--runs", str(tmp_path / "runs"),
                  "--results", str(tmp_path / "repair"), "--budgets", "0", "2", "--workers", "1", *extra)
        assert out.returncode == 0, out.stdout + out.stderr
        rows = list(csv.DictReader((tmp_path / "repair" / folder / "results.csv").open()))
        assert len(rows) == 4                                          # 2 instances x 2 budgets
        alone = [r for r in rows if float(r["tau_cp"]) == 0]
        repaired = [r for r in rows if float(r["tau_cp"]) == 2]
        assert all(r["J"] == r["J_policy"] and r["feasible"] == r["feasible_policy"] for r in alone)
        assert all(r["feasible"] == "True" for r in repaired)          # pool instances are CP-feasible
        assert all(float(r["J"]) >= float(r["J_cp"]) - 1e-6 for r in repaired)
        assert (tmp_path / "repair" / folder / "run.json").exists()
