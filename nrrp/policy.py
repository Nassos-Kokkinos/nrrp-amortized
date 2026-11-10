"""The policy network of Eq. (39), as a MaskablePPO policy (sb3-contrib).

Needs the ``rl`` extra. The observation of :class:`nrrp.env.RosterEnv` is n rows
psi_i (one per employee, d_psi features each) followed by the global vector g
(d_g features). The network

    h_i     = MLP([psi_i, g])             one MLP shared by all employees
    logit_i = w^T h_i + b                 score of assigning employee i, i < n
    logit_n = beta                        a learned scalar for the empty action
    V       = MLP_v(mean_i h_i)           value from the mean embedding

``published_prior`` = b0 (re-rostering, decision of 10/11/2026) adds a learned
prior b, initialised to b0, towards the published roster x°:

    logit_i += b * x°_{i,d,s}                      the employee x° had on this slot
    logit_n += b * (1 - max_{i allowed} x°_{i,d,s}) the empty action, if no such employee is allowed

so that, with a large b, the first policy reproduces x° wherever the mask allows
it, and training learns where to deviate from it. It reads x° and the mask from
the observation (columns X0 and ALLOWED of Table 2), so the network stays
equivariant in the employees.

is equivariant in the employees: permuting the rows psi_i permutes the first n
logits the same way and leaves logit_n and V unchanged (tests check this). The
mask (32) is applied by MaskablePPO on top of these logits.

``separate_critic=True`` gives the critic its own encoder MLP (same sizes), so
that the value loss does not shape the embedding the logits are read from; the
default shares one encoder, as Eq. (39) is written. No parameter depends on n:
a policy trained with one number of employees loads into a network for another
(same shift types and sizes), which ``scripts/train.py --init-from`` uses.

Implementation: sb3 builds a policy as features extractor -> ``mlp_extractor``
-> ``action_net`` / ``value_net``. Here the extractor is the identity flatten,
``EmployeeExtractor`` returns the n + 1 logits as the actor latent and the mean
embedding as the critic latent, ``action_net`` is the identity, and
``value_net`` is the linear layer sb3 adds on the critic latent.
"""
from __future__ import annotations

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


def _mlp(sizes, activation) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers)


class EmployeeExtractor(nn.Module):
    def __init__(self, n_emp: int, d_psi: int, d_g: int, hidden=(64, 64), value_hidden=(64,),
                 activation=nn.Tanh, separate_critic: bool = False, published_prior: float | None = None):
        super().__init__()
        from .env import RosterEnv
        self.n, self.d_psi, self.d_g = int(n_emp), int(d_psi), int(d_g)
        self.col_x0, self.col_allowed = RosterEnv.X0, RosterEnv.ALLOWED
        self.prior = None if published_prior is None else nn.Parameter(torch.tensor([float(published_prior)]))
        self.encoder = _mlp([d_psi + d_g, *hidden], activation)
        self.critic_encoder = _mlp([d_psi + d_g, *hidden], activation) if separate_critic else None
        self.score = nn.Linear(hidden[-1], 1)
        self.empty_logit = nn.Parameter(torch.zeros(1))
        self.value_body = _mlp([hidden[-1], *value_hidden], activation) if value_hidden else nn.Identity()
        self.latent_dim_pi = self.n + 1
        self.latent_dim_vf = value_hidden[-1] if value_hidden else hidden[-1]

    def split(self, obs: torch.Tensor):
        """(batch, n, d_psi) rows and (batch, d_g) global features."""
        b = obs.shape[0]
        rows = obs[:, : self.n * self.d_psi].reshape(b, self.n, self.d_psi)
        g = obs[:, self.n * self.d_psi:]
        return rows, g

    def inputs(self, obs: torch.Tensor) -> torch.Tensor:
        rows, g = self.split(obs)
        return torch.cat([rows, g.unsqueeze(1).expand(-1, self.n, -1)], dim=-1)   # (batch, n, d_psi + d_g)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.inputs(obs))                     # (batch, n, hidden)

    def _logits(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)                        # (batch, n)
        empty = self.empty_logit.expand(scores.shape[0], 1)
        if self.prior is not None:                                # towards the published roster x°
            x0 = x[..., self.col_x0]                              # (batch, n): x° had i on this slot
            offered = (x0 * x[..., self.col_allowed]).amax(dim=1, keepdim=True)   # such an i is allowed
            scores = scores + self.prior * x0
            empty = empty + self.prior * (1.0 - offered)
        return torch.cat([scores, empty], dim=1)                  # (batch, n + 1)

    def _value(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        hv = h if self.critic_encoder is None else self.critic_encoder(x)
        return self.value_body(hv.mean(dim=1))

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.inputs(obs)
        return self._logits(self.encoder(x), x)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.inputs(obs)
        if self.critic_encoder is not None:
            return self.value_body(self.critic_encoder(x).mean(dim=1))
        return self.value_body(self.encoder(x).mean(dim=1))

    def forward(self, obs: torch.Tensor):
        x = self.inputs(obs)
        h = self.encoder(x)
        return self._logits(h, x), self._value(x, h)


class EmployeePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy with the architecture of Eq. (39).

    policy_kwargs: ``n_emp``, ``n_shifts`` (to split the observation), and
    optionally ``hidden`` (encoder sizes), ``value_hidden``, ``separate_critic``,
    ``features`` (the state encoding of the environment, "table2" or "extended") and
    ``published_prior`` (initial b of the prior towards x°, re-rostering; None = no prior).
    """

    def __init__(self, observation_space, action_space, lr_schedule, *, n_emp: int, n_shifts: int,
                 hidden=(64, 64), value_hidden=(64,), separate_critic: bool = False,
                 features: str = "table2", published_prior: float | None = None,
                 activation_fn=nn.Tanh, **kwargs):
        self._n_emp, self._n_shifts = int(n_emp), int(n_shifts)
        self._hidden, self._value_hidden = tuple(hidden), tuple(value_hidden)
        self._separate_critic = bool(separate_critic)
        self._published_prior = None if published_prior is None else float(published_prior)
        self._features = features
        from .env import RosterEnv
        self._d_psi, self._d_g = RosterEnv.dims(self._n_shifts, features)
        want = self._n_emp * self._d_psi + self._d_g
        if observation_space.shape != (want,):
            raise ValueError(f"observation of shape {observation_space.shape}, expected ({want},) "
                             f"for {n_emp} employees and {n_shifts} shift types ({features} features)")
        if action_space.n != self._n_emp + 1:
            raise ValueError(f"{action_space.n} actions, expected {self._n_emp + 1}")
        super().__init__(observation_space, action_space, lr_schedule, activation_fn=activation_fn,
                         **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = EmployeeExtractor(self._n_emp, self._d_psi, self._d_g, self._hidden,
                                               self._value_hidden, self.activation_fn, self._separate_critic,
                                               self._published_prior)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # the actor latent already holds the n + 1 logits
        self.action_net = nn.Identity()
        # sb3 gives the policy head a small initial gain (0.01) so that the first
        # policy is near uniform over the admissible actions; here the head is ``score``
        if self.ortho_init:
            nn.init.orthogonal_(self.mlp_extractor.score.weight, gain=0.01)
            nn.init.zeros_(self.mlp_extractor.score.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_emp=self._n_emp, n_shifts=self._n_shifts, hidden=self._hidden,
                    value_hidden=self._value_hidden, separate_critic=self._separate_critic,
                    features=self._features, published_prior=self._published_prior)
        return data


def transfer_state(policy: EmployeePolicy, state: dict) -> list:
    """Load the parameters of another EmployeePolicy (any number of employees, same shift
    types and sizes) into ``policy``. A shared-encoder source gives its encoder to both the
    actor and the critic encoder of a separate-critic target. Returns the names copied."""
    own = policy.state_dict()
    state = dict(state)
    pre = "mlp_extractor."
    if any(k.startswith(pre + "critic_encoder.") for k in own) and \
            not any(k.startswith(pre + "critic_encoder.") for k in state):
        for k in list(state):
            if k.startswith(pre + "encoder."):
                state[pre + "critic_" + k[len(pre):]] = state[k]
    copied, bad = [], []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape != v.shape:
            bad.append(f"{k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
            continue
        own[k] = v
        copied.append(k)
    if bad:
        raise ValueError("cannot transfer, different sizes (shift types or --hidden?): " + "; ".join(bad))
    # the prior towards x° keeps its initial value when the source policy had none
    missing = [k for k in own if k not in copied and k != pre + "prior"]
    if missing:
        raise ValueError(f"source policy lacks {missing}")
    policy.load_state_dict(own)
    return copied


# ------------------------------------------------------------------ learned LNS (nrrp/lns_env.py)

class LNSExtractor(nn.Module):
    """Scores for the release decision of the learned LNS:

        h_i     = MLP([e_i, g])        one MLP shared by all employees (as Eq. (39))
        logit_i = w^T h_i + b          log-odds of releasing employee i for CP-SAT
        u_d     = MLP_day([f_d, g])    with n_days > 0 (release="both"): one MLP shared by the days
        logit_d = w'^T u_d + b'        log-odds of releasing day d for every employee
        V       = MLP_v([mean_i h'_i, mean_d u'_d])   value from separate encoders

    equivariant in the employees, with no parameter that depends on n. The biases b, b'
    start at ``init_bias`` (-2: about one employee, or day, in eight per step)."""

    def __init__(self, n_emp: int, d_e: int, d_g: int, hidden=(64, 64), value_hidden=(64,),
                 activation=nn.Tanh, init_bias: float = -2.0, n_days: int = 0, d_day: int = 0):
        super().__init__()
        self.n, self.d_e, self.d_g, self.init_bias = int(n_emp), int(d_e), int(d_g), float(init_bias)
        self.m, self.d_day = int(n_days), int(d_day)
        self.encoder = _mlp([d_e + d_g, *hidden], activation)
        self.critic_encoder = _mlp([d_e + d_g, *hidden], activation)
        self.score = nn.Linear(hidden[-1], 1)
        width = hidden[-1]
        if self.m:
            self.day_encoder = _mlp([d_day + d_g, *hidden], activation)
            self.day_critic_encoder = _mlp([d_day + d_g, *hidden], activation)
            self.day_score = nn.Linear(hidden[-1], 1)
            width = 2 * hidden[-1]
        self.value_body = _mlp([width, *value_hidden], activation) if value_hidden else nn.Identity()
        self.latent_dim_pi = self.n + self.m
        self.latent_dim_vf = value_hidden[-1] if value_hidden else width

    def inputs(self, obs: torch.Tensor):
        b, ne = obs.shape[0], self.n * self.d_e
        nd = self.m * self.d_day
        g = obs[:, ne + nd:]
        rows = torch.cat([obs[:, :ne].reshape(b, self.n, self.d_e), g.unsqueeze(1).expand(-1, self.n, -1)], dim=-1)
        if not self.m:
            return rows, None
        days = obs[:, ne: ne + nd].reshape(b, self.m, self.d_day)
        return rows, torch.cat([days, g.unsqueeze(1).expand(-1, self.m, -1)], dim=-1)

    def _actor(self, rows, days):
        logits = self.score(self.encoder(rows)).squeeze(-1)                     # (batch, n)
        if days is not None:
            logits = torch.cat([logits, self.day_score(self.day_encoder(days)).squeeze(-1)], dim=1)
        return logits

    def _critic(self, rows, days):
        z = self.critic_encoder(rows).mean(dim=1)
        if days is not None:
            z = torch.cat([z, self.day_critic_encoder(days).mean(dim=1)], dim=1)
        return self.value_body(z)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self._actor(*self.inputs(obs))                                  # (batch, n [+ m]) log-odds

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self._critic(*self.inputs(obs))

    def forward(self, obs: torch.Tensor):
        rows, days = self.inputs(obs)
        return self._actor(rows, days), self._critic(rows, days)


class LNSPolicy(ActorCriticPolicy):
    """PPO policy for :class:`nrrp.lns_env.LNSEnv`: a MultiBinary action, one Bernoulli per
    employee (and per day, with ``n_days`` > 0, i.e. release="both"), with the logits of
    :class:`LNSExtractor`. policy_kwargs: ``n_emp``, and optionally ``n_days``, ``hidden``,
    ``value_hidden``, ``init_bias``."""

    def __init__(self, observation_space, action_space, lr_schedule, *, n_emp: int, n_days: int = 0,
                 hidden=(64, 64), value_hidden=(64,), init_bias: float = -2.0,
                 activation_fn=nn.Tanh, **kwargs):
        from .lns_env import LNSEnv
        self._n_emp, self._n_days = int(n_emp), int(n_days)
        self._hidden, self._value_hidden = tuple(hidden), tuple(value_hidden)
        self._init_bias = float(init_bias)
        self._d_e, self._d_g = len(LNSEnv.FEATURES), len(LNSEnv.GLOBAL)
        self._d_day = len(LNSEnv.DAY_FEATURES) if self._n_days else 0
        want = self._n_emp * self._d_e + self._n_days * self._d_day + self._d_g
        if observation_space.shape != (want,):
            raise ValueError(f"observation of shape {observation_space.shape}, expected ({want},) for "
                             f"{n_emp} employees and {n_days} days")
        if action_space.n != self._n_emp + self._n_days:
            raise ValueError(f"{action_space.n} binary actions, expected {self._n_emp + self._n_days}")
        super().__init__(observation_space, action_space, lr_schedule, activation_fn=activation_fn, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LNSExtractor(self._n_emp, self._d_e, self._d_g, self._hidden, self._value_hidden,
                                          self.activation_fn, self._init_bias, self._n_days, self._d_day)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()               # the actor latent already holds the log-odds
        scores = [self.mlp_extractor.score] + ([self.mlp_extractor.day_score] if self._n_days else [])
        for sc in scores:
            nn.init.orthogonal_(sc.weight, gain=0.01)
            nn.init.constant_(sc.bias, self._init_bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_emp=self._n_emp, n_days=self._n_days, hidden=self._hidden, value_hidden=self._value_hidden,
                    init_bias=self._init_bias)
        return data
