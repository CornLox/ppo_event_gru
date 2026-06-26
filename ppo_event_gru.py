# ppo_recurrent_lstm_event.py
# ---------------------------------------------------------------------------
# Recurrent PPO following the LSTM implementation from
#   "The 37 Implementation Details of Proximal Policy Optimization"
#   (Huang et al., ICLR Blog Track 2022; CleanRL's ppo_atari_lstm.py),
# with a *swappable recurrent core* selected by
#   --recurrent-core {lstm, gru, mgu, evlstm, evgru, evmgu}:
#   * lstm   : the standard LSTM core, exactly as in the post.
#   * gru    : the standard GRU core (nn.GRU), same init convention as `lstm`.
#   * mgu    : the Minimal Gated Unit (Zhou et al., "Minimal Gated Unit for
#              Recurrent Neural Networks", 2016) -- a GRU whose reset and update
#              gates are merged into a SINGLE forget gate f, same init convention
#              as `gru`. The GRU analogue with one gate instead of two.
#   * evlstm : the SAME LSTM with the activity-sparsity mechanism from
#              "Efficient Recurrent Architectures Through Activity Sparsity and
#               Sparse Back-Propagation Through Time" (Subramoney et al.,
#               ICLR 2023 / EGRU) grafted on, so it differs from `lstm` ONLY in
#               the sparsity. The paper itself applies the mechanism to a GRU.
#   * evgru  : the canonical EGRU -- the SAME activity-sparsity mechanism on its
#              native GRU backbone, i.e. the model the paper actually proposes.
#   * evmgu  : the SAME activity-sparsity mechanism on the MGU backbone, i.e. the
#              event twin of `mgu` (single forget gate + EGRU sparsity), the MGU
#              analogue of evgru.
#
# The three event cores (evlstm, evgru, evmgu) share a single sparsity
# implementation (thresholded events, sparse recurrence, reset-by-subtraction,
# surrogate-gradient BPTT), so comparing them isolates the recurrent backbone;
# comparing each event core against its dense twin (lstm / gru / mgu) isolates the
# sparsity mechanism.
#
# The default environment is Breakout (BreakoutNoFrameskip-v4) with the exact
# Atari preprocessing + Nature-CNN encoder from ppo_atari_lstm.py. The encoder
# is auto-selected: CNN for image observations, MLP for vector observations
# (so classic-control env ids like CartPole-v1 still work for quick tests).
#
# Faithfulness notes are inlined as  # [37-#N]  for the PPO/LSTM details and
#   # [EGRU ...]  for the sparse-RNN mechanism.
# ---------------------------------------------------------------------------
import ale_py
import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

try:
    import gymnasium as gym
except ImportError:  # fall back to classic gym if gymnasium is unavailable
    import gym

# Gymnasium 1.0 removed the plugin that auto-registered ALE envs, so Atari ids
# (ALE/Breakout-v5, BreakoutNoFrameskip-v4, ...) only exist after ale_py is
# imported and registered. Harmless on older stacks. ale_py >= 0.9 also bundles
# the ROMs, so AutoROM is no longer needed there.
try:
    import ale_py
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
except ImportError:
    ale_py = None


# ===========================================================================
#  Argument parsing
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # --- experiment ---
    p.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic",
                   type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--cuda", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)

    # --- PPO core (the original "13 core details") ---
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", type=lambda x: x.lower()
                   == "true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", type=lambda x: x.lower()
                   == "true", default=True)
    # 0.1 is the blog's Atari value
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--clip-vloss", type=lambda x: x.lower()
                   == "true", default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)

    # --- recurrent core selection ---
    p.add_argument("--recurrent-core", type=str, default="evmgu",
                   choices=["lstm", "gru", "mgu", "evlstm", "evgru", "evmgu"],
                   help="lstm   ('37 details' baseline LSTM core) | "
                        "gru    (traditional GRU core) | "
                        "mgu    (Minimal Gated Unit: GRU with its reset+update gates "
                        "merged into a single forget gate, Zhou et al. 2016) | "
                        "evlstm (event LSTM: LSTM gating + the EGRU activity-sparsity "
                        "mechanism grafted on) | "
                        "evgru  (event GRU: the canonical EGRU -- GRU gating + the "
                        "same activity-sparsity mechanism) | "
                        "evmgu  (event MGU: MGU gating + the same activity-sparsity "
                        "mechanism)")
    # TRUE -> ignore --recurrent-core and run one full training run per core, in
    # turn, so a single invocation produces every dense/event pair under the same
    # seed and hyperparameters. Each core's trained parameters land in models/
    # (see save_model), one checkpoint per core, mirroring its runs/ TensorBoard
    # folder. FALSE -> a single run with the selected --recurrent-core.
    p.add_argument("--sweep-all-cores", type=lambda x: x.lower() == "true", default=False,
                   help="TRUE -> run a full training run for every core "
                        "(lstm, evlstm, gru, evgru, mgu, evmgu) sequentially, saving "
                        "each core's parameters under models/. FALSE -> one run with "
                        "--recurrent-core (default).")
    p.add_argument("--encoder-hidden", type=int, default=64,
                   help="MLP encoder hidden size (used for vector observations)")
    p.add_argument("--rnn-hidden", type=int, default=128,
                   help="recurrent hidden / state size")

    # --- event-sparsity knobs (used when --recurrent-core is evlstm or evgru) ---
    p.add_argument("--event-threshold-init", type=float, default=0.4,
                   help="initial value of the trainable per-unit threshold vartheta")
    p.add_argument("--event-surrogate-width", type=float, default=0.5,
                   help="epsilon: half-width of the surrogate-gradient window")
    p.add_argument("--event-surrogate-scale", type=float, default=1.0,
                   help="gamma: peak height / dampening of the surrogate gradient")
    p.add_argument("--event-sparse-bptt", type=lambda x: x.lower() == "true", default=True,
                   help="TRUE -> windowed (triangular) surrogate => sparse BPTT (paper). "
                        "FALSE -> straight-through estimator => dense backward (ablation). "
                        "Forward activity-sparsity (thresholded events) is unaffected.")
    p.add_argument("--learn-threshold", type=lambda x: x.lower()
                   == "true", default=True)
    # --- activity-rate regularizer (drives firing toward a target rate) ---
    # Off by default (coef 0) so existing baselines are unchanged. When > 0, the
    # loss gains  coef * (mean_firing_rate - target)^2 , a set-point penalty that
    # pushes the trainable threshold up/down until activity hits `target`. Use a
    # set-point (not plain L1) so activity can't collapse to 0 and starve the policy.
    p.add_argument("--event-activity-coef", type=float, default=0.1,
                   help="lambda for the activity-rate penalty (0 disables it)")
    p.add_argument("--event-target-activity", type=float, default=0.15,
                   help="rho: target fraction of units firing per step")
    p.add_argument("--event-softplus-threshold", type=lambda x: x.lower() == "true", default=False,
                   help="TRUE -> threshold = softplus(theta_raw) >= 0, so the dead-zone "
                        "can never invert (a negative threshold forces ~100% firing)")

    args = p.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    # For recurrent PPO we mini-batch over *environments*, not flattened timesteps,
    # so that each minibatch sequence stays contiguous in time. [37-LSTM #4]
    assert args.num_envs % args.num_minibatches == 0, "num_envs must divide num_minibatches"
    args.envs_per_batch = int(args.num_envs // args.num_minibatches)
    return args


# ===========================================================================
#  Weight init helper  [37-#2: orthogonal weights, constant bias]
# ===========================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def inverse_softplus(y):
    """Raw value r such that softplus(r) == y (y > 0). Used so the softplus-
    parametrized threshold *starts* at the requested threshold_init."""
    y = float(y)
    return float(np.log(np.expm1(y))) if y > 0 else -6.0


# ===========================================================================
#  Encoders
# ===========================================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.out_dim = hidden
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x.float())


class CNNEncoder(nn.Module):
    """Nature-CNN, identical to ppo_atari_lstm.py (expects (B, C, 84, 84))."""

    def __init__(self, obs_shape):
        super().__init__()
        self.out_dim = 512
        c = obs_shape[0]
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


# ===========================================================================
#  Recurrent cores -- uniform interface
#    .initial_state(batch, device) -> tuple of state tensors, each (1, B, H)
#    .step(x_t, state) -> (out_t (B, H), new_state)
#  The leading dim of size 1 mirrors nn.LSTM's (num_layers*dirs, B, H) so the
#  done-masking in Agent.get_states is identical across cores. [37-LSTM #3]
# ===========================================================================
class LSTMCore(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(in_dim, hidden)
        # [37-LSTM #2] orthogonal weights, zero biases for the LSTM itself
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def initial_state(self, batch, device):
        return (torch.zeros(1, batch, self.hidden, device=device),   # h  [37-LSTM #1]
                torch.zeros(1, batch, self.hidden, device=device))   # c

    def step(self, x_t, state):
        out, state = self.lstm(x_t.unsqueeze(0), state)  # out: (1, B, H)
        return out.squeeze(0), state


class GRUCore(nn.Module):
    """Traditional GRU core -- the GRU analogue of LSTMCore, using cuDNN nn.GRU
    and the same orthogonal-weight / zero-bias init. The GRU has a single hidden
    state (no cell), so its state is a 1-tuple (h,). The done-masking in
    Agent.get_states maps over the state tuple, so it works unchanged. [37-LSTM #3]
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(in_dim, hidden)
        # [37-LSTM #2] orthogonal weights, zero biases -- identical to LSTMCore
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def initial_state(self, batch, device):
        # single hidden state (no cell); 1-tuple keeps the uniform interface [37-LSTM #1]
        return (torch.zeros(1, batch, self.hidden, device=device),)

    def step(self, x_t, state):
        out, h = self.gru(x_t.unsqueeze(0), state[0])   # out, h: (1, B, H)
        return out.squeeze(0), (h,)


class MGUCore(nn.Module):
    """Minimal Gated Unit (Zhou, Wu, Zhang & Zhou, "Minimal Gated Unit for
    Recurrent Neural Networks", 2016). The MGU is the GRU with its two gates --
    the reset gate r and the update gate z -- *merged into a single forget gate*
    f, which is the minimal possible gated design. The single f serves both
    roles: it gates h_{t-1} into the candidate (the reset role) and it
    interpolates between h_{t-1} and the candidate (the update role).

    Per-step equations (paper Eq. 5-7):
        f_t  = sigmoid(W_f x_t + U_f h_{t-1} + b_f)            # the only gate
        h~_t = tanh(W_h x_t + U_h (f_t * h_{t-1}) + b_h)       # candidate
        h_t  = (1 - f_t) * h_{t-1} + f_t * h~_t                # update

    There is no cuDNN MGU, so it is implemented explicitly with a single hidden
    state (no cell), so its state is a 1-tuple (h,) like GRUCore. The same
    orthogonal-weight / zero-bias init as LSTMCore/GRUCore is used, so mgu vs gru
    isolates the single-gate simplification. Like the paper (and EventGRUCore),
    the gate is applied to h_{t-1} *before* the recurrent weight, U_h (f * h),
    whereas nn.GRU applies its reset gate *after* (r * (U_h h)); this is the same
    minor convention difference already noted for evgru vs gru. [37-LSTM #2/#3]
    """

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        # [f, h~] from x (with bias); f and h~ from h (no bias), kept separate so
        # the candidate's recurrent weight reads the *gated* state (f * h).
        self.x2h = layer_init(nn.Linear(in_dim, 2 * hidden), std=1.0)
        self.h2f = nn.Linear(hidden, hidden, bias=False)     # f   from h
        self.h2c = nn.Linear(hidden, hidden, bias=False)     # h~  from (f * h)
        # [37-LSTM #2] orthogonal recurrent weights -- identical to the other cores
        nn.init.orthogonal_(self.h2f.weight, 1.0)
        nn.init.orthogonal_(self.h2c.weight, 1.0)

    def initial_state(self, batch, device):
        # single hidden state (no cell); 1-tuple keeps the uniform interface [37-LSTM #1]
        return (torch.zeros(1, batch, self.hidden, device=device),)

    def step(self, x_t, state):
        h_prev = state[0].squeeze(0)                          # (B, H)
        xf, xh = self.x2h(x_t).chunk(2, dim=-1)
        # forget gate (only gate)
        f = torch.sigmoid(xf + self.h2f(h_prev))
        # candidate (f as reset)
        h_tilde = torch.tanh(xh + self.h2c(f * h_prev))
        # update (f as coupling)
        h = (1.0 - f) * h_prev + f * h_tilde
        return h, (h.unsqueeze(0),)


class _HeavisideSurrogate(torch.autograd.Function):
    """Forward: hard Heaviside  H(v)=1[v>0]  (gives sparse events).
       Backward: surrogate gradient.
         sparse=True  -> triangular window, nonzero only on |v| < eps  (paper, sparse BPTT)
         sparse=False -> constant 'scale' everywhere   (straight-through, dense backward)
    """
    @staticmethod
    def forward(ctx, v, eps, scale, sparse):
        ctx.save_for_backward(v)
        ctx.eps = eps
        ctx.scale = scale
        ctx.sparse = sparse
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        if ctx.sparse:
            # piecewise-linear triangle: scale * max(0, 1 - |v|/eps)   (EGRU Fig.1C)
            surrogate = ctx.scale * \
                torch.clamp(1.0 - v.abs() / ctx.eps, min=0.0)
        else:
            surrogate = ctx.scale * torch.ones_like(v)
        return grad_output * surrogate, None, None, None


def heaviside_surrogate(v, eps, scale, sparse):
    return _HeavisideSurrogate.apply(v, eps, scale, sparse)


class EventLSTMCore(nn.Module):
    """Event-based LSTM: the activity-sparsity mechanism from the EGRU paper
    (Subramoney et al., 2023) grafted onto an LSTM instead of a GRU. The paper
    treats the GRU only as a case study and notes the mechanism is general; this
    keeps the same LSTM gating and weight init as the baseline LSTMCore so the
    two cores are directly comparable, and adds the three event ingredients:

        * bipolar event output    y_t = h^_t * (H(h^_t - θ) + H(-h^_t - θ))
                                   (fires on either sign; ternary state in {-1,0,+1})
        * sparse recurrence       gates read the sparse y_{t-1}, not dense h_{t-1}
        * reset-by-subtraction    c_t <- c~_t - y_t   (deplete the cell after firing)
        * surrogate-gradient BPTT (sparse backward via the windowed pseudo-derivative)

    Per-step equations (standard LSTM gating, then the event readout):
        i,f,g,o = gates(x_t, y_{t-1})
        c~_t    = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h^_t    = o_t ⊙ tanh(c~_t)                 # dense candidate output
        s_t     = H(h^_t - θ) - H(-h^_t - θ)        # balanced-ternary state in {-1,0,+1}
        y_t     = h^_t * |s_t|                       # signed graded event (bipolar firing)
        c_t     = c~_t - y_t                        # reset-by-subtraction (symmetric)

    State = (y, c). Both reset to zero on episode boundaries (done mask), exactly
    like the LSTM hidden/cell states. Note: thresholding the gated output h^_t (the
    quantity that is both emitted and fed back) and depleting the cell by it is a
    design choice -- the paper specifies the mechanism for the GRU's single state,
    not the LSTM's (c, h) pair. The reset strength is implicitly 1.0 here.
    """

    def __init__(self, in_dim, hidden, threshold_init, eps, scale, sparse, learn_threshold,
                 softplus_threshold=False):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.scale = scale
        self.sparse = sparse
        self.use_softplus = softplus_threshold
        self.last_activity = 0.0  # logged: fraction of units that fired last step
        # differentiable (B,H) firing mask, for the activity reg
        self.last_fire = None

        # Same init choices as the baseline LSTMCore (orthogonal weights, zero bias)
        # so the only difference between the cores is the sparsity mechanism. [37-LSTM #2]
        self.x2h = layer_init(nn.Linear(in_dim, 4 * hidden),
                              std=1.0)   # [i, f, g, o] from x
        # recurrent, on sparse y
        self.y2h = nn.Linear(hidden, 4 * hidden, bias=False)
        nn.init.orthogonal_(self.y2h.weight, 1.0)

        raw = inverse_softplus(
            threshold_init) if softplus_threshold else float(threshold_init)
        thr = torch.full((hidden,), raw)
        self.threshold = nn.Parameter(thr, requires_grad=learn_threshold)

    def theta(self):
        # softplus keeps the threshold (and thus the dead-zone) non-negative
        return nn.functional.softplus(self.threshold) if self.use_softplus else self.threshold

    def initial_state(self, batch, device):
        return (torch.zeros(1, batch, self.hidden, device=device),   # y (events / output)
                torch.zeros(1, batch, self.hidden, device=device))   # c (cell)

    def step(self, x_t, state):
        y_prev = state[0].squeeze(0)   # (B, H)
        c_prev = state[1].squeeze(0)   # (B, H)

        i, f, g, o = (self.x2h(x_t) + self.y2h(y_prev)).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # cell accumulation (standard LSTM)
        c = f * c_prev + i * g
        h_hat = o * torch.tanh(c)               # dense candidate output
        # Balanced-ternary (bipolar) event: a unit fires +1 above +vartheta and
        # -1 below -vartheta, and is silent in the symmetric dead-zone. Built from
        # two surrogate Heavisides so the sparse-BPTT surrogate gradient is reused
        # unchanged; the +vartheta branch alone recovers the original one-sided unit.
        theta = self.theta()
        pos = heaviside_surrogate(
            h_hat - theta, self.eps, self.scale, self.sparse)   # {0,1} fired high
        neg = heaviside_surrogate(
            -h_hat - theta, self.eps, self.scale, self.sparse)  # {0,1} fired low
        # {0,1}: fired on either polarity
        fire = pos + neg
        # signed graded event (== h_hat when firing)
        y = h_hat * fire
        # reset-by-subtraction (symmetric toward 0)
        c = c - y

        # ternary firing state s = pos - neg in {-1,0,+1}; last_activity logs the
        # fraction of units that fired (either sign), i.e. the activity-sparsity.
        # differentiable, consumed by the activity reg
        self.last_fire = fire
        self.last_activity = float(fire.detach().mean().item())
        return y, (y.unsqueeze(0), c.unsqueeze(0))


class EventGRUCore(nn.Module):
    """Event-based GRU -- the *canonical* EGRU (Subramoney et al., ICLR 2023):
    the activity-sparsity mechanism on its native GRU backbone. This is the model
    the paper actually proposes; EventLSTMCore grafts the same mechanism onto an
    LSTM. Running evgru alongside evlstm therefore isolates the recurrent backbone
    while holding the event mechanism fixed, and comparing evgru to the dense
    GRUCore isolates the sparsity.

    The event ingredients are the same as EventLSTMCore (and share the same
    surrogate-gradient function and event-sparsity knobs):
        * bipolar event output    y_t = c_t * (H(c_t - θ) + H(-c_t - θ))
                                   (fires on either sign; ternary state in {-1,0,+1})
        * sparse recurrence       gates read the sparse y_{t-1}, not a dense state
        * reset-by-subtraction    the -y_{t-1} term depletes the cell after firing
        * surrogate-gradient BPTT (shared heaviside_surrogate)

    Per-step equations (paper Eq. 1-2; u = update gate, r = reset gate,
    z = candidate, c = auxiliary internal/local state, y = communicated event):
        u_t = sigmoid(W_u [x_t, y_{t-1}])
        r_t = sigmoid(W_r [x_t, y_{t-1}])
        z_t = tanh(W_zx x_t + W_zy (r_t * y_{t-1}))          # Cho et al. reset placement
        c_t = u_t * z_t + (1 - u_t) * c_{t-1} - y_{t-1}      # update + reset-by-subtraction
        s_t = H(c_t - θ) - H(-c_t - θ)                       # balanced-ternary state {-1,0,+1}
        y_t = c_t * |s_t|                                     # signed graded event (bipolar)

    State = (y, c). Both reset to zero on episode boundaries (done mask), exactly
    like LSTMCore's (h, c). Two faithful-to-the-paper differences from
    EventLSTMCore are worth noting: (a) the threshold is applied to the internal
    state c (the GRU's single state) rather than to a gated output h_hat, and
    (b) the *previous* event y_{t-1} is subtracted inside the c update, rather than
    the current y after the fact.

    Note also that the dense GRUCore uses nn.GRU, whose candidate applies the reset
    gate *after* the recurrent weight (r * (W_hn h)); the paper (and this core)
    apply it *before* (W_zy (r * y)). So evgru vs gru differ in this minor
    convention in addition to the sparsity; the gating structure is otherwise the
    same.
    """

    def __init__(self, in_dim, hidden, threshold_init, eps, scale, sparse, learn_threshold,
                 softplus_threshold=False):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.scale = scale
        self.sparse = sparse
        self.use_softplus = softplus_threshold
        self.last_activity = 0.0  # logged: fraction of units that fired last step
        # differentiable (B,H) firing mask, for the activity reg
        self.last_fire = None

        # Same init choices as the baseline cores (orthogonal weights, zero bias)
        # so the only differences from GRUCore are the sparsity + reset placement.
        self.x2h = layer_init(nn.Linear(in_dim, 3 * hidden),
                              std=1.0)            # [u, r, z] from x
        # [u, r] from sparse y
        self.y2gate = nn.Linear(hidden, 2 * hidden, bias=False)
        self.y2cand = nn.Linear(
            hidden, hidden, bias=False)      # z from (r * y)
        nn.init.orthogonal_(self.y2gate.weight, 1.0)
        nn.init.orthogonal_(self.y2cand.weight, 1.0)

        raw = inverse_softplus(
            threshold_init) if softplus_threshold else float(threshold_init)
        thr = torch.full((hidden,), raw)
        self.threshold = nn.Parameter(thr, requires_grad=learn_threshold)

    def theta(self):
        # softplus keeps the threshold (and thus the dead-zone) non-negative
        return nn.functional.softplus(self.threshold) if self.use_softplus else self.threshold

    def initial_state(self, batch, device):
        return (torch.zeros(1, batch, self.hidden, device=device),   # y (events / output)
                torch.zeros(1, batch, self.hidden, device=device))   # c (cell)

    def step(self, x_t, state):
        # (B, H)  sparse event from previous step
        y_prev = state[0].squeeze(0)
        c_prev = state[1].squeeze(0)   # (B, H)  auxiliary internal state

        xu, xr, xz = self.x2h(x_t).chunk(3, dim=-1)
        yu, yr = self.y2gate(y_prev).chunk(2, dim=-1)

        u = torch.sigmoid(xu + yu)                          # update gate
        r = torch.sigmoid(xr + yr)                          # reset gate
        # candidate: reset gate applied to y_{t-1} *before* the recurrent weight
        z = torch.tanh(xz + self.y2cand(r * y_prev))

        # GRU update on the auxiliary state, with reset-by-subtraction of the
        # previous event (paper Eq. 2). High u -> take the new candidate.
        c = u * z + (1.0 - u) * c_prev - y_prev
        # Balanced-ternary (bipolar) event on the internal state c: fires +1 above
        # +vartheta and -1 below -vartheta, silent in the symmetric dead-zone. The
        # +vartheta branch alone recovers the canonical one-sided EGRU unit.
        theta = self.theta()
        pos = heaviside_surrogate(
            c - theta, self.eps, self.scale, self.sparse)   # {0,1} fired high
        neg = heaviside_surrogate(
            -c - theta, self.eps, self.scale, self.sparse)  # {0,1} fired low
        # {0,1}: fired either polarity
        fire = pos + neg
        y = c * fire                                        # signed graded event
        # differentiable, for activity reg
        self.last_fire = fire
        self.last_activity = float(fire.detach().mean().item())
        return y, (y.unsqueeze(0), c.unsqueeze(0))


class EventMGUCore(nn.Module):
    """Event-based MGU -- the EGRU activity-sparsity mechanism (Subramoney et al.,
    ICLR 2023) on a Minimal Gated Unit backbone (Zhou et al., 2016). This is the
    event twin of EventGRUCore: identical machinery, but the GRU's two gates
    (update u + reset r) are merged into the MGU's single forget gate f, which
    serves both roles. Running evmgu alongside evgru/evlstm isolates the
    recurrent backbone while holding the event mechanism fixed; comparing evmgu to
    the dense MGUCore isolates the sparsity.

    The event ingredients are the same as EventGRUCore (and share the same
    surrogate-gradient function and event-sparsity knobs):
        * bipolar event output    y_t = c_t * (H(c_t - θ) + H(-c_t - θ))
                                   (fires on either sign; ternary state in {-1,0,+1})
        * sparse recurrence       gates read the sparse y_{t-1}, not a dense state
        * reset-by-subtraction    the -y_{t-1} term depletes the cell after firing
        * surrogate-gradient BPTT (shared heaviside_surrogate)

    Per-step equations (MGU gating -- single forget gate f -- on the EGRU's
    auxiliary internal state c, paralleling EventGRUCore Eq. 1-2):
        f_t = sigmoid(W_f [x_t, y_{t-1}])                     # single forget gate
        z_t = tanh(W_zx x_t + W_zy (f_t * y_{t-1}))           # candidate, f as reset
        c_t = (1 - f_t) * c_{t-1} + f_t * z_t - y_{t-1}       # MGU update + reset-by-subtraction
        s_t = H(c_t - θ) - H(-c_t - θ)                        # balanced-ternary state {-1,0,+1}
        y_t = c_t * |s_t|                                      # signed graded event (bipolar)

    State = (y, c). Both reset to zero on episode boundaries (done mask). The only
    difference from EventGRUCore is the gating: f replaces both u and r, so the
    update interpolation uses (1 - f, f) -- matching MGU's h = (1-f) h + f h~ --
    and the candidate's reset gate is the same f rather than a separate r.
    """

    def __init__(self, in_dim, hidden, threshold_init, eps, scale, sparse, learn_threshold,
                 softplus_threshold=False):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.scale = scale
        self.sparse = sparse
        self.use_softplus = softplus_threshold
        self.last_activity = 0.0  # logged: fraction of units that fired last step
        # differentiable (B,H) firing mask, for the activity reg
        self.last_fire = None

        # Same init choices as the baseline cores (orthogonal weights, zero bias)
        # so the only differences from MGUCore are the sparsity + reset placement.
        self.x2h = layer_init(nn.Linear(in_dim, 2 * hidden),
                              std=1.0)            # [f, z] from x
        # f (single forget gate) from sparse y
        self.y2gate = nn.Linear(hidden, hidden, bias=False)
        self.y2cand = nn.Linear(
            hidden, hidden, bias=False)      # z from (f * y)
        nn.init.orthogonal_(self.y2gate.weight, 1.0)
        nn.init.orthogonal_(self.y2cand.weight, 1.0)

        raw = inverse_softplus(
            threshold_init) if softplus_threshold else float(threshold_init)
        thr = torch.full((hidden,), raw)
        self.threshold = nn.Parameter(thr, requires_grad=learn_threshold)

    def theta(self):
        # softplus keeps the threshold (and thus the dead-zone) non-negative
        return nn.functional.softplus(self.threshold) if self.use_softplus else self.threshold

    def initial_state(self, batch, device):
        return (torch.zeros(1, batch, self.hidden, device=device),   # y (events / output)
                torch.zeros(1, batch, self.hidden, device=device))   # c (cell)

    def step(self, x_t, state):
        # (B, H)  sparse event from previous step
        y_prev = state[0].squeeze(0)
        c_prev = state[1].squeeze(0)   # (B, H)  auxiliary internal state

        xf, xz = self.x2h(x_t).chunk(2, dim=-1)
        yf = self.y2gate(y_prev)

        # single forget gate (MGU)
        f = torch.sigmoid(xf + yf)
        # candidate: forget gate applied to y_{t-1} *before* the recurrent weight
        z = torch.tanh(xz + self.y2cand(f * y_prev))

        # MGU update on the auxiliary state -- the single gate f plays both the
        # update role (1 - f, f interpolation) and the reset role (in z above) --
        # with reset-by-subtraction of the previous event. High f -> take candidate.
        c = (1.0 - f) * c_prev + f * z - y_prev
        # Balanced-ternary (bipolar) event on the internal state c: fires +1 above
        # +vartheta and -1 below -vartheta, silent in the symmetric dead-zone. The
        # +vartheta branch alone recovers the canonical one-sided event unit.
        theta = self.theta()
        pos = heaviside_surrogate(
            c - theta, self.eps, self.scale, self.sparse)   # {0,1} fired high
        neg = heaviside_surrogate(
            -c - theta, self.eps, self.scale, self.sparse)  # {0,1} fired low
        # {0,1}: fired either polarity
        fire = pos + neg
        y = c * fire                                        # signed graded event
        # differentiable, for activity reg
        self.last_fire = fire
        self.last_activity = float(fire.detach().mean().item())
        return y, (y.unsqueeze(0), c.unsqueeze(0))


def make_core(name, in_dim, hidden, args):
    if name == "lstm":
        return LSTMCore(in_dim, hidden)
    if name == "gru":
        return GRUCore(in_dim, hidden)
    if name == "mgu":
        return MGUCore(in_dim, hidden)
    if name == "evlstm":
        return EventLSTMCore(in_dim, hidden,
                             threshold_init=args.event_threshold_init,
                             eps=args.event_surrogate_width,
                             scale=args.event_surrogate_scale,
                             sparse=args.event_sparse_bptt,
                             learn_threshold=args.learn_threshold,
                             softplus_threshold=args.event_softplus_threshold)
    if name == "evgru":
        return EventGRUCore(in_dim, hidden,
                            threshold_init=args.event_threshold_init,
                            eps=args.event_surrogate_width,
                            scale=args.event_surrogate_scale,
                            sparse=args.event_sparse_bptt,
                            learn_threshold=args.learn_threshold,
                            softplus_threshold=args.event_softplus_threshold)
    if name == "evmgu":
        return EventMGUCore(in_dim, hidden,
                            threshold_init=args.event_threshold_init,
                            eps=args.event_surrogate_width,
                            scale=args.event_surrogate_scale,
                            sparse=args.event_sparse_bptt,
                            learn_threshold=args.learn_threshold,
                            softplus_threshold=args.event_softplus_threshold)
    raise ValueError(name)


def is_event_core(name):
    """True for cores carrying the EGRU activity-sparsity machinery (trainable
    threshold + a `last_activity` readout to log)."""
    return name in ("evlstm", "evgru", "evmgu")


# ===========================================================================
#  Recurrent agent
# ===========================================================================
class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n

        if len(obs_shape) == 3:                       # image obs -> CNN (Atari path)
            self.encoder = CNNEncoder(obs_shape)
        else:                                          # vector obs -> MLP
            self.encoder = MLPEncoder(
                int(np.prod(obs_shape)), args.encoder_hidden)

        self.core = make_core(args.recurrent_core,
                              self.encoder.out_dim, args.rnn_hidden, args)
        self.actor = layer_init(
            nn.Linear(args.rnn_hidden, n_actions), std=0.01)  # [37-#?]
        self.critic = layer_init(nn.Linear(args.rnn_hidden, 1), std=1.0)
        self.mean_activity = None  # set by get_states for event cores; used by activity reg

    def initial_state(self, batch, device):
        return self.core.initial_state(batch, device)

    def get_states(self, x, state, done):
        """Run the recurrent core over a (T*B, ...) flat batch, resetting the
        hidden state to zero at episode boundaries. Identical bookkeeping to
        ppo_atari_lstm.py, generalised over the state tuple. [37-LSTM #3]"""
        hidden = self.encoder(x)                                # (T*B, feat)
        batch_size = state[0].shape[1]
        hidden = hidden.reshape(
            (-1, batch_size, hidden.shape[-1]))  # (T, B, feat)
        done = done.reshape((-1, batch_size))                        # (T, B)

        outs = []
        fires = []
        for h_t, d in zip(hidden, done):
            reset = (1.0 - d).view(1, -1, 1)
            # zero state where done=1
            state = tuple(reset * s for s in state)
            out_t, state = self.core.step(h_t, state)           # out_t: (B, H)
            outs.append(out_t)
            if getattr(self.core, "last_fire", None) is not None:
                # (B, H), differentiable
                fires.append(self.core.last_fire)
        new_hidden = torch.flatten(torch.stack(outs), 0, 1)     # (T*B, H)
        # differentiable mean firing rate over this forward pass (event cores only);
        # consumed by the activity-rate regularizer in the update loop.
        self.mean_activity = torch.stack(fires).mean() if fires else None
        return new_hidden, state

    def get_value(self, x, state, done):
        hidden, _ = self.get_states(x, state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, state, done, action=None):
        hidden, state = self.get_states(x, state, done)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), state


# ===========================================================================
#  Environment factory
# ===========================================================================
def is_atari(env_id):
    return "NoFrameskip" in env_id or env_id.startswith("ALE/")


def make_env(env_id, idx, capture_video, run_name):
    """For Atari, reproduce the exact preprocessing stack from ppo_atari_lstm.py:
    NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset -> ClipReward ->
    Resize(84) -> Grayscale -> FrameStack(4), yielding (4, 84, 84) uint8 frames.
    RecordEpisodeStatistics is applied *before* the Atari wrappers so it logs the
    true full-game score (unclipped, all lives), as the blog reports it.

    Works on both the old and new Atari stacks:
      * id 'ALE/Breakout-v5' (modern): created with frameskip=1 and
        repeat_action_probability=0.0 so it matches the blog's NoFrameskip-v4
        dynamics, then MaxAndSkip(4) supplies the frame-skipping itself.
      * id 'BreakoutNoFrameskip-v4' (legacy): used as-is.
    Gymnasium 1.0 renamed two wrappers; resolved here via getattr so the same
    code runs on gymnasium 0.28-0.29 and >=1.0.

    Install (modern): pip install "gymnasium[atari,other]" ale-py stable-baselines3
    (ROMs are bundled with ale-py >= 0.9; older stacks also need autorom.)
    """
    def thunk():
        make_kwargs = {}
        if env_id.startswith("ALE/"):
            # reproduce NoFrameskip-v4 behaviour for the blog's preprocessing
            make_kwargs = dict(frameskip=1, repeat_action_probability=0.0)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **make_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if is_atari(env_id):
            from stable_baselines3.common.atari_wrappers import (
                ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
            )
            # wrapper names that changed in gymnasium 1.0
            Grayscale = getattr(gym.wrappers, "GrayscaleObservation", None) \
                or gym.wrappers.GrayScaleObservation
            FrameStack = getattr(gym.wrappers, "FrameStackObservation", None) \
                or gym.wrappers.FrameStack

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = Grayscale(env)
            env = FrameStack(env, 4)
        return env
    return thunk


def iter_episode_stats(info):
    """Yield (episodic_return, episodic_length) for every sub-env that finished
    this step. Works across Gymnasium info schemas:
      - vector RecordEpisodeStatistics: info['episode']['r'/'l'] with mask info['_episode']
      - autoreset / per-env wrapper:    info['final_info'] -> list of per-env dicts
    """
    # schema A: batched arrays under 'episode'
    if "episode" in info and isinstance(info["episode"], dict):
        r = np.asarray(info["episode"].get("r"))
        l = np.asarray(info["episode"].get("l"))
        mask = np.asarray(info.get("_episode", np.ones_like(r, dtype=bool)))
        for ri, li in zip(r[mask], l[mask]):
            yield float(ri), float(li)
        return
    # schema B: per-env dicts under 'final_info'
    if "final_info" in info:
        for item in info["final_info"]:
            if item and "episode" in item:
                yield float(item["episode"]["r"]), float(item["episode"]["l"])


# ===========================================================================
#  Training loop  (recurrent PPO)
# ===========================================================================
# ===========================================================================
#  Checkpoint saving
# ===========================================================================
def save_model(agent, optimizer, args, global_step, run_name):
    """Persist a finished run's trained parameters to models/{run_name}.pt,
    mirroring the runs/{run_name} TensorBoard folder so each core's checkpoint
    sits beside its logs. Called at the end of every run; after a single
    --sweep-all-cores invocation the models/ folder holds one checkpoint per core
    (lstm, evlstm, gru, evgru, mgu, evmgu). The env id can contain a '/', so the
    parent directory is created before saving (e.g. models/ALE/...)."""
    model_path = f"models/{run_name}.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "recurrent_core": args.recurrent_core,
            "global_step": global_step,
            "model_state_dict": agent.state_dict(),       # the trained parameters
            "optimizer_state_dict": optimizer.state_dict(),  # for resuming / fine-tuning
            "args": vars(args),                            # full hyperparameters
        },
        model_path,
    )
    print(f"saved trained parameters -> {model_path}")


# ===========================================================================
#  Training loop  (recurrent PPO) -- one full run for args.recurrent_core
# ===========================================================================
def train(args):
    """Run a single full PPO training run for args.recurrent_core, then save the
    trained parameters to models/. Factored out of main() so --sweep-all-cores can
    call it once per core, re-seeding identically each time for a fair comparison."""
    run_name = f"{args.env_id}__{args.recurrent_core}__{args.seed}__{int(time.time())}"

    # TensorBoard: every run streams to its own folder under runs/. Launch with
    #   tensorboard --logdir runs
    # so multiple runs (e.g. lstm vs evlstm) overlay on the same charts. [37-#1]
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" +
        "\n".join(f"|{k}|{v}|" for k, v in sorted(vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, run_name)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "this template uses a Categorical policy (discrete actions)"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5)  # [37-#3 Adam eps]

    # rollout storage
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_state = agent.initial_state(
        args.num_envs, device)        # zeros [37-LSTM #1]

    num_updates = args.total_timesteps // args.batch_size
    # rolling window of recent episodic rewards
    ep_returns = deque(maxlen=100)

    for update in range(1, num_updates + 1):
        # snapshot the recurrent state at the start of the rollout; needed to
        # re-run the sequences during optimization. [37-LSTM #4]
        initial_state = tuple(s.clone() for s in next_state)

        # [37-#4 lr anneal]
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---------------- rollout ----------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state = agent.get_action_and_value(
                    next_obs, next_state, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, term, trunc, info = envs.step(
                action.cpu().numpy())
            done_np = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done_np.astype(np.float32)).to(device)

            for ep_return, ep_length in iter_episode_stats(info):
                ep_returns.append(ep_return)
                extra = (f"  act={agent.core.last_activity:.3f}"
                         if is_event_core(args.recurrent_core) else "")
                print(
                    f"global_step={global_step}  episodic_return={ep_return:.1f}{extra}")
                # --- episodic return (== summed reward per episode) ---
                writer.add_scalar("charts/episodic_return",
                                  ep_return, global_step)
                writer.add_scalar("charts/episodic_length",
                                  ep_length, global_step)

        # ---------------- GAE ----------------  [37-#5 GAE]
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, next_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # ---------------- flatten (keep T x N layout for env-wise minibatching) ----------------
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---------------- optimization ----------------  [37-LSTM #4 env-wise minibatch]
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, args.envs_per_batch):
                end = start + args.envs_per_batch
                mbenvinds = envinds[start:end]
                # contiguous-in-time per env
                mb_inds = flatinds[:, mbenvinds].ravel()

                mb_state = tuple(s[:, mbenvinds] for s in initial_state)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], mb_state, b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # [37-#7 approx KL]
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                # [37-#6 minibatch adv norm]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss (clipped surrogate)                  [37-#8]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (optionally clipped)                  [37-#9]
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -
                        args.clip_coef, args.clip_coef
                    )
                    v_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # [37-#10 entropy bonus]
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # activity-rate regularizer (event cores only): a set-point penalty
                # lambda * (mean_firing_rate - target)^2 that drives sparsity toward
                # `event_target_activity`. Gradients flow through the surrogate windows,
                # so this raises/lowers the per-unit threshold to hit the target rate.
                act_loss = torch.zeros((), device=device)
                if (args.event_activity_coef > 0.0
                        and getattr(agent, "mean_activity", None) is not None):
                    act_loss = args.event_activity_coef * \
                        (agent.mean_activity - args.event_target_activity) ** 2
                    loss = loss + act_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)  # [37-#11]
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # explained variance: how much of the return variance the value fn captures
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))

        # ---------------- TensorBoard scalars ----------------  [37-#1]
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac",
                          float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        if ep_returns:
            # smoothed metric: mean episodic return over the last <=100 episodes,
            # logged every update so the curve is dense even when episodes are long
            writer.add_scalar("charts/episodic_return_mean",
                              float(np.mean(ep_returns)), global_step)
        if is_event_core(args.recurrent_core):
            # activity sparsity = fraction of units firing on the last step seen
            writer.add_scalar("charts/event_activity",
                              agent.core.last_activity, global_step)
            # HONEST sparsity accounting: the fraction of recurrent multiply-
            # accumulates that COULD be skipped on sparse-capable hardware
            # (== 1 - firing rate). This is *theoretical*: on a dense GPU the
            # nn.Linear matmuls run at full cost regardless, so this does NOT
            # speed up the current run -- it quantifies the payoff sparse
            # hardware would give, which is the thing worth reporting.
            writer.add_scalar("sparsity/recurrent_mac_reduction",
                              1.0 - agent.core.last_activity, global_step)
            if args.event_activity_coef > 0.0:
                writer.add_scalar("losses/activity_loss",
                                  float(act_loss.item()), global_step)
            # mean per-unit threshold (after softplus, if enabled) -- handy to watch
            # the regularizer push it up toward the target rate
            writer.add_scalar("charts/event_threshold_mean",
                              float(agent.core.theta().mean().item()), global_step)

        print(f"update={update}/{num_updates}  global_step={global_step}  "
              f"v_loss={v_loss.item():.3f}  pg_loss={pg_loss.item():.3f}  "
              f"approx_kl={approx_kl.item():.4f}  SPS={sps}")

    envs.close()
    writer.close()

    # run finished: store this core's trained parameters under models/ (mirrors
    # the runs/ folder). After a --sweep-all-cores pass, models/ holds one file
    # per core.
    save_model(agent, optimizer, args, global_step, run_name)


# ===========================================================================
#  Entry point: single run, or a full sweep over every recurrent core
# ===========================================================================
def main():
    args = parse_args()
    if args.sweep_all_cores:
        # one full run per core, dense/event paired, sharing seed + hyperparameters
        # so the only thing that changes between runs is the recurrent backbone.
        for core in ["lstm", "evlstm", "gru", "evgru", "mgu", "evmgu"]:
            args.recurrent_core = core
            print(f"\n===== sweep-all-cores: training recurrent-core={core} =====")
            train(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
