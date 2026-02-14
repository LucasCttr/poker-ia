import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        # Increased capacity for 6-max poker (more expressive network)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        # recurrent layer to handle partial observability
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        """Forward pass.

        x: [B, feat]
        hx: optional (h,c) each shape [num_layers, B, hidden_size]
        Returns (logits [B,n_actions], value [B], (hn,cn))
        """
        z = self.net(x)  # [B, 512]
        # LSTM expects [B, seq_len, input_size]; we use seq_len=1 for stepwise inference
        z_seq = z.unsqueeze(1)  # [B,1,512]
        if hx is None:
            batch = x.size(0)
            h0 = z.new_zeros((1, batch, 512))
            c0 = z.new_zeros((1, batch, 512))
            hx = (h0, c0)
        out_seq, (hn, cn) = self.lstm(z_seq, hx)
        out = out_seq.squeeze(1)  # [B, 512]
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)
        return logits, value, (hn, cn)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    # `values` should be an array with the same length as `rewards` (values[t] = V(s_t)).
    # `last_value` is V(s_{T}) used for bootstrapping.
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_value = last_value if t + 1 == T else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


class PPO:
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_actions = n_actions

    def act(self, obs: np.ndarray, action_mask=None):
        # Unwrap gym/gymnasium (obs, info) and handle dict observations
        if isinstance(obs, (tuple, list)):
            obs = obs[0]
        if isinstance(obs, dict):
            # flatten dict values (ordered by key) into a vector
            obs = np.concatenate([np.asarray(v).ravel() for k, v in sorted(obs.items())])

        # Normalize incoming observation to a batched tensor
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_t = torch.as_tensor(obs_arr[None, :], dtype=torch.float32, device=self.device)
        else:
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)

        # model now returns logits, value, and new hidden state
        logits, value, new_hx = self.model(obs_t)
        # action_mask: accept list/np array/torch tensor of shape [n_actions] or [B, n_actions]
        if action_mask is not None:
            # convert to tensor on device
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            if mask_t.dim() == 1:
                # expand to batch dim
                mask_t = mask_t.unsqueeze(0)
            # ensure mask matches batch size
            if mask_t.shape[0] != logits.shape[0]:
                mask_t = mask_t.expand(logits.shape[0], -1)
            # if all actions masked for a row, fallback to allow all actions
            all_masked = ~mask_t.any(dim=-1)
            if all_masked.any():
                mask_t[all_masked] = True
            # set logits for illegal actions to a large negative value
            logits = logits.masked_fill(~mask_t, -1e10)
        # logits: [B, n_actions], value: [B]
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)

        # return scalars for single-step use
        # return new hidden state so caller can carry it across timesteps
        return action.item(), logp.item(), value.squeeze(0).item(), new_hx

    def act_batch(self, states: np.ndarray, action_masks: Optional[np.ndarray] = None, hx=None, deterministic: bool = False):
        """Vectorized action selection for a batch of states.

        states: np.ndarray shape [B, feat]
        action_masks: optional bool array [B, n_actions]
        hx: optional tuple (h,c) tensors on device with shapes [num_layers, B, hidden]
        Returns: actions (B,), logps (B,), values (B,), new_hx (h,c)
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if hx is not None:
            # assume hx provided as numpy arrays or tensors already on device
            if isinstance(hx, tuple):
                h, c = hx
                if not isinstance(h, torch.Tensor):
                    h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
                    c = torch.as_tensor(c, dtype=torch.float32, device=self.device)
                hx_t = (h, c)
            else:
                hx_t = hx
        else:
            hx_t = None

        logits, values, new_hx = self.model(states_t, hx=hx_t)

        if action_masks is not None:
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
            if mask_t.shape[0] != logits.shape[0]:
                mask_t = mask_t.expand(logits.shape[0], -1)
            all_masked = ~mask_t.any(dim=-1)
            if all_masked.any():
                mask_t[all_masked] = True
            logits = logits.masked_fill(~mask_t, -1e10)

        probs = F.softmax(logits, dim=-1)
        if deterministic:
            actions = probs.argmax(dim=-1)
            # compute log prob of chosen actions
            logps = torch.log(torch.clamp(probs[range(probs.size(0)), actions], min=1e-12))
        else:
            dist = Categorical(probs)
            actions = dist.sample()
            logps = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            logps.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            (new_hx[0].detach().cpu().numpy(), new_hx[1].detach().cpu().numpy()),
        )

    def update(self, batch, epochs=4, minibatch_size=64):
        # batch: dict with keys states, actions, old_log_probs, returns, advantages
        states = torch.as_tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(batch['old_log_probs'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        # optional action masks
        action_masks = None
        if 'action_masks' in batch and batch['action_masks'] is not None:
            action_masks = torch.as_tensor(batch['action_masks'], dtype=torch.bool, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        for _ in range(epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                logits, values, _ = self.model(mb_states)
                # apply action mask if available
                if action_masks is not None:
                    mb_mask = action_masks[mb_idx]
                    if mb_mask.dim() == 1:
                        mb_mask = mb_mask.unsqueeze(0)
                    if mb_mask.shape[0] != logits.shape[0]:
                        mb_mask = mb_mask.expand(logits.shape[0], -1)
                    all_masked = ~mb_mask.any(dim=-1)
                    if all_masked.any():
                        mb_mask[all_masked] = True
                    logits = logits.masked_fill(~mb_mask, -1e10)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        return
