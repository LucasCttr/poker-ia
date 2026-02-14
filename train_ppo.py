"""Minimal PPO training scaffold.

This script demonstrates how to use `ppo_agent.PPO` with a Gym environment.
It is environment-agnostic and collects on-policy rollouts until a batch
of `batch_size` steps is gathered, then performs PPO updates.

Usage examples:
  python train_ppo.py --env CartPole-v1

For your poker environment pass the gym-compatible env id or replace the
collection loop to integrate your multi-player table logic.
"""
import argparse
import time
from collections import deque

import numpy as np
import torch

from ppo_agent import PPO, compute_gae


def collect_rollout(env, agent: PPO, batch_size: int):
    states = []
    actions = []
    old_log_probs = []
    rewards = []
    dones = []
    values = []

    reset_res = env.reset()
    # gymnasium returns (obs, info) while older gym returns obs directly
    if isinstance(reset_res, tuple) or isinstance(reset_res, list):
        obs, info = reset_res
    else:
        obs = reset_res
        info = {}
    if isinstance(obs, tuple) or isinstance(obs, list):
        obs = obs[0]
    ep_len = 0
    masks = []
    # maintain recurrent hidden state across steps in the same episode
    hx = None
    while len(states) < batch_size:
        # extract action mask from obs or info if provided by the environment
        action_mask = None
        if isinstance(obs, dict) and 'action_mask' in obs:
            action_mask = np.asarray(obs['action_mask'], dtype=bool)
        elif isinstance(info, dict) and 'action_mask' in info:
            action_mask = np.asarray(info['action_mask'], dtype=bool)
        else:
            # default: all actions allowed
            action_mask = np.ones(agent.n_actions, dtype=bool)

        action, logp, value, hx = agent.act(obs, action_mask=action_mask)
        step_res = env.step(action)
        # gymnasium.step returns (obs, reward, terminated, truncated, info)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_res

        # unwrap if gymnasium returns (obs, info)
        if isinstance(next_obs, tuple) or isinstance(next_obs, list):
            next_obs = next_obs[0]

        # flatten obs (if dict) and store numpy arrays to keep consistent shapes
        if isinstance(obs, (tuple, list)):
            obs_to_store = obs[0]
        else:
            obs_to_store = obs
        if isinstance(obs_to_store, dict):
            parts = [np.asarray(obs_to_store[k]).ravel() for k in sorted(obs_to_store.keys())]
            obs_arr = np.concatenate(parts).astype(np.float32)
        else:
            obs_arr = np.asarray(obs_to_store, dtype=np.float32)
        states.append(obs_arr)
        masks.append(np.asarray(action_mask, dtype=bool))
        actions.append(action)
        old_log_probs.append(logp)
        rewards.append(reward)
        # store a shallow copy of current hidden state (cpu numpy) if needed later
        # we do not currently use hidden states during update; proper recurrent
        # training requires collecting sequences and hidden states per sequence.
        dones.append(float(done))
        values.append(value)

        obs = next_obs
        ep_len += 1
        if done:
            obs = env.reset()
            if isinstance(obs, tuple) or isinstance(obs, list):
                obs = obs[0]
            ep_len = 0
            hx = None

    # Add final bootstrap value for last observation
    with torch.no_grad():
        import torch as _torch

        obs_t = np.asarray(obs, dtype=np.float32)
        if obs_t.ndim == 1:
            obs_t = _torch.as_tensor(obs_t[None, :], dtype=_torch.float32)
        else:
            obs_t = _torch.as_tensor(obs_t, dtype=_torch.float32)
        _, last_value_t = agent.model(obs_t)
        last_value = float(last_value_t.squeeze().item())

    values = np.array(values, dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, last_value)

    # Flatten and stack states to ensure a 2D array [N, feat]
    states_np = np.vstack([s.ravel() for s in states])
    masks_np = np.vstack([m.ravel() for m in masks]) if len(masks) > 0 else None

    batch = {
        'states': states_np,
        'actions': np.asarray(actions, dtype=np.int64),
        'old_log_probs': np.asarray(old_log_probs, dtype=np.float32),
        'advantages': advantages,
        'returns': returns,
        'action_masks': masks_np,
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_actions', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # Prefer gymnasium (maintained) but fall back to gym if necessary
    try:
        import gymnasium as gym
    except Exception:
        import gym

    env = gym.make(args.env)
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise RuntimeError('Environment observation shape is None')
    input_dim = int(np.prod(obs_shape))

    agent = PPO(input_dim=input_dim, n_actions=args.n_actions, lr=args.lr)

    print('Starting training on', args.env)
    reward_hist = deque(maxlen=100)
    for epoch in range(args.epochs):
        start = time.time()
        batch = collect_rollout(env, agent, args.batch_size)
        agent.update(batch, epochs=4, minibatch_size=256)

        # quick evaluation: run a few episodes
        eval_rewards = []
        for _ in range(3):
            o = env.reset()
            if isinstance(o, tuple) or isinstance(o, list):
                o = o[0]
            rsum = 0.0
            done = False
            while not done:
                a, _, _ = agent.act(o)
                step_res = env.step(a)
                if len(step_res) == 5:
                    o, r, terminated, truncated, _ = step_res
                    done = bool(terminated or truncated)
                else:
                    o, r, done, _ = step_res
                if isinstance(o, tuple) or isinstance(o, list):
                    o = o[0]
                rsum += r
            eval_rewards.append(rsum)
        reward_hist.append(np.mean(eval_rewards))

        print(f'Epoch {epoch:4d} | batch_steps {args.batch_size} | eval_reward={np.mean(eval_rewards):.2f} | avg100={np.mean(reward_hist):.2f} | time={time.time()-start:.1f}s')


if __name__ == '__main__':
    main()
