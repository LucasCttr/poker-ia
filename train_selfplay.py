"""Simple self-play training scaffold using PokerGymEnv and shared PPO policy.

This is a minimal prototype: the poker env is a simplified simulator (see poker_env.py).
It runs one table at a time and collects per-player transitions until a batch of steps
is gathered, then performs a PPO update on the pooled experiences.

This is intentionally simple to demonstrate self-play integration; for production
training you'd want vectorized tables, opponent pools, checkpoints, and
sequence-aware recurrent updates.
"""
import argparse
import time
from collections import defaultdict, deque

import numpy as np

from ppo_agent import PPO
from poker_env import PokerGymEnv


def train_selfplay(env: PokerGymEnv, agent: PPO, batch_size: int = 4096, epochs: int = 1000):
    # storage for pooled experiences
    states = []
    actions = []
    old_log_probs = []
    rewards = []
    dones = []
    masks = []

    reward_hist = deque(maxlen=100)

    for epoch in range(epochs):
        obs, info = env.reset()
        if isinstance(obs, tuple) or isinstance(obs, list):
            obs = obs[0]
        current_player = info.get('player_id', 0)
        # reset per-hand trackers
        hand_start_stacks = env.starting_stacks
        per_hand_transitions = defaultdict(list)  # player_id -> list of indices into pooled lists

        while len(states) < batch_size:
            # obs is a dict with 'state' and 'action_mask'
            action_mask = obs.get('action_mask') if isinstance(obs, dict) else None
            st_vec = obs['state'] if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
            a, logp, val, hx = agent.act(st_vec, action_mask=action_mask)

            res = env.step(a)
            # gymnasium style
            if len(res) == 5:
                next_obs, r, terminated, truncated, info = res
                done = bool(terminated or truncated)
            else:
                next_obs, r, done, info = res

            # record transition for this acting player
            idx = len(states)
            states.append(st_vec)
            actions.append(a)
            old_log_probs.append(logp)
            rewards.append(0.0)  # intermediate rewards are 0; final reward will be set at hand end
            dones.append(float(done))
            masks.append(action_mask if action_mask is not None else np.ones(agent.n_actions, dtype=bool))
            per_hand_transitions[current_player].append(idx)

            if done:
                # hand finished; compute per-player final reward in BB units
                final_stacks = info.get('final_stacks', env.stacks)
                for pid, idx_list in per_hand_transitions.items():
                    start = hand_start_stacks[pid]
                    final = final_stacks[pid]
                    # reward normalized by big blind
                    reward_val = (final - start) / env.big_blind
                    # set the reward for all transitions of this player in the hand
                    for i in idx_list:
                        rewards[i] = reward_val
                per_hand_transitions.clear()
                # new hand
                obs, info = env.reset()
                current_player = info.get('player_id', 0)
                hand_start_stacks = env.starting_stacks
                continue

            # move to next acting player
            if isinstance(next_obs, tuple) or isinstance(next_obs, list):
                next_obs = next_obs[0]
            obs = next_obs
            current_player = info.get('player_id', current_player + 1)

        # prepare batch
        batch = {
            'states': np.asarray(states, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.int64),
            'old_log_probs': np.asarray(old_log_probs, dtype=np.float32),
            'advantages': np.asarray(rewards, dtype=np.float32),  # simple advantage = reward - V omitted
            'returns': np.asarray(rewards, dtype=np.float32),
            'action_masks': np.asarray(masks, dtype=bool),
        }

        agent.update(batch, epochs=4, minibatch_size=256)

        avg_reward = np.mean(batch['returns'])
        reward_hist.append(avg_reward)
        print(f'Epoch {epoch:4d} | pooled_steps {len(states)} | avg_return={avg_reward:.3f} | avg100={np.mean(reward_hist):.3f}')

        # clear pooled lists
        states.clear(); actions.clear(); old_log_probs.clear(); rewards.clear(); dones.clear(); masks.clear()

        # periodic evaluation
        if (epoch + 1) % 1 == 0:
            stats = evaluate_policy(agent, n_eval=32, n_envs=4)
            print(f"Eval mean_reward={stats['mean']:.3f}")


def evaluate_policy(agent: PPO, n_eval: int = 100, n_envs: int = 8):
    from gymnasium.vector import AsyncVectorEnv

    def make_env(seed):
        def _thunk():
            e = PokerGymEnv()
            e.seed(seed)
            return e
        return _thunk

    env_fns = [make_env(7000 + i) for i in range(n_envs)]
    vec = AsyncVectorEnv(env_fns)
    obs, infos = vec.reset()
    collected = 0
    rewards_acc = []
    while collected < n_eval:
        if isinstance(obs, dict):
            states_batch = obs['state']
            masks_batch = obs.get('action_mask')
        else:
            states_batch = np.asarray(obs, dtype=np.float32)
            masks_batch = np.ones((n_envs, agent.n_actions), dtype=bool)

        actions, logps, vals, _ = agent.act_batch(states_batch, action_masks=masks_batch, hx=None, deterministic=True)
        res = vec.step(actions)
        if len(res) == 5:
            next_obs, rews, terminated, truncated, infos = res
            dones = np.logical_or(terminated, truncated)
        else:
            next_obs, rews, dones, infos = res

        for i in range(len(dones)):
            if dones[i]:
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                final = info_i.get('final_stacks', None)
                start = info_i.get('starting_stacks', None)
                if final is not None and start is not None:
                    # mean reward per seat for that hand
                    seat_rewards = [ (final[j] - start[j]) / env.big_blind for j in range(len(final)) ]
                    rewards_acc.extend(seat_rewards)
                    collected += 1
        obs = next_obs

    vec.close()
    return { 'mean': float(np.mean(rewards_acc)) if rewards_acc else 0.0 }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    args = parser.parse_args()

    env = PokerGymEnv()
    agent = PPO(input_dim=422, n_actions=7)
    train_selfplay(env, agent, batch_size=args.batch_size, epochs=args.epochs)
