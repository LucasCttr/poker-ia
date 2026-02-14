"""Vectorized self-play training using AsyncVectorEnv and shared PPO policy.

Runs multiple `PokerGymEnv` instances in parallel (processes) and collects transitions
from the acting player in each env. Pools transitions across envs and updates the shared PPO.
"""
import argparse
import time
from collections import defaultdict, deque

import numpy as np
from gymnasium.vector import AsyncVectorEnv

from ppo_agent import PPO
from poker_env import PokerGymEnv


def make_env(seed):
    def _thunk():
        env = PokerGymEnv()
        env.seed(seed)
        return env
    return _thunk


def evaluate_policy(agent: PPO, n_eval: int = 100, n_envs: int = 8):
    """Run deterministic evaluation over `n_eval` hands using `n_envs` parallel envs.
    Returns dict with overall mean reward and per-seat means.
    """
    from gymnasium.vector import AsyncVectorEnv

    def make_env_inner(seed):
        def _thunk():
            e = PokerGymEnv()
            e.seed(seed)
            return e
        return _thunk

    env_fns = [make_env_inner(9000 + i) for i in range(n_envs)]
    vec = AsyncVectorEnv(env_fns)

    obs, infos = vec.reset()
    total_rewards = []
    per_seat = []
    hands = 0
    # run until we collect n_eval completed hands
    while hands < n_eval:
        if isinstance(obs, dict):
            states_batch = obs['state']
            masks_batch = obs.get('action_mask')
        else:
            states_batch = np.asarray(obs, dtype=np.float32)
            masks_batch = np.ones((n_envs, agent.n_actions), dtype=bool)

        actions, logps, vals, _ = agent.act_batch(states_batch, action_masks=masks_batch, hx=None, deterministic=True)
        res = vec.step(actions)
        if len(res) == 5:
            next_obs, rewards, terminated, truncated, infos = res
            dones = np.logical_or(terminated, truncated)
        else:
            next_obs, rewards, dones, infos = res

        for i in range(len(dones)):
            if dones[i]:
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                final = info_i.get('final_stacks', None)
                start = info_i.get('starting_stacks', None)
                if final is not None and start is not None:
                    # compute reward for seat that acted in that hand: aggregate as mean over seats
                    seat_rewards = [ (final[j] - start[j]) / vec.envs[0].big_blind if hasattr(vec, 'envs') and vec.envs else (final[j] - start[j]) / PokerGymEnv().big_blind for j in range(len(final)) ]
                    total_rewards.extend(seat_rewards)
                    per_seat.append(np.mean(seat_rewards))
                    hands += 1
        obs = next_obs

    vec.close()
    return { 'mean': float(np.mean(total_rewards)) if total_rewards else 0.0, 'per_seat_mean': [float(x) for x in np.mean(per_seat) * np.ones(agent.n_actions) ] }


def train_vectorized(n_envs=8, batch_size=4096, epochs=100):
    env_fns = [make_env(1000 + i) for i in range(n_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    agent = PPO(input_dim=422, n_actions=7)

    # initialize vectorized env
    obs, infos = vec_env.reset()
    # obs is a dict with keys 'state' shaped (n_envs,422) and 'action_mask' (n_envs, n_actions)

    # recurrent hidden states per env
    h = np.zeros((1, n_envs, 512), dtype=np.float32)
    c = np.zeros((1, n_envs, 512), dtype=np.float32)
    hx = (h, c)

    reward_hist = deque(maxlen=100)

    for epoch in range(epochs):
        pooled_states = []
        pooled_actions = []
        pooled_logp = []
        pooled_rewards = []
        pooled_masks = []
        pooled_values = []

        # trackers for per-env per-hand transitions
        per_hand_trans = [defaultdict(list) for _ in range(n_envs)]
        hand_starts = [None] * n_envs

        steps_collected = 0
        while steps_collected < batch_size:
            # prepare batch inputs from obs
            if isinstance(obs, dict):
                states_batch = obs['state']
                masks_batch = obs.get('action_mask')
            else:
                # fallback: obs is array
                states_batch = np.asarray(obs, dtype=np.float32)
                masks_batch = np.ones((n_envs, agent.n_actions), dtype=bool)

            # vectorized action selection
            actions, logps, values, new_hx = agent.act_batch(states_batch, action_masks=masks_batch, hx=hx)

            # step envs with actions
            res = vec_env.step(actions)
            # res: (next_obs, rewards, terminated, truncated, infos)
            if len(res) == 5:
                next_obs, rewards, terminated, truncated, infos = res
                dones = np.logical_or(terminated, truncated)
            else:
                next_obs, rewards, dones, infos = res

            # record transitions per env
            def _info_at(infos_obj, idx):
                if isinstance(infos_obj, (list, tuple)):
                    return infos_obj[idx]
                if isinstance(infos_obj, dict):
                    # some vector API variants return a dict keyed by index
                    if idx in infos_obj:
                        return infos_obj[idx]
                    return infos_obj
                return {}

            for i in range(n_envs):
                info_i = _info_at(infos, i)
                pid = info_i.get('player_id', None)
                if pid is not None:
                    try:
                        pid = int(pid)
                    except Exception:
                        pid = None
                # store hand starting stacks
                if hand_starts[i] is None:
                    info_i = _info_at(infos, i)
                    hand_starts[i] = info_i.get('starting_stacks', None)

                pooled_states.append(states_batch[i].astype(np.float32))
                pooled_actions.append(int(actions[i]))
                pooled_logp.append(float(logps[i]))
                pooled_values.append(float(values[i]))
                pooled_masks.append(masks_batch[i].astype(bool))
                # temporary reward placeholder 0; fill at hand end
                pooled_rewards.append(0.0)

                if pid is not None:
                    per_hand_trans[i][pid].append(len(pooled_rewards) - 1)

                # if done, compute final rewards for that env
                if dones[i]:
                    info = _info_at(infos, i)
                    final_stacks = info.get('final_stacks', None)
                    if final_stacks is not None and hand_starts[i] is not None:
                        for pid_k, idx_list in per_hand_trans[i].items():
                            start = hand_starts[i][pid_k]
                            final = final_stacks[pid_k]
                            reward_val = (final - start) / vec_env.envs[i].big_blind
                            for idx in idx_list:
                                pooled_rewards[idx] = reward_val
                    per_hand_trans[i].clear()
                    hand_starts[i] = None

            steps_collected = len(pooled_states)
            obs = next_obs
            # update recurrent states
            h, c = new_hx
            hx = (h, c)

        # prepare batch arrays
        batch = {
            'states': np.asarray(pooled_states, dtype=np.float32),
            'actions': np.asarray(pooled_actions, dtype=np.int64),
            'old_log_probs': np.asarray(pooled_logp, dtype=np.float32),
            'advantages': np.asarray(pooled_rewards, dtype=np.float32),
            'returns': np.asarray(pooled_rewards, dtype=np.float32),
            'action_masks': np.asarray(pooled_masks, dtype=bool),
        }

        agent.update(batch, epochs=4, minibatch_size=256)

        avg_ret = np.mean(batch['returns'])
        reward_hist.append(avg_ret)
        print(f'Epoch {epoch:4d} | steps {len(pooled_states)} | avg_return={avg_ret:.3f} | avg100={np.mean(reward_hist):.3f}')

        # periodic evaluation: run small deterministic eval and log per-seat rewards
        if (epoch + 1) % 1 == 0:
            eval_stats = evaluate_policy(agent, n_eval=32, n_envs=4)
            print(f"Eval: mean_reward={eval_stats['mean']:.3f} | per_seat_mean={eval_stats['per_seat_mean']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    train_vectorized(n_envs=args.n_envs, batch_size=args.batch_size, epochs=args.epochs)
    train_vectorized(n_envs=args.n_envs, batch_size=args.batch_size, epochs=args.epochs)
