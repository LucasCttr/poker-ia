import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
obs = env.reset()
print('reset raw:', type(obs))
if isinstance(obs, (tuple, list)):
    obs = obs[0]
print('after unwrap:', type(obs), getattr(obs, 'shape', None))

for i in range(50):
    a = env.action_space.sample()
    step = env.step(a)
    print(f'step {i}: step type {type(step)}, len={len(step)}')
    if len(step) == 5:
        next_obs, reward, terminated, truncated, info = step
        done = bool(terminated or truncated)
    else:
        next_obs, reward, done, info = step
    print(f'  next_obs type {type(next_obs)}, shape {np.asarray(next_obs).shape if not isinstance(next_obs, (tuple, list, dict)) else "composite"}')
    if isinstance(next_obs, (tuple, list)):
        print('  next_obs is tuple/list; elements types:', [type(x) for x in next_obs])
    obs = next_obs
print('done')
