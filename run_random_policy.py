import numpy as np
from custom_fanet_sim import CustomFANETSim

EPISODES = 5
STEPS = 100

env = CustomFANETSim()
(action_low, action_high), (obs_low, obs_high) = env.get_spaces()
n_agents = env.get_num_agents()

def sample_action():
    return np.random.uniform(action_low, action_high)

for ep in range(EPISODES):
    obs = env.reset()
    total_reward = 0
    for step in range(STEPS):
        actions = sample_action()
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        if done:
            break
    print(f"Episode {ep+1}: Total reward = {total_reward:.2f}, Info = {info}") 