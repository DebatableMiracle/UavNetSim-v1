import numpy as np
from custom_fanet_env import CustomFANETEnv
import matplotlib.pyplot as plt

# ===== Constants =====
EPISODES = 200
STEPS = 100
STATE_BINS = 5
ACTION_BINS = 3
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.99

# Discretization helpers
def discretize(val, low, high, bins):
    return np.clip(((val - low) / (high - low) * bins).astype(int), 0, bins - 1)

def get_state(env, uav_idx):
    # Use UAV pos, energy, nearest node buffer
    uav_pos = env.uav_pos[uav_idx]
    uav_energy = env.uav_energy[uav_idx]
    dists = np.linalg.norm(env.node_pos - uav_pos, axis=1)
    nearest = np.argmin(dists)
    node_buffer = env.node_buffer[nearest]
    s = np.concatenate([uav_pos, [uav_energy, node_buffer]])
    return tuple(discretize(s, np.array([0,0,0,0,0]), np.array([20,20,10,100,5]), STATE_BINS))

def get_action(a_idx):
    # Map discrete action idx to (dx,dy,dz)
    grid = np.linspace(-1, 1, ACTION_BINS)
    dz = a_idx % ACTION_BINS
    dy = (a_idx // ACTION_BINS) % ACTION_BINS
    dx = (a_idx // (ACTION_BINS**2)) % ACTION_BINS
    return np.array([grid[dx], grid[dy], grid[dz]])

# Q-tables for each UAV
Q_tables = [np.zeros((STATE_BINS,STATE_BINS,STATE_BINS,STATE_BINS,STATE_BINS,ACTION_BINS**3)) for _ in range(4)]

env = CustomFANETEnv()
rewards_per_ep = []
info_lost_per_ep = []
energy_used_per_ep = []
for ep in range(EPISODES):
    env.reset()
    total_reward = 0
    total_info_lost = 0
    total_energy_used = 0
    for step in range(STEPS):
        actions = []
        states = []
        action_idxs = []
        for u in range(4):
            s = get_state(env, u)
            states.append(s)
            if np.random.rand() < EPSILON:
                a_idx = np.random.randint(ACTION_BINS**3)
            else:
                a_idx = np.argmax(Q_tables[u][s])
            action_idxs.append(a_idx)
            actions.append(get_action(a_idx))
        obs, reward, done, info = env.step(np.array(actions))
        for u in range(4):
            s = states[u]
            s_ = get_state(env, u)
            a = action_idxs[u]
            r = reward / 4.0  # split reward
            Q = Q_tables[u]
            Q[s][a] += ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])
        total_reward += reward
        total_info_lost += info['info_lost']
        total_energy_used += info['total_energy_used']
        if done:
            break
    rewards_per_ep.append(total_reward)
    info_lost_per_ep.append(total_info_lost)
    energy_used_per_ep.append(total_energy_used)
    if (ep+1) % 10 == 0:
        print(f"Episode {ep+1}")

# Plotting
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(rewards_per_ep)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.subplot(1,3,2)
plt.plot(info_lost_per_ep)
plt.title('Info Lost per Episode')
plt.xlabel('Episode')
plt.ylabel('Info Lost')
plt.subplot(1,3,3)
plt.plot(energy_used_per_ep)
plt.title('Energy Used per Episode')
plt.xlabel('Episode')
plt.ylabel('Energy Used')
plt.tight_layout()
plt.show() 