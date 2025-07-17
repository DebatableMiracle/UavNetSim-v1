import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from custom_fanet_env import CustomFANETEnv

# ===== Hyperparameters =====
EPISODES = 200
STEPS = 100
STATE_DIM = 5 
ACTION_DIM = 3 
N_AGENTS = 4
ACT_LIMIT = 1.0
REPLAY_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2 
LR = 3e-4

# ===== Replay Buffer =====
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# ===== Neural Networks =====
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, act_limit):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.act_limit = act_limit
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std
    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.act_limit
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q = nn.Linear(128, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# ===== MASAC Agent =====
class MASACAgent:
    def __init__(self, state_dim, action_dim, act_limit):
        self.actor = Actor(state_dim, action_dim, act_limit)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR)
        self.update_targets(tau=1.0)
    def update_targets(self, tau=1.0):
        for target, source in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target.data.copy_(tau * source.data + (1 - tau) * target.data)
        for target, source in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target.data.copy_(tau * source.data + (1 - tau) * target.data)

# ===== Utils =====
def get_state(env, uav_idx):
    uav_pos = env.uav_pos[uav_idx]
    uav_energy = env.uav_energy[uav_idx]
    dists = np.linalg.norm(env.node_pos - uav_pos, axis=1)
    nearest = np.argmin(dists)
    node_buffer = env.node_buffer[nearest]
    s = np.concatenate([uav_pos, [uav_energy, node_buffer]])
    return s.astype(np.float32)

def to_tensor(x):
    return torch.FloatTensor(np.array(x))

# ===== Main Training Loop =====
replay_buffer = ReplayBuffer(REPLAY_SIZE)
agents = [MASACAgent(STATE_DIM, ACTION_DIM, ACT_LIMIT) for _ in range(N_AGENTS)]
env = CustomFANETEnv()
rewards_per_ep = []
info_lost_per_ep = []
energy_used_per_ep = []

for ep in range(EPISODES):
    env.reset()
    episode_reward = 0
    episode_info_lost = 0
    episode_energy_used = 0
    states = [get_state(env, u) for u in range(N_AGENTS)]
    for step in range(STEPS):
        actions = []
        for u in range(N_AGENTS):
            state_tensor = to_tensor(states[u]).unsqueeze(0)
            with torch.no_grad():
                action, _ = agents[u].actor.sample(state_tensor)
            actions.append(action.squeeze(0).numpy())
        next_obs, reward, done, info = env.step(np.array(actions))
        next_states = [get_state(env, u) for u in range(N_AGENTS)]
        for u in range(N_AGENTS):
            replay_buffer.push(states[u], actions[u], reward / N_AGENTS, next_states[u], float(done))
        states = next_states
        episode_reward += reward
        episode_info_lost += info['info_lost']
        episode_energy_used += info['total_energy_used']
        if len(replay_buffer) > BATCH_SIZE:
            for u in range(N_AGENTS):
                # Sample batch
                batch = replay_buffer.sample(BATCH_SIZE)
                state = to_tensor(batch.state)
                action = to_tensor(batch.action)
                reward_b = to_tensor(batch.reward).unsqueeze(1)
                next_state = to_tensor(batch.next_state)
                done_b = to_tensor(batch.done).unsqueeze(1)
                # Critic update
                with torch.no_grad():
                    next_action, next_log_prob = agents[u].actor.sample(next_state)
                    target_q1 = agents[u].target_critic1(next_state, next_action)
                    target_q2 = agents[u].target_critic2(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob
                    target = reward_b + GAMMA * (1 - done_b) * target_q
                current_q1 = agents[u].critic1(state, action)
                current_q2 = agents[u].critic2(state, action)
                critic1_loss = F.mse_loss(current_q1, target)
                critic2_loss = F.mse_loss(current_q2, target)
                agents[u].critic1_optimizer.zero_grad()
                critic1_loss.backward()
                agents[u].critic1_optimizer.step()
                agents[u].critic2_optimizer.zero_grad()
                critic2_loss.backward()
                agents[u].critic2_optimizer.step()
                # Actor update
                new_action, log_prob = agents[u].actor.sample(state)
                q1_new = agents[u].critic1(state, new_action)
                q2_new = agents[u].critic2(state, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (ALPHA * log_prob - q_new).mean()
                agents[u].actor_optimizer.zero_grad()
                actor_loss.backward()
                agents[u].actor_optimizer.step()
                # Target update
                agents[u].update_targets(tau=TAU)
        if done:
            break
    rewards_per_ep.append(episode_reward)
    info_lost_per_ep.append(episode_info_lost)
    energy_used_per_ep.append(episode_energy_used)
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