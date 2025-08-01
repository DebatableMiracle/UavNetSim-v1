# SAC (Soft Actor-Critic) Algorithm Implementation

## Overview

This implementation provides a complete Soft Actor-Critic (SAC) algorithm for reinforcement learning. SAC is an off-policy actor-critic deep RL algorithm that uses entropy regularization to encourage exploration and improve sample efficiency.

## Algorithm Components

### 1. **Actor Network (Policy)**
- **Purpose**: Outputs action distribution for continuous control
- **Architecture**: Neural network that outputs mean and log standard deviation
- **Output**: Gaussian distribution parameters (μ, σ)
- **Sampling**: Uses reparameterization trick for differentiable sampling

```python
class ActorNetwork(nn.Module):
    def forward(self, state) -> (mean, logstd)
    def sample(self, state) -> (action, log_prob)
```

### 2. **Critic Networks (Q-functions)**
- **Purpose**: Estimate state-action values (Q-values)
- **Architecture**: Two separate networks for stability
- **Input**: State-action pairs
- **Output**: Q-value estimates

```python
class CriticNetwork(nn.Module):
    def forward(self, state, action) -> q_value
```

### 3. **Value Network**
- **Purpose**: Estimate state values (V-values)
- **Architecture**: Neural network for state value estimation
- **Input**: States only
- **Output**: State value estimates

```python
class ValueNetwork(nn.Module):
    def forward(self, state) -> value
```

### 4. **Target Networks**
- **Purpose**: Provide stable targets for training
- **Update**: Soft updates using parameter τ
- **Networks**: Target critic and value networks

## SAC Algorithm Steps

### 1. **Value Network Update**
```python
# Current value estimate
current_values = self.value(states)

# Next actions and log probs from actor
next_actions, next_log_probs = self.actor.sample(next_states)

# Q-values for next state-action pairs
next_q1 = self.target_critic1(next_states, next_actions)
next_q2 = self.target_critic2(next_states, next_actions)
next_q = torch.min(next_q1, next_q2)

# Target values with entropy regularization
alpha = self.log_alpha.exp()  # Temperature parameter
target_values = next_q - alpha * next_log_probs
target_values = rewards + gamma * target_values * (1 - dones)

# Value loss
value_loss = F.mse_loss(current_values, target_values.detach())
```

### 2. **Critic Networks Update**
```python
# Current Q-values
current_q1 = self.critic1(states, actions)
current_q2 = self.critic2(states, actions)

# Next Q-values with entropy regularization
next_actions, next_log_probs = self.actor.sample(next_states)
next_q1 = self.target_critic1(next_states, next_actions)
next_q2 = self.target_critic2(next_states, next_actions)
next_q = torch.min(next_q1, next_q2)

# Target Q-values
target_q = next_q - alpha * next_log_probs
target_q = rewards + gamma * target_q * (1 - dones)

# Critic losses
critic1_loss = F.mse_loss(current_q1, target_q.detach())
critic2_loss = F.mse_loss(current_q2, target_q.detach())
critic_loss = critic1_loss + critic2_loss
```

### 3. **Actor Network Update**
```python
# Actions and log probs from current policy
actions, log_probs = self.actor.sample(states)

# Q-values for current state-action pairs
q1 = self.critic1(states, actions)
q2 = self.critic2(states, actions)
q = torch.min(q1, q2)

# Actor loss with entropy regularization
actor_loss = alpha * log_probs - q
actor_loss = actor_loss.mean()
```

### 4. **Temperature Parameter Update (Auto-α)**
```python
# Get actions and log probs from current policy
actions, log_probs = self.actor.sample(states)

# Alpha loss for automatic temperature adjustment
alpha = self.log_alpha.exp()
alpha_loss = alpha * (-log_probs - target_entropy)
alpha_loss = alpha_loss.mean()
```

## Key Features

### 1. **Entropy Regularization**
- **Purpose**: Balances exploration vs exploitation
- **Implementation**: Automatic temperature adjustment
- **Target Entropy**: Usually set to -|A| (negative action dimension)

### 2. **Double Q-Learning**
- **Purpose**: Reduces overestimation bias
- **Implementation**: Two critic networks with minimum Q-value

### 3. **Soft Updates**
- **Purpose**: Stable training with target networks
- **Implementation**: τ-weighted parameter updates
- **Formula**: θ_target = τ * θ + (1 - τ) * θ_target

### 4. **Reparameterization Trick**
- **Purpose**: Differentiable action sampling
- **Implementation**: Action = μ + ε * σ, where ε ~ N(0,1)

## Usage Example

```python
# Initialize SAC algorithm
sac = SACAlgorithm(
    state_dim=10,
    action_dim=3,
    hidden_dim=256,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    auto_alpha=True,
    buffer_size=100000,
    batch_size=64,
    device='cpu'
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Get action from policy
        action, log_prob = sac.get_action(state, training=True)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        sac.store_transition(state, action, reward, next_state, done)
        
        # Update networks
        losses = sac.update()
        
        state = next_state
        if done:
            break
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | Adam optimizer learning rate |
| Gamma | 0.99 | Discount factor |
| Tau | 0.005 | Target network update rate |
| Alpha | 0.2 | Initial entropy coefficient |
| Buffer Size | 100,000 | Replay buffer capacity |
| Batch Size | 64 | Training batch size |
| Hidden Dim | 256 | Network hidden layer size |

## Network Architectures

### Actor Network
```
Input (state_dim) → FC(256) → ReLU → FC(256) → ReLU → 
FC(action_dim) → Tanh (mean) + FC(action_dim) → Clamp (logstd)
```

### Critic Network
```
Input (state_dim + action_dim) → FC(256) → ReLU → 
FC(256) → ReLU → FC(1) → Q-value
```

### Value Network
```
Input (state_dim) → FC(256) → ReLU → 
FC(256) → ReLU → FC(1) → Value
```

## Training Statistics

The algorithm tracks various training metrics:

- **Actor Loss**: Policy optimization loss
- **Critic Loss**: Q-function approximation loss
- **Value Loss**: State value approximation loss
- **Alpha Loss**: Temperature parameter loss
- **Entropy**: Policy entropy for exploration monitoring
- **Q Values**: Q-value estimates for performance monitoring

## Model Persistence

```python
# Save model
sac.save_model("sac_model.pth")

# Load model
sac.load_model("sac_model.pth")
```

## Advantages of SAC

1. **Sample Efficiency**: Off-policy learning with replay buffer
2. **Exploration**: Entropy regularization encourages exploration
3. **Stability**: Double Q-learning and target networks
4. **Continuous Control**: Natural fit for continuous action spaces
5. **Automatic Tuning**: Automatic temperature adjustment

## Mathematical Foundation

### SAC Objective
```
J(π) = E[Σ(γ^t * (r_t + α * H(π(·|s_t))))]
```

Where:
- π: Policy
- γ: Discount factor
- α: Temperature parameter
- H: Entropy

### Policy Gradient
```
∇J(π) = E[∇_θ log π(a|s) * (Q(s,a) - α * log π(a|s))]
```

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α * (r + γ * (min(Q1(s',a'), Q2(s',a')) - α * log π(a'|s') - Q(s,a))
```

## Testing

Run the test script to verify implementation:

```bash
python test_sac_algorithm.py
```

This will test:
- Action generation
- Replay buffer functionality
- Training updates
- Model saving/loading
- Network components

## Integration with Multi-Agent System

The SAC algorithm is integrated into the Multi-Agent SAC system for UAV control:

```python
# Each UAV agent gets its own SAC instance
for i in range(num_agents):
    sac_agent = SACAlgorithm(
        state_dim=state_dim,
        action_dim=action_dim,
        # ... other parameters
    )
    agents.append(sac_agent)
```

This allows each UAV to learn its own policy while coordinating with other agents through the shared environment. 