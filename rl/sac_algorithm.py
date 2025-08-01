import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict
import math


class SACAlgorithm:
    """
    Soft Actor-Critic (SAC) Algorithm Implementation
    
    SAC is an off-policy actor-critic deep RL algorithm that uses entropy regularization
    to encourage exploration and improve sample efficiency.
    
    Key Components:
    1. Actor Network (Policy) - outputs action distribution
    2. Critic Networks (Q-functions) - estimate state-action values
    3. Value Network - estimates state values
    4. Target Networks - for stable training
    5. Replay Buffer - stores experience tuples
    6. Temperature Parameter - controls exploration vs exploitation
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 target_entropy: float = None,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Initialize temperature parameter for auto-adjustment
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'value_loss': [],
            'alpha_loss': [],
            'entropy': [],
            'q_values': []
        }
    
    def _init_networks(self):
        """Initialize all neural networks"""
        
        # Actor Network (Policy)
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        # Critic Networks (Q-functions)
        self.critic1 = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic2 = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        # Value Network
        self.value = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        
        # Target Networks
        self.target_critic1 = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_critic2 = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_value = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        
        # Initialize target networks
        self._update_target_networks(tau=1.0)
    
    def _init_optimizers(self):
        """Initialize optimizers for all networks"""
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.learning_rate
        )
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.learning_rate)
    
    def _update_target_networks(self, tau: float = None):
        """Soft update of target networks"""
        if tau is None:
            tau = self.tau
        
        # Update target critic networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Update target value network
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action from current policy
        
        Args:
            state: Current state
            training: Whether in training mode (adds exploration noise)
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action distribution from actor
        action, log_prob = self.actor.sample(state_tensor)
        
        if training:
            # Add exploration noise
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.clamp(action, -1, 1)
        
        return action.squeeze(0).cpu().detach().numpy(), log_prob.squeeze(0).cpu().detach().numpy()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update(self) -> Dict[str, float]:
        """
        Perform one update step of SAC algorithm
        
        Returns:
            Dictionary containing loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update value network
        value_loss = self._update_value_network(states, actions, rewards, next_states, dones)
        
        # Update critic networks
        critic_loss = self._update_critic_networks(states, actions, rewards, next_states, dones)
        
        # Update actor network
        actor_loss = self._update_actor_network(states)
        
        # Update temperature parameter (if auto-adjustment is enabled)
        alpha_loss = self._update_temperature(states)
        
        # Update target networks
        self._update_target_networks()
        
        # Store statistics
        losses = {
            'value_loss': value_loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss
        }
        
        for key, value in losses.items():
            if value is not None:
                self.training_stats[key].append(value)
        
        return losses
    
    def _update_value_network(self, states: torch.Tensor, actions: torch.Tensor, 
                            rewards: torch.Tensor, next_states: torch.Tensor, 
                            dones: torch.Tensor) -> float:
        """Update value network"""
        
        # Get current value estimates
        current_values = self.value(states)
        
        # Get next actions and log probs from actor
        next_actions, next_log_probs = self.actor.sample(next_states)
        
        # Get Q-values for next state-action pairs
        next_q1 = self.target_critic1(next_states, next_actions)
        next_q2 = self.target_critic2(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2)
        
        # Compute target values
        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        target_values = next_q - alpha * next_log_probs
        target_values = rewards + self.gamma * target_values * (1 - dones)
        
        # Compute value loss
        value_loss = F.mse_loss(current_values, target_values.detach())
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def _update_critic_networks(self, states: torch.Tensor, actions: torch.Tensor,
                               rewards: torch.Tensor, next_states: torch.Tensor,
                               dones: torch.Tensor) -> float:
        """Update critic networks"""
        
        # Get current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Get next actions and log probs from actor
        next_actions, next_log_probs = self.actor.sample(next_states)
        
        # Get next Q-values
        next_q1 = self.target_critic1(next_states, next_actions)
        next_q2 = self.target_critic2(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2)
        
        # Compute target Q-values
        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        target_q = next_q - alpha * next_log_probs
        target_q = rewards + self.gamma * target_q * (1 - dones)
        
        # Compute critic losses
        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())
        critic_loss = critic1_loss + critic2_loss
        
        # Update critic networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor_network(self, states: torch.Tensor) -> float:
        """Update actor network"""
        
        # Get actions and log probs from current policy
        actions, log_probs = self.actor.sample(states)
        
        # Get Q-values for current state-action pairs
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Compute actor loss with entropy regularization
        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        actor_loss = alpha * log_probs - q
        actor_loss = actor_loss.mean()
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_temperature(self, states: torch.Tensor) -> float:
        """Update temperature parameter (alpha)"""
        
        if not self.auto_alpha:
            return 0.0
        
        # Get actions and log probs from current policy
        actions, log_probs = self.actor.sample(states)
        
        # Compute alpha loss
        alpha = self.log_alpha.exp()
        alpha_loss = alpha * (-log_probs - self.target_entropy)
        alpha_loss = alpha_loss.mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.training_stats
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'target_value_state_dict': self.target_value.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.target_value.load_state_dict(checkpoint['target_value_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action distribution
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action mean and log standard deviation"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.fc_mean(x))  # Output in [-1, 1]
        logstd = self.fc_logstd(x)
        logstd = torch.clamp(logstd, -20, 2)  # Clamp for numerical stability
        
        return mean, logstd
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy"""
        mean, logstd = self.forward(state)
        std = logstd.exp()
        
        # Reparameterization trick
        noise = torch.randn_like(mean)
        action = mean + noise * std
        
        # Compute log probability
        log_prob = self._log_prob(action, mean, std)
        
        return action, log_prob
    
    def _log_prob(self, action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action"""
        log_prob = -0.5 * ((action - mean) / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)
        return log_prob.sum(dim=-1, keepdim=True)


class CriticNetwork(nn.Module):
    """
    Critic network that estimates Q-values
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value"""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value 