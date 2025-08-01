import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import simpy
from simulator.simulator import Simulator
from utils import config
from rl.sac_algorithm import SACAlgorithm


class SACNetwork(nn.Module):
    """
    Neural network for SAC (Soft Actor-Critic)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACNetwork, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] for continuous actions
        )
        
        # Critic networks (Q-functions)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass - returns actor output by default
        """
        return self.actor(x)


class MultiAgentSAC:
    """
    Multi-Agent SAC for UAV data collection from IoT nodes
    """
    def __init__(self, simulator: Simulator, num_agents: int = None):
        self.simulator = simulator
        self.num_agents = num_agents if num_agents else len(simulator.drones)
        
        # Environment parameters
        self.state_dim = self._get_state_dim()
        self.action_dim = 3  # [dx, dy, dz] movement in 3D space
        
        # SAC parameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # entropy coefficient
        
        # Networks for each agent
        self.agents = []
        self._initialize_agents()
        
        # Training metrics
        self.training_rewards = []
        self.information_losses = []
        self.energy_losses = []
        self.episode_lengths = []
        
        # Visualization data
        self.visualization_data = {
            'uav_positions': [],
            'iot_buffer_levels': [],
            'energy_levels': [],
            'collection_events': []
        }

    def _get_state_dim(self) -> int:
        """
        Calculate state dimension based on environment
        """
        # Basic UAV state: position (3) + energy (1) = 4
        uav_state_dim = 4
        
        # IoT nodes state: for each IoT node: position (3) + buffer_level (1) + energy (1) = 5
        iot_state_dim = len(self.simulator.iot_nodes) * 5
        
        # Other agents' positions: (num_agents - 1) * 3
        other_agents_dim = (self.num_agents - 1) * 3
        
        return uav_state_dim + iot_state_dim + other_agents_dim

    def _initialize_agents(self):
        """
        Initialize SAC agents for each UAV
        """
        for i in range(self.num_agents):
            # Create SAC algorithm instance for each agent
            sac_agent = SACAlgorithm(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=256,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                tau=self.tau,
                alpha=self.alpha,
                auto_alpha=True,
                buffer_size=100000,
                batch_size=64,
                device='cpu'
            )
            
            self.agents.append(sac_agent)

    def _update_target_networks(self, agent, tau=None):
        """
        Update target networks using soft update
        """
        # This is now handled by the SACAlgorithm class
        pass

    def get_state(self, agent_id: int) -> np.ndarray:
        """
        Get state for a specific agent
        """
        uav = self.simulator.drones[agent_id]
        
        # UAV state: [x, y, z, energy]
        uav_state = np.array([
            uav.coords[0], uav.coords[1], uav.coords[2],
            uav.residual_energy
        ])
        
        # IoT nodes state
        iot_state = []
        for iot_node in self.simulator.iot_nodes:
            buffer_status = iot_node.get_buffer_status()
            energy_status = iot_node.get_energy_status()
            
            iot_state.extend([
                iot_node.coords[0], iot_node.coords[1], iot_node.coords[2],
                buffer_status['current_size'] / buffer_status['max_size'],  # normalized buffer
                energy_status['energy_percentage'] / 100.0  # normalized energy
            ])
        
        # Other agents' positions
        other_agents_state = []
        for i in range(self.num_agents):
            if i != agent_id and i < len(self.simulator.drones):
                other_uav = self.simulator.drones[i]
                other_agents_state.extend([
                    other_uav.coords[0], other_uav.coords[1], other_uav.coords[2]
                ])
        
        return np.concatenate([uav_state, np.array(iot_state), np.array(other_agents_state)])

    def get_action(self, agent_id: int, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get action for a specific agent
        """
        agent = self.agents[agent_id]
        action, log_prob = agent.get_action(state, training)
        return action

    def calculate_reward(self, agent_id: int, action: np.ndarray, 
                        next_state: np.ndarray, info_loss: float, energy_loss: float) -> float:
        """
        Calculate reward for the agent
        """
        # Base reward components
        collection_reward = 0.0
        movement_penalty = -0.01 * np.linalg.norm(action)  # Penalize large movements
        
        # Information loss penalty (higher penalty for more loss)
        info_loss_penalty = -10.0 * info_loss
        
        # Energy loss penalty
        energy_loss_penalty = -5.0 * energy_loss
        
        # Data collection bonus (if UAV is near IoT nodes)
        uav = self.simulator.drones[agent_id]
        for iot_node in self.simulator.iot_nodes:
            distance = np.linalg.norm(np.array(uav.coords) - np.array(iot_node.coords))
            if distance < 50:  # Collection range
                buffer_status = iot_node.get_buffer_status()
                if buffer_status['current_size'] > 0:
                    collection_reward += 1.0
        
        total_reward = collection_reward + movement_penalty + info_loss_penalty + energy_loss_penalty
        return total_reward

    def train_step(self, agent_id: int, state: np.ndarray, action: np.ndarray, 
                   reward: float, next_state: np.ndarray, done: bool):
        """
        Perform one training step for an agent
        """
        agent = self.agents[agent_id]
        
        # Store experience in replay buffer
        agent.store_transition(state, action, reward, next_state, done)
        
        # Perform SAC update
        losses = agent.update()
        
        return losses

    def run_episode(self, max_steps: int = 600) -> Dict:
        """
        Run one episode of training
        """
        # Reset environment
        env = simpy.Environment()
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
        sim = Simulator(seed=random.randint(0, 10000), env=env, channel_states=channel_states, 
                       n_drones=self.num_agents)
        
        episode_reward = 0
        total_info_loss = 0
        total_energy_loss = 0
        step_count = 0
        
        # Store visualization data
        episode_data = {
            'uav_positions': [],
            'iot_buffer_levels': [],
            'energy_levels': [],
            'collection_events': []
        }
        
        while step_count < max_steps and env.now < config.SIM_TIME:
            # Get states for all agents
            states = [self.get_state(i) for i in range(self.num_agents)]
            actions = []
            
            # Get actions for all agents
            for i in range(self.num_agents):
                action = self.get_action(i, states[i], training=True)
                actions.append(action)
            
            # Apply actions to environment
            for i, action in enumerate(actions):
                uav = sim.drones[i]
                # Move UAV based on action (normalize to reasonable movement range)
                movement = action * 10  # Scale action to reasonable movement
                new_coords = [
                    np.clip(uav.coords[0] + movement[0], 0, config.MAP_LENGTH),
                    np.clip(uav.coords[1] + movement[1], 0, config.MAP_WIDTH),
                    np.clip(uav.coords[2] + movement[2], 0, config.MAP_HEIGHT)
                ]
                uav.coords = new_coords
            
            # Run simulation for one step
            env.run(until=env.now + 100000)  # 0.1 seconds
            
            # Calculate rewards and losses
            info_loss = self._calculate_information_loss(sim)
            energy_loss = self._calculate_energy_loss(sim)
            
            # Get next states
            next_states = [self.get_state(i) for i in range(self.num_agents)]
            
            # Calculate rewards for each agent
            rewards = []
            for i in range(self.num_agents):
                reward = self.calculate_reward(i, actions[i], next_states[i], info_loss, energy_loss)
                rewards.append(reward)
                episode_reward += reward
            
            # Train agents
            for i in range(self.num_agents):
                self.train_step(i, states[i], actions[i], rewards[i], next_states[i], False)
            
            # Store visualization data
            self._store_visualization_data(sim, episode_data)
            
            total_info_loss += info_loss
            total_energy_loss += energy_loss
            step_count += 1
        
        return {
            'episode_reward': episode_reward,
            'total_info_loss': total_info_loss,
            'total_energy_loss': total_energy_loss,
            'step_count': step_count,
            'episode_data': episode_data
        }

    def _calculate_information_loss(self, sim: Simulator) -> float:
        """
        Calculate total information loss from IoT node buffer overflows
        """
        total_loss = 0
        for iot_node in sim.iot_nodes:
            buffer_status = iot_node.get_buffer_status()
            if buffer_status['overflow_count'] > 0:
                total_loss += buffer_status['overflow_count']
        return total_loss

    def _calculate_energy_loss(self, sim: Simulator) -> float:
        """
        Calculate total energy loss from UAVs
        """
        total_loss = 0
        for uav in sim.drones:
            energy_status = uav.get_energy_status()
            total_loss += (energy_status['battery_capacity'] - energy_status['residual_energy'])
        return total_loss

    def _store_visualization_data(self, sim: Simulator, episode_data: Dict):
        """
        Store data for visualization
        """
        # UAV positions
        uav_positions = []
        for uav in sim.drones:
            uav_positions.append(uav.coords.copy())
        episode_data['uav_positions'].append(uav_positions)
        
        # IoT buffer levels
        iot_buffer_levels = []
        for iot_node in sim.iot_nodes:
            buffer_status = iot_node.get_buffer_status()
            iot_buffer_levels.append(buffer_status['current_size'] / buffer_status['max_size'])
        episode_data['iot_buffer_levels'].append(iot_buffer_levels)
        
        # Energy levels
        energy_levels = []
        for uav in sim.drones:
            energy_status = uav.get_energy_status()
            energy_levels.append(energy_status['energy_percentage'])
        episode_data['energy_levels'].append(energy_levels)

    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 1000):
        """
        Train the multi-agent SAC system
        """
        print(f"Starting Multi-Agent SAC training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_result = self.run_episode(max_steps_per_episode)
            
            # Store metrics
            self.training_rewards.append(episode_result['episode_reward'])
            self.information_losses.append(episode_result['total_info_loss'])
            self.energy_losses.append(episode_result['total_energy_loss'])
            self.episode_lengths.append(episode_result['step_count'])
            
            # Store visualization data every 10th episode
            if episode % 10 == 0:
                self.visualization_data['uav_positions'].append(episode_result['episode_data']['uav_positions'])
                self.visualization_data['iot_buffer_levels'].append(episode_result['episode_data']['iot_buffer_levels'])
                self.visualization_data['energy_levels'].append(episode_result['episode_data']['energy_levels'])
            
            if episode % 50 == 0:
                avg_reward = np.mean(self.training_rewards[-50:]) if len(self.training_rewards) >= 50 else np.mean(self.training_rewards)
                avg_info_loss = np.mean(self.information_losses[-50:]) if len(self.information_losses) >= 50 else np.mean(self.information_losses)
                avg_energy_loss = np.mean(self.energy_losses[-50:]) if len(self.energy_losses) >= 50 else np.mean(self.energy_losses)
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                      f"Avg Info Loss: {avg_info_loss:.2f}, Avg Energy Loss: {avg_energy_loss:.2f}")
        
        print("Training completed!")
        self.plot_training_results()

    def plot_training_results(self):
        """
        Plot training results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training rewards
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Information loss
        axes[0, 1].plot(self.information_losses)
        axes[0, 1].set_title('Information Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        
        # Energy loss
        axes[1, 0].plot(self.energy_losses)
        axes[1, 0].set_title('Energy Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save training data
        np.savez('training_data.npz',
                 rewards=np.array(self.training_rewards),
                 info_losses=np.array(self.information_losses),
                 energy_losses=np.array(self.energy_losses),
                 episode_lengths=np.array(self.episode_lengths)) 