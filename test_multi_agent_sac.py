#!/usr/bin/env python3
"""
Test script for Multi-Agent SAC implementation
"""

import simpy
import numpy as np
import torch
from utils import config
from simulator.simulator import Simulator
from rl.multi_agent_sac import MultiAgentSAC

def test_multi_agent_sac():
    """Test Multi-Agent SAC functionality"""
    
    print("Testing Multi-Agent SAC Implementation")
    print("=" * 50)
    
    # Create simulation environment
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    
    # Create simulator with IoT nodes
    sim = Simulator(
        seed=42, 
        env=env, 
        channel_states=channel_states, 
        n_drones=config.NUMBER_OF_DRONES,
        n_iot_nodes=config.NUMBER_OF_IOT_NODES
    )
    
    print(f"Simulator created with {len(sim.drones)} drones and {len(sim.iot_nodes)} IoT nodes")
    
    # Initialize Multi-Agent SAC
    print("\nInitializing Multi-Agent SAC...")
    masac = MultiAgentSAC(sim, num_agents=config.NUMBER_OF_DRONES)
    
    print(f"  - State Dimension: {masac.state_dim}")
    print(f"  - Action Dimension: {masac.action_dim}")
    print(f"  - Number of Agents: {masac.num_agents}")
    
    # Test state generation
    print("\nTesting state generation...")
    for i in range(min(3, masac.num_agents)):  # Test first 3 agents
        state = masac.get_state(i)
        print(f"  Agent {i} state shape: {state.shape}")
        print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
    
    # Test action generation
    print("\nTesting action generation...")
    for i in range(min(3, masac.num_agents)):
        state = masac.get_state(i)
        action = masac.get_action(i, state, training=True)
        print(f"  Agent {i} action: {action}")
        print(f"  Action norm: {np.linalg.norm(action):.3f}")
    
    # Test reward calculation
    print("\nTesting reward calculation...")
    for i in range(min(3, masac.num_agents)):
        state = masac.get_state(i)
        action = masac.get_action(i, state, training=True)
        next_state = masac.get_state(i)  # Same state for testing
        reward = masac.calculate_reward(i, action, next_state, 0.1, 0.05)
        print(f"  Agent {i} reward: {reward:.3f}")
    
    # Test information and energy loss calculation
    print("\nTesting loss calculations...")
    info_loss = masac._calculate_information_loss(sim)
    energy_loss = masac._calculate_energy_loss(sim)
    print(f"  Information loss: {info_loss}")
    print(f"  Energy loss: {energy_loss}")
    
    # Test single episode
    print("\nTesting single episode...")
    episode_result = masac.run_episode(max_steps=10)  # Short episode for testing
    
    print(f"  Episode reward: {episode_result['episode_reward']:.3f}")
    print(f"  Total info loss: {episode_result['total_info_loss']}")
    print(f"  Total energy loss: {episode_result['total_energy_loss']}")
    print(f"  Steps taken: {episode_result['step_count']}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_multi_agent_sac() 