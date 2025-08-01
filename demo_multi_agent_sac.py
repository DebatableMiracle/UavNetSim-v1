#!/usr/bin/env python3
"""
Demo script for Multi-Agent SAC UAV Data Collection System

This script demonstrates the complete system with visualization
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import config
from simulator.simulator import Simulator
from rl.multi_agent_sac import MultiAgentSAC
import torch

def run_demo():
    """
    Run a demonstration of the Multi-Agent SAC system
    """
    print("=" * 60)
    print("Multi-Agent SAC Demo: UAV Data Collection from IoT Nodes")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simulation environment
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    
    # Create simulator
    sim = Simulator(seed=42, env=env, channel_states=channel_states, 
                   n_drones=config.NUMBER_OF_DRONES, n_iot_nodes=config.NUMBER_OF_IOT_NODES)
    
    print(f"Environment Setup:")
    print(f"  - UAVs: {len(sim.drones)}")
    print(f"  - IoT Nodes: {len(sim.iot_nodes)}")
    print(f"  - Map Size: {config.MAP_LENGTH} x {config.MAP_WIDTH} x {config.MAP_HEIGHT}")
    
    # Initialize Multi-Agent SAC
    masac = MultiAgentSAC(sim, num_agents=config.NUMBER_OF_DRONES)
    
    print(f"\nMulti-Agent SAC Setup:")
    print(f"  - State Dimension: {masac.state_dim}")
    print(f"  - Action Dimension: {masac.action_dim}")
    print(f"  - Number of Agents: {masac.num_agents}")
    
    # Run a short episode for demonstration
    print(f"\nRunning Demo Episode...")
    episode_result = masac.run_episode(max_steps=50)
    
    print(f"Demo Results:")
    print(f"  - Episode Reward: {episode_result['episode_reward']:.2f}")
    print(f"  - Information Loss: {episode_result['total_info_loss']}")
    print(f"  - Energy Loss: {episode_result['total_energy_loss']:.2f}")
    print(f"  - Steps: {episode_result['step_count']}")
    
    # Visualize the episode
    visualize_episode(episode_result['episode_data'], sim)
    
    print(f"\nDemo completed successfully!")

def visualize_episode(episode_data, sim):
    """
    Visualize the episode data
    """
    uav_positions = episode_data['uav_positions']
    iot_buffer_levels = episode_data['iot_buffer_levels']
    energy_levels = episode_data['energy_levels']
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot IoT nodes
    for iot_node in sim.iot_nodes:
        ax1.scatter(iot_node.coords[0], iot_node.coords[1], iot_node.coords[2], 
                   c='green', s=50, marker='o', alpha=0.7, label='IoT Node' if iot_node.identifier == sim.n_drones else "")
    
    # Plot UAV trajectories
    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for i, uav in enumerate(sim.drones):
        if i < len(uav_positions) and len(uav_positions[i]) > 0:
            positions = np.array(uav_positions[i])
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    color=colors[i % len(colors)], linewidth=2, label=f'UAV {i}')
            # Mark start and end positions
            ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                       color=colors[i % len(colors)], s=100, marker='^')
            ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                       color=colors[i % len(colors)], s=100, marker='s')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('UAV Trajectories and IoT Nodes')
    ax1.legend()
    
    # IoT buffer levels over time
    ax2 = fig.add_subplot(2, 3, 2)
    if len(iot_buffer_levels) > 0:
        buffer_data = np.array(iot_buffer_levels)
        for i in range(min(10, buffer_data.shape[1])):  # Show first 10 IoT nodes
            ax2.plot(buffer_data[:, i], label=f'IoT {i+sim.n_drones}', alpha=0.7)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Buffer Level (Normalized)')
    ax2.set_title('IoT Node Buffer Levels')
    ax2.legend()
    ax2.grid(True)
    
    # UAV energy levels over time
    ax3 = fig.add_subplot(2, 3, 3)
    if len(energy_levels) > 0:
        energy_data = np.array(energy_levels)
        for i in range(energy_data.shape[1]):
            ax3.plot(energy_data[:, i], label=f'UAV {i}', color=colors[i % len(colors)])
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Energy Level (%)')
    ax3.set_title('UAV Energy Levels')
    ax3.legend()
    ax3.grid(True)
    
    # Coverage map (top view)
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Plot IoT nodes
    for iot_node in sim.iot_nodes:
        ax4.scatter(iot_node.coords[0], iot_node.coords[1], 
                   c='green', s=50, marker='o', alpha=0.7)
    
    # Plot UAV final positions
    for i, uav in enumerate(sim.drones):
        ax4.scatter(uav.coords[0], uav.coords[1], 
                   color=colors[i % len(colors)], s=100, marker='^', label=f'UAV {i}')
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Coverage Map (Top View)')
    ax4.legend()
    ax4.grid(True)
    
    # Statistics
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Calculate statistics
    if len(iot_buffer_levels) > 0:
        buffer_data = np.array(iot_buffer_levels)
        avg_buffer = np.mean(buffer_data, axis=1)
        max_buffer = np.max(buffer_data, axis=1)
        
        ax5.plot(avg_buffer, label='Average Buffer', color='blue')
        ax5.plot(max_buffer, label='Max Buffer', color='red')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Buffer Level')
        ax5.set_title('Buffer Statistics')
        ax5.legend()
        ax5.grid(True)
    
    # Energy statistics
    ax6 = fig.add_subplot(2, 3, 6)
    if len(energy_levels) > 0:
        energy_data = np.array(energy_levels)
        avg_energy = np.mean(energy_data, axis=1)
        min_energy = np.min(energy_data, axis=1)
        
        ax6.plot(avg_energy, label='Average Energy', color='green')
        ax6.plot(min_energy, label='Min Energy', color='orange')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Energy Level (%)')
        ax6.set_title('Energy Statistics')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'demo_visualization.png'")

def print_system_info():
    """
    Print system information and configuration
    """
    print("System Configuration:")
    print(f"  - Map Dimensions: {config.MAP_LENGTH} x {config.MAP_WIDTH} x {config.MAP_HEIGHT}")
    print(f"  - Number of UAVs: {config.NUMBER_OF_DRONES}")
    print(f"  - Number of IoT Nodes: {config.NUMBER_OF_IOT_NODES}")
    print(f"  - IoT Node Density: {config.IOT_NODE_DENSITY}")
    print(f"  - Simulation Time: {config.SIM_TIME / 1e6:.1f} seconds")
    print(f"  - UAV Initial Energy: {config.INITIAL_ENERGY / 1000:.1f} kJ")
    print(f"  - IoT Node Battery: {config.IOT_NODE_BATTERY_CAPACITY / 1000:.1f} kJ")
    print(f"  - IoT Node Buffer Size: {config.IOT_NODE_MAX_BUFFER_SIZE}")
    print(f"  - IoT Data Generation Rate: {config.IOT_NODE_DATA_GENERATION_RATE} packets/sec")

if __name__ == "__main__":
    print_system_info()
    run_demo() 