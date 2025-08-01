#!/usr/bin/env python3
"""
Multi-Agent SAC Training Script for UAV Data Collection from IoT Nodes

This script trains multiple UAV agents using SAC to minimize information loss
in IoT nodes while minimizing energy consumption in drones.
"""



import simpy
import numpy as np
import matplotlib.pyplot as plt
from utils import config
from simulator.simulator import Simulator
from rl.multi_agent_sac import MultiAgentSAC
import torch
import os

def main():
    """
    Main training function
    """
    print("=" * 60)
    print("Multi-Agent SAC Training for UAV Data Collection")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create initial simulator to get environment parameters
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=42, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)
    
    print(f"Environment Setup:")
    print(f"  - Number of UAVs: {len(sim.drones)}")
    print(f"  - Number of IoT Nodes: {len(sim.iot_nodes)}")
    print(f"  - Map Size: {config.MAP_LENGTH} x {config.MAP_WIDTH} x {config.MAP_HEIGHT}")
    print(f"  - Simulation Time: {config.SIM_TIME / 1e6:.1f} seconds")
    
    # Initialize Multi-Agent SAC
    print("\nInitializing Multi-Agent SAC...")
    masac = MultiAgentSAC(sim, num_agents=config.NUMBER_OF_DRONES)
    
    print(f"  - State Dimension: {masac.state_dim}")
    print(f"  - Action Dimension: {masac.action_dim}")
    print(f"  - Number of Agents: {masac.num_agents}")
    
    # Training parameters
    num_episodes = 1000  # Reduced for testing, can increase later
    max_steps_per_episode = 600  # 60 seconds (60,000,000 microseconds / 100,000 per step)
    
    print(f"\nTraining Parameters:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Max Steps per Episode: {max_steps_per_episode}")
    print(f"  - Learning Rate: {masac.learning_rate}")
    print(f"  - Gamma: {masac.gamma}")
    print(f"  - Tau: {masac.tau}")
    print(f"  - Alpha: {masac.alpha}")
    
    # Create output directory for results
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput Directory: {output_dir}")
    
    # Start training
    print("\nStarting Training...")
    print("=" * 60)
    
    try:
        masac.train(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)
        
        # Save final results
        print(f"\nSaving results to {output_dir}/...")
        
        # Save training curves
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(masac.training_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot information loss
        plt.subplot(2, 2, 2)
        plt.plot(masac.information_losses)
        plt.title('Information Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot energy loss
        plt.subplot(2, 2, 3)
        plt.plot(masac.energy_losses)
        plt.title('Energy Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 2, 4)
        plt.plot(masac.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save training data
        np.savez(f'{output_dir}/training_data.npz',
                 rewards=np.array(masac.training_rewards),
                 info_losses=np.array(masac.information_losses),
                 energy_losses=np.array(masac.energy_losses),
                 episode_lengths=np.array(masac.episode_lengths))
        
        # Save model weights
        for i, agent in enumerate(masac.agents):
            torch.save({
                'actor_state_dict': agent['actor'].state_dict(),
                'critic1_state_dict': agent['critic1'].state_dict(),
                'critic2_state_dict': agent['critic2'].state_dict(),
                'value_state_dict': agent['value'].state_dict(),
            }, f'{output_dir}/agent_{i}_model.pth')
        
        print(f"Training completed successfully!")
        print(f"Results saved to: {output_dir}/")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  - Average Reward (last 100 episodes): {np.mean(masac.training_rewards[-100:]):.2f}")
        print(f"  - Average Information Loss (last 100 episodes): {np.mean(masac.information_losses[-100:]):.2f}")
        print(f"  - Average Energy Loss (last 100 episodes): {np.mean(masac.energy_losses[-100:]):.2f}")
        print(f"  - Best Episode Reward: {np.max(masac.training_rewards):.2f}")
        print(f"  - Worst Episode Reward: {np.min(masac.training_rewards):.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving intermediate results...")
        
        # Save intermediate results
        np.savez(f'{output_dir}/intermediate_training_data.npz',
                 rewards=np.array(masac.training_rewards),
                 info_losses=np.array(masac.information_losses),
                 energy_losses=np.array(masac.energy_losses),
                 episode_lengths=np.array(masac.episode_lengths))
        
        print(f"Intermediate results saved to: {output_dir}/intermediate_training_data.npz")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

def visualize_training_results():
    """
    Visualize training results from saved data
    """
    try:
        # Load training data
        data = np.load('training_results/training_data.npz')
        
        rewards = data['rewards']
        info_losses = data['info_losses']
        energy_losses = data['energy_losses']
        episode_lengths = data['episode_lengths']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training rewards
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Information loss
        axes[0, 1].plot(info_losses)
        axes[0, 1].set_title('Information Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Energy loss
        axes[0, 2].plot(energy_losses)
        axes[0, 2].set_title('Energy Loss')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Moving averages
        window = 50
        if len(rewards) >= window:
            moving_avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(moving_avg_rewards)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)
        
        # Combined loss
        combined_loss = info_losses + energy_losses
        axes[1, 2].plot(combined_loss)
        axes[1, 2].set_title('Combined Loss (Info + Energy)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Combined Loss')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results/comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training results visualization completed!")
        
    except FileNotFoundError:
        print("No training data found. Run training first.")
    except Exception as e:
        print(f"Error visualizing results: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        visualize_training_results()
    else:
        main() 