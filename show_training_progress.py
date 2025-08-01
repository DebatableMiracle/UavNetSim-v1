#!/usr/bin/env python3
"""
Show training progress in real-time
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

def show_training_progress():
    """Show current training progress"""
    
    # Check for training data
    if os.path.exists('training_results/intermediate_training_data.npz'):
        data = np.load('training_results/intermediate_training_data.npz')
        
        rewards = data['rewards']
        info_losses = data['info_losses']
        energy_losses = data['energy_losses']
        episode_lengths = data['episode_lengths']
        
        print(f"Training Progress:")
        print(f"  Episodes completed: {len(rewards)}")
        print(f"  Average reward: {np.mean(rewards):.2f}")
        print(f"  Average info loss: {np.mean(info_losses):.2f}")
        print(f"  Average energy loss: {np.mean(energy_losses):.2f}")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f} steps")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Rewards over time
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
        axes[1, 0].plot(energy_losses)
        axes[1, 0].set_title('Energy Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Episode lengths
        axes[1, 1].plot(episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Progress visualization saved as 'training_progress.png'")
        
    elif os.path.exists('training_results/training_data.npz'):
        data = np.load('training_results/training_data.npz')
        
        rewards = data['rewards']
        info_losses = data['info_losses']
        energy_losses = data['energy_losses']
        episode_lengths = data['episode_lengths']
        
        print(f"Training Completed!")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Final average reward: {np.mean(rewards[-100:]):.2f}")
        print(f"  Final average info loss: {np.mean(info_losses[-100:]):.2f}")
        print(f"  Final average energy loss: {np.mean(energy_losses[-100:]):.2f}")
        
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
        plt.savefig('final_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Final results visualization saved as 'final_training_results.png'")
        
    else:
        print("No training data found yet. Training may still be in progress.")

if __name__ == "__main__":
    show_training_progress() 