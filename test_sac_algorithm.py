#!/usr/bin/env python3
"""
Test script for SAC Algorithm implementation
"""

import numpy as np
import torch
from rl.sac_algorithm import SACAlgorithm

def test_sac_algorithm():
    """Test SAC algorithm functionality"""
    
    print("Testing SAC Algorithm Implementation")
    print("=" * 50)
    
    # Test parameters
    state_dim = 10
    action_dim = 3
    hidden_dim = 64
    
    # Initialize SAC algorithm
    sac = SACAlgorithm(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=1000,
        batch_size=32,
        device='cpu'
    )
    
    print(f"SAC Algorithm initialized:")
    print(f"  - State Dimension: {sac.state_dim}")
    print(f"  - Action Dimension: {sac.action_dim}")
    print(f"  - Hidden Dimension: {sac.hidden_dim}")
    print(f"  - Learning Rate: {sac.learning_rate}")
    print(f"  - Gamma: {sac.gamma}")
    print(f"  - Tau: {sac.tau}")
    print(f"  - Alpha: {sac.alpha}")
    print(f"  - Auto Alpha: {sac.auto_alpha}")
    
    # Test action generation
    print("\nTesting action generation...")
    test_state = np.random.randn(state_dim)
    action, log_prob = sac.get_action(test_state, training=True)
    
    print(f"  Input state shape: {test_state.shape}")
    print(f"  Output action shape: {action.shape}")
    print(f"  Output log_prob shape: {log_prob.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  Log probability: {log_prob.item():.3f}")
    
    # Test replay buffer
    print("\nTesting replay buffer...")
    for i in range(50):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.choice([True, False])
        
        sac.store_transition(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {len(sac.replay_buffer)}")
    
    # Test training update
    print("\nTesting training update...")
    losses = sac.update()
    
    if losses:
        print("  Training update successful!")
        for key, value in losses.items():
            print(f"    {key}: {value:.6f}")
    else:
        print("  Training update skipped (buffer too small)")
    
    # Test multiple updates
    print("\nTesting multiple training updates...")
    for i in range(10):
        # Add more transitions
        for j in range(10):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = np.random.choice([True, False])
            
            sac.store_transition(state, action, reward, next_state, done)
        
        # Perform update
        losses = sac.update()
        if losses:
            print(f"  Update {i+1}: Actor Loss = {losses.get('actor_loss', 0):.6f}, "
                  f"Critic Loss = {losses.get('critic_loss', 0):.6f}")
    
    # Test training statistics
    print("\nTesting training statistics...")
    stats = sac.get_training_stats()
    for key, values in stats.items():
        if values:
            print(f"  {key}: {len(values)} values, last = {values[-1]:.6f}")
    
    # Test model saving/loading
    print("\nTesting model saving/loading...")
    test_filepath = "test_sac_model.pth"
    
    # Save model
    sac.save_model(test_filepath)
    print(f"  Model saved to {test_filepath}")
    
    # Create new SAC instance and load model
    sac2 = SACAlgorithm(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device='cpu'
    )
    
    sac2.load_model(test_filepath)
    print(f"  Model loaded from {test_filepath}")
    
    # Test that loaded model produces similar actions
    test_state = np.random.randn(state_dim)
    action1, _ = sac.get_action(test_state, training=False)
    action2, _ = sac2.get_action(test_state, training=False)
    
    action_diff = np.linalg.norm(action1 - action2)
    print(f"  Action difference between original and loaded model: {action_diff:.6f}")
    
    # Clean up
    import os
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
        print(f"  Test file {test_filepath} removed")
    
    print("\nSAC Algorithm test completed successfully!")

def test_networks():
    """Test individual network components"""
    
    print("\nTesting Network Components")
    print("=" * 30)
    
    from rl.sac_algorithm import ActorNetwork, CriticNetwork, ValueNetwork
    
    state_dim = 10
    action_dim = 3
    hidden_dim = 64
    
    # Test Actor Network
    print("Testing Actor Network...")
    actor = ActorNetwork(state_dim, action_dim, hidden_dim)
    
    test_state = torch.randn(1, state_dim)
    mean, logstd = actor.forward(test_state)
    action, log_prob = actor.sample(test_state)
    
    print(f"  Mean shape: {mean.shape}")
    print(f"  Logstd shape: {logstd.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log_prob shape: {log_prob.shape}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test Critic Network
    print("\nTesting Critic Network...")
    critic = CriticNetwork(state_dim, action_dim, hidden_dim)
    
    test_action = torch.randn(1, action_dim)
    q_value = critic(test_state, test_action)
    
    print(f"  Q-value shape: {q_value.shape}")
    print(f"  Q-value: {q_value.item():.3f}")
    
    # Test Value Network
    print("\nTesting Value Network...")
    value_net = ValueNetwork(state_dim, hidden_dim)
    
    value = value_net(test_state)
    
    print(f"  Value shape: {value.shape}")
    print(f"  Value: {value.item():.3f}")
    
    print("\nNetwork component tests completed!")

if __name__ == "__main__":
    test_sac_algorithm()
    test_networks() 