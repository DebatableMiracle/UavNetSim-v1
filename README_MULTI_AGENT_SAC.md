# Multi-Agent SAC for UAV Data Collection from IoT Nodes

## Overview

This project implements a Multi-Agent Soft Actor-Critic (SAC) reinforcement learning system to control UAVs for efficient data collection from IoT nodes. The goal is to minimize information loss (buffer overflow) in IoT nodes while minimizing energy consumption in UAVs.

## System Architecture

### Components

1. **IoT Nodes** (`entities/iot_nodes.py`)
   - Stationary ground-based sensors
   - Generate data packets following Poisson process
   - Limited buffer capacity with overflow handling
   - Battery management with sleep/wake cycles
   - Energy consumption model

2. **UAVs** (`entities/drone.py`)
   - Mobile agents controlled by SAC
   - 3D movement capabilities
   - Energy consumption tracking
   - Data collection from IoT nodes

3. **Multi-Agent SAC** (`rl/multi_agent_sac.py`)
   - Soft Actor-Critic implementation
   - Multi-agent coordination
   - Continuous action space (3D movement)
   - Comprehensive state representation

4. **Simulation Environment** (`simulator/simulator.py`)
   - Integrated IoT nodes and UAVs
   - Real-time simulation with SimPy
   - Visualization and metrics tracking

## State Space

Each UAV agent observes:
- **UAV State**: Position (x, y, z) + Energy level
- **IoT Nodes State**: For each IoT node: Position + Buffer level + Energy level
- **Other Agents**: Positions of all other UAVs

**Total State Dimension**: 4 + (num_iot_nodes × 5) + (num_agents - 1) × 3

## Action Space

Each UAV can perform:
- **3D Movement**: Continuous actions in [dx, dy, dz] format
- **Range**: [-1, 1] normalized, scaled to reasonable movement distances
- **Collection**: Automatic data collection when near IoT nodes

## Reward Function

The reward combines multiple objectives:

```
Reward = Collection_Bonus + Movement_Penalty + Info_Loss_Penalty + Energy_Loss_Penalty
```

Where:
- **Collection Bonus**: +1.0 for each IoT node with data when UAV is within range
- **Movement Penalty**: -0.01 × action_norm (encourages efficient movement)
- **Info Loss Penalty**: -10.0 × buffer_overflow_count
- **Energy Loss Penalty**: -5.0 × energy_consumed

## Training Process

### Environment Setup
```python
# Create simulator with IoT nodes
sim = Simulator(seed=42, env=env, channel_states=channel_states, 
               n_drones=5, n_iot_nodes=20)

# Initialize Multi-Agent SAC
masac = MultiAgentSAC(sim, num_agents=5)
```

### Training Parameters
- **Learning Rate**: 3e-4
- **Gamma (Discount)**: 0.99
- **Tau (Target Update)**: 0.005
- **Alpha (Entropy)**: 0.2
- **Batch Size**: 64
- **Replay Buffer**: 100,000 experiences

### Training Loop
```python
# Train for specified episodes
masac.train(num_episodes=500, max_steps_per_episode=500)
```

## Usage

### 1. Test IoT Nodes
```bash
conda activate uavnetsim
python test_iot_nodes.py
```

### 2. Test Multi-Agent SAC
```bash
conda activate uavnetsim
python test_multi_agent_sac.py
```

### 3. Run Training
```bash
conda activate uavnetsim
python train_multi_agent_sac.py
```

### 4. Visualize Results
```bash
conda activate uavnetsim
python train_multi_agent_sac.py visualize
```

## Configuration

### IoT Node Parameters (`utils/config.py`)
```python
NUMBER_OF_IOT_NODES = 20
IOT_NODE_DENSITY = 0.00008  # nodes per square meter
IOT_NODE_BATTERY_CAPACITY = 5000  # Joule
IOT_NODE_ENERGY_THRESHOLD = 500  # Joule
IOT_NODE_MAX_BUFFER_SIZE = 50  # packets
IOT_NODE_DATA_GENERATION_RATE = 2.0  # packets per second
```

### UAV Parameters
```python
NUMBER_OF_DRONES = 5
INITIAL_ENERGY = 20 * 1e3  # Joule
ENERGY_THRESHOLD = 2000  # Joule
```

## Output and Visualization

### Training Results
- **Training Curves**: Rewards, losses, episode lengths
- **Model Weights**: Saved for each agent
- **Metrics**: Average performance statistics

### Visualization
- **UAV Trajectories**: 3D movement paths
- **IoT Buffer Levels**: Real-time buffer status
- **Energy Consumption**: UAV and IoT node energy levels
- **Collection Events**: Data collection timestamps

### Output Files
```
training_results/
├── training_curves.png
├── comprehensive_results.png
├── training_data.npz
├── agent_0_model.pth
├── agent_1_model.pth
└── ...
```

## Key Features

### 1. Multi-Agent Coordination
- Independent agents with shared environment
- Coordinated coverage of IoT nodes
- Collision avoidance through reward shaping

### 2. Energy Efficiency
- Energy-aware decision making
- Sleep/wake cycles for IoT nodes
- Energy consumption tracking

### 3. Information Loss Minimization
- Buffer overflow prevention
- Priority-based collection strategies
- Real-time buffer monitoring

### 4. Scalable Architecture
- Configurable number of agents
- Modular design for easy extension
- Comprehensive logging and metrics

## Performance Metrics

### Training Metrics
- **Episode Rewards**: Overall performance
- **Information Loss**: Buffer overflow count
- **Energy Loss**: Total energy consumption
- **Episode Lengths**: Time efficiency

### Evaluation Metrics
- **Collection Efficiency**: Percentage of data collected
- **Energy Efficiency**: Energy per collected packet
- **Coverage**: Area covered by UAVs
- **Latency**: Time to collect data

## Future Enhancements

1. **Advanced Coordination**
   - Centralized vs decentralized approaches
   - Communication between agents
   - Dynamic task allocation

2. **Enhanced State Representation**
   - Historical data patterns
   - Predictive modeling
   - Attention mechanisms

3. **Improved Reward Design**
   - Multi-objective optimization
   - Adaptive reward shaping
   - Hierarchical rewards

4. **Real-world Integration**
   - Hardware-in-the-loop testing
   - Real UAV control
   - Sensor integration

## Dependencies

```python
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
simpy>=3.0.11
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{uavnetsim2024,
  title={Multi-Agent SAC for UAV Data Collection from IoT Nodes},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on the GitHub repository. 