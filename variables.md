# UAV Network Simulation Variables Documentation

This document provides a comprehensive overview of all important variables used in the UAV network simulation and multi-agent SAC implementation for research purposes.

## Table of Contents
1. [Simulation Environment Parameters](#simulation-environment-parameters)
2. [UAV Hardware Parameters](#uav-hardware-parameters)
3. [IoT Node Parameters](#iot-node-parameters)
4. [Radio Communication Parameters](#radio-communication-parameters)
5. [Packet and Protocol Parameters](#packet-and-protocol-parameters)
6. [Physical Layer Parameters](#physical-layer-parameters)
7. [MAC Layer Parameters](#mac-layer-parameters)
8. [Multi-Agent SAC Parameters](#multi-agent-sac-parameters)
9. [Energy Model Parameters](#energy-model-parameters)
10. [IEEE 802.11 Standards](#ieee-80211-standards)

---

## Simulation Environment Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `MAP_LENGTH` | 500 | m | Length of the simulation map |
| `MAP_WIDTH` | 500 | m | Width of the simulation map |
| `MAP_HEIGHT` | 200 | m | Height of the simulation map |
| `SIM_TIME` | 100 × 10⁶ | μs | Total simulation time |
| `NUMBER_OF_DRONES` | 5 | - | Number of UAVs in the network |
| `NUMBER_OF_IOT_NODES` | 20 | - | Number of IoT nodes in the network |
| `IOT_NODE_DENSITY` | 0.00008 | nodes/m² | IoT nodes per square meter |
| `GRID_RESOLUTION` | 20 | m | Grid resolution for path planning |
| `STATIC_CASE` | 0 | - | Whether to simulate a static network |
| `HETEROGENEOUS` | 0 | - | Heterogeneous network support (speed variation) |
| `LOGGING_LEVEL` | INFO | - | Logging level for simulation details |

---

## UAV Hardware Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `PROFILE_DRAG_COEFFICIENT` | 0.012 | - | Profile drag coefficient for UAV aerodynamics |
| `AIR_DENSITY` | 1.225 | kg/m³ | Air density at sea level |
| `ROTOR_SOLIDITY` | 0.05 | - | Ratio of total blade area to disc area |
| `ROTOR_DISC_AREA` | 0.79 | m² | Rotor disc area |
| `BLADE_ANGULAR_VELOCITY` | 400 | rad/s | Blade angular velocity |
| `ROTOR_RADIUS` | 0.5 | m | Rotor radius |
| `INCREMENTAL_CORRECTION_FACTOR` | 0.1 | - | Incremental correction factor for induced power |
| `AIRCRAFT_WEIGHT` | 100 | N | Aircraft weight in Newtons |
| `ROTOR_BLADE_TIP_SPEED` | 500 | m/s | Rotor blade tip speed |
| `MEAN_ROTOR_VELOCITY` | 7.2 | m/s | Mean rotor induced velocity in hover |
| `FUSELAGE_DRAG_RATIO` | 0.3 | - | Fuselage drag ratio |
| `INITIAL_ENERGY` | 20 × 10³ | J | Initial energy capacity of UAV |
| `ENERGY_THRESHOLD` | 2000 | J | Energy threshold for UAV sleep mode |
| `MAX_QUEUE_SIZE` | 200 | packets | Maximum size of UAV's packet queue |

---

## IoT Node Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `IOT_NODE_BATTERY_CAPACITY` | 5000 | J | Initial battery capacity of IoT nodes |
| `IOT_NODE_ENERGY_THRESHOLD` | 500 | J | Energy threshold for IoT node sleep mode |
| `IOT_NODE_MAX_BUFFER_SIZE` | 50 | packets | Maximum number of packets in IoT node buffer |
| `IOT_NODE_DATA_GENERATION_RATE` | 2.0 | packets/s | Data generation rate per IoT node |
| `IOT_NODE_PACKET_SIZE` | 1024 × 8 | bits | Size of IoT node data packets (1024 bytes) |
| `IOT_NODE_SENSING_ENERGY` | 0.05 | J | Energy consumed per sensing operation |
| `IOT_NODE_TRANSMISSION_ENERGY` | 0.2 | J | Energy consumed per packet transmission |
| `IOT_NODE_IDLE_ENERGY` | 0.01 | J/s | Energy consumed per second when idle |

---

## Radio Communication Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `TRANSMITTING_POWER` | 0.1 | W | Transmitting power of nodes |
| `LIGHT_SPEED` | 3 × 10⁸ | m/s | Speed of light |
| `CARRIER_FREQUENCY` | 2.4 × 10⁹ | Hz | Carrier frequency (IEEE 802.11b) |
| `NOISE_POWER` | 4 × 10⁻¹¹ | W | Noise power |
| `RADIO_SWITCHING_TIME` | 100 | μs | Transceiver mode switching time |
| `SNR_THRESHOLD` | 6 | dB | Signal-to-Noise Ratio threshold |
| `SENSING_RANGE` | 750 | m | Range for sensing interference |

---

## Packet and Protocol Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `VARIABLE_PAYLOAD_LENGTH` | 0 | - | Whether to use random payload length |
| `AVERAGE_PAYLOAD_LENGTH` | 1024 × 8 | bits | Average payload length (1024 bytes) |
| `MAXIMUM_PAYLOAD_VARIATION` | 1600 | bits | Maximum payload variation |
| `MAX_TTL` | NUMBER_OF_DRONES + 1 | - | Maximum time-to-live value |
| `PACKET_LIFETIME` | 10 × 10⁶ | μs | Packet lifetime (10 seconds) |
| `IP_HEADER_LENGTH` | 20 × 8 | bits | IP header length (20 bytes) |
| `MAC_HEADER_LENGTH` | 14 × 8 | bits | MAC header length (14 bytes) |

### Packet Type IDs
| Variable | Value | Description |
|----------|-------|-------------|
| `GL_ID_DATA_PACKET` | 0 | Data packet identifier |
| `GL_ID_HELLO_PACKET` | 10000 | Hello packet identifier |
| `GL_ID_ACK_PACKET` | 20000 | Acknowledgment packet identifier |
| `GL_ID_VF_PACKET` | 30000 | Virtual Force packet identifier |
| `GL_ID_GRAD_MESSAGE` | 40000 | Gradient message identifier |

---

## Physical Layer Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `PATH_LOSS_EXPONENT` | 2 | - | Path loss exponent for large-scale fading |
| `PLCP_PREAMBLE` | 128 + 16 | bits | PLCP preamble including sync and SFD |
| `PLCP_HEADER` | 8 + 8 + 16 + 16 | bits | PLCP header (signal, service, length, HEC) |
| `PHY_HEADER_LENGTH` | PLCP_PREAMBLE + PLCP_HEADER | bits | Total physical layer header length |
| `ACK_HEADER_LENGTH` | 16 × 8 | bits | ACK packet header length (16 bytes) |
| `ACK_PACKET_LENGTH` | ACK_HEADER_LENGTH + 14 × 8 | bits | Total ACK packet length |
| `HELLO_PACKET_PAYLOAD_LENGTH` | 256 | bits | Hello packet payload length |
| `HELLO_PACKET_LENGTH` | IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + HELLO_PACKET_PAYLOAD_LENGTH | bits | Total hello packet length |

---

## MAC Layer Parameters

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `SLOT_DURATION` | 20 | μs | Duration of one time slot |
| `SIFS_DURATION` | 10 | μs | Short Inter-Frame Space duration |
| `DIFS_DURATION` | SIFS_DURATION + (2 × SLOT_DURATION) | μs | DCF Inter-Frame Space duration |
| `CW_MIN` | 31 | - | Initial contention window size |
| `ACK_TIMEOUT` | ACK_PACKET_LENGTH / BIT_RATE × 10⁶ + SIFS_DURATION + 50 | μs | Maximum waiting time for ACK |
| `MAX_RETRANSMISSION_ATTEMPT` | 5 | - | Maximum number of retransmission attempts |

---

## Multi-Agent SAC Parameters

### Network Architecture Parameters
| Variable | Value | Description |
|----------|-------|-------------|
| `state_dim` | Calculated dynamically | State dimension based on environment |
| `action_dim` | 3 | Action dimension [dx, dy, dz] for 3D movement |
| `hidden_dim` | 256 | Hidden layer dimension for neural networks |

### SAC Algorithm Parameters
| Variable | Value | Description |
|----------|-------|-------------|
| `learning_rate` | 3e-4 | Learning rate for all networks |
| `gamma` | 0.99 | Discount factor for future rewards |
| `tau` | 0.005 | Soft update parameter for target networks |
| `alpha` | 0.2 | Entropy coefficient for exploration |
| `auto_alpha` | True | Whether to automatically adjust alpha |
| `target_entropy` | -action_dim | Target entropy for automatic alpha adjustment |
| `buffer_size` | 100000 | Size of replay buffer |
| `batch_size` | 64 | Batch size for training |

### State Space Components
- **UAV State**: Position (3) + Energy (1) = 4 dimensions
- **IoT Nodes State**: For each IoT node: Position (3) + Buffer Level (1) + Energy (1) = 5 dimensions × number of IoT nodes
- **Other Agents State**: (num_agents - 1) × 3 dimensions for other UAV positions

### Reward Function Components
- **Collection Reward**: +1.0 for each data collection event
- **Movement Penalty**: -0.01 × movement magnitude
- **Information Loss Penalty**: -10.0 × information loss
- **Energy Loss Penalty**: -5.0 × energy loss

---

## Energy Model Parameters

### UAV Energy Model (Y. Zeng 2019)
| Variable | Symbol | Value | Unit | Description |
|----------|--------|-------|------|-------------|
| `delta` | δ | 0.012 | - | Profile drag coefficient |
| `rho` | ρ | 1.225 | kg/m³ | Air density |
| `s` | σ | 0.05 | - | Rotor solidity |
| `a` | A | 0.79 | m² | Rotor disc area |
| `omega` | ω | 400 | rad/s | Blade angular velocity |
| `r` | R | 0.5 | m | Rotor radius |
| `k` | k | 0.1 | - | Incremental correction factor |
| `w` | W | 100 | N | Aircraft weight |
| `u_tip` | U_tip | 500 | m/s | Tip speed of rotor blade |
| `v0` | v₀ | 7.2 | m/s | Mean rotor induced velocity in hover |
| `d0` | d₀ | 0.3 | - | Fuselage drag ratio |

### Power Consumption Components
- **Blade Profile Power**: P₀ × (1 + 3v²/U_tip²)
- **Induced Power**: Pᵢ × √(1 + v⁴/(4v₀⁴)) - v²/(2v₀²)
- **Parasite Power**: 0.5 × d₀ × ρ × σ × A × v³

---

## IEEE 802.11 Standards

### IEEE 802.11b (Default)
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `carrier_frequency` | 2.4 × 10⁹ | Hz | Carrier frequency |
| `bit_rate` | 2 × 10⁶ | bps | Bit rate (up to 11 Mbps) |
| `bandwidth` | 22 × 10⁶ | Hz | Channel bandwidth |
| `slot_duration` | 20 | μs | Time slot duration |
| `SIFS` | 10 | μs | Short Inter-Frame Space |
| `snr_threshold` | 6 | dB | SNR threshold |

### IEEE 802.11a
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `carrier_frequency` | 5 × 10⁹ | Hz | Carrier frequency |
| `bit_rate` | 54 × 10⁶ | bps | Bit rate |
| `bandwidth` | 20 × 10⁶ | Hz | Channel bandwidth |
| `slot_duration` | 9 | μs | Time slot duration |
| `SIFS` | 16 | μs | Short Inter-Frame Space |
| `snr_threshold` | 21 | dB | SNR threshold |

### IEEE 802.11g
| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `carrier_frequency` | 2.4 × 10⁹ | Hz | Carrier frequency |
| `bit_rate` | 54 × 10⁶ | bps | Bit rate |
| `bandwidth` | 20 × 10⁶ | Hz | Channel bandwidth |
| `slot_duration` | 9 | μs | Time slot duration |
| `SIFS` | 10 | μs | Short Inter-Frame Space |
| `snr_threshold` | 21 | dB | SNR threshold |

---

## Research-Relevant Metrics

### Performance Metrics
- **Information Loss**: Total packets dropped due to buffer overflow
- **Energy Loss**: Total energy consumed by UAVs and IoT nodes
- **Collection Efficiency**: Ratio of collected packets to generated packets
- **Network Lifetime**: Time until first node runs out of energy
- **Coverage Area**: Area covered by UAVs for data collection

### Training Metrics
- **Episode Reward**: Total reward accumulated per episode
- **Actor Loss**: Policy network loss during training
- **Critic Loss**: Q-function network loss during training
- **Value Loss**: Value network loss during training
- **Entropy**: Policy entropy for exploration measurement

### Visualization Data
- **UAV Positions**: 3D coordinates of all UAVs over time
- **IoT Buffer Levels**: Buffer utilization of all IoT nodes
- **Energy Levels**: Energy consumption of all nodes
- **Collection Events**: Data collection events and their locations

---

## Notes for Research

1. **Scalability**: The simulation supports variable numbers of UAVs and IoT nodes
2. **Energy Efficiency**: Both UAV and IoT node energy models are implemented
3. **Real-time Adaptation**: Multi-agent SAC enables real-time path planning
4. **Interference Modeling**: CSMA/CA protocol with interference detection
5. **Mobility Models**: Multiple 3D mobility models available (Gauss-Markov, Random Walk, Random Waypoint)
6. **Routing Protocols**: Multiple routing protocols available (DSDV, GPSR, Q-Routing, etc.)
7. **MAC Protocols**: CSMA/CA and Pure ALOHA implementations
8. **Path Planning**: A* algorithm for optimal path planning
9. **Channel Assignment**: Centralized channel assignment for interference mitigation
10. **Data Collection**: UAV-based data collection from ground IoT nodes

This comprehensive variable documentation provides all necessary parameters for reproducing and extending the UAV network simulation research. 