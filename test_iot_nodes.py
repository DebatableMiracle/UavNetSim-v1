#!/usr/bin/env python3
"""
Test script for IoT nodes functionality
"""

import simpy
from utils import config
from simulator.simulator import Simulator
from entities.iot_nodes import IoTNode

def test_iot_nodes():
    """Test IoT nodes functionality"""
    
    print("Testing IoT Nodes Implementation")
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
    
    # Test IoT node properties
    if sim.iot_nodes:
        iot_node = sim.iot_nodes[0]
        print(f"\nIoT Node {iot_node.identifier} properties:")
        print(f"  Position: {iot_node.coords}")
        print(f"  Buffer size: {iot_node.max_buffer_size}")
        print(f"  Battery capacity: {iot_node.battery_capacity} J")
        print(f"  Data generation rate: {iot_node.data_generation_rate} packets/sec")
        
        # Test buffer status
        buffer_status = iot_node.get_buffer_status()
        print(f"  Buffer status: {buffer_status}")
        
        # Test energy status
        energy_status = iot_node.get_energy_status()
        print(f"  Energy status: {energy_status}")
    
    # Run simulation for a short time to test functionality
    print(f"\nRunning simulation for 1 second...")
    env.run(until=1e6)  # 1 second
    
    # Check IoT nodes after simulation
    if sim.iot_nodes:
        iot_node = sim.iot_nodes[0]
        buffer_status = iot_node.get_buffer_status()
        energy_status = iot_node.get_energy_status()
        
        print(f"\nAfter 1 second simulation:")
        print(f"  Packets generated: {buffer_status['total_generated']}")
        print(f"  Buffer usage: {buffer_status['current_size']}/{buffer_status['max_size']}")
        print(f"  Energy remaining: {energy_status['residual_energy']:.2f} J ({energy_status['energy_percentage']:.1f}%)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_iot_nodes() 