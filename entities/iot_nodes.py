import simpy
import numpy as np
import random
import math
import queue
from simulator.log import logger
from entities.packet import DataPacket
from energy.energy_model import EnergyModel
from utils import config


class IoTNode:
    """
    IoT Node implementation for stationary ground-based sensors

    IoT nodes are stationary entities that generate data packets and have limited buffer capacity.
    They are designed to be collected by UAVs to minimize information loss when buffer overflows.
    
    Attributes:
        simulator: the simulation platform that contains everything
        env: simulation environment created by simpy
        identifier: used to uniquely represent an IoT node
        coords: the 3-D position of the IoT node (stationary, z=0 for ground level)
        buffer: queue for storing generated data packets
        max_buffer_size: maximum number of packets the buffer can hold
        residual_energy: the residual energy of IoT node in Joule
        sleep: if the IoT node is in a "sleep" state, it cannot perform operations
        energy_model: energy consumption model for IoT node
        data_generation_rate: rate at which data packets are generated (packets per second)
        packet_size: size of generated data packets in bits
        battery_capacity: initial battery capacity in Joule
        energy_threshold: energy level below which node goes to sleep
        
    Author: Assistant
    Created at: 2024/12/19
    """

    def __init__(self,
                 env,
                 node_id,
                 coords,
                 simulator,
                 data_generation_rate=2.0,  # packets per second
                 max_buffer_size=50,
                 battery_capacity=5000,  # Joule
                 energy_threshold=500,   # Joule
                 packet_size=1024*8):   # 1024 bytes in bits
    
        self.simulator = simulator
        self.env = env
        self.identifier = node_id
        self.coords = coords  # [x, y, 0] - stationary on ground
        self.start_coords = coords
        
        # Random number generator for this node
        self.rng_node = random.Random(self.identifier + self.simulator.seed)
        
        # Buffer and queue management
        self.buffer = queue.Queue()
        self.max_buffer_size = max_buffer_size
        self.buffer_overflow_count = 0
        self.total_packets_generated = 0
        self.total_packets_dropped = 0
        
        # Energy management
        self.residual_energy = battery_capacity
        self.battery_capacity = battery_capacity
        self.energy_threshold = energy_threshold
        self.sleep = False
        
        # Data generation parameters
        self.data_generation_rate = data_generation_rate
        self.packet_size = packet_size
        
        # Energy model for IoT node (simplified compared to UAV)
        self.energy_model = IoTEnergyModel(self)
        
        # Statistics
        self.packets_collected_by_uav = 0
        self.last_collection_time = 0
        
        # Start processes
        self.env.process(self.generate_data_packet())
        self.env.process(self.energy_monitor())
        
        logger.info('IoT Node %s initialized at position %s with buffer size %d and battery capacity %.2f J',
                   self.identifier, self.coords, self.max_buffer_size, self.battery_capacity)

    def generate_data_packet(self):
        """
        Generate data packets following Poisson process
        """
        while True:
            if not self.sleep:
                # Generate packet following exponential distribution
                inter_arrival_time = self.rng_node.expovariate(self.data_generation_rate)
                yield self.env.timeout(inter_arrival_time * 1e6)  # Convert to microseconds
                
                # Check if buffer has space
                if self.buffer.qsize() < self.max_buffer_size:
                    # Create data packet
                    config.GL_ID_DATA_PACKET += 1
                    
                    # Calculate packet length (payload + headers)
                    data_packet_length = (config.IP_HEADER_LENGTH + config.MAC_HEADER_LENGTH +
                                        config.PHY_HEADER_LENGTH + self.packet_size)
                    
                    # For IoT nodes, destination is typically a UAV (we'll use a placeholder for now)
                    # In RL scenarios, UAVs will collect from IoT nodes
                    destination = None  # Will be set when UAV collects
                    
                    # Create packet
                    packet = DataPacket(
                        src_drone=self,  # IoT node as source
                        dst_drone=destination,
                        creation_time=self.env.now,
                        data_packet_id=config.GL_ID_DATA_PACKET,
                        data_packet_length=data_packet_length,
                        simulator=self.simulator,
                        channel_id=0  # Default channel
                    )
                    
                    # Add to buffer
                    self.buffer.put(packet)
                    self.total_packets_generated += 1
                    
                    # Consume energy for packet generation
                    generation_energy = 0.1  # Joule per packet generation
                    self.residual_energy -= generation_energy
                    
                    logger.info('IoT Node %s generated packet %s at time %s (buffer: %d/%d)',
                              self.identifier, packet.packet_id, self.env.now, 
                              self.buffer.qsize(), self.max_buffer_size)
                    
                else:
                    # Buffer overflow - packet dropped
                    self.buffer_overflow_count += 1
                    self.total_packets_dropped += 1
                    
                    logger.warning('IoT Node %s buffer overflow at time %s - packet dropped (overflow count: %d)',
                                 self.identifier, self.env.now, self.buffer_overflow_count)
            else:
                # Node is sleeping, wait before checking again
                yield self.env.timeout(100000)  # 0.1 seconds

    def energy_monitor(self):
        """
        Monitor energy consumption and put node to sleep when energy is low
        """
        while True:
            yield self.env.timeout(1e6)  # Check every second
            
            if self.residual_energy <= self.energy_threshold:
                if not self.sleep:
                    self.sleep = True
                    logger.warning('IoT Node %s going to sleep due to low energy (%.2f J)',
                                 self.identifier, self.residual_energy)
            else:
                if self.sleep:
                    self.sleep = False
                    logger.info('IoT Node %s waking up (energy: %.2f J)',
                              self.identifier, self.residual_energy)

    def get_buffer_status(self):
        """
        Get current buffer status
        
        Returns:
            dict: Buffer status information
        """
        return {
            'current_size': self.buffer.qsize(),
            'max_size': self.max_buffer_size,
            'overflow_count': self.buffer_overflow_count,
            'total_generated': self.total_packets_generated,
            'total_dropped': self.total_packets_dropped,
            'packets_collected': self.packets_collected_by_uav
        }

    def collect_packets(self, collector_uav):
        """
        Collect all packets from this IoT node's buffer
        
        Args:
            collector_uav: UAV that is collecting the packets
            
        Returns:
            list: List of collected packets
        """
        collected_packets = []
        
        while not self.buffer.empty():
            packet = self.buffer.get()
            packet.dst_drone = collector_uav
            collected_packets.append(packet)
            self.packets_collected_by_uav += 1
        
        self.last_collection_time = self.env.now
        
        logger.info('UAV %s collected %d packets from IoT Node %s at time %s',
                   collector_uav.identifier, len(collected_packets), self.identifier, self.env.now)
        
        return collected_packets

    def get_position(self):
        """
        Get current position of IoT node
        
        Returns:
            list: [x, y, z] coordinates
        """
        return self.coords

    def get_energy_status(self):
        """
        Get current energy status
        
        Returns:
            dict: Energy status information
        """
        return {
            'residual_energy': self.residual_energy,
            'battery_capacity': self.battery_capacity,
            'energy_percentage': (self.residual_energy / self.battery_capacity) * 100,
            'sleep': self.sleep
        }


class IoTEnergyModel:
    """
    Simplified energy model for IoT nodes
    
    IoT nodes have much simpler energy consumption compared to UAVs.
    They mainly consume energy for:
    - Sensing/data generation
    - Packet transmission (when UAVs collect)
    - Basic operations
    
    Author: Assistant
    Created at: 2024/12/19
    """
    
    def __init__(self, iot_node):
        self.iot_node = iot_node
        
        # Energy consumption rates (Joule per operation)
        self.sensing_energy = 0.05      # Energy per sensing operation
        self.transmission_energy = 0.2   # Energy per packet transmission
        self.idle_energy = 0.01         # Energy per second when idle
        
        # Start energy monitoring
        self.iot_node.simulator.env.process(self.energy_monitor())
    
    def energy_monitor(self):
        """
        Monitor and consume idle energy
        """
        while True:
            yield self.iot_node.simulator.env.timeout(1e6)  # Every second
            
            if not self.iot_node.sleep:
                # Consume idle energy
                self.iot_node.residual_energy -= self.idle_energy
                
                # Ensure energy doesn't go below 0
                if self.iot_node.residual_energy < 0:
                    self.iot_node.residual_energy = 0 