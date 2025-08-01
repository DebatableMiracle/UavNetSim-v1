import random
import numpy as np
import matplotlib.pyplot as plt
from phy.channel import Channel
from entities.drone import Drone
from entities.iot_nodes import IoTNode
from entities.obstacle import SphericalObstacle, CubeObstacle
from simulator.metrics import Metrics
from mobility import start_coords
from path_planning.astar import astar
from utils import config
from utils.util_function import grid_map
from allocation.central_controller import CentralController
from visualization.static_drawing import scatter_plot, scatter_plot_with_obstacles


class Simulator:
    """
    Description: simulation environment

    Attributes:
        env: simpy environment
        total_simulation_time: discrete time steps, in nanosecond
        n_drones: number of the drones
        n_iot_nodes: number of the IoT nodes
        channel_states: a dictionary, used to describe the channel usage
        channel: wireless channel
        metrics: Metrics class, used to record the network performance
        drones: a list, contains all drone instances
        iot_nodes: a list, contains all IoT node instances

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2025/7/8
    """

    def __init__(self,
                 seed,
                 env,
                 channel_states,
                 n_drones,
                 n_iot_nodes=None,
                 total_simulation_time=config.SIM_TIME):

        self.env = env
        self.seed = seed
        self.total_simulation_time = total_simulation_time  # total simulation time (ns)

        self.n_drones = n_drones  # total number of drones in the simulation
        self.n_iot_nodes = n_iot_nodes if n_iot_nodes else config.NUMBER_OF_IOT_NODES
        self.channel_states = channel_states
        self.channel = Channel(self.env)

        self.metrics = Metrics(self)  # use to record the network performance

        # NOTE: if distributed optimization is adopted, remember to comment this to speed up simulation
        # self.central_controller = CentralController(self)

        start_position = start_coords.get_random_start_point_3d(seed)
        # start_position = start_coords.get_customized_start_point_3d()

        self.drones = []
        print('Seed is: ', self.seed)
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 10

            print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env,
                          node_id=i,
                          coords=start_position[i],
                          speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i),
                          simulator=self)

            self.drones.append(drone)

        # Initialize IoT nodes
        self.iot_nodes = []
        self._initialize_iot_nodes()

        # scatter_plot_with_spherical_obstacles(self)
        # scatter_plot(self)  # Disabled during training to avoid interruptions

        self.env.process(self.show_performance())
        self.env.process(self.show_time())

    def _initialize_iot_nodes(self):
        """
        Initialize IoT nodes with proper distribution on the ground
        """
        print(f'Initializing {self.n_iot_nodes} IoT nodes...')
        
        # Calculate area and determine node density
        area = config.MAP_LENGTH * config.MAP_WIDTH
        target_density = config.IOT_NODE_DENSITY
        
        # Generate IoT node positions on the ground (z=0)
        iot_positions = self._generate_iot_positions()
        
        for i in range(self.n_iot_nodes):
            if i < len(iot_positions):
                coords = iot_positions[i]
            else:
                # Fallback: random position on ground
                coords = [
                    random.uniform(0, config.MAP_LENGTH),
                    random.uniform(0, config.MAP_WIDTH),
                    0  # Ground level
                ]
            
            # Create IoT node with configuration parameters
            iot_node = IoTNode(
                env=self.env,
                node_id=i + self.n_drones,  # Offset to avoid ID conflicts with drones
                coords=coords,
                simulator=self,
                data_generation_rate=config.IOT_NODE_DATA_GENERATION_RATE,
                max_buffer_size=config.IOT_NODE_MAX_BUFFER_SIZE,
                battery_capacity=config.IOT_NODE_BATTERY_CAPACITY,
                energy_threshold=config.IOT_NODE_ENERGY_THRESHOLD,
                packet_size=config.IOT_NODE_PACKET_SIZE
            )
            
            self.iot_nodes.append(iot_node)
            print(f'IoT Node: {i + self.n_drones} initialized at position: {coords}')

    def _generate_iot_positions(self):
        """
        Generate IoT node positions based on density requirements
        Uses a grid-based approach to ensure proper distribution
        """
        positions = []
        
        # Calculate grid spacing based on density
        area = config.MAP_LENGTH * config.MAP_WIDTH
        target_nodes = min(self.n_iot_nodes, int(area * config.IOT_NODE_DENSITY))
        
        # Use a grid-based approach for better distribution
        grid_size = int(np.sqrt(target_nodes)) + 1
        cell_length = config.MAP_LENGTH / grid_size
        cell_width = config.MAP_WIDTH / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= target_nodes:
                    break
                    
                # Add some randomness within each cell
                x = (i * cell_length) + random.uniform(0, cell_length * 0.8)
                y = (j * cell_width) + random.uniform(0, cell_width * 0.8)
                z = 0  # Ground level
                
                positions.append([x, y, z])
        
        # If we need more positions, add random ones
        while len(positions) < self.n_iot_nodes:
            x = random.uniform(0, config.MAP_LENGTH)
            y = random.uniform(0, config.MAP_WIDTH)
            z = 0
            positions.append([x, y, z])
        
        return positions[:self.n_iot_nodes]

    def get_all_nodes(self):
        """
        Get all nodes (drones + IoT nodes) in the simulation
        
        Returns:
            list: Combined list of all nodes
        """
        return self.drones + self.iot_nodes

    def get_iot_nodes_in_range(self, position, range_distance):
        """
        Get IoT nodes within a certain range of a position
        
        Args:
            position: [x, y, z] coordinates
            range_distance: maximum distance in meters
            
        Returns:
            list: IoT nodes within range
        """
        nodes_in_range = []
        for iot_node in self.iot_nodes:
            distance = np.linalg.norm(np.array(position) - np.array(iot_node.coords))
            if distance <= range_distance:
                nodes_in_range.append(iot_node)
        return nodes_in_range

    def show_time(self):
        while True:
            # print('At time: ', self.env.now / 1e6, ' s.')  # Disabled during training

            # the simulation process is displayed every 0.5s
            yield self.env.timeout(0.5*1e6)

    def show_performance(self):
        yield self.env.timeout(self.total_simulation_time - 1)

        # scatter_plot(self)  # Disabled during training

        # self.metrics.print_metrics()  # Disabled during training
