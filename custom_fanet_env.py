import numpy as np

# ===== Constants =====
NUM_UAVS = 4
NUM_NODES = 10
UAV_INIT_ENERGY = 100.0
NODE_INIT_ENERGY = 100.0
NODE_BUFFER_SIZE = 5
NODE_PACKET_PER_STEP = 1
UAV_COLLECT_RANGE = 2.0
UAV_MAX_VEL = 1.0
ALPHA = 0.1  # energy cost weight
BETA = 1.0   # info loss weight
SPACE_LOW = np.array([0, 0, 0])
SPACE_HIGH = np.array([20, 20, 10])

class CustomFANETEnv:
    """
    Multi-agent FANET environment for continuous-action RL 
    4 UAVs (agents), 10 fixed IoT nodes. State and action spaces are continuous.
    """
    def __init__(self, seed=None):
        self.num_uavs = NUM_UAVS
        self.num_nodes = NUM_NODES
        self.uav_pos = np.zeros((NUM_UAVS, 3))
        self.uav_energy = np.ones(NUM_UAVS) * UAV_INIT_ENERGY
        self.node_pos = np.zeros((NUM_NODES, 3))
        self.node_energy = np.ones(NUM_NODES) * NODE_INIT_ENERGY
        self.node_buffer = np.zeros(NUM_NODES, dtype=int)
        self._rng = np.random.RandomState(seed)
        self.seed(seed)
        self.reset()
        # Action/observation space bounds
        self._action_low = np.full((NUM_UAVS, 3), -UAV_MAX_VEL)
        self._action_high = np.full((NUM_UAVS, 3), UAV_MAX_VEL)
        obs_dim = NUM_UAVS * 4 + NUM_NODES * 4
        self._obs_low = np.concatenate([
            np.tile(SPACE_LOW, NUM_UAVS),
            np.zeros(NUM_UAVS),
            np.tile(SPACE_LOW, NUM_NODES),
            np.zeros(NUM_NODES),
            np.zeros(NUM_NODES)
        ])
        self._obs_high = np.concatenate([
            np.tile(SPACE_HIGH, NUM_UAVS),
            np.ones(NUM_UAVS) * UAV_INIT_ENERGY,
            np.tile(SPACE_HIGH, NUM_NODES),
            np.ones(NUM_NODES) * NODE_INIT_ENERGY,
            np.ones(NUM_NODES) * NODE_BUFFER_SIZE
        ])

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self._rng = np.random.RandomState(seed)

    def reset(self):
        """Reset the environment to the initial state."""
        self.uav_pos = self._rng.uniform(SPACE_LOW, SPACE_HIGH, (NUM_UAVS, 3))
        self.uav_energy = np.ones(NUM_UAVS) * UAV_INIT_ENERGY
        grid_x = np.linspace(2, 18, int(np.ceil(np.sqrt(NUM_NODES))))
        grid_y = np.linspace(2, 18, int(np.ceil(np.sqrt(NUM_NODES))))
        coords = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)[:NUM_NODES]
        self.node_pos = np.hstack([coords, np.zeros((NUM_NODES, 1))])
        self.node_energy = np.ones(NUM_NODES) * NODE_INIT_ENERGY
        self.node_buffer = np.zeros(NUM_NODES, dtype=int)
        return self._get_obs()

    def step(self, actions):
        """
        Take a step in the environment.
        Args:
            actions: np.ndarray of shape (NUM_UAVS, 3), continuous (dx, dy, dz) for each UAV.
        Returns:
            obs, reward, done, info
        """
        actions = np.clip(actions, self._action_low, self._action_high)
        self.uav_pos += actions
        self.uav_pos = np.clip(self.uav_pos, SPACE_LOW, SPACE_HIGH)
        move_energy = np.linalg.norm(actions, axis=1)
        self.uav_energy -= move_energy
        self.uav_energy = np.clip(self.uav_energy, 0, UAV_INIT_ENERGY)
        info_lost = 0
        for i in range(self.num_nodes):
            for _ in range(NODE_PACKET_PER_STEP):
                if self.node_buffer[i] < NODE_BUFFER_SIZE:
                    self.node_buffer[i] += 1
                else:
                    info_lost += 1
        for u in range(self.num_uavs):
            if self.uav_energy[u] <= 0:
                continue
            dists = np.linalg.norm(self.node_pos - self.uav_pos[u], axis=1)
            in_range = np.where(dists <= UAV_COLLECT_RANGE)[0]
            for n in in_range:
                if self.node_buffer[n] > 0:
                    self.node_buffer[n] = 0
        total_energy_used = np.sum(move_energy)
        reward = - (ALPHA * total_energy_used + BETA * info_lost)
        done = np.all(self.uav_energy <= 0)
        info = {'info_lost': info_lost, 'total_energy_used': total_energy_used}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Return the full environment observation as a flat array."""
        obs = np.concatenate([
            self.uav_pos.flatten(),
            self.uav_energy.flatten(),
            self.node_pos.flatten(),
            self.node_energy.flatten(),
            self.node_buffer.flatten()
        ])
        return obs

    def get_spaces(self):
        """
        Returns (action_space, observation_space) as tuples of (low, high) arrays.
        action_space: (low, high) of shape (NUM_UAVS, 3)
        observation_space: (low, high) of shape (obs_dim,)
        """
        return (self._action_low, self._action_high), (self._obs_low, self._obs_high)

    def get_agent_obs(self, uav_idx):
        """
        Returns the observation for a single UAV agent.
        Includes: its pos (3), energy (1), all node pos (10x3), node buffer (10), node energy (10).
        """
        obs = np.concatenate([
            self.uav_pos[uav_idx],
            [self.uav_energy[uav_idx]],
            self.node_pos.flatten(),
            self.node_buffer.flatten(),
            self.node_energy.flatten()
        ])
        return obs

    def get_agent_obs_space(self):
        """
        Returns (low, high) arrays for a single agent's observation.
        """
        low = np.concatenate([
            SPACE_LOW,
            [0],
            np.tile(SPACE_LOW, NUM_NODES),
            np.zeros(NUM_NODES),
            np.zeros(NUM_NODES)
        ])
        high = np.concatenate([
            SPACE_HIGH,
            [UAV_INIT_ENERGY],
            np.tile(SPACE_HIGH, NUM_NODES),
            np.ones(NUM_NODES) * NODE_BUFFER_SIZE,
            np.ones(NUM_NODES) * NODE_INIT_ENERGY
        ])
        return low, high

    def get_num_agents(self):
        """Return the number of UAV agents."""
        return self.num_uavs 