import numpy as np
import simpy
from simulator.simulator import Simulator
from utils import config

class CustomFANETSim:
    """
    Multi-agent RL environment wrapping the real Simulator.
    Each agent controls a drone's movement. Observation is global. Step advances SimPy by a fixed interval.
    """
    def __init__(self, n_drones=None, sim_time=None, step_interval=1e5, seed=0):
        """
        Args:
            n_drones: number of drones/agents (default: config.NUMBER_OF_DRONES)
            sim_time: total simulation time (default: config.SIM_TIME)
            step_interval: RL step interval in microseconds (default: 1e5 = 0.1s)
            seed: random seed
        """
        self.n_drones = n_drones or config.NUMBER_OF_DRONES
        self.sim_time = sim_time or config.SIM_TIME
        self.step_interval = step_interval
        self.seed(seed)
        self._init_sim()

    def _init_sim(self):
        self.env = simpy.Environment()
        self.channel_states = {i: simpy.Resource(self.env, capacity=1) for i in range(self.n_drones)}
        self.sim = Simulator(seed=self._seed, env=self.env, channel_states=self.channel_states, n_drones=self.n_drones, total_simulation_time=self.sim_time)
        self._done = False
        self._last_time = 0

    def reset(self):
        """Reset the simulator and return the initial global observation."""
        self._init_sim()
        self.env.run(until=0.0)
        self._done = False
        self._last_time = 0
        return self._get_obs()

    def step(self, actions):
        """
        Apply actions (movement for each drone), advance SimPy, return (obs, reward, done, info).
        Args:
            actions: np.ndarray of shape (n_drones, 3) (e.g., velocity or waypoint for each drone)
        Returns:
            obs: global observation
            reward: global reward
            done: episode done
            info: dict with extra info
        """
        # Set each drone's velocity (or target position)
        for i, drone in enumerate(self.sim.drones):
            # TODO: Optionally support waypoints or more complex control
            drone.velocity = np.array(actions[i])
        # Advance SimPy by step_interval
        self.env.run(until=self._last_time + self.step_interval)
        self._last_time += self.step_interval
        obs = self._get_obs()
        reward, info = self._get_reward()
        done = self._check_done()
        self._done = done
        return obs, reward, done, info

    def _get_obs(self):
        """Return the global observation (positions, energies, queues, etc. of all drones and nodes)."""
        obs = []
        for drone in self.sim.drones:
            obs.extend(drone.coords)
            obs.append(drone.residual_energy)
            # TODO: add drone queue/buffer info if needed
        # TODO: add node info if needed (currently only drones)
        return np.array(obs, dtype=np.float32)

    def _get_reward(self):
        """Compute global reward and info dict (e.g., delivery ratio, energy, info loss, etc.)."""
        # Example: negative sum of energy used so far, or use sim.metrics
        # TODO: customize reward as needed
        total_energy = sum([config.INITIAL_ENERGY - d.residual_energy for d in self.sim.drones])
        # Use metrics if available
        pdr = getattr(self.sim.metrics, 'datapacket_arrived', set())
        pdr = len(pdr) / (self.sim.metrics.datapacket_generated_num or 1)
        reward = pdr - 0.01 * total_energy
        info = {'pdr': pdr, 'energy_used': total_energy}
        return reward, info

    def _check_done(self):
        """Check if episode is done (e.g., all drones out of energy or sim time exceeded)."""
        if self._last_time >= self.sim_time:
            return True
        if all(d.residual_energy <= 0 for d in self.sim.drones):
            return True
        return False

    def get_spaces(self):
        """
        Returns (action_space, observation_space) as tuples of (low, high) arrays.
        Action: (n_drones, 3), e.g., velocity in each axis, bounded by config or drone speed.
        Observation: global state vector.
        """
        action_low = np.full((self.n_drones, 3), -20)  # TODO: set proper bounds
        action_high = np.full((self.n_drones, 3), 20)
        obs_dim = self.n_drones * 4  # [x,y,z,energy] per drone
        obs_low = np.full(obs_dim, -np.inf)
        obs_high = np.full(obs_dim, np.inf)
        return (action_low, action_high), (obs_low, obs_high)

    def get_num_agents(self):
        """Return the number of agents (drones)."""
        return self.n_drones

    def seed(self, seed=0):
        """Set the random seed for reproducibility."""
        self._seed = seed
        np.random.seed(seed) 