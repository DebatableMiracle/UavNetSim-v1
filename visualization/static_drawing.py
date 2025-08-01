import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import config
from utils.util_function import euclidean_distance_3d
from phy.large_scale_fading import maximum_communication_range


def scatter_plot(simulator):
    """Draw a static scatter plot, includes communication edges (without obstacles)"""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes(Axes3D(fig))

    # Plot drones (red dots in 3D space)
    for drone1 in simulator.drones:
        for drone2 in simulator.drones:
            if drone1.identifier != drone2.identifier:
                ax.scatter(drone1.coords[0], drone1.coords[1], drone1.coords[2], c='red', s=50, marker='^', label='UAV' if drone1.identifier == 0 else "")
                distance = euclidean_distance_3d(drone1.coords, drone2.coords)
                if distance <= maximum_communication_range():
                    x = [drone1.coords[0], drone2.coords[0]]
                    y = [drone1.coords[1], drone2.coords[1]]
                    z = [drone1.coords[2], drone2.coords[2]]
                    ax.plot(x, y, z, color='black', linestyle='dashed', linewidth=1, alpha=0.5)

    # Plot IoT nodes (green dots on ground level)
    if hasattr(simulator, 'iot_nodes') and simulator.iot_nodes:
        for iot_node in simulator.iot_nodes:
            ax.scatter(iot_node.coords[0], iot_node.coords[1], iot_node.coords[2], 
                      c='green', s=30, marker='o', label='IoT Node' if iot_node.identifier == simulator.n_drones else "")
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax.set_xlim(0, config.MAP_LENGTH)
    ax.set_ylim(0, config.MAP_WIDTH)
    ax.set_zlim(0, config.MAP_HEIGHT)

    # maintain the proportion of the x, y and z axes
    ax.set_box_aspect([config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'UAV Network with IoT Nodes\n({len(simulator.drones)} UAVs, {len(simulator.iot_nodes) if hasattr(simulator, "iot_nodes") else 0} IoT Nodes)')

    plt.show()

def scatter_plot_with_obstacles(simulator, grid, path_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for obst_type in simulator.obstacle_type:
        obstacle_points = np.argwhere(grid == obst_type)
        if obstacle_points.size > 0:
            ax.scatter(obstacle_points[:, 0] * config.GRID_RESOLUTION,
                       obstacle_points[:, 1] * config.GRID_RESOLUTION,
                       obstacle_points[:, 2] * config.GRID_RESOLUTION)

    for path in path_list:
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linewidth=3)

    ax.set_xlim(0, config.MAP_LENGTH)
    ax.set_ylim(0, config.MAP_WIDTH)
    ax.set_zlim(0, config.MAP_HEIGHT)

    # maintain the proportion of the x, y and z axes
    ax.set_box_aspect([config.MAP_LENGTH, config.MAP_WIDTH, config.MAP_HEIGHT])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    plt.show()
