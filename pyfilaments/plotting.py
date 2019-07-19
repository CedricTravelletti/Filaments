import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def plot_filament(filament_inds, sim_points_vals, simulation_grid):
    plot_vals = sim_points_vals.copy()

    # In order to be able to see heigts, we keep the values, but where
    # there is a filament we replace it with twice the maximum.
    plot_vals[filament_inds] = 2 * np.max(sim_points_vals)
    
    # Normalize so colors wotk.
    plot_vals = plot_vals/plot_vals.max()  # normalize 0..1
    
    # Dark red is hard to see, so change to light.
    plot_vals[plot_vals > 0.6] = 0.8
    
    # Convert to grid.
    heights_2d = simulation_grid.reshape_to_2d(sim_points_vals)
    plot_vals_2d = simulation_grid.reshape_to_2d(plot_vals)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plt.hold(True)
    X, Y = np.meshgrid(simulation_grid.x_spacings, simulation_grid.y_spacings)
    surf = ax.plot_surface(
                X, Y, heights_2d, rstride=1, cstride=1,
                    facecolors=cm.jet(plot_vals_2d), alpha=1.0)
    
    plt.show()
