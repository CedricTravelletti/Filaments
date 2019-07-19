# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

SIMULATION will mean the points at which we predict, conditioning the ones at
which we condition.

"""
from pyfilaments.gaussian_process import GaussianProcess
from pyfilaments.grid import Grid
import volcapy.covariance.covariance_tools as cl

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import os

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

n_cond_x = 5
n_cond_y = 2

n_sim_x = 100
n_sim_y = 100

# We noticed its not good to have conditioning points on the boundary.
x_spacings_cond = np.linspace(0.2, 0.8, n_cond_x)
y_spacings_cond = np.linspace(0.2, 0.8, n_cond_y)
conditioning_grid = Grid(x_spacings_cond, y_spacings_cond)

# 1D part of regular grid for simulation points.
x_spacings_sim = np.linspace(0, 1, n_sim_x)
y_spacings_sim = np.linspace(0, 1, n_sim_y)
simulation_grid = Grid(x_spacings_sim, y_spacings_sim)

sigma0 = 0.2
lambda0 = 0.2
m0 = 0.0

mygp = GaussianProcess(conditioning_grid, simulation_grid, sigma0, lambda0)

# simulation_grid.plot_list(mygp.cov_sim[0, :])
cond_points_vals = torch.from_numpy(
        np.array(100*[0.0]).astype(np.float32))
cond_points_vals[30] = 2.0


# Generate random vals.
mean = np.zeros(mygp.cov_cond.shape[0])
cond_points_vals = np.random.multivariate_normal(mean, mygp.cov_cond)
cond_points_vals = torch.from_numpy(cond_points_vals.astype(np.float32))
sim_points_vals = mygp.condition(cond_points_vals)
# simulation_grid.plot_list(sim_points_vals)

# simulation_grid.plot_list_3d(sim_points_vals)

gradient = mygp.gradient()
norm = torch.einsum("ij,ij->i", (gradient, gradient))
norm = torch.sqrt(norm)
simulation_grid.plot_list_3d(sim_points_vals, norm)

hessian = mygp.hessian()

# To filament estimation.
filament_inds, filament_vals = mygp.compute_filament_criterion(
        gradient, hessian, 1e-2)

# Plot results.
# Create a zero vector.
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
