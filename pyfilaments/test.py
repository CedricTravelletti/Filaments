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
import os

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

n_cond_x = 10
n_cond_y = 10

n_sim_x = 50
n_sim_y = 50


x_spacings_cond = np.linspace(0, 1, n_cond_x)
y_spacings_cond = np.linspace(0, 1, n_cond_y)
conditioning_grid = Grid(x_spacings_cond, y_spacings_cond)

# 1D part of regular grid for simulation points.
x_spacings_sim = np.linspace(0, 1, n_sim_x)
y_spacings_sim = np.linspace(0, 1, n_sim_y)
simulation_grid = Grid(x_spacings_sim, y_spacings_sim)

sigma0 = 1.0
lambda0 = 0.05
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
simulation_grid.plot_list(sim_points_vals)

simulation_grid.plot_list_3d(sim_points_vals)