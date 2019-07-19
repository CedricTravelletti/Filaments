# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Given a set of hyperparameters, compute the *kriging* predictor and the
cross validation error.

SIMULATION will mean the points at which we predict, conditioning the ones at
which we condition.

"""
from pyfilaments.gaussian_process import GaussianProcess
from pyfilaments.grid import Grid
from pyfilaments.plotting import plot_filament
import volcapy.covariance.covariance_tools as cl

import numpy as np
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

sigma0 = 0.2
lambda0 = 0.2
m0 = 0.0

# We noticed its not good to have conditioning points on the boundary.
x_spacings_cond = np.linspace(0.2, 0.8, n_cond_x)
y_spacings_cond = np.linspace(0.2, 0.8, n_cond_y)
conditioning_grid = Grid(x_spacings_cond, y_spacings_cond)

# 1D part of regular grid for simulation points.
x_spacings_sim = np.linspace(0, 1, n_sim_x)
y_spacings_sim = np.linspace(0, 1, n_sim_y)
simulation_grid = Grid(x_spacings_sim, y_spacings_sim)

mygp = GaussianProcess(conditioning_grid, simulation_grid, sigma0, lambda0)

n_realizations = 100
# Generate pseudorealizations and see if filaments.
for i in range(n_realizations):
    sim_points_vals = mygp.generate_pseudorealization()
    gradient = mygp.gradient()
    hessian = mygp.hessian()

    # To filament estimation.
    filament_inds, filament_vals = mygp.compute_filament_criterion(
        gradient, hessian, 1e-3)

    if len(filament_inds) > 40:
        print("FOUND FILAMENT")
        plot_filament(filament_inds, sim_points_vals, mygp.simulation_grid)
    else:
        print("found none")


