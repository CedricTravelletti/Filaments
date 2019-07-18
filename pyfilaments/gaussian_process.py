""" Gaussian Process for Conditional simulations class.


Covariance Matrices
-------------------
Most kernels have a variance parameter sigma0^2 that just appears as a
multiplicative constant. To make its optimization easier, we strip it of the
covariance matrix, store it as a model parameter (i.e. as an attribute of the
class) and include it manually in the experessions where it shows up.

This means that when one sees a covariance matrix in the code, it generally
doesn't include the sigma0 factor, which has to be included by hand.


The model covariance matrix K is to big to be stored, but during conditioning
it only shows up in the form K * F^t. It is thus sufficient to compute this
product once and for all. We call it the *covariance pushforward*.

"""
import volcapy.covariance.covariance_tools as cl
import numpy as np
import torch
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


class GaussianProcess():
    def __init__(self, conditioning_grid, simulation_grid,
            sigma0, lambda0):
        """

        Parameters
        ----------
       conditioning_grid: Grid
    A grid (see pyFilaments.grid.Grid) object defining the points on
            which the full realization will be conditioned.
        simulation_grid: Grid
            Full grid on which we want to generate pseudo-realizations.
        sigma0: float
            Variance parameter for the gaussian kernel.
        lambda0: float
            Lengthscale parameter for the gaussian kernel.

        """
        self.sigma0 = torch.tensor(sigma0, dtype=torch.float32)
        self.lambda0 = torch.tensor(lambda0, dtype=torch.float32)

        self.ncond = conditioning_grid.n_cells
        self.nsim = simulation_grid.n_cells


        # Create a tempory array containing ALL cells.
        # The conditioninig ones are first, then the simulation ones.
        # This trick alllows to compute the covariance between everything in
        # one go.
        full_cells = torch.from_numpy(
                np.vstack((
                        conditioning_grid.cells_list,
                        simulation_grid.cells_list)))

        tmp_cov = cl.compute_full_cov(self.lambda0,
                full_cells,
                device=cpu)

        # Extract back the values.
        self.cov_cond = tmp_cov[:self.ncond, :self.ncond]
        self.cov_sim = tmp_cov[-self.nsim:, -self.nsim:]
        self.cov_sim_cond = tmp_cov[-self.nsim:, :self.ncond]


        """
        cov_full = sigma0**2 * cov_full

        # Do not need the full one, hence extract covariance between
        # conditioning points themselves, and between conditioning points and
        # simulation points.
        # EXTRACT THE BLOCKS.
        self.cov_cond= cov_full[:self.n_conds, :self.n_conds]

        # Covarianve between simulation points and data points.
        self.cov_sims_cond = cov_full[self.n_conds+1:, :self.n_conds]
        self.cov_full = cov_full
        """

    def condition(self, cond_points_vals):
        """ Condition on the values at the conditioning points.

        Parameters
        ----------
        cond_points_vals: array
            Array of length n_conds containing the values to condition on.
                (Order should be the same as ordering of the conditioning
                points.

        Returns
        -------
        sim_points_vals: array
            Krigging mean at the simulation points.

        """
        sim_points_vals = torch.mm(
            self.cov_sim_cond,
            torch.mm(
                torch.inverse(self.cov_cond),
                cond_points_vals.reshape(-1, 1)))

        return sim_points_vals.numpy()