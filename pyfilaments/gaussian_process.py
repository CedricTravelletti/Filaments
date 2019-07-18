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

        self.simulation_grid = simulation_grid
        self.conditioning_grid = conditioning_grid


        # Create a tempory array containing ALL cells.
        # The conditioninig ones are first, then the simulation ones.
        # This trick alllows to compute the covariance between everything in
        # one go.
        full_cells = torch.from_numpy(
                np.vstack((
                        conditioning_grid.cells_list,
                        simulation_grid.cells_list)))

        tmp_cov = self.sigma0**2 *cl.compute_full_cov(
                self.lambda0,full_cells,device=cpu)

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
        self.kriging_weights = torch.mm(
                torch.inverse(self.cov_cond),
                cond_points_vals.reshape(-1, 1))

        sim_points_vals = torch.mm(
            self.cov_sim_cond,
            self.kriging_weights)

        return sim_points_vals.numpy()

    def gradient(self):
        """ Computes gradient of the field.

        Returns
        -------
        Tensor
            nsim * ndim array containing components of the gradient at every
            point.

        """
        sim_cells = torch.from_numpy(self.simulation_grid.cells_list)
        cond_cells = torch.from_numpy(self.conditioning_grid.cells_list)

        nsim = self.nsim
        ncond = self.ncond
        ndim = 2

        # y_alpha - x_i, along each dimension.
        # Hence has dim nsim * ncond * ndim.
        diff = (sim_cells.unsqueeze(1).expand(nsim, ncond, ndim)
                - cond_cells.unsqueeze(0).expand(nsim, ncond, ndim))
        tmp = (
            self.cov_sim_cond.unsqueeze(2).expand(nsim, ncond, ndim)
            * diff)
        tmp = - (1 / self.lambda0**2) * tmp

        # Notice the dim of tmp: nsim * ncond * ndim.

        gradient = torch.einsum("ijk,jl->ik", (tmp, self.kriging_weights))

        return gradient

    def hessian(self):
        """ Computes hessian of the field.

        Returns
        -------
        Tensor
            nsim * ndim * ndim array containing components of the gradient at every
            point.

        """
        sim_cells = torch.from_numpy(self.simulation_grid.cells_list)
        cond_cells = torch.from_numpy(self.conditioning_grid.cells_list)

        nsim = self.nsim
        ncond = self.ncond
        ndim = 2

        # y_alpha - x_i, along each dimension.
        # Hence has dim nsim * ncond * ndim.
        diff = (sim_cells.unsqueeze(1).expand(nsim, ncond, ndim)
                - cond_cells.unsqueeze(0).expand(nsim, ncond, ndim))

        # Now add a dimension containing all mixed products to compute the
        # hessian.
        # This has dimension nsim * ncond * ndim * ndim.
        diff_mat = torch.einsum("abi,abj->abij", (diff, diff))

        # The full factor come with ones, hence have to squash the identity in
        # there.
        identity = torch.eye(ndim).unsqueeze(0).expand(nsim, ncond, ndim, ndim)

        # Whole factor to multiply with cov.
        tmp = (1 / self.lambda0**4) * (diff_mat - self.lambda0**2 * identity)

        # Compute all products.
        expanded_cov = self.cov_sim_cond.unsqueeze(2).expand(nsim, ncond, ndim)
        expanded_cov = expanded_cov.unsqueeze(3).expand(nsim, ncond, ndim, ndim)
        tmp = (expanded_cov * tmp)

        # Notice the dim of tmp: nsim * ncond * ndim * ndim

        hessian = torch.einsum("ijkm,jl->ikm", (tmp, self.kriging_weights))

        return hessian

    def compute_filament_criterion(self, gradient, hessian, tol):
        filament_inds = []
        filament_vals = []
    
        # Loop over all cells.
        for i in range(self.nsim):
            current_hess = hessian[i, :, :]
            # Compute eigenvalues and eigenvectors.
            e, v = torch.symeig(current_hess, eigenvectors=True)
        
            # Notice the order.
            eigv1 = v[:, 0]
            eigv2 = v[:, 1]
        
            # Find smallest eigenvalue and belonging eigenvector.
            small_val, small_ind = torch.min(e, 0)
            small_eigv = v[:, small_ind]
        
            # If negative curavture.
            if small_val < 1e-9:
                # If scalar product of smallest eigenvector with gradient
                # is zero, we are on a filament.
                criterion = torch.dot(gradient[i, :], small_eigv)

                if np.abs(criterion) < tol:
                    filament_inds.append(i)
                    filament_vals.append(criterion)
    
        return filament_inds, filament_vals
