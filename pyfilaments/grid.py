""" Regular mesh grid helper class.

IMPORTANT
---------
x axis corresponds to second index of matrices (columns).

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


class Grid():
    """

    Attributes
    ----------
    x_cells: array
        Array containing coordinates of x-cells. Note that this arrray and the
        y one are already for meshgrid functionality, hence y is a column
        vector.
    y_cells: array
    nx: number of cells along x-dimension.
    ny: number of cells along y-dimension.

    """
    def __init__(self, x_spacings, y_spacings):
        """
        self,x_cells, self.y_cells = np.meshgrid(x_spacings, y_spacings,
                sparse=True)
        """
        self.x_spacings = x_spacings.astype(np.float32)
        self.y_spacings = y_spacings.astype(np.float32)

        self.nx = len(x_spacings)
        self.ny = len(y_spacings)

        self.n_cells = self.nx * self.ny

    def __getitem__(self, tup):
        y, x = tup
        return (self.y_spacings[y], self.x_spacings[x])

    def one_dim_index(self, i, j):
        """ Convert two dimensional index to 1d.

        """
        return int(i + j * self.nx)

    def two_dim_index(self, k):
        """ Converts a one dimensional index to 2d.

        WARNING: Check the order of the return. x-axis is the horizontal one,
        so columns, so second index of matrix.

        """
        ind_x = k % self.nx
        ind_y = (k - ind_x) / self.nx
        return (int(ind_y), int(ind_x))

    @property
    def cells_list(self):
        """ Return a list of all the cells in the grid. We start increasing x
        first, i.e. 0-th cell is the first cell, 1-cell is the one with the
        next x in the list and y unchanged, ....

        Return
        ------
        array
            An array of size n_cells * n_dims.

        """
        xx, yy = np.meshgrid(self.x_spacings, self.y_spacings)
        return np.vstack([yy.ravel(), xx.ravel()]).transpose()

    def reshape_to_2d(self, array):
        return np.reshape(array, (-1, self.nx))

    def plot_list(self, vals):
        # Put the values back to a 2d array.
        vals_2d = self.reshape_to_2d(vals)

        xx, yy = np.meshgrid(self.x_spacings, self.y_spacings)
        # plt.contour(xx, yy, vals_2d)
        plt.pcolormesh(xx, yy, vals_2d)
        plt.show()

    def plot_list_3d(self, vals):
        # Put the values back to a 2d array.
        vals_2d = self.reshape_to_2d(vals)

        xx, yy = np.meshgrid(self.x_spacings, self.y_spacings)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, vals_2d, cmap='viridis')
        plt.show()
