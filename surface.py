import numpy as np
from elements import *
from spectral import *
from const import Constants

import numpy as np

class Surface:
    """
    Represents a rectangular physical surface (e.g., wall, window, ceiling) 
    discretized into a grid of smaller elements.

    This class automatically generates the coordinate points for a grid based on 
    dimensions and resolution, and initializes both Transmitter (Tx) and 
    Receiver (Rx) elements at each grid point to simulate reflection and emission.

    Attributes:
        r_surface (np.ndarray): Coordinates of the center of each grid element (N, 3).
        A (float): Area of a single grid element (patch area).
        Tx_elements (OpticalTxElements): The transmission properties associated with the surface elements.
        Rx_elements (OpticalRxElements): The reception/reflection properties associated with the surface elements.
    """
    def __init__(self, center, dims, const_axis, resolution, nT = np.array([0,0,1]), nR = np.array([0,0,1]),refl = 0.8, type = 'Wall',name=None, P = None):
        """
        Initialize the Surface.

        Args:
            center (array-like): The (x, y, z) center coordinate of the surface.
            dims (tuple): Dimensions (dim_1, dim_2) along the non-constant axes.
            const_axis (int): The axis index (0=x, 1=y, 2=z) that remains constant (normal to surface).
            resolution (tuple): The number of grid points (res_1, res_2) along dim_1 and dim_2.
            nT (array-like, optional): Normal vector for Transmitters (emission direction). Defaults to [0,0,1].
            nR (array-like, optional): Normal vector for Receivers (detection direction). Defaults to [0,0,1].
            refl (float, optional): Reflection coefficient of the surface. Defaults to 0.8.
            type (str, optional): Surface type ('Wall', 'window', etc.). 'window' triggers solar power calc.
            name (str, optional): Label for the surface.
            P (float, optional): Optical power emitted by the surface. If None, defaults to 0 (unless type='window').
        """
        self.r_surface, self.A = self.gen_surface_points(center=center, dims=dims, const_axis = const_axis,
                                                 resolution = resolution)
        self.const_axis = const_axis
        self.name = name

        # Normals will be tiled inside Element init, but we can prep them here
        self.nT = nT
        self.nR = nR
        self.refl = refl
        self.type = type
        self.P = 0 if P is None else P

        if self.type == "window":
            self.P = Constants.pd_peak * self.A * SpectralPhysics.sun_power()


        count = self.r_surface.shape[0]


        self.Tx_elements = OpticalTxElements(r = self.r_surface, n = self.nT, p = self.P, m = 1)
        self.Rx_elements = OpticalRxElements(r = self.r_surface, n = self.nR, A = self.A, refl = self.refl, fov = np.pi/2)



    @staticmethod
    def gen_surface_points(center, dims, const_axis, resolution):
      """
      Generates a mesh grid of points centered at a specific location on a 2D plane 
      in 3D space.

      Args:
          center (array-like): (x, y, z) center.
          dims (tuple): (length_1, length_2) of the rectangle.
          const_axis (int): 0 (YZ plane), 1 (XZ plane), or 2 (XY plane).
          resolution (tuple): (n_points_1, n_points_2) division count.

      Returns:
          tuple: 
              - points (np.ndarray): Array of shape (N, 3) containing grid centers.
              - patch_area (float): The area of a single grid patch.
      """
      dim_1, dim_2 = dims
      res_1, res_2 = resolution

      grid_1 = np.linspace(-dim_1/2 + dim_1/res_1/2, dim_1/2 - dim_1/res_1/2, res_1)
      grid_2 = np.linspace(-dim_2/2 + dim_2/res_2/2, dim_2/2 - dim_2/res_2/2, res_2)
      mesh_1, mesh_2 = np.meshgrid(grid_1, grid_2)
      zeros = np.zeros_like(mesh_1)

      if const_axis == 0: offsets = np.stack([zeros, mesh_1, mesh_2], axis=-1)
      elif const_axis == 1: offsets = np.stack([mesh_1, zeros, mesh_2], axis=-1)
      else: offsets = np.stack([mesh_1, mesh_2, zeros], axis=-1)

      points = center + offsets.reshape(-1, 3)
      patch_area = float((dim_1/res_1)*(dim_2/res_2))
      return points, patch_area
