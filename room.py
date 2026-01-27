import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from elements import *
from surface import Surface
from builder import RoomBuilder
from const import *
class Room:
    """
    Constructs the physical room environment for the simulation.

    This class initializes the standard room surfaces (walls, floor, ceiling) 
    and handles the integration of special surfaces like Windows and RIS 
    (Reconfigurable Intelligent Surfaces) by managing spatial overlaps and 
    reflectivity modifications.

    Attributes:
        windows (list): List of window Surface objects.
        RIS (list): List of RIS Surface objects.
        Tx_wall_elements (OpticalTxElements): Combined transmitter elements for all walls.
        Rx_wall_elements (OpticalRxElements): Combined receiver elements for all walls.
        Tx_RIS_elements (OpticalTxElements): Combined transmitter elements for all RIS units.
        Tx_windows_elements (OpticalTxElements): Combined transmitter elements for all windows.
        h_ww (np.ndarray): Channel gain matrix representing intra-wall reflections (Wall-to-Wall).
    """
    def __init__(self, rb : RoomBuilder, ignore_RIS = False, ignore_windows = False, console = False):
        """
        Initialize the Room using specifications from a RoomBuilder.

        Args:
            rb (RoomBuilder): The builder object containing parsed design parameters.
            ignore_RIS (bool): If True, skips adding RIS units.
            ignore_windows (bool): If True, skips adding windows.
            console (bool): If True, enables debug printing.
        """
        # 1. Initialize Standard Surfaces
        #Helpers
        L = rb.L
        W = rb.W
        H = rb.H
        res = rb.res
        refl = rb.refl

        #CAUTION!!! room.windows & room.RIS is a list of surfaces
        #CAUTION!!! rb.windows & rb.RIS is a list of dictionaries
        self.windows = []
        self.RIS = []

        self.Tx_RIS_elements = None
        self.Tx_windows_elements = None
        self.h_ww = None

        #Create wall surfaces
        self.floor = Surface(np.array([L/2, W/2, 0]), (L, W), 2, res, nR = SimulationDefaults.zp, nT = SimulationDefaults.zp, refl=refl[0], name='Floor')
        self.ceiling = Surface(np.array([L/2, W/2, H]), (L, W), 2,  res, nR = SimulationDefaults.zm, nT = SimulationDefaults.zm, refl=refl[1], name='Ceiling')
        self.west_wall = Surface(np.array([0, W/2, H/2]), (W, H), 0, res, nR = SimulationDefaults.xp, nT = SimulationDefaults.xp, refl=refl[2], name='West Wall')
        self.east_wall = Surface(np.array([L, W/2, H/2]), (W, H), 0, res, nR = SimulationDefaults.xm, nT = SimulationDefaults.xm, refl=refl[2], name='East Wall')
        self.south_wall = Surface(np.array([L/2, 0, H/2]), (L, H), 1, res, nR = SimulationDefaults.yp, nT = SimulationDefaults.yp, refl=refl[2], name='South Wall')
        self.north_wall = Surface(np.array([L/2, W, H/2]), (L, H), 1, res, nR = SimulationDefaults.ym, nT = SimulationDefaults.ym, refl=refl[2], name='East Wall')

        self.walls = [self.floor, self.ceiling, self.west_wall, self.east_wall, self.south_wall, self.north_wall]
        self._build_master_element()

        if not ignore_RIS:
          for ris in rb.RIS:
            ris_surface = Surface(ris['center'], ris['dims'], ris['const_axis'], ris['resolution'], ris['normal'], ris['normal'], ris['reflectivity'], ris['type'], ris['name'])
            self.add_surface(ris_surface)
        if not ignore_windows:
          for window in rb.windows:
            window_surface = Surface(window['center'], window['dims'], window['const_axis'], window['resolution'], window['normal'], window['normal'], window['reflectivity'], window['type'], window['name'])
            self.add_surface(window_surface)
        if console:
          print("Room built successfully!")


    def _build_master_element(self):
      """
      Combines all individual wall surfaces into the main self.wall_elms object.
      
      Merges the elements from Floor, Ceiling, West, East, South, and North walls
      into single unified OpticalTxElements and OpticalRxElements batches.
      """
      Tx_wall_elements = [s.Tx_elements for s in self.walls]
      Rx_wall_elements = [s.Rx_elements for s in self.walls]
      self.Tx_wall_elements = None
      self.Rx_wall_elements = None
      self.Tx_wall_elements = OpticalTxElements.merge(Tx_wall_elements)
      self.Rx_wall_elements = OpticalRxElements.merge(Rx_wall_elements)


    def plot_surface_addition(self, new_surface, overlap_mask):
        """
        Visualizes the process of adding a new surface (like a Window or RIS) 
        onto an existing wall.

        Plots:
        1.  Blue: Existing wall tiles that remain active.
        2.  Red: Wall tiles being removed/deactivated (because they are behind the new surface).
        3.  Green: The new surface elements.
        4.  Black: A wireframe border around the new surface.

        Args:
            new_surface (Surface): The surface object being added.
            overlap_mask (np.ndarray): Boolean array indicating which wall elements are overlapped.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- 1. Plot Wall Elements (Active vs Removed) ---
        wall_r = self.Rx_wall_elements.r
        kept_r = wall_r[~overlap_mask]
        removed_r = wall_r[overlap_mask]

        ax.scatter(kept_r[:, 0], kept_r[:, 1], kept_r[:, 2],
                   c='blue', alpha=0.05, s=2, label='Existing Wall (Kept)')

        if removed_r.shape[0] > 0:
            ax.scatter(removed_r[:, 0], removed_r[:, 1], removed_r[:, 2],
                       c='red', alpha=0.8, s=10, label='Wall Tiles Removed')

        # --- 2. Plot The New Surface ---
        new_r = new_surface.Tx_elements.r
        ax.scatter(new_r[:, 0], new_r[:, 1], new_r[:, 2],
                   c='green', alpha=0.9, s=15, marker='s', label=f'New {new_surface.name}')

        # --- 3. Draw Borders (New Logic) ---
        # Find which axes vary (not the constant axis)
        const_axis = new_surface.const_axis
        active_axes = [i for i in range(3) if i != const_axis]

        # Calculate limits for the active axes
        limits = {}
        for axis in active_axes:
            vals = new_r[:, axis]
            # Calculate step size (tile width) to find true edge
            unique = np.unique(np.round(vals, 5))
            if len(unique) > 1:
                step = np.mean(np.diff(unique))
            else:
                # If 1x1 grid, infer step from Area (assuming square tile for estimation)
                step = np.sqrt(new_surface.Tx_elements.A[0,0])

            limits[axis] = (vals.min() - step/2, vals.max() + step/2)

        # Construct the 4 corners
        # Start with a base point at the center of the constant axis
        base_val = new_r[0, const_axis]

        # Determine the cycle of corners (rectangular path)
        a1, a2 = active_axes
        min1, max1 = limits[a1]
        min2, max2 = limits[a2]

        # The 5 points to draw the loop (Start -> TopRight -> BottomRight -> BottomLeft -> Start)
        corners_2d = [
            (min1, min2),
            (max1, min2),
            (max1, max2),
            (min1, max2),
            (min1, min2) # Close loop
        ]

        # Convert back to 3D coordinates
        x_line, y_line, z_line = [], [], []
        for (c1, c2) in corners_2d:
            pt = [0, 0, 0]
            pt[const_axis] = base_val
            pt[a1] = c1
            pt[a2] = c2

            x_line.append(pt[0])
            y_line.append(pt[1])
            z_line.append(pt[2])

        ax.plot(x_line, y_line, z_line, color='black', linewidth=3, label='Border')

        # --- Formatting ---
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Adding '{new_surface.name}' to Room")
        ax.legend()

        # Fix Aspect Ratio
        max_range = np.array([self.floor.Tx_elements.r[:,0].max(),
                              self.floor.Tx_elements.r[:,1].max(),
                              self.ceiling.Tx_elements.r[:,2].max()]).max()
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)

        plt.show()

    def add_surface(self, new_surface):
        """
        Integrates a special surface (Window or RIS) into the room.

        This method detects which existing wall tiles lie 'behind' the new 
        surface and effectively 'turns them off' by setting their reflectivity 
        to zero. This prevents double-counting reflections at those coordinates.

        Args:
            new_surface (Surface): The surface object to be added.
        """
        if new_surface.type == 'RIS':
            self.RIS.append(new_surface)
            if self.Tx_RIS_elements == None:
                self.Tx_RIS_elements = new_surface.Tx_elements
            else:
                self.Tx_RIS_elements = self.Tx_RIS_elements + new_surface.Tx_elements

        elif new_surface.type == 'window':
            self.windows.append(new_surface)
            if self.Tx_windows_elements == None:
                self.Tx_windows_elements = new_surface.Tx_elements
            else:
                self.Tx_windows_elements = self.Tx_windows_elements + new_surface.Tx_elements

        # Check Overlap
        new_r = new_surface.Tx_elements.r # (N, 3)
        old_r = self.Rx_wall_elements.r   # (M, 3)
        const_axis = new_surface.const_axis
        const_val = new_r[0, const_axis]
        active_axes = [i for i in range(3) if i != const_axis]

        # Mask
        overlap_mask = np.abs(old_r[:, const_axis] - const_val) < 1e-4

        for axis in active_axes:
            vals = new_r[:, axis]
            unique_vals = np.unique(np.round(vals, 5))
            if len(unique_vals) > 1:
                step = np.mean(np.diff(unique_vals))
            else:
                step = np.sqrt(new_surface.Tx_elements.A[0])

            min_edge = np.min(vals) - step/2
            max_edge = np.max(vals) + step/2
            buffer = 1e-5
            is_inside_dim = (old_r[:, axis] > min_edge + buffer) & (old_r[:, axis] < max_edge - buffer)
            overlap_mask = overlap_mask & is_inside_dim

        removed_count = np.sum(overlap_mask)
        print(f"Adding '{new_surface.name}'. Removed {removed_count} tiles.")
        if 0 and removed_count > 0:
            self.plot_surface_addition(new_surface, overlap_mask)

        # Set reflectivity to 0 (note overlap_mask is 1D array of size M)
        # self.Wall_elms.refl is (M, 1)
        self.Rx_wall_elements.refl[overlap_mask,0] = 0
