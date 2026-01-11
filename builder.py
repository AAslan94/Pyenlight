import numpy as np
from typing import Dict
from spatial import *

class RoomBuilder:
    """
    Parses a design dictionary to extract and configure physical room parameters.

    This class serves as a parser and state container for the room's geometry,
    surface properties, and special elements (like Windows or RIS units) before
    the actual simulation objects are constructed.

    Attributes:
        design (Dict): The source configuration dictionary.
        L, W, H (float): Room Length, Width, and Height.
        res (int/tuple): Wall grid resolution.
        floor_refl, ceiling_refl, wall_refl (float): Reflectivity coefficients.
        windows (list): List of dictionaries defining window parameters.
        RIS (list): List of dictionaries defining RIS (Reconfigurable Intelligent Surface) parameters.
    """

    def __init__(self, design: Dict, console=False):
        """
        Initialize the RoomBuilder.

        Args:
            design (Dict): Configuration dictionary containing 'environment' keys.
            console (bool): If True, prints extraction details to stdout.
        """
        self.design = design
        self.get_dimensions_and_res(console)
        self.get_reflectivity(console)
        self.get_surfaces_by_type("RIS", console)
        self.get_surfaces_by_type("window", console)

    def get_dimensions_and_res(self, console=False):
        """
        Extracts room dimensions (L, W, H) and meshing resolution.
        
        Args:
            console (bool): Enable debug printing.
        """
        self.L = self.design['environment']['dimensions'][0]
        self.W = self.design['environment']['dimensions'][1]
        self.H = self.design['environment']['dimensions'][2]
        self.res = self.design['environment']['wall_resolution']
        
        if console:
            print(f"Retrieving design parameters: ")
            print(f"L: {self.L}")
            print(f"W: {self.W}")
            print(f"H: {self.H}")
            print(f"Res: {self.res}")

    def get_reflectivity(self, console=False):
        """
        Extracts reflectivity coefficients for standard room surfaces.
        
        Sets:
            self.floor_refl
            self.ceiling_refl
            self.wall_refl
            self.refl (list of the above)
        """
        self.floor_refl = self.design['environment']['reflectivity']['floor']
        self.ceiling_refl = self.design['environment']['reflectivity']['ceiling']
        self.wall_refl = self.design['environment']['reflectivity']['walls']
        self.refl = [self.floor_refl, self.ceiling_refl, self.wall_refl]

        if isinstance(self.res, int):
            self.res = (self.res, self.res)
        
        if console:
            print(f"Retrieving design parameters: ")
            print(f"Floor Reflectivity: {self.floor_refl}")
            print(f"Ceiling Reflectivity: {self.ceiling_refl}")
            print(f"Wall Reflectivity: {self.wall_refl}")

    def get_surfaces_by_type(self, surface_type, console=False):
        """
        Extracts specific special surface configurations from the design dictionary.

        Populates self.windows or self.RIS depending on the requested type.

        Args:
            surface_type (str): Either 'window' or 'RIS'.
            console (bool): Enable debug printing.

        Raises:
            ValueError: If surface_type is not 'window' or 'RIS'.
        """
        env = self.design.get('environment', {})
        surfaces = env.get('special_surfaces', [])
        
        if surface_type == 'window':
            self.windows = [s for s in surfaces if s.get('type') == surface_type]
            if console:
                print(f"Found {len(self.windows)} {surface_type} surfaces")
        elif surface_type == 'RIS':
            self.RIS = [s for s in surfaces if s.get('type') == surface_type]
            if console:
                print(f"Found {len(self.RIS)} {surface_type} surfaces")
        else:
            raise ValueError("Invalid surface type. Must be 'window' or 'RIS'.")


class NodeBuilder:
    """
    Parses design configurations to prepare parameters for node creation.

    This class acts as a factory helper, extracting raw data from the design dictionary
    and normalizing it (e.g., reshaping arrays, applying defaults) so that 
    Master, Sensor, or AmbientNode objects can be instantiated cleanly.
    """

    def __init__(self, design: Dict, node_type, console=False):
        """
        Initialize the NodeBuilder.

        Args:
            design (Dict): The main configuration dictionary containing a "nodes" key.
            node_type (str): The specific category to build ('masters', 'sensors', or 'ambient_nodes').
            console (bool): If True, enables debug printing (currently unused).
        """
        self.design = design
        self.node_type = node_type
        self.get_node_params(node_type, console)
        self.sanity_check()

    def get_node_params(self, node_type, console=False):
        """
        Extracts params for specific node types from the design dictionary.

        Populates instance attributes including physical location, orientation (nT/nR),
        electrical properties (battery, power), and optical properties (FOV, bandwidth).

        Args:
            node_type (str): 'masters', 'sensors', or 'ambient_nodes'.
        """
        node_cfg = self.design["nodes"][node_type]
        
        self.node_type = node_type
        self.rx_area = node_cfg.get("rx_area")
        self.nT = node_cfg.get("nT")
        self.nR = node_cfg.get("nR")
        self.m = node_cfg.get("m", 1)
        self.tx_power = node_cfg.get("tx_power")
        self.rx_type = np.atleast_1d(node_cfg.get("rx_type",0))
        self.uplink_type = np.atleast_1d(node_cfg.get("uplink_type", 0))
        self.battery_capacity_J = node_cfg.get("battery_capacity_J")
        self.initial_charge = node_cfg.get("initial_charge")
        self.positions = node_cfg.get("positions").reshape(-1, 3)
        self.FOV = node_cfg.get("FOV", np.pi/2)
        self.BW = node_cfg.get("BW", 10e3)
        self.tia = self.design.get("TIA", None)
        self.IR_tx_power = node_cfg.get("IR_tx_power", 0)
        self.RF_tx_power = node_cfg.get("RF_tx_power", 0)
        
        if node_type == "sensors":
            self.VLC_pass_filter = node_cfg.get("VLC_pass_filter", True)
        elif node_type == "masters":
            self.IR_pass_filter = node_cfg.get("IR_pass_filter", True)
            self.sensitivity = node_cfg.get("sensitivity", -100)
        else:
            self.IR_pass_filter = None
            self.VLC_pass_filter = None

    def sanity_check(self):
        """
        Validates parameter dimensions and filters optical properties for hybrid networks.

        Logic:
            - If node_type is 'sensors', it checks 'uplink_type' (0=Optical, 1=RF).
            - It ensures that optical transmitter properties (nT, m) are only assigned 
              to sensors that actually have optical uplinks.
            - If 'nT' or 'm' arrays match the total number of positions but include RF nodes,
              this method slices them to keep only the optical nodes' parameters.
        """
        if self.node_type == "sensors":
            # Assuming to_scal_Nx1 is defined elsewhere in your project
            self.uplink_type = to_scal_Nx1(self.positions.reshape(-1, 3).shape[0], self.uplink_type).flatten()
            self.no_optical_uplinks = np.where(np.array([self.uplink_type]) == 0)[0].size
            self.no_RF_uplinks = np.where(np.array([self.uplink_type]) == 1)[0].size
            self.nT = np.array(self.nT)
            self.m = np.array(self.m)

            # Check nT shape compatibility
            if self.nT.reshape(-1, 3).shape[0] != 1 and self.nT.reshape(-1, 3).shape[0] != self.no_optical_uplinks:
                if self.nT.reshape(-1, 3).shape[0] == self.positions.reshape(-1, 3).shape[0]:
                    nT_x = self.nT[self.uplink_type == 0]
                    self.nT = nT_x

            # Check m shape compatibility
            if self.m.reshape(-1, 1).shape[0] != 1 and self.m.reshape(-1, 1).shape[0] != self.no_optical_uplinks:
                if self.m.reshape(-1, 1).shape[0] == self.positions.reshape(-1, 3).shape[0]:
                    m_x = self.m[self.uplink_type == 0]
                    self.m = m_x
