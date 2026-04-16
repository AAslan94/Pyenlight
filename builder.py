import numpy as np
from typing import Dict
from spatial import *
from const import *

class NodeBuilder():
  """
  Parses design configurations to prepare parameters for node creation.

  This class acts as a factory helper, extracting raw data from the design dictionary
  and normalizing it (e.g., reshaping arrays, applying defaults) so that 
  Master, Sensor, or AmbientNode objects can be instantiated cleanly.
  """
  def __init__(self, design: Dict, node_type, console=False):
        self.design = design
        self.node_type = node_type
        self.get_node_params(node_type, console)
        self.sanity_check()

  def get_node_params(self, node_type, console=False):
        """
        Extracts params with dynamic fallbacks to SimulationDefaults.
        """
        # Retrieve the specific node group from the design
        node_group = self.design.get("nodes", {}).get(node_type, {})
        
        # --- 1. Position & Orientation ---
        # Position is the only strictly required field in practice
        self.positions = node_group.get("positions").reshape(-1, 3)
        self.N_nodes = self.positions.shape[0]

        # --- 2. Geometric & Optical Defaults ---
        # fallbacks now point to SimulationDefaults
        self.rx_area = node_group.get("rx_area", SimulationDefaults.rx_area)
        self.m = node_group.get("m", SimulationDefaults.m)
        self.FOV = node_group.get("FOV", SimulationDefaults.fov)

        # --- 3. Type & Uplink Defaults ---
        
        self.rx_type = as_array_of_size(node_group.get("rx_type", 0), self.N_nodes)
        self.uplink_type = node_group.get("uplink_type", SimulationDefaults.uplink_type)
        
        # --- 4. Electrical & Transmit Power ---
        self.tx_power = node_group.get("tx_power", SimulationDefaults.VLC_Tx_power)
        self.IR_tx_power = node_group.get("IR_tx_power", SimulationDefaults.IR_Tx_power)
        self.RF_tx_power = node_group.get("RF_tx_power", SimulationDefaults.rf_driver['p_min'])
        
        # --- 5. TIA with fallback ---
        self.tia = self.design.get("TIA", SimulationDefaults.tia)

        # --- 6. Communication Metrics ---
        energy_prof = self.design.get('energy_profile', {})
        comm_cfg = energy_prof.get('communication', {})

        
        self.n_sp_d = comm_cfg.get("n_sp_d", SimulationDefaults.n_sp)
        
        # Bit rate fallbacks check specific IR/RF/VLC defaults
        self.Rb_down = comm_cfg.get("Rb_down", SimulationDefaults.bit_rate_dw)
        
        if node_type == "sensors":

            self.nT = node_group.get("nT", SimulationDefaults.zp)
            self.nR = node_group.get("nR", SimulationDefaults.zp)

            self.Rb_up = np.zeros(self.N_nodes)
            self.n_sp_u = np.zeros(self.N_nodes)

            design_rb = comm_cfg.get('Rb_up')

            design_nsp = comm_cfg.get('n_sp_u', SimulationDefaults.n_sp)

            ir_mask = (self.uplink_type == 0)
            rf_mask = (self.uplink_type == 1)
            
            if design_rb is not None:
                self.Rb_up = as_array_of_size(design_rb, self.N_nodes)
            else:
                self.Rb_up[ir_mask] = SimulationDefaults.bit_rate_up_ir
                self.Rb_up[rf_mask] = SimulationDefaults.bit_rate_up_rf
                
            #n_sp_u = as_array_of_size(design_nsp, self.N_nodes)
            n_sp_u = as_array_of_size(design_nsp, self.N_nodes)
            self.n_sp_u = n_sp_u[ir_mask]
            self.Rb_up_ir = self.Rb_up[ir_mask] 
            self.VLC_pass_filter = node_group.get("VLC_pass_filter", True)
            self.IR_pass_filter = None
              
        elif node_type == "masters":
            self.nT = node_group.get("nT", SimulationDefaults.zm)
            self.nR = node_group.get("nR", SimulationDefaults.zm)
            self.IR_pass_filter = node_group.get("IR_pass_filter", True)
            self.sensitivity = node_group.get("sensitivity", SimulationDefaults.sensitivity)
            self.VLC_pass_filter = None

        else:
            self.nT = node_group.get("nT", SimulationDefaults.zm)
            self.nR = node_group.get("nR", SimulationDefaults.zm)
  
  def sanity_check(self):
      """
      Validates parameter dimensions and filters optical properties for hybrid networks.

      Logic:
            - If node_type is 'sensors', it checks 'uplink_type' (0=Optical, 1=RF).
            - It ensures that optical transmitter properties (nT, m) are only assigned 
              to sensors that actually have optical uplinks.
            - If 'nT' or 'm' arrays match the total number of positions but include RF nodes,
              this method slices them to keep only the optical nodes' parameters to avoid shape mismatches.
      """
        #size sanity check for different input styles for uplinks
      if self.node_type != "sensors":
          pass
      else:
          self.uplink_type = to_scal_Nx1(self.positions.reshape(-1,3).shape[0], self.uplink_type).flatten()
          self.no_optical_uplinks = np.where(np.array([self.uplink_type])==0)[0].size
          self.no_RF_uplinks = np.where(np.array([self.uplink_type])==1)[0].size
          self.nT = np.array(self.nT)
          self.m = np.array(self.m)


          if self.nT.reshape(-1,3).shape[0] != 1 and self.nT.reshape(-1,3).shape[0] != self.no_optical_uplinks:
            if self.nT.reshape(-1,3).shape[0] == self.positions.reshape(-1,3).shape[0]:
              nT_x = self.nT[self.uplink_type == 0]
              self.nT = nT_x


          if self.m.reshape(-1,1).shape[0] != 1 and self.m.reshape(-1,1).shape[0] != self.no_optical_uplinks:
            if self.nT.reshape(-1,1).shape[0] == self.positions.reshape(-1,3).shape[0]:
              m_x = self.m[self.uplink_type == 0]
              self.m = m_x

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
    def __init__(self, design: Dict, console = False):
        """
        Initialize the RoomBuilder.

        Args:
            design (Dict): Configuration dictionary containing 'environment' keys.
            console (bool): If True, prints extraction details to stdout.
        """
        self.design = design
        self.env = self.design.get('environment', {})
        self.get_dimensions_and_res(console)
        self.get_reflectivity(console)
        self.get_surfaces_by_type("RIS", console)
        self.get_surfaces_by_type("window", console)


    def get_dimensions_and_res(self, console = False):
        """
        Extracts room dimensions (L, W, H) and meshing resolution.
        
        Args:
            console (bool): Enable debug printing.
        """
        dims = self.env.get('dimensions', SimulationDefaults.room_dim)
        self.L, self.W, self.H = dims[0], dims[1], dims[2]

        self.res = self.env.get('wall_resolution', SimulationDefaults.wall_resolution)

        if isinstance(self.res, int):
          self.res = (self.res, self.res)
        
        if console:
          print(f"Retrieving design parameters: ")
          print(f"L: {self.L}")
          print(f"W: {self.W}")
          print(f"H: {self.H}")
          print(f"Res: {self.res}")

    def get_reflectivity(self, console = False):
        """
        Extracts reflectivity coefficients for standard room surfaces.
        
        Sets:
            self.floor_refl
            self.ceiling_refl
            self.wall_refl
            self.refl (list of the above)
        """
        refl_cfg = self.env.get('reflectivity', {})
        
        #fallback
        def_refl = SimulationDefaults.reflectivity
        
        self.floor_refl = refl_cfg.get('floor', def_refl)
        self.ceiling_refl = refl_cfg.get('ceiling', def_refl)
        self.wall_refl = refl_cfg.get('walls', def_refl)
        
        self.refl = [self.floor_refl, self.ceiling_refl, self.wall_refl]
       
        if console:
          print(f"Retrieving design parameters: ")
          print(f"Floor Reflectivity: {self.floor_refl}")
          print(f"Ceiling Reflectivity: {self.ceiling_refl}")
          print(f"Wall Reflectivity: {self.wall_refl}")

    def get_surfaces_by_type(self, surface_type, console = False):
      """
      Extracts specific special surface configurations from the design dictionary.

      Populates self.windows or self.RIS depending on the requested type.

      Args:
          surface_type (str): Either 'window' or 'RIS'.
          console (bool): Enable debug printing.

      Raises:
          ValueError: If surface_type is not 'window' or 'RIS'.
      """
      surfaces = self.env.get('special_surfaces', [])
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


