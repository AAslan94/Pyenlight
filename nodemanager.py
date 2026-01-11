import numpy as np
from typing import Dict
from elements import *
from builder import *
from spectral import *
from models import *
from spatial import *
from gains import *
from const import *
from room import Room

class SNManager:
    """
    Sensor Node Manager.
    
    Orchestrates the initialization of sensor nodes and computes the electro-optical 
    conversion efficiency based on spectral matching.
    
    This class handles the creation of physical simulation elements (Optical/RF 
    transceivers) and pre-calculates the **Effective Responsivity** ($R_{eff}$) 
    for both Signal (Downlink) and Noise (Solar Background) sources.

    Attributes:
        tia (TIA): Transimpedance Amplifier model defining electrical noise and gain.
        no_sensors (int): Total count of receiver nodes.
        ORx_elements (OpticalRxElements): Photodiode and Solar Panel receiver batch.
        OTx_elements (OpticalTxElements): Optical uplinks (IR emitters).
        RFTx_elements (RFTxElements): RF uplinks (if hybrid system is active).
        c_d (np.ndarray): Signal Effective Responsivity vector [A/W].
        c_d_n (np.ndarray): Noise/Background Spectral Matching Factor.
    """
    def __init__(self,nb: NodeBuilder):
      """
      Initialize the Sensor Manager.

      Segregates nodes into Optical and RF categories and initializes the 
      spectral response matrices.

      Args:
          nb (NodeBuilder): The configured builder containing raw node parameters.
      """
      self.tia = TIA(**nb.tia)
      self.no_sensors = nb.positions.shape[0]
      self.BW = to_scal_Nx1(self.no_sensors,nb.BW)
      
      self.rx_type = np.atleast_1d(nb.rx_type)
      
      self.rf_flag = 0
      self.ir_flag = 0
      self.ORx_elements = OpticalRxElements( r = nb.positions, n = nb.nR, type_Rx = nb.rx_type, fov = nb.FOV, A = nb.rx_area)
      if nb.no_optical_uplinks > 0:
        self.OTx_elements = OpticalTxElements( r = nb.positions[nb.uplink_type == 0 ], n = nb.nT, m = nb.m, p = nb.IR_tx_power)
        self.ir_flag = nb.no_optical_uplinks
      if nb.no_RF_uplinks > 0:
        self.RFTx_elements = RFTxElements(r = nb.positions[nb.uplink_type == 1], p = to_scal_Nx1(nb.no_RF_uplinks,nb.RF_tx_power))
        self.rf_flag = nb.no_RF_uplinks
      #make the masks for the effective responsivity calculations
      self.mask_VLC_filter = normalize_bool_array(nb.VLC_pass_filter,self.no_sensors) & (self.rx_type.reshape(-1,) == 0)
      self.mask_pd_no_filter = ~normalize_bool_array(nb.VLC_pass_filter,self.no_sensors) & (self.rx_type.reshape(-1,) == 0)
      self.mask_pv = (self.rx_type.reshape(-1,) == 1)

      self.c_d = np.zeros([self.no_sensors]) #Array to hold effective responsivity values
      self.c_d_n = np.zeros([self.no_sensors]) #Array to hold effective responsivity values

      self.downlink_effective_responsivity()


    def downlink_effective_responsivity(self):
      """
      Computes the Effective Responsivity ($R_{eff}$) via the spectral overlap integral.

      The received optical power is converted to current based on the matching 
      between the Source Spectrum ($S(\lambda)$), Filter Transmission ($T(\lambda)$), 
      and Detector Responsivity ($R(\lambda)$).

      **1. Signal Responsivity ($c_d$):**
      $$c_d = \\frac{\\int S_{LED}(\\lambda) \\cdot T_{filt}(\\lambda) \\cdot R_{det}(\\lambda) d\\lambda}{\\int S_{LED}(\\lambda) d\\lambda}$$
      
      **2. Noise Responsivity ($c_{d,n}$):**
      Calculated similarly using the Solar Spectrum ($S_{Sun}(\\lambda)$) as the source. 
      This determines the DC background current ($I_{bg}$) which generates Shot Noise.

      **Special Case - Solar Panels (PV):**
      For PVs, the responsivity is normalized against the solar spectrum to model 
      performance relative to standard STC conditions:
      $$c_{d, PV} = \\frac{R_{eff}(LED \\to PV)}{R_{eff}(Sun \\to PV)}$$
      """
      self.c_d[self.mask_VLC_filter] = SpectralPhysics.get_responsivity_by_name("WLED2PDwF")
      self.c_d[self.mask_pv] = SpectralPhysics.get_responsivity_by_name("WLED2PV")/SpectralPhysics.get_responsivity_by_name("SUN2PV") #different approach for PVs
      self.c_d[self.mask_pd_no_filter] = SpectralPhysics.get_responsivity_by_name("WLED2PD")
      #for noise calculations - sun to pd / pv
      self.c_d_n[self.mask_VLC_filter] = SpectralPhysics.get_responsivity_by_name("SUN2PDwFv")
      self.c_d_n[self.mask_pd_no_filter] = SpectralPhysics.get_responsivity_by_name("SUN2PD")
      self.c_d_n[self.mask_pv] = 1 # for irradiance the sun's spectrum is already taken into account
      
      
class MNManager:
    """
    Master Node Manager (Base Stations / Access Points).

    Manages the physical and spectral properties of the central transmitters (Masters).
    Unlike sensors, Masters typically act as the primary sources of Downlink optical power 
    (Visible Light) and receivers of Uplink data (Infrared).

    Attributes:
        tia (TIA): Transimpedance Amplifier model for the Master's receiver circuit.
        no_masters (int): Total number of master nodes.
        sensitivity (np.ndarray): Optical sensitivity.
        ORx_elements (OpticalRxElements): The Master's receiver hardware (detecting Uplink).
        OTx_elements (OpticalTxElements): The Master's transmitter hardware (sending Downlink).
        c_d (np.ndarray): Signal Effective Responsivity vector ($A/W$) for Uplink.
        c_d_n (np.ndarray): Noise Effective Responsivity vector ($A/W$).
    """
    def __init__(self,nb: NodeBuilder):
      """
      Initialize the Master Node Manager.

      Configures the Masters with specific Bandwidth, Sensitivity, and Optical Filters 
      (typically IR-pass filters to reject visible downlink reflections).

      Args:
          nb (NodeBuilder): The configured builder containing raw node parameters.
      """
      self.tia = TIA(**nb.tia)
      self.no_masters = nb.positions.shape[0]
      self.BW = to_scal_Nx1(self.no_masters,nb.BW)
      self.sensitivity = to_scal_Nx1(self.no_masters,nb.sensitivity)
      #make the masks for the effective responsivity calculations
      self.mask_IR_filter = normalize_bool_array(nb.IR_pass_filter,self.no_masters)
      self.mask_pd_no_filter = ~normalize_bool_array(nb.IR_pass_filter,self.no_masters)

      self.ORx_elements = OpticalRxElements( r = nb.positions, n = nb.nR, type_Rx = 0, fov = nb.FOV, A = nb.rx_area)
      self.OTx_elements = OpticalTxElements( r = nb.positions, n = nb.nT, m = nb.m, p = nb.tx_power)


      self.c_d = np.zeros([self.no_masters]) #Array to hold effective responsivity values
      self.c_d_n = np.zeros([self.no_masters]) #Array to hold effective responsivity values

      self.uplink_effective_responsivity()


    def uplink_effective_responsivity(self):
      """
      Computes the Effective Responsivity ($c_d$) for the Uplink Channel.

      This defines how efficiently the Master's Photodiode converts the received 
      Infrared (IR) signal from sensors into electrical current.

      **Spectral Overlap Integral:**
      $$c_d = \\frac{\\int S_{IR-Tx}(\\lambda) \\cdot T_{IR-Filter}(\\lambda) \\cdot R_{PD}(\\lambda) d\\lambda}{\\int S_{IR-Tx}(\\lambda) d\\lambda}$$

      Where:
      * $S_{IR-Tx}$: Spectrum of the Sensor's IR LED (e.g., TSFF5210).
      * $T_{IR-Filter}$: Transmission curve of the Master's optical filter.
      * $R_{PD}$: Spectral responsivity of the Master's Photodiode.
      """
      self.c_d[self.mask_IR_filter] = SpectralPhysics.get_responsivity_by_name("IR2PDwF")
      self.c_d[self.mask_pd_no_filter] = SpectralPhysics.get_responsivity_by_name("IR2PD")

      self.c_d_n[self.mask_IR_filter] = SpectralPhysics.get_responsivity_by_name("IR2PDwF")
      self.c_d_n[self.mask_pd_no_filter] = SpectralPhysics.get_responsivity_by_name("IR2PD") 
      
class ANManager:
    """
    Ambient Node Manager.

    Manages auxiliary light sources in the environment that are not data-carrying 
    Masters but contribute to the total optical power (and potentially interference). 
    These could represent standard room lighting (ceiling lamps, desk lamps) that 
    act as optical interference sources for the sensors.

    Attributes:
        OTx_elements (OpticalTxElements): The transmitter elements representing the ambient light sources.
    """
    def __init__(self,nb: NodeBuilder):
      """
      Initialize the Ambient Node Manager.

      Creates the optical transmitter elements for ambient sources based on the 
      builder configuration.

      Args:
          nb (NodeBuilder): The configured builder containing ambient node parameters.
      """
      self.OTx_elements = OpticalTxElements( r = nb.positions, n = nb.nT, m = nb.m, p = nb.tx_power)
      
      
class oPhyGains:
  """
  Optical Physical Layer Gain Calculator.

  This class acts as the central physics engine for the link budget. It orchestrates the 
  calculation of geometric channel states and converts them into received optical power 
  and electrical photocurrents.

  It handles three distinct propagation environments:
  1.  **Downlink (Visible Light):** From Master LEDs to Sensor Photodiodes/PVs.
  2.  **Uplink (Infrared/RF):** From Sensor IR LEDs (or RF antennas) to Master receivers.
  3.  **Ambient (Interference):** From Artificial sources (Lamps) and Natural sources (Windows/Sun).

  Attributes:
      h_*: Geometric Channel Gains (dimensionless or path loss).
      p_*: Received Optical Power [Watts].
      i_*: Induced Photocurrent [Amps].
  """
  def __init__(self, room, masters: MNManager, sensors: SNManager, ambient: ANManager):
    """
    Initialize the Physics Engine.

    Args:
        room (Room): The physical environment (walls, windows, RIS).
        masters (MNManager): Master nodes (Base Stations).
        sensors (SNManager): Sensor nodes (User Terminals).
        ambient (ANManager): Ambient light sources (Interference).
    """
    self.room = room
    self.mn = masters
    self.sn = sensors
    self.ambient = ambient
     
    self.compute_gains()
    self.compute_ambient()



  def compute_gains(self):
    """
    Computes the Geometric Channel Gains ($H$) for all active links.

    This method isolates the geometry-dependent components of the link.
    
    **Downlink Channels:**
    * $H_{LOS}$: Direct Line-of-Sight gain (Lambertian).
    * $H_{Diff}$: Non-Line-of-Sight gain via wall reflections (Multi-bounce).
    * $H_{RIS}$: Controlled reflection gain via Intelligent Surfaces.

    **Uplink Channels:**
    * Checks `ir_flag` to compute Optical Uplink gains (IR).
    * Checks `rf_flag` to compute RF Path Loss (if hybrid system).
    """
    #downlink gains

    gains = Gains(self.room, self.sn.ORx_elements, self.mn.OTx_elements)
    gains.los_channel_gains()
    gains.diffuse_channel_gains()
    gains.ris_channel_gains()

    self.h_d_los = gains.h_los
    self.h_d_diff = gains.h_diff
    self.h_d_ris = gains.h_ris

    #uplink gains
    if self.sn.ir_flag > 0:
      gains = Gains(self.room, self.mn.ORx_elements, self.sn.OTx_elements)
      gains.los_channel_gains()
      gains.diffuse_channel_gains()
      gains.ris_channel_gains()

      self.h_u_los = gains.h_los
      self.h_u_diff = gains.h_diff
      self.h_u_ris = gains.h_ris

    if self.sn.rf_flag > 0:
      self.h_u_rf = Gains.calc_h_rf(self.sn.RFTx_elements, self.mn.ORx_elements)


  def compute_downlink(self):
    """
    Calculates the Downlink Optical Power and Photocurrent.

    **1. Received Power ($P_{rx}$):**
    Scales the geometric gain $H$ by the transmitted optical power ($P_{tx}$).
    $$P_{rx} = H \cdot P_{tx}$$

    **2. Signal Current ($I_{sig}$):**
    Converts optical power to current using the Effective Responsivity ($c_d$).
    $$I_{sig} = P_{rx} \cdot c_d$$
    
    The total signal current is the sum of LoS, Diffuse, and RIS contributions.
    """

    self.p_d_los = self.h_d_los * self.mn.OTx_elements.p
    self.p_d_diff = self.h_d_diff * self.mn.OTx_elements.p
    self.p_d_ris = self.h_d_ris * self.mn.OTx_elements.p

    self.i_d_los = self.p_d_los * self.sn.c_d
    self.i_d_diff = self.p_d_diff * self.sn.c_d
    self.i_d_ris = self.p_d_ris * self.sn.c_d

    self.i_d_signal = self.i_d_los + self.i_d_diff + self.i_d_ris

  def compute_uplink(self):
    """
    Calculates the Uplink Optical Power and Photocurrent.

    Similar to downlink, but flows from Sensors to Masters.
    Uses `sn.OTx_elements.p` (Sensor Tx Power) and `mn.c_d` (Master Responsivity).
    """

    self.p_u_los = self.h_u_los * self.sn.OTx_elements.p
    self.p_u_diff = self.h_u_diff * self.sn.OTx_elements.p
    self.p_u_ris = self.h_u_ris * self.sn.OTx_elements.p

    self.i_u_los = self.p_u_los * self.mn.c_d
    self.i_u_diff = self.p_u_diff * self.mn.c_d
    self.i_u_ris = self.p_u_ris * self.mn.c_d

    self.i_u_signal = self.i_u_los + self.i_u_diff + self.i_u_ris

  def compute_ambient(self):
    """
    Computes Optical Interference (Shot Noise precursors).

    Calculates the background current generated by non-signal sources.

    **1. Artificial Ambient (Lamps):**
    * Models interference from room lighting ($ix$).
    * Calculated for both Downlink (at Sensors) and Uplink (at Masters).
    * Uses Signal Responsivity ($c_d$) assuming lamps have similar spectra to white LEDs.

    **2. Natural Ambient (Windows/Sun):**
    * Models strong background interference from sunlight ($is$).
    * Uses Noise Responsivity ($c_{d,n}$) because solar spectrum differs from LED spectrum.
    * Sums contributions from all window tiles to get total background current.
    """
    self.ix_d_noise = np.zeros((1, self.sn.ORx_elements.N))
    self.is_d_noise = np.zeros((1, self.sn.ORx_elements.N))

    # Uplink noise (at Masters)
    self.ix_u_noise = np.zeros((1, self.mn.ORx_elements.N))
    self.is_u_noise = np.zeros((1, self.mn.ORx_elements.N))
    
    if self.ambient.OTx_elements is not None:
      gains_down = Gains(self.room, self.sn.ORx_elements, self.ambient.OTx_elements)

      gains_down.los_channel_gains()
      gains_down.diffuse_channel_gains()

      self.hx_d_los = gains_down.h_los
      self.hx_d_diff = gains_down.h_diff

      self.px_d_los =  self.hx_d_los * self.ambient.OTx_elements.p
      self.px_d_diff = self.hx_d_diff * self.ambient.OTx_elements.p

      self.ix_d_los = self.px_d_los * self.sn.c_d
      self.ix_d_diff = self.px_d_diff * self.sn.c_d
      self.ix_d_noise = (self.ix_d_los + self.ix_d_diff).reshape(-1,self.sn.no_sensors)

      gains_up = Gains(self.room, self.mn.ORx_elements, self.ambient.OTx_elements)

      gains_up.los_channel_gains()
      gains_up.diffuse_channel_gains()

      self.hx_u_los = gains_up.h_los
      self.hx_u_diff = gains_up.h_diff

      self.px_u_los =  self.hx_u_los * self.ambient.OTx_elements.p
      self.px_u_diff = self.hx_u_diff * self.ambient.OTx_elements.p

      self.ix_u_los = self.px_u_los * self.mn.c_d
      self.ix_u_diff = self.px_u_diff * self.mn.c_d
      self.ix_u_noise = (self.ix_u_los + self.ix_u_diff).reshape(-1,self.mn.no_masters)


    if self.room.Tx_windows_elements is not None:
      #windows elems should be initialized correctly
      gains_s_up = Gains(self.room, self.mn.ORx_elements,self.room.Tx_windows_elements)

      gains_s_up.los_channel_gains()
      gains_s_up.diffuse_channel_gains()

      self.hs_u_los = gains_s_up.h_los
      self.hs_u_diff = gains_s_up.h_diff

      self.ps_u_los = self.hs_u_los * self.room.Tx_windows_elements.p
      self.ps_u_diff = self.hs_u_diff * self.room.Tx_windows_elements.p

      self.is_u_los = self.ps_u_los * self.mn.c_d_n
      self.is_u_diff = self.ps_u_diff * self.mn.c_d_n

      self.is_u_los_sum = np.sum(self.is_u_los,axis = 0)
      self.is_u_diff_sum = np.sum(self.is_u_diff, axis = 0)

      self.is_u_noise = (self.is_u_los_sum + self.is_u_diff_sum).reshape(-1,self.mn.no_masters)


      gains_s_down = Gains(self.room, self.sn.ORx_elements,self.room.Tx_windows_elements)

      gains_s_down.los_channel_gains()
      gains_s_down.diffuse_channel_gains()

      self.hs_d_los = gains_s_down.h_los
      self.hs_d_diff = gains_s_down.h_diff

      self.ps_d_los = self.hs_d_los * self.room.Tx_windows_elements.p
      self.ps_d_diff = self.hs_d_diff * self.room.Tx_windows_elements.p

      self.is_d_los = self.ps_d_los * self.sn.c_d_n
      self.is_d_diff = self.ps_d_diff * self.sn.c_d_n

      self.is_d_los_sum = np.sum(self.is_d_los,axis = 0)
      self.is_d_diff_sum = np.sum(self.is_d_diff, axis = 0)

      self.is_d_noise = (self.is_d_los_sum + self.is_d_diff_sum).reshape(-1,self.sn.no_sensors)      
      

      

