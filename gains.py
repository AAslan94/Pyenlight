import numpy as np
from elements import *
from room import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
ignores = np.seterr(all='ignore')

class Gains:
  """
  Calculates optical channel gains (Line-of-Sight, Diffuse, and RIS-assisted) 
  between transmitters and receivers in a specific room environment.
  It can also calculate RF channel gains based on the log-distance path loss
  model that can be computed directly without instantiating the class.

  This class acts as the core physics engine for the channel model, handling 
  vectorized geometric calculations for signal propagation.

  Attributes:
      room (Room): The physical environment containing walls and surfaces.
      masters (OpticalTxElements): The primary light sources (transmitters).
      sensors (OpticalRxElements): The detectors (receivers).
      h_los (np.ndarray): Line-of-Sight channel gain matrix (Nt, Nr).
      h_diff (np.ndarray): Diffuse (multi-bounce) channel gain matrix (Nt, Nr).
      h_ris (np.ndarray): RIS-reflected channel gain matrix (Nt, Nr).
  """
  def __init__(self, room: Room, optRx, optTx):
    self.room = room
    self.masters = optTx
    self.sensors = optRx

    self.h_diff = None
    self.h_los = None

    if self.room.h_ww is None:
        self.room.h_ww = self.calc_h(
        self.room.Tx_wall_elements, 
        self.room.Rx_wall_elements
        )



  @staticmethod
  def calc_h_rf(tx, rx, **kwargs):
    """
    Calculates the path loss for RF signals based on the log-distance path loss model.
    
    Formula: PL(d) = 10*n*log10(d) + PL_ref + 10*k*log10(f) + X_sigma
    
    Args:
        tx (RFTxElements): RF Transmitter elements.
        rx (RFRxElements): RF Receiver elements.
        **kwargs: Optional channel model coefficients.
            - n (float): Path loss exponent (default: 1.46)
            - pl_ref (float): Reference path loss constant (default: 34.62)
            - k (float): Frequency dependence coefficient (default: 2.03)
            - f (float): Frequency in GHz (default: 2.45)
            - sigma (float): Shadowing or additional loss term (default: 3.76)
            - sigma_factor (float): Multiplier for sigma (default: 2)

    Returns:
        np.ndarray: Path loss values in dB.
    """
    # Extract coefficients from kwargs with existing values as defaults
    n = kwargs.get('n', SimulationDefaults.rf_n)
    pl_ref = kwargs.get('pl_ref', SimulationDefaults.rf_pl_ref)
    k = kwargs.get('k', SimulationDefaults.rf_k)
    f = kwargs.get('f', SimulationDefaults.rf_f)
    sigma = kwargs.get('sigma', SimulationDefaults.rf_sigma)
    sigma_factor = kwargs.get('sigma_factor', SimulationDefaults.rf_sigma_factor)

    # Distance calculation remains geometric
    D_tx_rx = -(tx.r[:, None, :] - rx.r[None, :, :])
    d = np.linalg.norm(D_tx_rx, axis=2)

    return (10 * n * np.log10(d)) + pl_ref + (10 * k * np.log10(f)) + (sigma_factor * sigma)

  @staticmethod
  def calc_h(tx, rx):
    """
    Calculates channel gain matrix H (Nt, Nr) for optical links.

    Implements the standard Lambertian emission model.
    Gain = (A * (m+1) * cos(phi)^m * cos(psi)) / (2 * pi * d^2)

    Where:
        phi: Irradiance angle (at Tx)
        psi: Incidence angle (at Rx)
        m: Lambertian order
        d: Distance
        A: Detector Area

    Assumes tx and rx properties are already formatted as (N,3) or (N,1).

    Args:
        tx (Elements): Transmitter batch (Sources or Wall patches).
        rx (Elements): Receiver batch (Detectors or Wall patches).

    Returns:
        np.ndarray: Gain matrix of shape (Nt, Nr).
    """

    # Extract positions: (Nt, 3) and (Nr, 3)
    # No reshapes needed, Element class guarantees (N, 3)

    # Distance Vector (Nt, Nr, 3)
    # Broadcasting: (Nt, 1, 3) - (1, Nr, 3)
    D_tx_rx = -(tx.r[:, None, :] - rx.r[None, :, :])

    # Distance Norm (Nt, Nr)
    D_tx_rx_norm = np.linalg.norm(D_tx_rx, axis=2)

    # Unit Vectors (Nt, Nr, 3)
    D_tx_rx_unit = D_tx_rx / D_tx_rx_norm[..., None]

    # --- Cosines ---
    # tx.nT is (Nt, 3). Reshape to (Nt, 1, 3) for broadcasting
    # rx.nR is (Nr, 3). Reshape to (1, Nr, 3) for broadcasting

    # Cosine of irradiance angle (Tx -> Rx)
    cos_irr = np.maximum(0, np.sum(D_tx_rx_unit * tx.n[:, None, :], axis=2)) # (Nt, Nr)

    # Cosine of incidence angle (Rx <- Tx)
    # Note negative sign on D vector for Rx perspective
    cos_inc = np.maximum(0, np.sum(-D_tx_rx_unit * rx.n[None, :, :], axis=2)) # (Nt, Nr)

    # --- FOV Logic ---
    inc_angle = np.arccos(cos_inc) # (Nt, Nr)

    # rx.fov is (Nr, 1). Transpose to (1, Nr) to match columns
    fov_mask_pd = (inc_angle <= rx.fov.T) # (Nt, Nr)

    # rx.type_Rx is (Nr, 1). Transpose to (1, Nr) for boolean usage
    is_sp = (rx.type_Rx.T == 1) # (1, Nr) boolean mask

    # --- Gain Calculation ---
    # rx.A is (Nr, 1) -> Transpose to (1, Nr)
    # tx.m is (Nt, 1) -> Matches rows (Nt) perfectly

    h = (
        rx.A.T * (tx.m + 1) * (cos_irr ** tx.m) * cos_inc
        / (2 * np.pi * D_tx_rx_norm**2)
    ) # Result is (Nt, Nr)

    # Apply FOV Mask (Only for standard PDs, i.e., NOT solar panels)
    # is_sp is (1, Nr). ~is_sp masks columns.
    # We broadcast mask (Nt, Nr) against columns where ~is_sp is True.
    h = np.where(~is_sp, h * fov_mask_pd, h)

    # Apply Solar Panel Efficiency if any exist
    if np.any(is_sp):
        sp_factor = solar_panel_angular_efficiency(cos_inc)
        h = np.where(is_sp, h * sp_factor, h)

    h = np.nan_to_num(h, nan=0)
    return h


  def los_channel_gains(self):
    """Computes the Line-of-Sight (LoS) gain matrix between Masters and Sensors."""
    self.h_los = self.calc_h(tx=self.masters, rx=self.sensors)

  def diffuse_channel_gains(self, bounces = 4):
    """
    Computes the Non-Line-of-Sight (Diffuse) gain matrix using an iterative 
    multi-bounce approach.

    Algorithm:
    1. Calculate Source -> Wall gains (h_mw).
    2. Calculate Wall -> Sensor gains (h_ws).
    3. Calculate Wall -> Wall gains (h_ww) if not already done.
    4. Iteratively propagate power between wall elements for 'k' bounces.
    
    Args:
        bounces (int): Number of reflection bounces to simulate.
    """
    self.h_mw = self.calc_h(tx = self.masters, rx = self.room.Rx_wall_elements)
    self.h_ws = self.calc_h(tx = self.room.Tx_wall_elements, rx= self.sensors)


    R = np.diag(self.room.Rx_wall_elements.refl.flatten())
    current_wall_power = self.h_mw @ R
    m = self.masters.r.shape[0]
    s = self.sensors.r.shape[0]
    H_diffuse_total = np.zeros((m, s))

    for k in range(1, bounces + 1):
        bounce_contribution = current_wall_power @ self.h_ws
        H_diffuse_total += bounce_contribution
        if k < bounces:
            current_wall_power = (current_wall_power @ self.room.h_ww) @ R

    self.h_diff = H_diffuse_total

  def ris_channel_gains(self):
      """
      Computes the channel gain via a Reconfigurable Intelligent Surface (RIS).
      
      Calculates the two-hop path: Source -> RIS Element -> Sensor.
      This ignores phase shifts (pure amplitude gain), assuming perfect phase alignment 
      or amplitude-only modeling depending on context.
      """
      self.h_ris = np.zeros((self.masters.N, self.sensors.N))
      if self.room.Tx_RIS_elements is None:
          #print("You need to add a RIS surface first")
          return
      else:
          self.no_masters = self.masters.r.shape[0]
          self.no_sensors = self.sensors.r.shape[0]
          self.no_ris_elems = self.room.Tx_RIS_elements.r.shape[0]
          self.D_tx_ris = np.zeros([self.no_masters,self.no_ris_elems,3])
          self.D_tx_ris_unit = np.zeros([self.no_masters,self.no_ris_elems,3])
          self.D_tx_ris_norm = np.zeros([self.no_masters,self.no_ris_elems])
          self.D_rx_ris = np.zeros([self.no_sensors,self.no_ris_elems,3])
          self.D_rx_ris_unit = np.zeros([self.no_sensors,self.no_ris_elems,3])
          self.D_rx_ris_norm = np.zeros([self.no_sensors,self.no_ris_elems])
          self.d = np.zeros([self.no_masters,self.no_sensors,self.no_ris_elems])
          self.hris = np.zeros([self.no_masters,self.no_sensors,self.no_ris_elems])

          r_master = self.masters.r
          r_sensor = self.sensors.r
          r_ris = self.room.Tx_RIS_elements.r

          #BROADCASTING RIS CALCULATIONS
          self.D_tx_ris = -1*(r_master[:,None,:] - r_ris[None,:,:])
          self.D_tx_ris_norm = np.linalg.norm(self.D_tx_ris,axis=2)
          self.D_tx_ris_unit = self.D_tx_ris/self.D_tx_ris_norm[...,None]
          self.D_rx_ris = -1*(r_sensor[:,None,:] - r_ris[None,:,:])
          self.D_rx_ris_norm = np.linalg.norm(self.D_rx_ris,axis=2)
          self.D_rx_ris_unit = self.D_rx_ris/self.D_rx_ris_norm[...,None]
          self.cos_irr = np.maximum(0,np.sum(self.D_tx_ris_unit * self.masters.n.reshape(-1, 1, 3), axis=2))
          self.cos_inc = np.maximum(0,np.sum(self.D_rx_ris_unit * self.sensors.n.reshape(-1, 1, 3), axis=2))
          self.d = self.D_tx_ris_norm[:,None,:] + self.D_rx_ris_norm[None,...]
          self.hris = self.sensors.A[None,...] * (self.masters.m[...,None]+1) * self.cos_irr[:,None,:]**self.masters.m[...,None] * self.cos_inc[None,...]/(2*np.pi * self.d)
          inc_angle = np.arccos(self.cos_inc) # (Nt, Nr)
          is_sp = (self.sensors.type_Rx == 1)
          fov_mask_pd = (inc_angle <= self.sensors.fov)
          self.hris = np.where(~is_sp, self.hris * fov_mask_pd, self.hris)
          if np.any(is_sp):
              sp_factor = solar_panel_angular_efficiency(self.cos_inc)
              h = np.where(is_sp, self.hris * sp_factor, self.hris)
          self.h_ris = np.sum(self.hris,axis=2)
