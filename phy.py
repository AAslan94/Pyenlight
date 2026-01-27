import numpy as np
from typing import Dict, Optional, Any
from builder import *
from room import Room
from nodemanager import *
from pv import *
from spatial import *
from const import *


class PhyNet:
  """
  Physics Network Simulation Kernel.

  This class acts as the central controller for the entire optical wireless simulation.
  It orchestrates the initialization of the environment, nodes, and physics engines,
  and executes the main simulation pipeline:
  
  1.  **Build Phase:** Constructs Room, Sensors (SN), Masters (MN), and Ambient nodes (AN).
  2.  **Physics Phase:** Calculates Noise, Geometric Gains, and received signal powers.
  3.  **Optimization Phase (Optional):** Calculates minimum required Tx power to meet BER targets.
  4.  **Analysis Phase:** Computes SNR, Link Margins, and Bit Error Rates (BER).

  Attributes:
      room_builder, sn, mn, an: Builder objects for components.
      snm, mnm, amn: Manager objects for component logic.
      ogains: The Optical Physics Engine (oPhyGains).
      snr_d, ber_d: Downlink performance metrics.
      snr_u, ber_u: Uplink performance metrics.
  """
  def __init__(self, design, budget_run = False):
    """
    Initialize the Simulation Kernel.

    Args:
        design (dict): The master configuration dictionary.
        budget_run (bool): If True, triggers a 'Link Budget' run which calculates 
                           the minimum required Tx power to hit target BERs 
                           before running the full metric analysis.
    """
    self.room_builder = RoomBuilder(design)
    self.room = Room(self.room_builder)
    self.sn = NodeBuilder(design, "sensors")
    self.mn = NodeBuilder(design, "masters")
    
    if "ambient_nodes" in design.get("nodes", {}):
        self.an = NodeBuilder(design, "ambient_nodes")
    else:
        # Create a "Null Builder" or handle the None case in ANManager
        self.an = None
   
    self.snm = SNManager(self.sn)
    self.mnm = MNManager(self.mn)
    
    self.amn = ANManager(self.an) if self.an is not None else None
    
    self.ogains = oPhyGains(self.room, self.mnm, self.snm, self.amn)  
    

    #fetch PV circuit params if they exist 
    self.pv_kwargs = design.get("PV_circuit", {})  

    self.compute_noise()
    if budget_run == True:
      self.set_tx_power()
    self.ogains.compute_downlink()
    self.ogains.compute_uplink()
    self.compute_metrics()


  def calc_min_ow_tx_power(self,target_ber):
    """
    Calculates the minimum Optical Wireless (OW) transmit power required to achieve a target BER.

    Uses the Inverse Q-function ($Q^{-1}$) to determine the required SNR, then solves 
    the link budget equation in reverse:
    
    $$P_{tx, req} = \\frac{2 \cdot Q^{-1}(BER) \cdot \sqrt{\\sigma_{noise}^2}}{H \cdot c_d}$$

    Args:
        target_ber (float): The desired Bit Error Rate (e.g., 3.8e-3).
    """
    self.target_ber = to_scal_Nx1(self.snm.ir_flag,target_ber)
    self.target_g = Qinv(self.target_ber)
    self.target_snr = self.target_g**2
    self.p_req_los = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_los * self.mnm.c_d)
    self.p_req_diff = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_diff * self.mnm.c_d)
    self.p_req_ris = 2 * self.target_g * np.sqrt(self.x_u_noise) / (self.ogains.h_u_ris * self.mnm.c_d)
    self.p_req_total = 2 * self.target_g * np.sqrt(self.x_u_noise) / ((self.ogains.h_u_ris + self.ogains.h_u_los + self.ogains.h_u_diff) * self.mnm.c_d)

    self.p_req = np.min(self.p_req_total,axis = 1).reshape(-1,1) # minimum power required for the best link
    self.p_sel = np.argmin(self.p_req_total,axis = 1, keepdims = True) #find with which MN the best link occurs

  def calc_min_rf_tx_power(self):
    """
    Calculates the minimum RF transmit power to exceed receiver sensitivity.
    
    $$P_{tx} \\ge Sensitivity + PathLoss_{dB}$$
    """
    self.p_rf_x = self.mnm.sensitivity.T + self.ogains.h_u_rf
    self.p_rf = np.min(self.p_rf_x, axis = 1).reshape(-1,1) # minimum power required for the best link
    self.p_rf_sel = np.argmin(self.p_rf_x, axis = 1, keepdims = True) #find with which MN the best link occurs

  def set_tx_power(self,target_ber = 3.8e-3):
    """"
    Optimizes Transmit Power for Uplinks.
    
    Overwrites the default Tx power settings for both IR and RF uplinks based on
    the minimum power needed to meet communication requirements (Target BER for Optical,
    Sensitivity limit for RF).
    """
    if self.snm.rf_flag > 0:
        self.calc_min_rf_tx_power()
        self.snm.RFTx_elements.p = self.p_rf
        
    if self.snm.ir_flag > 0:  
        self.align_sensors_to_master()
        self.calc_min_ow_tx_power(target_ber)
        self.snm.OTx_elements.p = self.p_req
    



  def compute_noise(self):
    """
    Computes total noise power/irradiance for all link types.
    
    **1. Downlink (PV Receivers):**
    Calculates noise **Irradiance** ($W/m^2$).
    * Sum of Artificial Light (Lamps) + Natural Light (Windows).
    
    **2. Downlink (Photodiode Receivers):**
    Calculates noise **Current Variance** ($A^2$).
    * Thermal Noise (from TIA model).
    * Shot Noise (from background light: Lamps + Windows).
    
    **3. Uplink (Master Receivers):**
    Calculates noise **Current Variance** ($A^2$).
    * Thermal + Shot Noise at the Base Station receiver.
    """
    #downlink noise for the PV-based Rxs
    self.g_d_noise = None
    self.flag_pv = (self.snm.ORx_elements.type_Rx == 1).flatten()
    self.no_pv = np.sum(self.flag_pv)
    if self.flag_pv.any():
      self.g_d_noise = np.zeros((1,self.no_pv))
      if self.amn is not None:  
        self.gix_d_noise = self.ogains.ix_d_noise[:,self.flag_pv]/self.snm.ORx_elements.A.flatten()[self.flag_pv]
        self.g_d_noise = self.gix_d_noise
      if self.room.Tx_windows_elements is not None:
        self.gis_d_noise =  self.ogains.is_d_noise[:,self.flag_pv]/self.snm.ORx_elements.A.flatten()[self.flag_pv]
        self.g_d_noise = self.g_d_noise + self.gis_d_noise
    

    #for the downlink we calculate the required BW - we consider all MN to transmit at the same bit rate
    self.Rb_d = to_scal_Nx1(self.snm.no_sensors,self.mnm.Rb_down)
    self.n_sp_d = to_scal_Nx1(self.snm.no_sensors,self.mnm.n_sp_d)    
    self.BW_d = self.Rb_d/self.n_sp_d
     
        
      
    #downlink noise for the PD-based Rxs
    self.x_d_noise = None
    self.flag_pd = (self.snm.ORx_elements.type_Rx == 0).flatten()
    if self.flag_pd.any():
      self.tia_noise_downlink = self.snm.tia.calc_noise_power(self.BW_d.reshape(-1,))[self.flag_pd]
      self.x_d_noise = self.snm.tia.calc_noise_power(self.BW_d.reshape(-1,))[self.flag_pd]
      if self.room.Tx_windows_elements is not None:
        self.xis_d_noise = 2 * SimulationDefaults.q * self.ogains.is_d_noise[:,self.flag_pd] * self.BW_d.reshape(-1,)[self.flag_pd]
        self.x_d_noise = self.tia_noise_downlink + self.xis_d_noise
      if self.amn is not None:
        self.xix_d_noise = 2 * SimulationDefaults.q * self.ogains.ix_d_noise[:,self.flag_pd] * self.BW_d.reshape(-1,)[self.flag_pd]
        self.x_d_noise = self.x_d_noise + self.xix_d_noise

      
    #uplink noise for the PD-based Rxs
    self.x_u_noise = None
    self.Rb_u = self.snm.Rb_up.reshape(-1,1)
    if self.snm.ir_flag > 0:
      self.Rb_u_ir = to_scal_Nx1(self.snm.ir_flag,self.snm.Rb_up_ir)        
      self.n_sp_u = to_scal_Nx1(self.snm.ir_flag,self.snm.n_sp_u)   
      self.BW_u = self.Rb_u_ir/self.n_sp_u  
      self.tia_noise_uplink = self.mnm.tia.calc_noise_power(self.BW_u.reshape(-1,))
      self.x_u_noise = self.mnm.tia.calc_noise_power(self.BW_u.reshape(-1,)).reshape(-1,1)
      if self.room.Tx_windows_elements is not None:
        self.xis_u_noise = 2 * SimulationDefaults.q * self.ogains.is_u_noise * self.BW_u.reshape(-1,1)
        self.x_u_noise = self.tia_noise_uplink.reshape(-1,1) + self.xis_u_noise
      if self.amn is not None:
        self.xix_u_noise = 2 * SimulationDefaults.q * self.ogains.ix_u_noise * self.BW_u.reshape(-1,1)
        self.x_u_noise = self.x_u_noise + self.xix_u_noise

  def align_sensors_to_master(self):
      """
      Updates the orientation (nT) of all IR-enabled sensors to point 
      directly at the Master node. Re-runs the physics engine to apply changes.
      """
      # 1. Check if IR uplinks exist
      if self.snm.ir_flag == 0:
          print("No Optical Uplinks to align.")
          return

      # 2. Check for multiple masters
      m_pos = self.mnm.ORx_elements.r
      if m_pos.shape[0] > 1:
          print("Multiple master nodes, not implemented yet")
          return

      # 3. Calculate Pointing Vectors
      # Master Pos - Sensor Pos
      s_pos = self.snm.OTx_elements.r
      direction = m_pos - s_pos
      
      # Normalize
      norms = np.linalg.norm(direction, axis=1, keepdims=True)
      new_nT = np.divide(direction, norms, out=np.zeros_like(direction), where=norms!=0)

      # 4. Apply new vectors
      self.snm.OTx_elements.n = new_nT.reshape(-1,3)

      # 5. Re-compute Physics and Metrics
      print("Sensors aligned...")


    
  def compute_metrics(self):
    """
    Computes final performance metrics (SNR, BER, Link Margin).

    **Methodology:**
    1.  **PV Receivers:**
        * Instantiates a `PV` circuit model for each receiver.
        * Inputs calculated Signal and Noise Irradiance ($W/m^2$).
        * Solves for Maximum Power Point (MPP).
        * Calculates SNR based on AC small-signal power vs. Thermal/Shot noise power.
    
    2.  **PD Receivers:**
        * Standard SNR formulation: $SNR = \\frac{I_{sig}^2}{4 \\sigma_{noise}^2}$.
        * BER via Q-function: $BER = Q(\\sqrt{SNR})$.
    
    3.  **Hybrid Uplink (RF + Optical):**
        * **RF:** Calculates Link Margin ($P_{rx} - Sensitivity$).
        * **Optical:** Calculates SNR and BER.
        * Selects the best performing link (Max SNR / Max Margin).
    """
    #calculate metrics for downlink

    self.snr_d = np.zeros(self.snm.no_sensors)
    self.snr_d_dB = np.zeros(self.snm.no_sensors)

    #calculate for PV-based Rxs
    if self.flag_pv.any():
      self.g_d_los = self.ogains.i_d_los[:,self.flag_pv]/self.snm.ORx_elements.A.flatten()[self.flag_pv]
      self.g_d_diff = self.ogains.i_d_diff[:,self.flag_pv]/self.snm.ORx_elements.A.flatten()[self.flag_pv]
      self.g_d_ris = self.ogains.i_d_ris[:,self.flag_pv]/self.snm.ORx_elements.A.flatten()[self.flag_pv]


      self.g_d_signal = self.g_d_los + self.g_d_diff + self.g_d_ris

      #let's consider in this case all MN LEDs transmit the same data simultaneously
      self.g_d_signal_total = np.max(self.g_d_signal, axis = 0)

      self.pvx = PV(Gsignal = self.g_d_signal_total, Gamb = self.g_d_noise, 
                    A = self.snm.ORx_elements.A.flatten()[self.flag_pv] ,
                    unscaled = True, run = True, **self.pv_kwargs)

      self.signal_pv = np.take_along_axis(self.pvx.vac[..., -1], self.pvx.ind.reshape(-1,1), axis=1)/0.707


      self.noise_pv = 4 * (np.take_along_axis(self.pvx.th_noise, self.pvx.ind.reshape(-1,1), axis=1) + np.take_along_axis(self.pvx.sh_noise, self.pvx.ind.reshape(-1,1), axis=1))
      self.snr_pv = self.signal_pv**2 / self.noise_pv
      self.snr_pv_dB = 10 * np.log10(self.snr_pv)

      self.snr_d[self.flag_pv] = self.snr_pv.reshape(-1,)
      self.snr_d_dB[self.flag_pv] = self.snr_pv_dB.reshape(-1,)

    #for PD-based Rxs

    if self.flag_pd.any():
      self.x_d_los = self.ogains.i_d_los[:,self.flag_pd]
      self.x_d_diff = self.ogains.i_d_diff[:,self.flag_pd]
      self.x_d_ris = self.ogains.i_d_ris[:,self.flag_pd]

      self.x_d_signal = self.x_d_los + self.x_d_diff + self.x_d_ris

      self.x_d_signal_total = np.sum(self.x_d_signal, axis = 0)


      self.snr_pd = self.x_d_signal_total**2 / (4 * self.x_d_noise)
      self.snr_pd_dB = 10 * np.log10(self.snr_pd)

      self.snr_d[self.flag_pd] = self.snr_pd.reshape(-1,)
      self.snr_d_dB[self.flag_pd] = self.snr_pd_dB.reshape(-1,)


    self.BER_d = Qfunction(np.sqrt(self.snr_d))


    if self.snm.rf_flag > 0:

      #uplink - RF first
      self.hrf = self.ogains.h_u_rf
      #find incident RF power
      self.p_u_rf_m = self.snm.RFTx_elements.p - self.hrf

      #find power margin
      self.rf_margin = self.p_u_rf_m - self.mnm.sensitivity.T

      #find the best link

      self.rf_best_margin = np.max(self.rf_margin, axis = 1)
      self.u_sel_rf = np.argmax(self.rf_margin, axis = 1, keepdims=True)

      #find incident power for best margin
      self.p_u_rf = np.take_along_axis(self.p_u_rf_m, self.u_sel_rf, axis = 1)

    if self.snm.ir_flag > 0:  
       #optical wireless uplinks
      self.snr_u = self.ogains.i_u_signal**2/(4*self.x_u_noise)
      self.snr_u_dB = 10 * np.log10(self.snr_u)
      self.BER_u = Qfunction(np.sqrt(self.snr_u))

       #select best link
      self.snr_u_sel = np.max(self.snr_u,axis = 1)
      self.u_sel_ow = np.argmax(self.snr_u,axis = 1, keepdims = True)
      self.ber_u_sel = np.take_along_axis(self.BER_u, self.u_sel_ow, axis = 1)

