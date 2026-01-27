import numpy as np
from typing import Dict, Any, Optional

from phy import *
from models import *
from spatial import *
from const import *



class EnergyManager:
    """
    Energy Consumption & Harvesting Manager.

    This class simulates the power profile of sensor nodes over a defined operation cycle.
    It integrates hardware specifications, task computational loads, and communication 
    overhead to compute the total energy drain per cycle. It also calculates the 
    energy harvesting potential for PV-equipped nodes.

    **Energy Model:**
    The cycle is decomposed into discrete states:
    1.  **Initialization:** Wake-up and boot sequence.
    2.  **Sensing:** ADC sampling and sensor readout.
    3.  **Processing:** MCU computation (compression, formatting).
    4.  **Uplink (Tx):** Data transmission (IR LED or RF Radio).
    5.  **Turnaround (Wait):** Waiting for Downlink (ACK/Data).
    6.  **Downlink (Rx):** Receiving data.
    7.  **Sleep:** Low-power state for the remainder of the cycle period ($T_{cycle}$).

    Attributes:
        f_mcu, f_s (np.ndarray): Clock frequencies for MCU and Sampling [Hz].
        V (np.ndarray): Supply Voltage [V].
        I_* (np.ndarray): Current consumption for various hardware states [A].
        N_* (np.ndarray): Number of clock cycles for tasks.
        L_* (np.ndarray): Data packet lengths [bits].
        br_* (np.ndarray): Bit rates for communication [bps].
        harvesting_hours (np.ndarray): Effective hours of light exposure per day for PV nodes.
        batt_charge (np.ndarray): Total battery energy capacity in Joules.
    """
    def __init__(self, phy_net, design):
        """
        Initialize the Energy Manager.

        Parses the configuration dictionary to populate hardware and protocol parameters.
        Uses a hierarchical lookup (Design Profile -> Protocol -> Defaults) to ensure
        all parameters are set, broadcasting scalars to arrays of size $N$ (number of sensors).
        
        It also initializes the battery state and assigns harvesting hours specifically 
        to nodes identified as having PV capabilities.

        Args:
            phy_net (PhyNet): The main simulation object (access to physics models).
            design (dict): Configuration dictionary containing 'nodes', 'energy_profile', etc.
        """
        self.pn = phy_net
        self.N = self.pn.snm.no_sensors

        self.nodes = design['nodes']['sensors']
        self.u_prof = design.get('energy_profile', {})
        self.u_prot = design.get('protocol', {})
        self.u_mpp = design.get('MPP', {})

        # 0. Driver Initialization
        # Pulls polynomials/limits from SimulationDefaults internally
        self.ir_driver = IRdriver(**self.u_prof.get('IRDriver', {}))
        self.rf_config = self.u_prof.get('RFDriver', {})

         # 1. Hardware Specs
        self.f_mcu = self._v('f_mcu', SimulationDefaults, 'hardware')
        self.f_s   = self._v('f_s',   SimulationDefaults, 'hardware')
        self.V     = self._v('voltage', SimulationDefaults, 'hardware')
        self.I_mcu, self.I_adc = self._v('I_mcu', SimulationDefaults), self._v('I_adc', SimulationDefaults)
        self.I_ext = self._v('I_ext', SimulationDefaults)
        self.I_sleep, self.I_wake = self._v('I_sleep', SimulationDefaults), self._v('I_wake', SimulationDefaults)

        # 2. Task Loads
        self.N_s_up = self._v('N_s_up', SimulationDefaults, 'tasks')
        self.N_c_up = self._v('N_c_up', SimulationDefaults, 'tasks')
        self.L_up   = self._v('L_up_bits', SimulationDefaults, 'tasks')
        self.L_dw   = self._v('L_dw_bits', SimulationDefaults, 'tasks')

        # 3. Communication Overheads
        self.Rb_up = self.pn.Rb_u.flatten()     # Already handles Design -> Defaults fallback in SNManager
        self.Rb_down = self.pn.Rb_d.flatten()  # Already handles Design -> Defaults fallback in MNManagers
        
        
        self.t_init, self.t_wait = self._v('t_init', SimulationDefaults), self._v('t_wait', SimulationDefaults)
        self.T_cycle = self._v('T_cycle', SimulationDefaults)
               

        #4. Battery 
        self.batt_capacity_mAh = self._v('battery_capacity_mAh', SimulationDefaults, "battery")
        self.V_batt = self._v('V_batt', SimulationDefaults, "battery")
        self.initial_soc = self._v("initial_soc",  SimulationDefaults, "battery")
        self.batt_charge = self.batt_capacity_mAh * self.V_batt * 3.6 * self.initial_soc #mAh to joules

        # 1. Start with 0 for everyone
        self.harvesting_hours = np.zeros(self.N)

        hh_input = self.u_prot.get('harvesting_hours', SimulationDefaults.harvesting_hours)

        self.mpp_eff = self.u_mpp.get('mpp_eff', SimulationDefaults.mpp_eff)
        
        # 3. Apply ONLY to PV nodes
        if hasattr(self.pn, 'flag_pv') and np.any(self.pn.flag_pv):
            try:
                # This automatically handles Scalar (broadcasts) OR Array (if size matches mask)
                self.harvesting_hours[self.pn.flag_pv] = hh_input
            except ValueError:
                # If sizes don't match, warn and fail safe to 0
                print(f"Warning: 'harvesting_hours' array size does not match number of PV nodes. Using 0.")
        
        # Placeholders for daily stats
        self.E_day_consumed = np.zeros(self.N)
        self.E_day_harvested = np.zeros(self.N)
        self.E_day_net = np.zeros(self.N)
        self.days_to_empty = np.zeros(self.N)

        self.calc_cycle_energy()
        self.calc_harv_energy()
        self.calc_battery_life()

    def _v(self, key, default_source, profile_sub=None):
        """Standardizes lookup: energy_profile[sub] -> protocol -> SimulationDefaults."""
    
        # 1. Search energy_profile sub-categories (e.g., 'hardware', 'tasks')
        if profile_sub:
            d = self.u_prof.get(profile_sub, {})
            if key in d: 
                return as_array_of_size(d[key], self.N)
    
        # 2. Search root energy_profile
        if key in self.u_prof:
            return as_array_of_size(self.u_prof[key], self.N)

        # 3. Search root protocol
        if key in self.u_prot:
            return as_array_of_size(self.u_prot[key], self.N)

        # 4. Fallback to SimulationDefaults attributes 
        val = getattr(default_source, key, None)
    
        if val is None:
            raise AttributeError(f"Parameter '{key}' not found in design or SimulationDefaults.")
        
        return as_array_of_size(val, self.N)

    
    def calc_cycle_energy(self):
        """
        Calculates the total energy consumption per operation cycle.

        **Physics:**
        1.  **Duration ($t$):** Calculated based on task size ($N_{cycles}$) and frequency ($f$), 
            or packet size ($L$) and bit rate ($R$).
            $$t_{proc} = N_{cycles} / f_{clk}, \quad t_{tx} = L_{bits} / R_{bps}$$
        
        2.  **Energy ($E$):** Integrated over time for all states.
            $$E_{active} = V \cdot \sum (I_{state} \cdot t_{state})$$
            $$E_{sleep} = V \cdot I_{sleep} \cdot (T_{cycle} - t_{active})$$

        **Note:** During the 'Wait' phase (turnaround time), the node is assumed to be in 
        an Active/Idle state (`I_mcu`), not sleep mode, to quickly respond to downlink.

        Returns:
            None (updates self.E_cycle in place).
        """
        
        # --- DURATIONS (s) ---
        self.d_init   = self.t_init
        self.d_sens_u = self.N_s_up / self.f_s
        self.d_proc_u = self.N_c_up / self.f_mcu
        
        # Uplink Transmission Time
        self.ir_m, self.rf_m = (self.nodes['uplink_type'] == 0), (self.nodes['uplink_type'] == 1)
        self.d_tx = np.zeros(self.N)
        self.d_tx = self.L_up / self.Rb_up
       

        # Downlink Reception Time
        self.d_wait   = self.t_wait #turnaround time
        self.d_rx     = self.L_dw / self.Rb_down

        # --- CURRENTS (A) ---
        self.I_sens = (self.I_adc + self.I_mcu + self.I_ext) 
        self.I_proc = (self.I_mcu) 
        self.I_rx   = (self.I_mcu + self.I_adc + self.I_mcu) 

        self.I_tx = np.zeros(self.N)
        if np.any(self.ir_m):
            self.I_tx[self.ir_m] = (self.I_mcu[self.ir_m] + 0.5 * self.ir_driver.calc_I(self.pn.snm.OTx_elements.p.reshape(-1,))) 
        if np.any(self.rf_m):
            #clip values to transceiver limits
            self.I_tx[self.rf_m] = RF_calc_I(self.pn.snm.RFTx_elements.p.reshape(-1,), **self.rf_config) 
            #consumption from RF Transceiver during this phase usually already includes all relevant current

        # --- ENERGY INTEGRATION (J) ---
        
        self.E_active = self.V * (
            (self.I_wake * self.d_init) + #WU
            (self.I_sens * self.d_sens_u) + #SENS
             (self.I_proc * self.d_proc_u) + #PROC
              (self.I_tx * self.d_tx) + # UL
            (self.I_mcu * self.d_wait) + # Wait
            (self.I_rx * self.d_rx) # DL
        )

        
        self.d_total = self.d_init + self.d_sens_u + self.d_proc_u + self.d_tx + self.d_wait + self.d_rx 
        self.E_sleep = self.V * (self.I_sleep) * np.maximum(0, self.T_cycle - self.d_total)

        self.E_cycle = self.E_active + self.E_sleep

    def calc_harv_energy(self):
      """
      Calculates the harvestable power for PV-equipped nodes.

      Extracts the Maximum Power Point (MPP) voltage and current from the `PV` 
      physics model (calculated in `PhyNet`) and applies an efficiency factor 
      for the DC-DC converter / PMIC.

      Args:
          mpp_eff (float): Efficiency of the Maximum Power Point Tracking (MPPT) circuit (0.0 - 1.0).
      
      Sets:
          self.p_raw (np.ndarray): Raw power at MPP [Watts].
          self.p_harv (np.ndarray): Extractable power after conversion losses [Watts].
      """
      self.p_raw = np.zeros(self.N)
      self.p_harv = np.zeros(self.N)
      
      if self.pn.flag_pv.any():
        v = np.take_along_axis(self.pn.pvx.V, self.pn.pvx.ind.reshape(-1,1),axis = 1)
        i = np.take_along_axis(self.pn.pvx.I, self.pn.pvx.ind.reshape(-1,1),axis = 1)
        self.p_r = v*i
        self.p_h = v*i * self.mpp_eff
        self.p_raw[self.pn.flag_pv] = self.p_r.flatten()
        self.p_harv[self.pn.flag_pv] = self.p_h.flatten()        
      else:
        print("There are not any PV-based receivers.")

    def calc_battery_life(self):
        """
        Calculates daily energy budget and estimates battery lifetime.

        Scales the single-cycle energy consumption to a full 24-hour day and compares it
        against the daily harvested energy to determine net energy flow.
        
        **Formula:**
        $$E_{day, cons} = E_{cycle} \times (24h / T_{cycle})$$
        $$E_{day, net} = (P_{harv} \times t_{sun}) - E_{day, cons}$$
        $$Days_{left} = E_{battery} / |E_{day, net}| \quad (\text{if } E_{day, net} < 0)$$

        Prints a tabular report of the results.
        """
        if not hasattr(self, 'E_cycle'):
            print("Running cycle energy calc first...")
            self.calc_cycle_energy()
            
        self.current_energy = self.batt_charge
        self.cycles_per_hour = 3600 / self.T_cycle
        self.E_day_consumed = self.E_cycle * self.cycles_per_hour * 24.0

        if not hasattr(self, 'p_harv'):
            self.p_harv = np.zeros(self.N)
        
        self.E_day_harvested = self.p_harv * self.harvesting_hours * 3600.0
        self.E_day_net = self.E_day_harvested - self.E_day_consumed

        drain_mask = self.E_day_net < 0
        self.days_to_empty = np.full(self.N, np.inf)

        if np.any(drain_mask):
            loss_rate = np.abs(self.E_day_net[drain_mask])
            self.days_to_empty[drain_mask] = self.current_energy[drain_mask] / loss_rate

        # --- Updated Tabular Print ---
        header = f"{'Node ID':<8} {'Rx':<12} {'Uplink':<8} {'Cons/Day(J)':<15} {'Harv/Day(J)':<15} {'Net/Day(J)':<15} {'Life(Days)':<10}"
        print(header)
        print("-" * len(header))

        for i in range(self.N):
             # Determine Node Type (PV vs Battery Only)
             node_type = "PV" if self.pn.flag_pv[i] else "PD"
             
             # Determine Uplink Type from the NodeBuilder uplink_type array
             # 0 = IR, 1 = RF
             u_type_val = self.pn.sn.uplink_type[i]
             link_type = "IR" if u_type_val == 0 else "RF"
             
             life_str = "Inf" if self.days_to_empty[i] == np.inf else f"{self.days_to_empty[i]:.2f}"
             
             print(f"{i:<8} {node_type:<12} {link_type:<8} "
                   f"{self.E_day_consumed[i]:<15.2f} {self.E_day_harvested[i]:<15.2f} "
                   f"{self.E_day_net[i]:<15.2f} {life_str:<10}")
