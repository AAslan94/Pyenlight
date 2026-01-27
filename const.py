import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationDefaults:
    """
    Exhaustive central repository for all simulation constants and fallbacks.
    """
    # --- 1. Physics & Constants ---
    q: float = 1.60217663e-19       # Elementary charge
    kB: float = 1.380649e-23        # Boltzmann constant
    c0: float = 299792458.0         # Speed of light
    hP: float = 6.62607015e-34      # Planck constant
    T0: float = 300.0               # Standard Temperature (K)
    bK: float = 2.8977729e-3        # Wien Constant
    pd_peak: float = 2e9            # PD peak
    T: float = 298                  # Default Temperature
    eo: float = 8.854e-12           # Permittivity of free space
    zp =  np.array([0, 0, 1])
    zm = np.array([0, 0, -1])
    xp = np.array([1, 0, 0])
    xm = np.array([-1, 0, 0])
    yp = np.array([0, 1, 0])
    ym = np.array([0, -1, 0])

    # --- 2. Spectral Integration (SpectralPhysics) ---
    L_MIN: float = 300e-9           # Minimum wavelength (m)
    L_MAX: float = 1200e-9          # Maximum wavelength (m)
    GRID_POINTS: int = 1000         # Integration resolution
    T_sun: float = 5800             # Solar color temperature (K)

    # --- 3. Environmental & Geometric Defaults ---
    reflectivity: float = 0.6       # Default for walls/surfaces
    wall_resolution: tuple = (20, 20) # Default wall discretization
    m: int = 1                      # Default Lambertian order
    rx_area: float = 1e-4           # Default detector area (m^2)
    fov: float = np.pi/2            # Default half-angle FOV
    bounces: int = 4                # Default reflection bounces
    room_dim = np.array([5,5,3])    # room_dimensions
    
   # --- 4. Node & Communication Defaults ---
    IR_Tx_power: float = 15e-3          #IR optical Tx power (W)
    VLC_Tx_power: float = 1            # VLC optical Tx power (W)
    sensitivity: float = -100           # Master RX sensitivity (dBm)
    uplink_type: int = 0                # 0=Optical (IR), 1=RF
    

    # --- 4. Hardware & Energy Defaults (EnergyDefaults.hardware) ---
    f_mcu: float = 16e6             # MCU clock (Hz)
    f_s: float = 1e3                # Sampling freq (Hz)
    voltage: float = 3.3            # Operating voltage (V)
    I_mcu: float = 2.73e-3           # Active MCU current (A)
    I_adc: float = 0.7e-3           # ADC current (A)
    I_ext: float = 1.0e-3           # External sensor current (A)
    I_sleep: float = 2e-6        # Sleep current (A)
    I_wake: float = 1e-3          # Wake-up current (A)

    # --- 5. Communication & Task Defaults (EnergyDefaults.tasks/comm) ---
    L_up_bits: int = 1024           # UL Payload (bits)
    L_dw_bits: int = 128             # DL ACK (bits)
    N_s_up: int = 100               # Samples for UL
    N_c_up: float = 1e3             # Cycles for UL
    bit_rate_up_ir: float = 10e3   # IR UL rate (bps)
    bit_rate_up_rf: float = 250e3   # RF UL rate (bps)
    bit_rate_dw: float = 10e3        # VLC DL rate (bps)
    t_init: float = 5e-3            # Boot time (s)
    t_wait: float = 1e-3           # ACK wait time (s)
    T_cycle: float = 60           # Cycle period (s)
    harvesting_hours: float = 5     # Daily sun exposure
    n_sp: float = 0.4               # Spectral efficiency
    

    # --- 6. Battery Defaults (EnergyDefaults.battery) ---
    battery_capacity_mAh: float = 500
    initial_soc: float = 1          # 100%
    V_batt: float = 3.6             # Nominal
    mpp_eff: float = 0.8            # MPP circuit efficiency 

    # --- 7. RF Channel Model Defaults (Gains.calc_h_rf) ---
    rf_n: float = 1.46              # Path loss exponent
    rf_pl_ref: float = 34.62        # Ref path loss
    rf_k: float = 2.03              # Freq dependence
    rf_f: float = 2.45              # GHz
    rf_sigma: float = 3.76          # Shadowing
    rf_sigma_factor: float = 2      #

    
    # --- 8. TIA Model Defaults ---
    tia = {
        'RF': 1e6, 'CF': 1e-9, 'Vn': 15e-9, 'In': 400e-15,
        'fncV': 1e3, 'fncI': 1e3, 'temperature': 300.0
    }
    
    # --- 9. Photovoltaic (PV) Physics Defaults (PV.__init__) ---
    pv_circuit =  {
        'n': 1.6, 'Rs': 1.0, 'Rsh': 1000.0, 'Voc': 0.64, 'Jsc': 35e-3,
        'Lo': 1e-6, 'Co': 1e-6, 'Rc': 10.0, 
        'Na': 1.0e22, 'Nd': 1.0e25, 'L': 300e-6, 'er': 11.68, 'ni': 1.0e16,
        'Gref': 1000.0, 'f_steps': 600
    }

    # --- 10. Driver Model Defaults ---
    ir_driver =   {
        'imax': 100e-3, 'imin': 0.0,
        'pol': np.array([1.353e-01, 1.868e-01, -1.017e-04]),
        'polinv': np.array([-1.740e+01, 5.329e+00, 5.618e-04])
    }
    rf_driver =  {
        'p_min': -20.0, 'p_max': 5.0, 'pol': np.array([0.24, 8.8])
    }
    
 
