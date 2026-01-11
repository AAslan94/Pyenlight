import numpy as np

class Constants:
    """Physical constants used across the simulation."""
    q = 1.60217663e-19       # Elementary charge
    kB = 1.380649e-23        # Boltzmann constant
    c0 = 299792458.0         # Speed of light
    hP = 6.62607015e-34      # Planck constant
    T0 = 300.0               # Standard Temperature (K)
    bK = 2.8977729e-3        # Wien Constant
    pd_peak = 2e9
    T = 298                  # Temperature
    eo = 8.854e-12
    # Vectors
    zp = np.array([0, 0, 1])
    zm = np.array([0, 0, -1])
    xp = np.array([1, 0, 0])
    xm = np.array([-1, 0, 0])
    yp = np.array([0, 1, 0])
    ym = np.array([0, -1, 0])


class DefaultSimValues:
    fov = np.pi/2
    m = 1
    A = 1e-4
    n = Constants.zp
    sensitivity = -100 #dBm

class EnergyDefaults:
    # Hardware defaults
    hardware = {
        'f_mcu': 16e6, 'f_s': 1e3, 'voltage': 3.3,
        'I_mcu': 5.0e-3, 'I_adc': 2.0e-3, 'I_act': 3.0e-3, 'I_ext': 1.0e-3,
        'I_sleep': 0.01e-3, 'I_wake': 5.0e-3,
    }
    # Networking defaults
    communication = {
        'bit_rate_up_ir': 100e3, 'bit_rate_up_rf': 250e3, 'bit_rate_dw': 1e6,
        't_init': 5e-3, 't_wait': 10e-3,
        "T_cycle":1,
        'harvesting_hours': 5
    
    }
    # Computational/Sensing task defaults
    tasks = {
        'L_up_bits': 1024, 'L_dw_bits': 64,
        'N_s_up': 100, 'N_c_up': 1e5,
        'N_s_dw': 0, 
    }
    battery = {
        'battery_capacity_mAh': 500,
        'initial_soc': 1,
        'V_batt': 3.6,
        
    }
    
    
 
