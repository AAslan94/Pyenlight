import numpy as np
from const import *

class TIA:
    """
    Models the frequency response and noise performance of a Transimpedance Amplifier (TIA).
    
    Attributes:
        RF (float or np.array): Feedback resistance (Ohms).
        Vn (float or np.array): Op-amp input voltage noise density (V/sqrt(Hz)).
        In (float or np.array): Op-amp input current noise density (A/sqrt(Hz)).
        fncV (float or np.array): 1/f corner frequency for voltage noise (Hz).
        fncI (float or np.array): 1/f corner frequency for current noise (Hz).
        temperature (float or np.array): Temperature in Kelvin.
    """

    def __init__(self, **kwargs):
        # 1. Fetch the default dictionary
        d = SimulationDefaults.tia

        # 2. Assign parameters using pop logic or get fallback
        # We use .get() to avoid KeyError if the key is missing in kwargs
        self.RF = kwargs.get('RF', d['RF'])
        self.Vn = kwargs.get('Vn', d['Vn'])
        self.In = kwargs.get('In', d['In'])
        self.fncV = kwargs.get('fncV', d['fncV'])
        self.fncI = kwargs.get('fncI', d['fncI'])
        self.temperature = kwargs.get('temperature', SimulationDefaults.T)

    # -----------------------------
    # Vectorized transfer function
    # -----------------------------
    def CF(self, B):
        return 1 / (2 * np.pi * B * self.RF)   # (NB,)

    def ZF(self, f, B):
        """
        f: (Nf,)
        B: (NB,)
        return: (NB, Nf)
        """
        f = f[None, :]
        B = B[:, None]
        CF = self.CF(B)

        return self.RF / (1 + 1j * 2 * np.pi * f * CF * self.RF)

    # -----------------------------
    # PSDs (all vectorized)
    # -----------------------------
    def RF_psd(self, f, B):
        return (4 * SimulationDefaults.kB * self.temperature / self.RF) * np.ones((B.size, f.size))

    def SV_psd(self, f, B):
        f = f[None, :]
        Z = self.ZF(f.squeeze(), B)
        return (self.Vn**2 + self.Vn**2 * self.fncV / f) / np.abs(Z)**2

    def SI_psd(self, f, B):
        f = f[None, :]
        return self.In**2 + self.In**2 * self.fncI / f

    def psd(self, f, B):
        return (
            self.RF_psd(f, B)
            + self.SV_psd(f, B)
            + self.SI_psd(f, B)
        )

    # -----------------------------
    # Noise power (vectorized)
    # -----------------------------
    def calc_noise_power(self, B, Nf=1000, fmin=0.1):
        B = np.atleast_1d(B)

        # Normalized frequency grid (0..1), shared
        x = np.linspace(0, 1, Nf)
        f = fmin + x * (B[:, None] - fmin)  # (NB, Nf)

        psd_vals = self.psd(f[0], B)        # f[0] gives base grid
        psd_vals *= (f <= B[:, None])       # enforce per-B cutoff

        return np.trapz(psd_vals, f, axis=1)
        
        
        
class IRdriver:
    """
    Models the electro-optical characteristics of an Infrared LED driver.
    
    Robustly handles initialization via a configuration dictionary.
    """

    def __init__(self, **kwargs):
        """
        Initialize the IR Driver model using keyword arguments.
        
        Args:
            **kwargs: Dictionary containing:
                - imax (float): Saturation current limit (default: 100mA).
                - imin (float): Cut-off current limit (default: 0mA).
                - pol (array): Coefficients for I -> P (default provided).
                - polinv (array): Coefficients for P -> I (default provided).
        """
        # 0. fallback dictionary
        d = SimulationDefaults.ir_driver
        
        # 1. Extract params with defaults using kwargs.get()
        self.imax = kwargs.get('imax', d['imax'])
        self.imin = kwargs.get('imin', d['imin'])
        
        # Default polynomials (Polynomials for standard IR LED)
        self.pol = np.array(kwargs.get('pol', d['pol']))
        self.polinv = np.array(kwargs.get('polinv', d['polinv']))
        

        # 2. Pre-calculate limits
        self.Pmax = np.polyval(self.pol, self.imax)
        self.Pmin = np.polyval(self.pol, self.imin)

    def calc_I(self, P):
        """Calculates Drive Current from Optical Power."""
        I = np.polyval(self.polinv, P)
        
        # Vectorized Saturation Checks
        I = np.atleast_1d(I)
        P = np.atleast_1d(P)
        
        I[I >= self.imax] = np.inf
        I[P >= self.Pmax] = np.inf
        
        if I.size == 1: return I.item()
        return I

    def calc_P(self, I):
        """Calculates Optical Power from Drive Current."""
        return np.polyval(self.pol, I)
        
def RF_calc_I(P, **kwargs):
    """
    Estimates the power consumption current of the RF transmitter.
    
    Args:
        P (float or np.ndarray): RF Transmit Power (dBm).
        **kwargs: 
            - p_min (float): Min power limit. Default: -20
            - p_max (float): Max power limit. Default: 5
            - pol (array): [slope, intercept]. Default: [0.24, 8.8]
    """
    # 0. fallback values
    d = SimulationDefaults.rf_driver
    
    # 1. Extract parameters from kwargs with fallbacks
    p_min = kwargs.get('p_min', d['p_min'])
    p_max = kwargs.get('p_max', d['p_max'])
    pol = kwargs.get('pol', d['pol'])

    # 2. Ensure P is a numpy array for consistent masking/clipping
    P_bounded = np.array(P, copy=True)
    
    # 3. Apply the clipping/logic
    # Anything below p_min is treated as p_min
    P_bounded[P_bounded < p_min] = p_min
    
    # Identify values exceeding p_max for infinity current later
    over_limit_mask = P_bounded > p_max

    # 4. Calculation using the bounded power
    # np.atleast_1d handles the "Node 0" scalar case to prevent size mismatches
    I = np.atleast_1d(np.polyval(pol, P_bounded) * 1e-3)
    
    # 5. Apply infinity to current for values exceeding limits
    if np.any(over_limit_mask):
        I = I.astype(float) # Ensure we can hold np.inf
        I[over_limit_mask] = np.inf
        
    # Return as scalar if input was scalar, else return array
    return I.item() if np.isscalar(P) else I
