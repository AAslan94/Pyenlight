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

    def __init__(self, **argv):
        self.RF = argv.pop('RF')
        self.Vn = argv.pop('Vn')
        self.In = argv.pop('In')
        self.fncV = argv.pop('fncV')
        self.fncI = argv.pop('fncI')
        self.temperature = argv.pop('temperature')

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
        return (4 * Constants.kB * self.temperature / self.RF) * np.ones((B.size, f.size))

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
        # 1. Extract params with defaults using kwargs.get()
        self.imax = kwargs.get('imax', 100e-3)
        self.imin = kwargs.get('imin', 0e-3)
        
        # Default polynomials (Polynomials for standard IR LED)
        default_pol = np.array([ 1.35376064e-01,  1.86846949e-01, -1.01789073e-04])
        default_polinv = np.array([-1.74039667e+01, 5.32917840e+00, 5.61867428e-04])
        
        # Ensure inputs are numpy arrays
        self.pol = np.array(kwargs.get('pol', default_pol))
        self.polinv = np.array(kwargs.get('polinv', default_polinv))

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
      **kwargs: Optional dictionary containing:
          - pol (array): Polynomial coefficients [slope, intercept]. 
                         Default: [0.24, 8.8]
  
  Returns:
      float or np.ndarray: Supply Current (Amps).
  """
  # Extract 'pol' from kwargs, or use default if missing
  default_pol = np.array([0.24, 8.8])
  pol = kwargs.get('pol', default_pol)
  
  # Calculation
  I = np.polyval(pol, P) * 1e-3
  return I
