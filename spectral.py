import numpy as np
from typing import Callable
from const import Constants

class SpectralPhysics:
    """
    Physics engine for calculating effective responsivity by integrating spectral 
    overlaps of light sources, detectors, and optical filters.
    
    This class manages spectral data generation (using Gaussian approximations 
    or blackbody radiation), polynomial fitting for detector sensitivity, and 
    numerical integration to determine system efficiency.
    """
    # 1. Store constants clearly to avoid magic numbers
    L_MIN, L_MAX = 300e-9, 1200e-9
    GRID_POINTS = 1000
    # Pre-define the spectrum configurations for easy lookup
    # Format: "NAME": (Source_Func, Detector_Func, Filter_Name)
    CONFIGURATIONS = {
        "WLED2PD":   ("white_led_spectrum", "photodiode_responsivity", "ALL_PASS"),
        "WLED2PDwF": ("white_led_spectrum", "photodiode_responsivity", "VLC_PASS"),
        "IR2PD":     ("tsff5210_spectrum", "photodiode_responsivity", "ALL_PASS"),
        "IR2PDwF":   ("tsff5210_spectrum", "photodiode_responsivity", "IR_PASS"),
        "WLED2PV":   ("white_led_spectrum", "solar_panel_sensitivity", "ALL_PASS"),
        "SUN2PDwFv": ("sun_spectrum","photodiode_responsivity", "VLC_pass"),
        "SUN2PDwFi": ("sun_spectrum","photodiode_responsivity", "IR_pass"),
        "SUN2PD":    ("sun_spectrum","photodiode_responsivity", "ALL_pass"),
        "SUN2PV":    ("sun_spectrum","solar_panel_sensitivity", "ALL_pass"),

    }

    # --- Helper Methods to reduce Math Duplication ---
    @staticmethod
    def _gaussian(wl: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
        """Generates a Gaussian curve based on Peak and FWHM."""
        sigma = fwhm / (2 * np.sqrt(np.log(2)))
        return np.exp(-(wl - peak)**2.0 / sigma**2.0)

    @staticmethod
    def _poly_response(wl: np.ndarray, coeffs: np.ndarray,
                       l_min_nm: float, l_max_nm: float) -> np.ndarray:
        """Evaluates a polynomial response within a specific nanometer range."""
        wl_nm = wl / 1e-9
        response = np.zeros_like(wl)

        # Boolean mask for valid range
        mask = (wl_nm >= l_min_nm) & (wl_nm <= l_max_nm)

        if np.any(mask):
            # Preserve the specific scaling logic from your original code
            x_scaled = 2 * wl_nm[mask] / (l_min_nm + l_max_nm)
            response[mask] = np.polyval(coeffs, x_scaled)

        return response

    # --- Source Definitions ---
    @classmethod
    def white_led_spectrum(cls, wl: np.ndarray) -> np.ndarray:
        """
        Models a White LED spectrum as a sum of two Gaussians.
        
        Represents the Blue pump (peak ~470nm) and the Yellow Phosphor emission 
        (peak ~600nm).
        """
        
        # Sum of two Gaussians (Blue pump + Phosphor)
        blue = cls._gaussian(wl, peak=470e-9, fwhm=20e-9)
        phosphor = cls._gaussian(wl, peak=600e-9, fwhm=100e-9)
        #plt.plot(wl,blue+phosphor)
        return blue + phosphor

    @classmethod
    def tsff5210_spectrum(cls, wl: np.ndarray) -> np.ndarray:
        """Models the TSFF5210 IR Emitter spectrum (Peak ~870nm)."""
        return cls._gaussian(wl, peak=870e-9, fwhm=40e-9)

    # --- Detector Definitions ---
    @classmethod
    def photodiode_responsivity(cls, wl: np.ndarray) -> np.ndarray:
        """
        Returns the spectral responsivity of the system photodiode.
        
        Based on a 5th-order polynomial fit valid between 330nm and 1090nm.
        """
        coeffs = np.array([-6.39503882, 27.47316339, -45.57791267,
                           36.01964536, -12.8418451, 1.73076976])
        return cls._poly_response(wl, coeffs, 330, 1090)

    @classmethod
    def solar_panel_sensitivity(cls, wl: np.ndarray) -> np.ndarray:
        """
        Returns the spectral sensitivity of the system Solar Panel.
        
        Based on a 6th-order polynomial fit valid between 300nm and 1175nm.
        """
        
        coeffs = np.array([26.78555644, -160.24353775, 381.86564712, -463.07816469,
                           300.12488471, -97.25192023, 12.34949208])
        #plt.plot(wl,cls._poly_response(wl, coeffs, 300, 1175))
        return cls._poly_response(wl, coeffs, 300, 1175)

    @classmethod
    def sun_spectrum(cls,wl:np.ndarray) -> np.ndarray:
        """
        Approximates the Solar Spectrum using Planck's Blackbody radiation law.
        
        Assumes a color temperature T = 5800K. The spectrum is normalized 
        relative to the peak power.
        """
        
        T = 5800
        pmax = 1
        lmax = Constants.bK / T

        def blackbody(lam):
            return (2 * Constants.hP * Constants.c0**2 / lam**5 /
                    (np.exp(Constants.hP * Constants.c0 /
                            (lam * Constants.kB * T)) - 1))

        Pmax = blackbody(lmax)
        xd = pmax * blackbody(wl) / Pmax
        return xd

    @classmethod
    def sun_power(cls):
        """Calculates the total integrated power of the generated solar spectrum."""
        wl = np.linspace(cls.L_MIN, cls.L_MAX, cls.GRID_POINTS)
        xd = cls.sun_spectrum(wl)
        return np.trapz( xd, wl)

    # --- Filters ---
    @staticmethod
    def get_filter_transmission(name: str, wl: np.ndarray) -> np.ndarray:
        """
        Returns the transmission window (0.0 or 1.0) for a given filter name.
        
        Supported filters:
            - 'VLC_PASS': Visible light pass (320nm - 720nm)
            - 'IR_PASS': Infrared pass (770nm - 1100nm)
        """
        
        if name == "VLC_PASS":
            return ((wl >= 320e-9) & (wl <= 720e-9)).astype(float)
        elif name == "IR_PASS":
            return ((wl >= 770e-9) & (wl <= 1100e-9)).astype(float)
        return np.ones_like(wl)

    # --- Core Logic ---
    @classmethod
    def calculate_effective_responsivity(cls,
                                         source_func: Callable,
                                         detector_func: Callable,
                                         filter_name: str = "ALL_PASS") -> float:
        """Calculates overlap integral of Source * Detector * Filter / Integral(Source)."""
        # Create grid once
        wl = np.linspace(cls.L_MIN, cls.L_MAX, cls.GRID_POINTS)

        P_src = source_func(wl)
        R_det = detector_func(wl)
        T_filt = cls.get_filter_transmission(filter_name, wl)

        # Determine denominator (Total Source Power)
        total_power = np.trapz(P_src, wl)

        if total_power == 0:
            return 0.0

        # Determine numerator (detected power)
        detected_power = np.trapz(P_src * R_det * T_filt, wl)

        return float(detected_power / total_power)

    @classmethod
    def get_responsivity_by_name(cls, config_name: str) -> float:
        """Retrieves and calculates responsivity based on the configuration name."""
        if config_name not in cls.CONFIGURATIONS:
            raise ValueError(f"Unknown configuration: {config_name}. Available: {list(cls.CONFIGURATIONS.keys())}")

        src_name, det_name, filt_name = cls.CONFIGURATIONS[config_name]

        # Dynamically fetch the methods from the class
        src_func = getattr(cls, src_name)
        det_func = getattr(cls, det_name)

        return cls.calculate_effective_responsivity(src_func, det_func, filt_name)

