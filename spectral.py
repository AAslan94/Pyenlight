import numpy as np
from typing import Callable
from const import *

class SpectralPhysics:
    """
    Physics engine for calculating effective responsivity by integrating spectral 
    overlaps using SimulationDefaults for all constants and grid parameters.
    """
    # Grid settings pull directly from the merged SimulationDefaults attributes
    L_MIN = SimulationDefaults.L_MIN
    L_MAX = SimulationDefaults.L_MAX
    GRID_POINTS = SimulationDefaults.GRID_POINTS

    # Pre-define the spectrum configurations for easy lookup
    # Format: "NAME": (Source_Func, Detector_Func, Filter_Name)
    CONFIGURATIONS = {
        "WLED2PD":   ("white_led_spectrum", "photodiode_responsivity", "ALL_PASS"),
        "WLED2PDwF": ("white_led_spectrum", "photodiode_responsivity", "VLC_PASS"),
        "IR2PD":     ("tsff5210_spectrum", "photodiode_responsivity", "ALL_PASS"),
        "IR2PDwF":   ("tsff5210_spectrum", "photodiode_responsivity", "IR_PASS"),
        "WLED2PV":   ("white_led_spectrum", "solar_panel_sensitivity", "ALL_PASS"),
        "SUN2PDwFv": ("sun_spectrum","photodiode_responsivity", "VLC_PASS"),
        "SUN2PDwFi": ("sun_spectrum","photodiode_responsivity", "IR_PASS"),
        "SUN2PD":    ("sun_spectrum","photodiode_responsivity", "ALL_PASS"),
        "SUN2PV":    ("sun_spectrum","solar_panel_sensitivity", "ALL_PASS"),
    }

    # --- Helper Methods ---
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
        mask = (wl_nm >= l_min_nm) & (wl_nm <= l_max_nm)
        if np.any(mask):
            x_scaled = 2 * wl_nm[mask] / (l_min_nm + l_max_nm)
            response[mask] = np.polyval(coeffs, x_scaled)
        return response

    # --- Source Definitions ---
    @classmethod
    def white_led_spectrum(cls, wl: np.ndarray) -> np.ndarray:
        """Models a White LED spectrum as a sum of two Gaussians."""
        blue = cls._gaussian(wl, peak=470e-9, fwhm=20e-9)
        phosphor = cls._gaussian(wl, peak=600e-9, fwhm=100e-9)
        return blue + phosphor

    @classmethod
    def tsff5210_spectrum(cls, wl: np.ndarray) -> np.ndarray:
        """Models the TSFF5210 IR Emitter spectrum (Peak ~870nm)."""
        return cls._gaussian(wl, peak=870e-9, fwhm=40e-9)

    @classmethod
    def sun_spectrum(cls, wl: np.ndarray) -> np.ndarray:
        """Approximates Solar Spectrum using SimulationDefaults.T_sun."""
        T = SimulationDefaults.T_sun
        lmax = SimulationDefaults.bK / T
        def blackbody(lam):
            return (2 * SimulationDefaults.hP * SimulationDefaults.c0**2 / lam**5 /
                    (np.exp(SimulationDefaults.hP * SimulationDefaults.c0 /
                            (lam * SimulationDefaults.kB * T)) - 1))
        Pmax = blackbody(lmax)
        return blackbody(wl) / Pmax

    # --- Detector Definitions ---
    @classmethod
    def photodiode_responsivity(cls, wl: np.ndarray) -> np.ndarray:
        """Spectral responsivity of the system photodiode."""
        coeffs = np.array([-6.39503882, 27.47316339, -45.57791267,
                           36.01964536, -12.8418451, 1.73076976])
        return cls._poly_response(wl, coeffs, 330, 1090)

    @classmethod
    def solar_panel_sensitivity(cls, wl: np.ndarray) -> np.ndarray:
        """Spectral sensitivity of the system Solar Panel."""
        coeffs = np.array([26.78555644, -160.24353775, 381.86564712, -463.07816469,
                           300.12488471, -97.25192023, 12.34949208])
        return cls._poly_response(wl, coeffs, 300, 1175)

    # --- Core Physics Logic ---
    @classmethod
    def sun_power(cls):
        """Calculates total integrated power using SimulationDefaults limits."""
        wl = np.linspace(cls.L_MIN, cls.L_MAX, cls.GRID_POINTS)
        xd = cls.sun_spectrum(wl)
        return np.trapz(xd, wl)

    @staticmethod
    def get_filter_transmission(name: str, wl: np.ndarray) -> np.ndarray:
        """Returns transmission window (0.0 or 1.0) for a filter."""
        if name == "VLC_PASS":
            return ((wl >= 320e-9) & (wl <= 720e-9)).astype(float)
        elif name == "IR_PASS":
            return ((wl >= 770e-9) & (wl <= 1100e-9)).astype(float)
        return np.ones_like(wl)

    @classmethod
    def calculate_effective_responsivity(cls, source_func: Callable,
                                         detector_func: Callable,
                                         filter_name: str = "ALL_PASS") -> float:
        """Calculates Source * Detector * Filter overlap integral."""
        wl = np.linspace(cls.L_MIN, cls.L_MAX, cls.GRID_POINTS)
        P_src = source_func(wl)
        R_det = detector_func(wl)
        T_filt = cls.get_filter_transmission(filter_name, wl)
        total_power = np.trapz(P_src, wl)
        if total_power == 0: return 0.0
        detected_power = np.trapz(P_src * R_det * T_filt, wl)
        return float(detected_power / total_power)

    @classmethod
    def get_responsivity_by_name(cls, config_name: str) -> float:
        """Retrieves and calculates responsivity by configuration name."""
        if config_name not in cls.CONFIGURATIONS:
            raise ValueError(f"Unknown config: {config_name}")
        src_n, det_n, filt_n = cls.CONFIGURATIONS[config_name]
        src_f, det_f = getattr(cls, src_n), getattr(cls, det_n)
        return cls.calculate_effective_responsivity(src_f, det_f, filt_n)
