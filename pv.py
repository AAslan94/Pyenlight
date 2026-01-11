import numpy as np
from scipy.special import lambertw
from typing import Dict, Optional, Any
from spatial import to_scal_Nx1
from const import Constants

class PV:
    """
    Photovoltaic Cell Model.
    
    This class implements a comprehensive physical and electrical model of a solar cell,
    spanning DC operating point analysis, AC small-signal characteristics, and 
    frequency-dependent noise modeling.
    """
    def __init__(self, Gsignal, Gamb, unscaled=True, run = True, **kwargs):
        """
        Args:
            Gsignal: Signal Irradiance (W/m^2)
            Gamb: Ambient Irradiance (W/m^2)
            unscaled: If True, scales Rs, Rsh, Jsc by Area.
            **kwargs: Overrides for circuit (Rs, Rsh...) or physics (Na, Nd...) params.
        """
        # 1. Setup Input Vectors
        self.Gamb = np.array(Gamb).reshape(-1, 1)
        self.Gsignal = np.array(Gsignal).reshape(-1, 1)
        self.no_pv = self.Gamb.shape[0]
        self.Gref = 1000.0

        # 2. Initialize Parameters
        # 
        self._init_params(kwargs)

        # 3. Scale Circuit Parameters
        if unscaled:
            self._scale_params()

        # 4. DC Operating Point & Bias
        self._calc_dc_bias()

        # 5. IV Curve Calculation (Analytic Lambert W το avoid loops)
        # Sweep 0 to Voc (Open Circuit)
        # 
        self.V = self.Voc * np.linspace(0, 1, 100).reshape(1, -1) # Shape: (1, 100) -> broadcasts to (N, 100)
        self.I = self.pv_current(self.V)
        self.I[self.I <= 0] = 1e-20  # To avoid negative current

        # 6. DC Performance Metrics
        self.P = self.I * self.V
        self.ind = np.argmax(self.P, axis=1) # Index of MPP
        self.Pmax = np.take_along_axis(self.P, self.ind[:, None], axis=1)
        self.Rl = self.V / self.I

        # 7. Small Signal Parameters (Dynamic Resistance)
        # ID is diode current
        self.ID = self.I0 * np.exp((self.V + self.I * self.Rs) / (self.n * self.Vt))
        self.r = (self.n * self.Vt) / self.ID
        self.iac = self.Iph * self.Gsignal/self.Gamb

        # 8. Frequency & Noise Setup
        self.calc_capacitance() # Uses self.Na, self.Nd from kwargs
        self.find_bw()

        # Generate Frequency Vector based on Bandwidth
        # Creates frequency sweep from 100Hz up to cut-off freq of each panel
        self.bw_ind = np.take_along_axis(self.BW, self.ind[:, None], axis=1) # (N, 1)
        f_steps = 600
        self.f = np.linspace(100, self.bw_ind.flatten(), f_steps).T

        #If default frequency vector is used we can run everything here.
        #For other freq intervals use run = False at object initiation and run later
        if run == True:
          self.tf(self.f)
          self._thermal_noise_base()
          self.compute_all_noise(self.f)
          self.shot_noise(self.f)
          self.tf(self.f)
          self.vp2p(self.f)

    def _init_params(self, kwargs):
        """Unified handler for all parameters with defaults."""
        defaults = {
            # Circuit Model
            'A': 1e-4, 'n': 1.6, 'Rs': 1, 'Rsh': 1e3,
            'Voc': 0.64, 'Jsc': 35e-3,
            'Lo': 1e-6, 'Co': 1e-6, 'Rc': 10,
            # Physical Model (PV manufacturing params)
            'Na': 1e16 * 1e6, 'Nd': 1e19 * 1e6, 'L': 300e-6,
            'er': 11.68, 'ni': 1e10 * 1e6
        }

        for key, default in defaults.items():
            val = kwargs.get(key, default)
            setattr(self, key, to_scal_Nx1(self.no_pv, val))

    def _scale_params(self):
        """Scales area-dependent parameters."""
        scale = self.A * 1e4
        self.Rsh = self.Rsh/ scale
        self.Rs = self.Rs/ scale
        self.Isc = self.Jsc * scale

    def _calc_dc_bias(self):
        """Sets up Iph, Voc, I0 based on Irradiance."""
        if not hasattr(self, 'Isc'): self.Isc = self.Jsc

        self.Iph = self.Isc.copy()
        self.Vt = Constants.kB * Constants.T / Constants.q

        # Scale for Irradiance
        self.Iph *= (self.Gamb+self.Gsignal) / self.Gref
        self.Isc = self.Iph # Approximation for Isc at new G
        self.Rsh *= self.Gref / (self.Gamb + self.Gsignal)
        self.Voc += self.Vt * np.log((self.Gamb + self.Gsignal) / self.Gref + 1e-20) # Avoid log(0)

        # Calculate Saturation Current I0
        num = self.Isc - self.Voc / self.Rsh
        den = np.exp(self.Voc / (self.n * self.Vt)) - 1
        self.I0 = num / den

    def pv_current(self, V):
        """
        Vectorized Analytic Solution using Lambert W function.
        """
        V = np.asarray(V)
        Rs, Rsh = self.Rs, self.Rsh
        Iph, I0 = self.Iph, self.I0
        nVt = self.n * self.Vt

        R_sum = Rs + Rsh

        # 1. Linear Term (Ohmic limit)
        term_linear = (Rsh * (Iph + I0) - V) / R_sum

        # 2. Lambert Argument (Theta)
        common_factor = Rsh / (nVt * R_sum)
        exponent_term = np.exp(common_factor * (V + Rs * (Iph + I0)))
        pre_factor = I0 * Rs * common_factor

        theta = pre_factor * exponent_term

        # 3. Solve
        w_val = lambertw(theta).real

        return term_linear - (nVt / Rs) * w_val

    def calc_capacitance(self):
        """Calculates junction capacitance using physics parameters."""
        es = self.er * Constants.eo
        q = Constants.q

        no = self.ni**2 / self.Na
        vbi = self.Vt * np.log(self.Na * self.Nd / self.ni**2)

        # Depletion Capacitance
        denom = 2 * (self.Na + self.Nd) * (vbi - self.V + 1e-6)
        denom[denom < 0] = 1e-20
        c_dep = self.A * np.sqrt((q * es * self.Na * self.Nd) / denom)

        # Diffusion Capacitance
        c_dif = self.A * q * self.L * no * np.exp(self.V / self.Vt) / self.Vt

        self.C = c_dep + c_dif

    def find_bw(self, verbose=False):
        """Calculates 3dB Bandwidth."""
        r_eq_inv = 1/self.Rsh + 1/self.r + 1/(self.Rs + self.Rc)
        self.req = 1 / r_eq_inv

        self.BW = 1 / (2 * np.pi * self.req * self.C)
        if verbose: print(f"BW: {self.BW}")

    def tf(self, f):
        """Calculates Transfer Function H_pv(f)."""
        # Dimensions: [N, 1, M] for freq, [N, V, 1] for components
        w = 2 * np.pi * f[:, None, :]
        r, C, Rl = self.r[..., None], self.C[..., None], self.Rl[..., None]

        # Impedances
        Zp = 1 / (1/self.Rsh[..., None] + 1/r + 1j*w*C)
        Zdc = 1j*w*self.Lo[..., None] + Rl
        Zac = self.Rc[..., None] + 1/(1j*w*self.Co[..., None])

        # Divider Chain
        Zout = 1 / (1/Zac + 1/Zdc) + self.Rs[..., None]
        h1 = Zp / (Zp + Zout)
        h2 = Zdc / (Zac + Zdc)

        self.hpv = np.abs(h1 * h2 * self.Rc[..., None])

    # --- Noise Methods ---
    def _thermal_noise_base(self):
        """Helper to init thermal noise densities."""
        kT = 4 * Constants.kB * Constants.T
        self.No_r = kT * self.r
        self.No_Rs = kT * self.Rs
        self.No_Rsh = kT * self.Rsh
        self.No_Rl = kT * self.Rl
        self.No_Rc = kT * self.Rc

    def compute_all_noise(self, f):
        """
        Computes all 5 thermal noise sources (Rc, Rs, Rl, Rsh, r)
        and integrates them to get total thermal noise.
        """
        # 1. Initialize Thermal Noise Densities (4kTR)
        self._thermal_noise_base()

        # 2. Setup Vectorized Variables
        # Shape: (N, 1, M) for freq, (N, V, 1) for components
        w = 2 * np.pi * f[:, None, :]
        r = self.r[..., None]
        C = self.C[..., None]
        Rl = self.Rl[..., None]
        Rs = self.Rs[..., None]
        Rsh = self.Rsh[..., None]
        Rc = self.Rc[..., None]

        # 3. Common Impedance Blocks
        Z_Co = 1 / (1j * w * self.Co[..., None])
        Z_Lo = 1j * w * self.Lo[..., None]
        Z_Comm = Rc + Z_Co          # Branch with Rc and Co
        Z_EH = Rl + Z_Lo            # Branch with Rl and Lo (Energy Harvester)

        # --- A. Noise from Rc (Contact Resistance) ---
        J_p = 1/r + 1j*w*C + 1/Rsh
        Z_source = Rs + (1/J_p)
        Z_sp = 1 / (1/Z_source + 1/Z_EH)
        den_rc = Z_Comm + Z_sp
        self.n_rc = np.abs(Rc / den_rc)**2 * self.No_Rc[..., None]

        # --- B. Noise from Rs (Series Resistance) ---
        r2s = 1 / (1/Z_EH + 1/Z_Comm)
        u1_rs = r2s / (Rs + (1/J_p) + r2s)
        u2_rs = Rc / Z_Comm
        self.n_rs = np.abs(u1_rs * u2_rs)**2 * self.No_Rs[..., None]

        # --- C. Noise from Rl (Load Resistance) ---
        r1l = (1/J_p) + Rs
        r2l = Z_Comm
        r3l = 1 / (1/r1l + 1/r2l)
        u1_rl = Rc / Z_Comm
        u2_rl = r3l / (Rl + r3l + Z_Lo)
        self.n_rl = np.abs(u1_rl * u2_rl)**2 * self.No_Rl[..., None]

        # --- D. Noise from Rsh (Shunt Resistance) ---
        # Helper impedances for parallel branches
        h1 = 1 / Z_Comm
        h2 = 1 / Z_EH
        z_load_eq = 1 / (h1 + h2) # Equivalent load impedance seen from Rs

        # Impedance looking back into the diode/capacitor junction
        denom_rsh = 1/(Rs + z_load_eq) + 1/r + 1j*w*C
        z_node_rsh = 1 / denom_rsh

        u1_rsh = z_node_rsh / (Rsh + z_node_rsh)
        u2_rsh = z_load_eq / (z_load_eq + Rs)
        u3_rsh = Rc / Z_Comm

        self.n_rsh = np.abs(u1_rsh * u2_rsh * u3_rsh)**2 * self.No_Rsh[..., None]

        # --- E. Noise from r (Dynamic Resistance) ---
        # Similar topology to Rsh, but 'r' is the source
        denom_r = 1/(Rs + z_load_eq) + 1/Rsh + 1j*w*C
        z_node_r = 1 / denom_r

        u1_r = z_node_r / (r + z_node_r)
        u2_r = z_load_eq / (z_load_eq + Rs)
        u3_r = Rc / Z_Comm

        self.n_r = np.abs(u1_r * u2_r * u3_r)**2 * self.No_r[..., None]

        # 4. Integration (Total Noise Current^2)
        # Integrate PSD over frequency (axis=2)
        self.int_rc = np.trapz(self.n_rc, f[:, None, :], axis=2)
        self.int_rs = np.trapz(self.n_rs, f[:, None, :], axis=2)
        self.int_rl = np.trapz(self.n_rl, f[:, None, :], axis=2)
        self.int_rsh = np.trapz(self.n_rsh, f[:, None, :], axis=2)
        self.int_r = np.trapz(self.n_r, f[:, None, :], axis=2)

        # 5. Total Thermal Noise Sum
        self.th_noise = (self.int_rc + self.int_rs + self.int_rl +
                         self.int_rsh + self.int_r)

    def shot_noise(self, f):
        """Calculates shot noise contribution."""
        # Transfer function magnitude squared
        t = np.abs(self.hpv)**2
        # Integration
        integral = np.trapz(t, f[:, None, :], axis=2)
        self.sh_noise = 2 * Constants.q * self.Iph * integral

    def vp2p(self,f):
      self.vac = self.hpv * self.iac[:,None]
