# EnLight-IoT: Hybrid VLC/RF & Energy Harvesting Simulation Framework

**EnLight-IoT** is a high-fidelity Python simulation library designed for **Hybrid Optical Wireless (VLC/LiFi) and RF IoT networks**. It goes beyond simple channel modeling by integrating a comprehensive physical layer engine that accounts for spectral overlaps, realistic energy harvesting (PV) characteristics, and complex environmental interactions like Reconfigurable Intelligent Surfaces (RIS).

> **Documentation**: This codebase features **extensive internal documentation in Markdown format**. Each class and method is fully documented to guide usage, modification, and understanding of the underlying physics.

## 🚀 Key Features

* **Hybrid Physical Layer**: Simulates both **Optical (VLC/IR)** and **Radio Frequency (RF)** uplinks and downlinks, allowing for realistic handover and link budget analysis.
* **Spectral Physics Engine**: Unlike simulators that use constant gain, EnLight-IoT calculates **Effective Responsivity** ($R_{eff}$) by integrating source spectra (LEDs, Sun), filter transmission curves, and detector responsivity.
* **Advanced Environment Modeling**:
  * **Complex Geometries**: Configurable room dimensions and wall reflectivities.
  * **RIS Support**: Models Reconfigurable Intelligent Surfaces for controlled reflections.
  * **Ambient Interference**: Models noise from artificial lighting and solar background radiation through windows.
* **Detailed Photovoltaic (PV) Model**: Includes a dedicated `PV` class for simulating energy harvesting sensors (SLIPT). It models DC operating points, AC small-signal characteristics, and frequency-dependent noise.
* **Modular Architecture**: Uses a "Builder-Manager" pattern to separate configuration parsing from physical simulation logic.

## 📂 Project Structure

The library is organized into distinct modules handling specific aspects of the simulation:

| Module | Description |
|------|------------|
| `phy.py` | **Simulation Kernel**: The main entry point (`PhyNet`) that orchestrates the build, physics, and analysis phases. |
| `nodemanager.py` | **Node Logic**: Manages Sensors (`SNManager`), Masters (`MNManager`), and Ambient sources (`ANManager`). Handles optical/RF elements and gain calculations. |
| `room.py` | **Environment**: Constructs the physical room, managing walls, windows, and RIS surfaces. |
| `pv.py` | **Energy Harvesting**: A physics-based model for photovoltaic cells, calculating IV curves and noise equivalent circuits. |
| `spectral.py` | **Spectral Engine**: Handles spectral overlap integrals for LEDs, lasers, and solar radiation against specific photodiode/filter responses. |
| `builder.py` | **Configuration**: Factory classes (`NodeBuilder`, `RoomBuilder`) that parse design dictionaries into simulation objects. |
| `gains.py` | **Channel Models**: Calculates Line-of-Sight (LoS), Diffuse (multi-bounce), and RIS-assisted channel gains. |

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/EnLight-IoT.git
cd EnLight-IoT
pip install numpy scipy matplotlib
```

💻 Usage

The simulation is driven by a configuration dictionary passed to the PhyNet kernel.

## 🧩 Comprehensive Design Example

The following example demonstrates a **fully populated design dictionary** intended for a **reference template**.  
It exercises **all major subsystems** of EnLight-IoT, including environment modeling, RIS, hybrid VLC/RF nodes, PV energy harvesting, analog front-end overrides, and protocol-level energy profiling.

```python
import numpy as np
from defaults import SimulationDefaults

aster_design = {
    'meta': {
        'name': 'Comprehensive_Simulation_Template',
        'description': 'A design populating every available parameter for stress-testing and template use.'
    },

    'environment': {
        'dimensions': np.array([6.0, 6.0, 3.5]),  # [Length, Width, Height] in meters
        'wall_resolution': (30, 30),
        'reflectivity': {
            'floor': 0.2,
            'ceiling': 0.7,
            'walls': 0.5
        },
        'special_surfaces': [
            {
                'type': 'RIS',
                'name': 'ris_west_wall',
                'center': [6.0, 3.0, 1.75],
                'dims': [2.0, 2.0],
                'normal': SimulationDefaults.xm,  # Facing into room (-X)
                'const_axis': 0,
                'resolution': (10, 10),
                'reflectivity': 0.95
            },
            {
                'type': 'window',
                'name': 'south_window',
                'center': [3.0, 0.0, 1.5],
                'dims': [4.0, 1.5],
                'normal': SimulationDefaults.yp,  # Facing into room (+Y)
                'const_axis': 1,
                'resolution': (15, 15),
                'reflectivity': 0.1
            }
        ]
    },

    'nodes': {
        'masters': {
            'positions': np.array([[3.0, 3.0, 3.5], [1.0, 1.0, 3.5]]),
            'nT': SimulationDefaults.zm,
            'nR': SimulationDefaults.zm,
            'm': 1,
            'tx_power': 2.0,          # VLC Tx power (W)
            'rx_area': 1e-4,          # PD area (m²)
            'FOV': 65,
            'sensitivity': -110,      # dBm
            'IR_pass_filter': True
        },

        'sensors': {
            'positions': np.array([
                [2.0, 2.0, 0.85],
                [4.0, 4.0, 0.0],
                [1.0, 5.0, 0.85]
            ]),
            'rx_area': np.array([1e-4, 20e-4, 1e-4]),
            'nT': SimulationDefaults.zp,
            'nR': SimulationDefaults.zp,
            'rx_type': np.array([0, 1, 0]),       # [PD, PV, PD]
            'uplink_type': np.array([0, 0, 1]),   # [IR, IR, RF]
            'VLC_pass_filter': np.array([True, False, True]),
            'IR_tx_power': 0.05,                  # W
            'RF_tx_power': -15,                   # dBm
            'm': 1,
            'FOV': 70
        }
    },

    'TIA': {
        'RF': 1.2e6,
        'Vn': 10e-9,
        'In': 300e-15,
        'fncV': 1e3,
        'fncI': 1e3
    },

    'PV_circuit': {
        'Rs': 0.5,
        'Rsh': 2000,
        'Voc': 0.7,
        'Jsc': 40e-3,
        'n': 1.5
    },

    'energy_profile': {
        'hardware': {
            'f_mcu': 48e6,
            'f_s': 10e3,
            'voltage': 3.3,
            'I_mcu': 5e-3,
            'I_adc': 1e-3,
            'I_act': 2e-3,
            'I_sleep': 2e-6,
            'I_wake': 8e-3,
            'I_ext': 1e-3
        },
        'tasks': {
            'N_s_up': 1000,
            'N_c_up': 50000,
            'L_up_bits': 2048,
            'L_dw_bits': 128
        },
        'communication': {
            'Rb_up': 20e3,
            'Rb_down': 50e3,
            'n_sp_u': 0.45,
            'n_sp_d': 0.45,
            't_init': 10e-3,
            't_wait': 20e-3
        },
        'battery': {
            'battery_capacity_mAh': 1200,
            'V_batt': 3.7,
            'initial_soc': 0.95
        }
    },

    'protocol': {
        'T_cycle': 30.0,
        'harvesting_hours': 12.0
    }
}


🇪🇺 Funding & Acknowledgements

This work is supported by the EU OWIN6G DN (Optical and Wireless Internet for 6G Doctoral Network).

📝 License

This project is currently available for review.
Full open-source licensing details will be provided immediately following the formal publication of the associated research paper.

The codebase includes extensive documentation in Markdown to facilitate understanding and usage of the physics engines.
