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
| :--- | :--- |
| `phy.py` | **Simulation Kernel**: The main entry point (`PhyNet`) that orchestrates the build, physics, and analysis phases. |
| `nodemanager.py` | **Node Logic**: Manages Sensors (`SNManager`), Masters (`MNManager`), and Ambient sources. Handles optical/RF elements and gain calculations. |
| `room.py` | **Environment**: Constructs the physical room, managing walls, windows, and RIS surfaces. |
| `pv.py` | **Energy Harvesting**: A physics-based model for photovoltaic cells, calculating IV curves and noise equivalent circuits. |
| `spectral.py` | **Spectral Engine**: Handles spectral overlap integrals for LEDs, lasers, and solar radiation against specific photodiode/filter responses. |
| `builder.py` | **Configuration**: Factory classes (`NodeBuilder`, `RoomBuilder`) that parse design dictionaries into simulation objects. |
| `gains.py` | **Channel Models**: Calculates Line-of-Sight (LoS), Diffuse (multi-bounce), and RIS-assisted channel gains. |

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/EnLight-IoT.git](https://github.com/yourusername/EnLight-IoT.git)
cd EnLight-IoT
pip install numpy scipy matplotlib

💻 Usage
The simulation is driven by a configuration dictionary passed to the PhyNet kernel.

🇪🇺 Funding & Acknowledgements
This work is supported by the EU OWIN6G DN (Optical and Wireless Internet for 6G Doctoral Network).

<img src="https://www.google.com/search?q=https://owin6g.eu/wp-content/uploads/2023/01/logo-owin6g.png" alt="OWIN6G Logo" width="200"/>

📝 License
This project is currently available for review. Full open-source licensing details will be provided immediately following the formal publication of the associated research paper.

The codebase includes extensive documentation in Markdown to facilitate understanding and usage of the physics engines.
