import numpy as np
from const import *
from phy import *
from energy import *




DUMMY_DESIGN = {
    'meta': {
        'name': 'Hybrid_VLC_RF_IoT_Network_Daily',
        'description': 'Daily Energy Budget with RIS and Windows.'
    },

    'environment': {
        'dimensions': np.array([5.0, 5.0, 3.0]),
        'wall_resolution': (20,20),
        'reflectivity': {'floor': 0.3, 'ceiling': 0.8, 'walls': 0.5},

        # (RIS/Windows)
        'special_surfaces': [
            {
                'type': 'RIS', 'name': 'ris_west',
                'center': [5, 2.5, 1.5], 'dims': [1.0, 1.0],
                'const_axis': 0, 'normal': [1,0,0],
                'resolution': (2,2), 'mode': 10.0, 'reflectivity': 0.9
            },

            {
                'type': 'RIS', 'name': 'ris_east',
                'center': [0, 2.5, 1.5], 'dims': [1.0, 1.0],
                'const_axis': 0, 'normal': [1,0,0],
                'resolution': (2,2), 'mode': 10.0, 'reflectivity': 0.9
            },

            {
                'type': 'window', 'name': 'win_north',
                'center': [2.5, 5, 1.5], 'dims': [2.0, 1.0],
                'const_axis': 1, 'normal': [0,-1,0],
                'resolution': (2,2), 'reflectivity': 0.1
            }
        ],


    },

    'nodes': {
        'masters': {
            'positions': np.array([[2.5, 2.5, 3.0], [3,3,3]]),
            'nT':   Constants.zm,
            'nR':   Constants.zm,
            'm':      1,
            'tx_power':  1,
            'rx_area':   1e-4,
            'IR_pass_filter': True,
        },
        'sensors': {
            'positions': np.array([
                [1.0, 1.0, 0.85], [4.0, 4.0, 0.85],
                [2.5, 2.5, 0.85], [2,2,0]
            ]),
            'rx_area':    np.full(4, 1e-4),
            'nT': np.repeat(Constants.zp[None, :], 4, axis=0),
            'nR': np.repeat(Constants.zp[None, :], 4, axis=0),
            # 0=PD, 1=SP
            'rx_type':     np.array([1, 0, 1,0]),
            #for PVs filters are ignored
            'VLC_pass_filter': np.array([False,False,False,True]),
            # 1=RF, 0=IR
            'uplink_type': np.array([1, 1, 0,1]),
            "IR_tx_power": 0.015,
            "RF_tx_power": -20,
            # Energy Storage
            
        },
        'ambient_nodes': {
            'positions': np.array([[3, 3, 3.0]]),
            'nT':   np.array([[0, 0, -1]]),
            'tx_power':    2,
            'm' : 1
        }
    },

    'TIA': {
        'RF': 1e6, 'CF': 1e-9,
        'Vn': 15e-9, 'In': 400e-15,
        'fncV': 1e3, 'fncI': 1e3,
        'temperature': 300
    },

    'PV': {
        'Rs': 1, 'Rsh': 1000, 'Jsc': 35e-3, 'Voc': 0.64, 'n': 1.6

    },

    'MPP': {
        'efficiency' : 0.8
    },

    'energy_profile': {
        'hardware': {
            'f_mcu':   np.array([80e6, 80e6, 16e6, 48e6]),     # Clock Speed (Hz)
            'f_s':     np.array([1e3, 10e3, 1e3, 5e3]),        # Sampling Freq (Hz)
            'voltage': np.array([3.3, 3.3, 3.3, 1.8]),         # Operating Voltage (V)
            'I_mcu':   np.array([4.0, 12.0, 4.0, 8.0])*1e-3,   # Active MCU (mA)
            'I_adc':   2.0*1e-3,                               # ADC Current (mA)
            'I_act':   3.0*1e-3,                               # RX Front-end (mA)
            'I_ext':   np.array([0.5, 5.0, 1.0, 2.0])*1e-3,    # Ext. Sensors (mA)
            'I_sleep': 0.01*1e-3,                              # Sleep Current (mA)
            'I_wake':  5.0*1e-3,                               # Wake-up Current (mA)
        },
        'tasks': {
            'N_samples_up': np.array([100, 5000, 100, 500]), # Samples for UL
            'N_cycles_up':  np.array([1e5, 5e6, 1e5, 8e5]),  # Processing Cycles for UL
            'L_up_bits':    np.array([1024, 8192, 512, 2048]),# Payload size
            'L_down_bits':  64,                               # ACK size
           
        }
    ,
        'communication': {
            'bit_rate_up_ir': 100e3, # bps
            'bit_rate_up_rf': 250e3, # bps
            'bit_rate_down':  15e3,  # bps (VLC)
            't_init': 5e-3,          # Boot time (s)
            #'t_wait': np.array([10e-3, 10e-3, 10e-3, 20e-3]), # ACK wait (s)
        },
        'battery': {
            'battery_capacity_mAh': 500,
            'initial_soc':     1,
            'V_batt': 3.6
        }
    },

    'protocol': {
        'T_cycle': 6,
        #'harvesting_hours': 8
    }
}



ph_net = PhyNet(DUMMY_DESIGN,True)
em = EnergyManager(ph_net,DUMMY_DESIGN)

    
 
