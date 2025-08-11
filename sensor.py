import pandas as pd
import numpy as np
from parent import SimParent

class Sensor(SimParent):
    def __init__(self, node_id, init_energy=1e12):
        super().__init__()
        if node_id == '1':
            init_energy = 1
        if node_id == '2':
            init_energy = 0.4e12
            
        self.id = node_id # Sensor id
        self.init_energy = init_energy
        self.energy = init_energy
        self.dist_ratio_w_dict = {}
        self.hop_ratio_w_dict = {}
        self.p_ack_ratio_w_dict = {}
        self.sum_dist_hop_p_ack_w_dict = {}
        self.erx_downstrm = {}
        self.erx_upstrm = {}
        self.etx_upstrm = {}
        self.erx_data = {}
        self.etx_data = {}
        self.erx_etx_downstrm_upstrm = {}
        self.erx_etx_no_downstrm = {}
        self.erx_etx_no_downstrm_no_upstrm = {}
        self.energy_rem_try = self.energy
        self.sum_dist_hop_p_ack_w_arr = np.array([])
        self.e_use_rx_data = 0
        self.e_use_tx_data = 0
        self.e_use_rx_upstrm = 0
        self.e_use_tx_upstrm = 0
        self.e_use_rx_downstrm = 0