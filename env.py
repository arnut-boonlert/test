
import numpy as np
import pandas as pd
from controller import Controller
from sensor import Sensor
from parent import SimParent
from itertools import combinations
import pickle
import os
import logging
import random
import networkx as nx # type: ignore
import matplotlib.pyplot as plt
import time
import math
import copy

class SdWsnEnv(SimParent):
    # ===== Operates ===== (H)
    def __init__(self, decimal, state_method, reward_method, weights, hop_reward_cal, rt_num, cache_mode, data_size, upstrm_ctl_size, downstrm_ctl_size, rt_period, num_sensors, topo_range_x, topo_range_y, get_rt_method, create_topo, sim_no):
        super().__init__()
        self.data_size = data_size
        self.upstrm_ctl_size = upstrm_ctl_size
        self.downstrm_ctl_size = downstrm_ctl_size
        self.rt_period = rt_period
        self.decimal = decimal
        self.state_method = state_method
        self.reward_method = reward_method
        self.all_weights = weights
        self.hop_reward_cal = hop_reward_cal
        self.expect_rt_num = rt_num
        self.cache_mode = cache_mode
        self.name_topo = 'main_topo'

        self.num_sensors = num_sensors
        self.num_sensors_ctrl = self.num_sensors+1
        self.cache_fitness = set()

        self.controller_pos = (-20, 50)
        self.topo_range_x = topo_range_x
        self.topo_range_y = topo_range_y
        self.neighbor_range = 87 # 87 meters
        self.bttn_num = (self.num_sensors * 20)//100 # bottlenecks number = 30% of nodes
        self.bttn_num = 3
        self.get_rt_method = get_rt_method
        self.create_topo = create_topo
        self.unavailable_rt = set()
        self.unavailable_rt_dict = {}
        self.unavailable_rt_save = set()

        self.energy_cache = {}
        self.time_old_get_rt = np.nan
        self.time_old_cal_info = np.nan
        self.time_GA_get_cal_rt = np.nan
        self.time_random_topo = np.nan
        self.time_save_topo = np.nan
        self.time_get_all_rt = np.nan
        self.try_step_time = []

        self.sim_no = sim_no

        self.same_rt = False
        self.cur_rt = None

    def timeit(func):  # Decorator function
        def wrapper(*args, **kwargs):  # Wraps the original function
            start = time.perf_counter()
            result = func(*args, **kwargs)  # Calls the original function
            print(f"{func.__name__} took {time.perf_counter() - start:.6f} seconds")
            return result  # Returns original function's output
        return wrapper  # Returns the wrapped function
    
    @staticmethod
    def timeit_try_step(func):
        def wrapper(self, *args, **kwargs):  # Ensure 'self' is passed for methods
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            self.try_step_time.append(elapsed)  # ✅ Now self.timeit is accessible
            # print(f"{func.__name__} took {elapsed:.6f}  seconds")
            return result
        return wrapper

    def reset(self):
        if self.get_rt_method == 'GA':
            state, d_ratios, h_ratios, rts_list = self.GA_rt_method()
        elif self.get_rt_method == 'old':
            state, d_ratios, h_ratios, rts_list = self.old_rt_method()
        print(f'RTs shape: {self.RTs.shape}')
        print(f'Bottlenecks: {self.controller.bottlenecks}')

        start_time_save_topo = time.time()
        # if self.get_rt_method == 'GA':
        # self.save_rts_topo()
        # self.save_topo_connectivity()
        self.time_save_topo = time.time()-start_time_save_topo

        self.init_rts = self.controller.RTs

        self.controller.cal_sensor_rt() # Prepare calculate remaining and to be remove RT when any bottlenecks node die, and store them to bottleneck sensors.

        return state, d_ratios, h_ratios, rts_list

    def GA_rt_method(self):
        # Reset Environment
        self.done = False
        self.die_node = []
        # self.round_dies = []
        self.controller = Controller('C') # Create controller object and assign attribute controller
        self.controller.data_size = self.data_size
        self.controller.upstrm_ctl_size = self.upstrm_ctl_size
        self.controller.downstrm_ctl_size = self.downstrm_ctl_size
        self.controller.rt_period = self.rt_period

        start_GA = time.time()
        self.genetic_algo()
        self.time_GA_get_cal_rt = time.time()-start_GA

        # Reset controller
        self.controller.RTs = self.RTs
        self.controller.init_num_nodes = self.init_num_nodes
        self.controller.hop_reward_cal = self.hop_reward_cal 
        self.controller.expect_rt_num = self.expect_rt_num
        self.controller.cache_mode = self.cache_mode
        # self.controller.total_rt = self.total_rt

        weight_splited = self.all_weights.split(', ') # Split each weight parameter by ','.
        self.controller.dist_w = float(weight_splited[0]) # Weight of distance to the controller.
        self.controller.hop_w = float(weight_splited[1]) # Weight of hop count to the controller parameter.
        self.controller.energy_rem_w = float(weight_splited[2]) # Weight of energy remain parameter.
        self.controller.p_ack_w = 0 # Weight of packet acknowledgement parameter (P_ack) !ASSUMPTIOIN: we consider their are no loss in the network (perfect communication).

        self.controller.hop_max = len(self.controller.sensors) # The worst path (traval to every node before controller).

        # Calculate distance, hop, p_ack of each RT 
        self.controller.sensors_info_df[['NodeID', 'Paths', 'RT-No.']].apply(lambda x: self.controller.cal_dist_hop_p_ack_ratio(x['NodeID'], x['Paths'], x['RT-No.']), axis=1)
        # Calculate the best effort route (every node send data to the controller directly).
        self.controller.cal_dist_hop_p_ack_ratio_best_effort()
        # Calculate the best effort route's energy use.
        self.controller.cal_erx_etx_best_effort()

        # Convert the column names from strings to integers
        self.RTs.columns = self.RTs.columns.astype(int)
        # Sort the columns in ascending order
        self.RTs = self.RTs.loc[:, sorted(self.RTs.columns)]
        self.controller.RTs = self.RTs
        
        rts_list = self.controller.RTs.columns
        self.rts_sorted = rts_list
        state = self.get_state(self.RTs.columns[0]) # Get the first state of the simulation.
        
        return state, {sensor.id: {k: v for k, v in sensor.dist_ratio_w_dict.items() if k in rts_list} for sensor in self.controller.sensors.values()}, {sensor.id: {k: v for k, v in sensor.hop_ratio_w_dict.items() if k in rts_list} for sensor in self.controller.sensors.values()}, rts_list

    def old_rt_method(self):
        # Reset Environment
        self.done = False
        self.die_node = []
        # self.round_dies = []
        self.build_env_rts()
        # Reset controller
        self.controller.data_size = self.data_size
        self.controller.upstrm_ctl_size = self.upstrm_ctl_size
        self.controller.downstrm_ctl_size = self.downstrm_ctl_size
        self.controller.rt_period = self.rt_period
        self.controller.RTs = self.RTs
        self.controller.init_num_nodes = self.init_num_nodes
        self.controller.hop_reward_cal = self.hop_reward_cal 
        self.controller.expect_rt_num = self.expect_rt_num
        self.controller.cache_mode = self.cache_mode
        # self.controller.total_rt = self.total_rt
        start_setup = time.time()
        # Setup simulation informations.
        if self.cache_mode == 'n': # If we not use cache mode.
            start_cal_rt = time.time()

            self.controller.setup_sim_info(self.all_weights) # Setup informations.

            self.time_old_cal_info = time.time()-start_cal_rt
    
        else: # If we use cache mode.
            self.cache_sensors = f"{self.cache_dir}/{self.hop_reward_cal}/w({self.all_weights}), d_s({self.data_size}), ctl_s({self.upstrm_ctl_size}), upstm_s({self.upstrm_ctl_size}), dstm_ctl_s({self.downstrm_ctl_size}, rt_num({self.total_rt}), sensor_num({self.init_num_nodes})"
            # Create cache folder.
            if not os.path.exists(f"{self.cache_dir}/{self.hop_reward_cal}"):
                os.makedirs(f"{self.cache_dir}/{self.hop_reward_cal}")
            # If cache files exest, get informations.
            if os.path.exists(self.cache_sensors):
                with open(self.cache_sensors, "rb") as file:
                    self.controller.sensors, self.controller.sensors_info_df, self.controller.energy_rem_w, self.controller.RTs = pickle.load(file)

            # If cache file does not exist, calculate informatins and dump into cache file.
            else:
                self.controller.setup_sim_info(self.all_weights) # Call ... to calculate informations.
                with open(self.cache_sensors, "wb") as file:
                    pickle.dump((self.controller.sensors, self.controller.sensors_info_df, self.controller.energy_rem_w, self.controller.RTs), file)
        print(f'setup_sim_info: {time.time()-start_setup}')
        ############# change method while cal routes
        # Filter n route that lower energy use.
        # rts_list = self.controller.find_e_use_bottleneck(self.RTs, len(self.RTs.columns))
        rts_list = self.controller.find_e_use_bottleneck(self.RTs, max(self.RTs.shape[1], 500))
        self.rts_sorted = rts_list
        # print(self.rts_sorted)
        self.e_sorted_data = self.controller.e_sorted_data 
        ##############
        self.RTs = self.controller.RTs
        state = self.get_state(self.RTs.columns[0]) # Get the first state of the simulation.
        return state, {sensor.id: {k: v for k, v in sensor.dist_ratio_w_dict.items() if k in rts_list} for sensor in self.controller.sensors.values()}, {sensor.id: {k: v for k, v in sensor.hop_ratio_w_dict.items() if k in rts_list} for sensor in self.controller.sensors.values()}, rts_list

    def load_cache_topo(self):
        ####################################################################################################################
        ##################################### CACHE TEST TOPO with genetic #################################################
        ####################################################################################################################
        cache_f = f"{self.cache_dir}/topo"
        cache_n = f"{cache_f}/topo.pkl"

        # Create cache folder.
        if not os.path.exists(cache_f):
            os.makedirs(cache_f)

        # If cache file exists, load it.
        if os.path.exists(cache_n):
            with open(cache_n, "rb") as file:
                sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks, self.sensor_positions = pickle.load(file)
        else:
            sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks = self.random_topo()
            with open(cache_n, "wb") as file:
                pickle.dump((sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks, self.sensor_positions), file)

        return sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks, self.sensor_positions
        ####################################################################################################################
        ####################################################################################################################

    def save_rts_topo(self):
        """
        Save the routing topologies as images. This method uses the sensor positions and the controller position
        generated from random_topo() (stored in self.sensor_positions and self.controller_pos, respectively).
        It draws the network topology for each RT in self.RTs.
        """
        import os
        import networkx as nx
        import matplotlib.pyplot as plt

        # Create output directory using self.dtime (assumed to be defined)
        topo_dir = f"topo/ns{self.num_sensors}_{self.dtime}/sim{self.sim_no}_getRT-{self.get_rt_method}_show_RTs"

        if not os.path.exists(topo_dir):
            os.makedirs(topo_dir)

        # Build a position dictionary for drawing: include the controller and all sensors.
        pos = {"C": self.controller_pos}
        # self.sensor_positions is expected to be a dict mapping sensor ID (as string) to (x, y) coordinates.
        for sensor_id, sensor_pos in self.sensor_positions.items():
            pos[sensor_id] = sensor_pos

        # Iterate over each routing topology column in self.RTs
        for col in self.RTs.columns:
            # Create a directed graph for this routing topology
            G_rt = nx.DiGraph()
            # Nodes: Controller "C" and all sensor IDs (as in pos)
            nodes = ["C"] + list(self.sensor_positions.keys())
            G_rt.add_nodes_from(nodes)

            # For each sensor (row) in self.RTs, get its forwarder for this RT and add the edge if both nodes are valid.
            for row_idx in self.RTs.index:
                forwarder = self.RTs.at[row_idx, col]
                if (row_idx in G_rt.nodes) and (forwarder in G_rt.nodes):
                    G_rt.add_edge(row_idx, forwarder)
            # Draw the network topology.
            plt.figure(figsize=(8, 5))
            nx.draw_networkx_nodes(G_rt, pos,
                                node_size=300,
                                node_color="white",
                                edgecolors="gray")
            nx.draw_networkx_labels(G_rt, pos,
                                    font_size=10,
                                    font_color="black")
            nx.draw_networkx_edges(G_rt, pos,
                                arrowstyle="->",
                                arrowsize=15,
                                edge_color="gray")
            plt.title(f"Network Topology - RT: {col}", fontsize=16)
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            all_x = [pos[0] for pos in self.sensor_positions.values()] + [self.controller_pos[0]]
            all_y = [pos[1] for pos in self.sensor_positions.values()] + [self.controller_pos[1]]
            
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            x_margin = (max_x - min_x) * 0.1
            y_margin = (max_y - min_y) * 0.1
            
            plt.xlim(min_x - x_margin, max_x + x_margin)
            plt.ylim(min_y - y_margin, max_y + y_margin)
            
            # Set custom ticks for the grid using intervals of 10
            # Compute lower and upper bounds for ticks as multiples of 10
            lower_x = int(np.floor((min_x - x_margin) / 10) * 10)
            upper_x = int(np.ceil((max_x + x_margin) / 10) * 10)
            lower_y = int(np.floor((min_y - y_margin) / 10) * 10)
            upper_y = int(np.ceil((max_y + y_margin) / 10) * 10)
            
            x_ticks = np.arange(lower_x, upper_x + 1, 20)
            y_ticks = np.arange(lower_y, upper_y + 1, 20)
            ax = plt.gca()
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # Annotate vertical grid lines with their x coordinate as an integer
            for x in x_ticks:
                plt.text(x, min_y - y_margin*0.4, f'{x}', fontsize=8, color='blue', ha='center', va='center')
            
            # Annotate horizontal grid lines with their y coordinate as an integer
            for y in y_ticks:
                plt.text(lower_x - x_margin*0.4, y, f'{y}', fontsize=8, color='blue', ha='center', va='center')
            # Enable grid to show coordinate lines
            plt.grid(True)
            # Insert annotation block after limits and ticks, before tight_layout
            # Annotate each node with its (x,y) coordinates after limits set
            ax = plt.gca()
            for node, (x_coord, y_coord) in pos.items():
                ax.text(
                    x_coord, y_coord,
                    f"{node} ({x_coord},{y_coord})",
                    fontsize=6,
                    alpha=0.7,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )
            plt.tight_layout()
            # plt.show()


            # Save the plot to file.
            output_path = f"{topo_dir}/RT_{col}.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            # plt.close()
            plt.clf()
            plt.cla()
            plt.close()


            # Optionally log the saved topology path.
            # print(f"Saved topology for RT: {col} to {output_path}")

    def save_topo_connectivity(self):
        # Visualize the generated topology

        topo_dir = f"topo/ns{self.num_sensors}_{self.dtime}/sim{self.sim_no}_getRT-{self.get_rt_method}_show_topo"

        if not os.path.exists(topo_dir):
            os.makedirs(topo_dir)

        G_vis = nx.Graph()
        # Add controller node
        G_vis.add_node('C')
        pos = {'C': self.controller_pos}

        # Add sensor nodes with positions
        for sensor_id, coord in self.sensor_positions.items():
            G_vis.add_node(sensor_id)
            pos[sensor_id] = coord

        # Add edges between sensors if within neighbor_range
        for id1, coord1 in self.sensor_positions.items():
            for id2, coord2 in self.sensor_positions.items():
                if int(id2) <= int(id1):
                    continue
                if math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1]) <= self.neighbor_range:
                    G_vis.add_edge(id1, id2)

        # Add edges from controller to its bottleneck nodes
        for bn in self.controller.bottlenecks:
            G_vis.add_edge('C', str(bn))

        # Draw the topology with zone boundaries
        from matplotlib import patches

        fig, ax = plt.subplots(figsize=(16, 9))
        # Draw zones (50×50)
        zone_size = 50
        num_zones_x = self.topo_range_x // zone_size
        num_zones_y = self.topo_range_y // zone_size
        for ix in range(num_zones_x):
            for iy in range(num_zones_y):
                rect = patches.Rectangle((ix*zone_size, iy*zone_size), zone_size, zone_size,
                                         linewidth=0.5, edgecolor='gray', facecolor='none', linestyle='--')
                ax.add_patch(rect)

        # Draw network graph
        nx.draw(G_vis, pos, ax=ax, with_labels=True, node_size=200, font_size=8)
        # Annotate each node with its (x,y) coordinates below the node
        for node, (x_coord, y_coord) in pos.items():
            ax.text(
                x_coord, y_coord - zone_size * 0.02,
                f"({x_coord},{y_coord})",
                fontsize=6,
                horizontalalignment='center',
                verticalalignment='top'
            )
        # Adjust limits to include all nodes with margin
        xs = [coord[0] for coord in pos.values()]
        ys = [coord[1] for coord in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        x_margin = (max_x - min_x) * 0.1
        y_margin = (max_y - min_y) * 0.1
        ax.set_xlim(min_x - x_margin, max_x + x_margin)
        ax.set_ylim(min_y - y_margin, max_y + y_margin)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.title("Topology: Controller (C) and Sensor Neighbors (50×50 Zones)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(False)
        # plt.show()

        # Save the plot to file.
        output_path = f"{topo_dir}/topo.png"
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        # plt.close()
        plt.clf()
        plt.cla()
        plt.close()

    def compute_nodes_per_zone(self, total_nodes: int, num_zones: int) -> list[int]:
        """
        Return a list `nodes_per_zone[0..num_zones-1]` such that:
        1) sum(nodes_per_zone) == total_nodes
        2) each zone gets at least 1 node initially
        3) any “extra” (leftover) nodes are assigned one at a time
            to zones in a user-defined order (e.g. 0, 1, 2, … or 0, 2, 4, …).
        """

        # 1) Give each zone a base of 1 sensor:
        nodes_per_zone = [1] * num_zones

        # 2) How many “extra” sensors remain?
        leftover = total_nodes - num_zones
        if leftover < 0:
            raise ValueError(f"total_nodes={total_nodes} is smaller than num_zones={num_zones}")

        # 3) Distribute leftovers.  You can choose your own “pattern.”
        #    Below is one plausible rule: “First give to zone 0, then zone 2, then zone 4, …,
        #    then (once you run out of even indices) go to zone 1, zone 3, zone 5, ….”
        #
        idx = 0
        while leftover > 0:
            nodes_per_zone[idx] += 1
            leftover -= 1

            # Move to the next zone index according to our custom pattern:
            if idx + 2 < num_zones:
                idx += 2
            else:
                # Once even indices are exhausted, jump to the first odd index
                # and continue stepping by 2 (1 → 3 → 5 → …).
                idx = 1

        return nodes_per_zone

    def random_topo(self):
        """
        Generate random sensor positions, re-label sensor IDs by their proximity to the controller,
        build neighbor information, and ensure:
        1. All sensors have a path to the controller ("C").
        2. Each sensor (except Sensor 1 or those directly connected to "C")
            has at least one neighbor with a lower sensor ID than itself (forward routing).

        Each sensor's dictionary (in original_sensors_nbrs_dist_info) contains:
        'Neighbors': list of neighbors (computed from the communication range)
        'Distances': corresponding distances
        'Controller': the sensor's distance to the controller

        The function then computes the list of sensors with a real direct connection to "C"
        based solely on the original logic.

        For display purposes only, a separate copy is made (display_sensors_nbrs_dist_info)
        in which "C" is force‑added to every sensor’s neighbor list (if not already present).

        Returns:
        - display_sensors_nbrs_dist_info: sorted dict for display with forced "C" in neighbor list.
        - controller_nbrs_dist_info: dict including ALL sensors (with their distances from the controller).
        - original_directly_connected: list of sensor IDs (as strings) that originally have a direct connection to "C".
        """
        topo_range_x = self.topo_range_x
        topo_range_y = self.topo_range_y
        neighbor_range = self.neighbor_range

        # Generate grid points with fixed 10x10 m spacing
        x_coords = np.arange(0, topo_range_x + 10, 10)
        y_coords = np.arange(0, topo_range_y + 10, 10)
        # Create list of grid intersection points
        grid_points = [(x, y) for x in x_coords for y in y_coords]

        # Precompute global candidate sets
        bttn_count = int(self.bttn_num)
        # Global neighbor-range and outside-range candidates
        bottleneck_candidates = [
            pt for pt in grid_points
            if math.hypot(pt[0] - self.controller_pos[0], pt[1] - self.controller_pos[1]) <= neighbor_range
        ]
        outside_points_global = [
            pt for pt in grid_points
            if math.hypot(pt[0] - self.controller_pos[0], pt[1] - self.controller_pos[1]) > neighbor_range
        ]

        # Sample global bottleneck nodes
        bttn_count = int(self.bttn_num)
        if len(bottleneck_candidates) < bttn_count:
            bttn_count = len(bottleneck_candidates)
        bottleneck_positions = random.sample(bottleneck_candidates, bttn_count)

        # # Define 50×50 zones across the topology
        # zone_size = 50
        # num_zones_x = topo_range_x // zone_size
        # num_zones_y = topo_range_y // zone_size
        # num_zones = int(num_zones_x * num_zones_y)

        # # Compute total nodes per zone: ensure at least one node in each zone
        # total_nodes = self.num_sensors
        # min_per_zone = 1
        # remaining = total_nodes - min_per_zone * num_zones
        # base = remaining // num_zones
        # rem = remaining % num_zones
        # nodes_per_zone = [min_per_zone + base + (1 if i < rem else 0) for i in range(num_zones)]


        # 1) decide how many zones in X/Y:
        zone_size   = 50
        num_zones_x = topo_range_x // zone_size
        num_zones_y = topo_range_y // zone_size
        num_zones   = int(num_zones_x * num_zones_y)

        # 2) call our helper to build “nodes_per_zone”:
        total_nodes    = self.num_sensors
        nodes_per_zone = self.compute_nodes_per_zone(total_nodes, num_zones)
        #    e.g. if total_nodes=10 and num_zones=8, you’ll get [2,1,2,1,1,1,1,1].

        # Keep bottleneck_positions intact; sample non-bottleneck per zone
        other_positions = []
        for zone_idx in range(num_zones):
            zx = zone_idx // num_zones_y
            zy = zone_idx % num_zones_y
            x_min, x_max = zx * zone_size, (zx + 1) * zone_size
            y_min, y_max = zy * zone_size, (zy + 1) * zone_size

            # Count how many bottlenecks fall in this zone
            b_zone = [
                pt for pt in bottleneck_positions
                if x_min <= pt[0] < x_max and y_min <= pt[1] < y_max
            ]
            # Determine how many non-bottleneck sensors to sample here
            non_needed = nodes_per_zone[zone_idx] - len(b_zone)
            if non_needed < 0:
                non_needed = 0

            # Eligible outside-range points in this zone
            outside_zone = [
                pt for pt in outside_points_global
                if x_min <= pt[0] < x_max and y_min <= pt[1] < y_max
            ]
            # If fewer than needed, take all available
            if len(outside_zone) < non_needed:
                non_needed = len(outside_zone)
            other_positions.extend(random.sample(outside_zone, non_needed))

        # Combine bottleneck and non-bottleneck lists, trimming/padding non-bottlenecks only
        total = self.num_sensors
        needed_non = total - len(bottleneck_positions)

        # Trim down if too many non-bottlenecks
        if len(other_positions) > needed_non:
            other_positions = other_positions[:needed_non]
        # Pad if too few
        elif len(other_positions) < needed_non:
            pool = list(set(outside_points_global) - set(other_positions) - set(bottleneck_positions))
            extra = random.sample(pool, needed_non - len(other_positions))
            other_positions.extend(extra)

        # Final sensor list keeps bottlenecks untouched
        sensors = bottleneck_positions + other_positions

        # Debug: print sensor positions with zone indices
        # print("Sensor positions and their zones:")
        for idx, pos in enumerate(sensors, start=1):
            zx = int(pos[0] // zone_size)
            zy = int(pos[1] // zone_size)
            zone_idx = zx * num_zones_y + zy
            # print(f"Sensor {idx}: Position {pos}, Zone {zone_idx}")

        # Print zone-to-node mapping
        zone_centers_x = [
            ( (z // num_zones_y) * zone_size + zone_size / 2 )
            for z in range(num_zones)
        ]
        sorted_zones = sorted(
            range(num_zones),
            key=lambda z: abs(zone_centers_x[z] - self.controller_pos[0])
        )
        zone_map = {}
        for idx, coord in enumerate(sensors, start=1):
            zx = int(coord[0] // zone_size)
            zy = int(coord[1] // zone_size)
            zone_idx = zx * num_zones_y + zy
            zone_map.setdefault(zone_idx, []).append(idx)
        # print("Zone assignments (zone: [sensor IDs]):")
        # for z in sorted(zone_map):
        #     print(f"Zone {z}: {zone_map[z]}")

        # Determine dynamic enforcement limits
        total_nodes = self.num_sensors
        max_initial_zones = min(4, num_zones, total_nodes)
        # Ensure the first few zones have at least one sensor
        for zone_idx in range(max_initial_zones):
            if zone_idx not in zone_map or not zone_map[zone_idx]:
                zx = zone_idx // num_zones_y
                zy = zone_idx % num_zones_y
                x_min, x_max = zx * zone_size, (zx + 1) * zone_size
                y_min, y_max = zy * zone_size, (zy + 1) * zone_size
                zone_points = [
                    pt for pt in grid_points
                    if x_min <= pt[0] < x_max and y_min <= pt[1] < y_max
                ]
                used = set(sensors)
                available = [pt for pt in zone_points if pt not in used]
                if available:
                    new_pt = random.choice(available)
                    other_positions.insert(0, new_pt)
                    zone_map.setdefault(zone_idx, []).append(None)  # mark filled

        # Then ensure other zones up to total_nodes have at least one sensor
        enforce_limit = num_zones if total_nodes >= num_zones else total_nodes
        for zone_idx in sorted_zones:
            if zone_idx < max_initial_zones or zone_idx >= enforce_limit:
                continue
            if zone_idx not in zone_map or not zone_map[zone_idx]:
                zx = zone_idx // num_zones_y
                zy = zone_idx % num_zones_y
                x_min, x_max = zx * zone_size, (zx + 1) * zone_size
                y_min, y_max = zy * zone_size, (zy + 1) * zone_size
                zone_points = [
                    pt for pt in grid_points
                    if x_min <= pt[0] < x_max and y_min <= pt[1] < y_max
                ]
                used = set(sensors)
                available = [pt for pt in zone_points if pt not in used]
                if available:
                    new_pt = random.choice(available)
                    other_positions.insert(0, new_pt)
                    zone_map.setdefault(zone_idx, []).append(None)  # mark filled

        # Rebuild sensors list and enforce exact total count
        sensors = bottleneck_positions + other_positions
        # Trim or pad to total_nodes
        if len(sensors) > total_nodes:
            sensors = sensors[:total_nodes]
        elif len(sensors) < total_nodes:
            pool = [pt for pt in outside_points_global if pt not in sensors]
            sensors.extend(random.sample(pool, total_nodes - len(sensors)))
        # Sync other_positions to match non-bottleneck sensors
        other_positions = [pt for pt in sensors if pt not in bottleneck_positions]

        # Debug: print final zone-to-node mapping after ensure and trimming
        zone_map_final = {}
        for idx, coord in enumerate(sensors, start=1):
            zx = int(coord[0] // zone_size)
            zy = int(coord[1] // zone_size)
            zone_idx = zx * num_zones_y + zy
            zone_map_final.setdefault(zone_idx, []).append(idx)
        # print("Final zone assignments (zone: [sensor IDs]):")
        # for z in range(num_zones):
        #     print(f"Zone {z}: {zone_map_final.get(z, [])}")

        # Prepare sets for repositioning
        bottleneck_candidates_set = set(bottleneck_candidates)
        outside_points_set = set(outside_points_global)

        all_ok = False
        while not all_ok:
            # Calculate each sensor's distance from the controller.
            sensor_distances = []
            for i, pos in enumerate(sensors):
                dist = math.hypot(pos[0] - self.controller_pos[0], pos[1] - self.controller_pos[1])
                sensor_distances.append((i, pos, dist))

            # Sort sensors by distance (nearest first).
            sensor_distances_sorted = sorted(sensor_distances, key=lambda x: x[2])

            # Create a mapping: original sensor index -> new sensor ID (nearest gets ID 1).
            id_mapping = {orig_id: new_id for new_id, (orig_id, pos, dist) in enumerate(sensor_distances_sorted, start=1)}

            # Build the original sensor neighbors info.
            original_sensors_nbrs_dist_info = {}
            for i, pos in enumerate(sensors):
                sensor_new_id = str(id_mapping[i])
                sensor_neighbors = []
                sensor_distances_list = []

                # Add direct connection to "C" if within neighbor_range.
                controller_distance = math.hypot(pos[0] - self.controller_pos[0], pos[1] - self.controller_pos[1])
                if controller_distance <= neighbor_range:
                    sensor_neighbors.append("C")
                    sensor_distances_list.append(round(controller_distance, 2))

                # Check distances to all other sensors.
                for j, pos_j in enumerate(sensors):
                    if i == j:
                        continue  # Skip itself.
                    d = math.hypot(pos[0] - pos_j[0], pos[1] - pos_j[1])
                    if d <= neighbor_range:
                        sensor_neighbors.append(str(id_mapping[j]))
                        sensor_distances_list.append(round(d, 2))

                original_sensors_nbrs_dist_info[sensor_new_id] = {
                    'Neighbors': sensor_neighbors,
                    'Distances': sensor_distances_list
                }

            # Build network graph from the original neighbor info.
            G = nx.Graph()
            G.add_node("C")
            for sensor_id in original_sensors_nbrs_dist_info.keys():
                G.add_node(sensor_id)
            for sensor_id, info in original_sensors_nbrs_dist_info.items():
                for neighbor in info['Neighbors']:
                    G.add_edge(sensor_id, neighbor)

            # Check connectivity: each sensor must have a path to "C".
            disconnected = [sensor_id for sensor_id in original_sensors_nbrs_dist_info.keys() if not nx.has_path(G, sensor_id, "C")]

            if disconnected:
                for sensor_new_id in disconnected:
                    original_id = next(orig for orig, new in id_mapping.items() if new == int(sensor_new_id))
                    if original_id < bttn_count:
                        # Reposition a bottleneck node: pick from neighbor_range candidates
                        available = list(bottleneck_candidates_set - set(sensors[:bttn_count]))
                        if not available:
                            raise ValueError("No available bottleneck candidate for repositioning")
                        new_pt = random.choice(available)
                        sensors[original_id] = new_pt
                        bottleneck_positions[original_id] = new_pt
                    else:
                        # Reposition a non-bottleneck node: pick from outside_range candidates
                        current_others = sensors[bttn_count:]
                        available = list(outside_points_set - set(current_others))
                        if not available:
                            raise ValueError("No available outside candidate for repositioning")
                        new_pt = random.choice(available)
                        sensors[original_id] = new_pt
                        other_positions[original_id - bttn_count] = new_pt
                continue

            # Forward-routing check:
            forward_issue = []
            for sensor_id, info in original_sensors_nbrs_dist_info.items():
                int_id = int(sensor_id)
                if int_id == 1:
                    continue
                if "C" in info["Neighbors"]:
                    continue
                if not any((nbr != "C" and int(nbr) < int_id) for nbr in info["Neighbors"]):
                    forward_issue.append(sensor_id)

            if forward_issue:
                for sensor_new_id in forward_issue:
                    original_id = next(orig for orig, new in id_mapping.items() if new == int(sensor_new_id))
                    if original_id < bttn_count:
                        # Reposition a bottleneck node: pick from neighbor_range candidates
                        available = list(bottleneck_candidates_set - set(sensors[:bttn_count]))
                        if not available:
                            raise ValueError("No available bottleneck candidate for repositioning")
                        new_pt = random.choice(available)
                        sensors[original_id] = new_pt
                        bottleneck_positions[original_id] = new_pt
                    else:
                        # Reposition a non-bottleneck node: pick from outside_range candidates
                        current_others = sensors[bttn_count:]
                        available = list(outside_points_set - set(current_others))
                        if not available:
                            raise ValueError("No available outside candidate for repositioning")
                        new_pt = random.choice(available)
                        sensors[original_id] = new_pt
                        other_positions[original_id - bttn_count] = new_pt
                continue

            all_ok = True

        # Build controller neighbor info: include ALL sensors with their distances from the controller.
        controller_neighbors = []
        controller_distances = []
        for i, pos in enumerate(sensors):
            sensor_new_id = str(id_mapping[i])
            d = math.hypot(pos[0] - self.controller_pos[0], pos[1] - self.controller_pos[1])
            controller_neighbors.append(sensor_new_id)
            controller_distances.append(round(d, 2))
        controller_nbrs_dist_info = {
            'Neighbors': controller_neighbors,
            'Distances': controller_distances
        }

        # Add the "Controller" field to each sensor (distance from sensor to controller).
        reverse_id_mapping = {v: k for k, v in id_mapping.items()}
        for sensor_id in original_sensors_nbrs_dist_info.keys():
            orig_index = reverse_id_mapping[int(sensor_id)]
            d = math.hypot(sensors[orig_index][0] - self.controller_pos[0], sensors[orig_index][1] - self.controller_pos[1])
            original_sensors_nbrs_dist_info[sensor_id]["Controller"] = round(d, 2)
        
        # Compute the list of sensors with a real direct connection to "C" based on original logic.
        original_directly_connected = [sensor_id for sensor_id, info in original_sensors_nbrs_dist_info.items() if "C" in info["Neighbors"]]
        # Enforce exact bottleneck count
        bttn_count = int(self.bttn_num)
        # Sort sensors by their distance to the controller
        sorted_sensors = sorted(
            original_sensors_nbrs_dist_info.keys(),
            key=lambda k: original_sensors_nbrs_dist_info[k]["Controller"]
        )
        # Pick sensors with a direct controller connection, up to bttn_count
        new_bottlenecks = [
            sid for sid in sorted_sensors
            if "C" in original_sensors_nbrs_dist_info[sid]["Neighbors"]
        ][:bttn_count]
        # If fewer than needed, add the nearest sensors until count reached
        if len(new_bottlenecks) < bttn_count:
            for sid in sorted_sensors:
                if sid not in new_bottlenecks:
                    new_bottlenecks.append(sid)
                    if len(new_bottlenecks) == bttn_count:
                        break
        # Replace original_directly_connected with the enforced list
        original_directly_connected = new_bottlenecks
        
        # --- Sorting Phase ---
        # Sort original_sensors_nbrs_dist_info by sensor id.
        original_sensors_nbrs_dist_info = dict(sorted(original_sensors_nbrs_dist_info.items(), key=lambda x: int(x[0])))
        
        # For each sensor, sort its neighbor list (keeping "C" first if present).
        for sensor_id, info in original_sensors_nbrs_dist_info.items():
            pairs = list(zip(info["Neighbors"], info["Distances"]))
            c_pairs = [pair for pair in pairs if pair[0] == "C"]
            other_pairs = [pair for pair in pairs if pair[0] != "C"]
            other_pairs_sorted = sorted(other_pairs, key=lambda x: int(x[0]))
            sorted_pairs = c_pairs + other_pairs_sorted
            original_sensors_nbrs_dist_info[sensor_id]["Neighbors"] = [p[0] for p in sorted_pairs]
            original_sensors_nbrs_dist_info[sensor_id]["Distances"] = [p[1] for p in sorted_pairs]
        
        # Similarly, sort controller_nbrs_dist_info by sensor id.
        ctrl_pairs = list(zip(controller_nbrs_dist_info["Neighbors"], controller_nbrs_dist_info["Distances"]))
        ctrl_pairs_sorted = sorted(ctrl_pairs, key=lambda x: int(x[0]))
        controller_nbrs_dist_info["Neighbors"] = [p[0] for p in ctrl_pairs_sorted]
        controller_nbrs_dist_info["Distances"] = [p[1] for p in ctrl_pairs_sorted]
        
        # --- Create display copy with forced "C" in each sensor's neighbor list ---
        display_sensors_nbrs_dist_info = copy.deepcopy(original_sensors_nbrs_dist_info)
        for sensor_id, info in display_sensors_nbrs_dist_info.items():
            if "C" not in info["Neighbors"]:
                info["Neighbors"].insert(0, "C")
                info["Distances"].insert(0, info["Controller"])

        # --- Save final sensor positions in self.sensor_positions ---
        self.sensor_positions = {}
        for i, pos in enumerate(sensors):
            sensor_new_id = str(id_mapping[i])
            self.sensor_positions[sensor_new_id] = pos

        # print("\nFinal Sensor Neighbors Info (Display Copy) [Forced 'C' in each sensor]:")
        # for sensor_id, info in display_sensors_nbrs_dist_info.items():
        #     print(f"Sensor {sensor_id}:")
        #     print("  Neighbors:", info['Neighbors'])
        #     print("  Distances:", info['Distances'])
        #     print("  Controller:", info["Controller"])
        
        # print("\nFinal Controller Neighbors Info (all sensors in the network):")
        # print("  Neighbors:", controller_nbrs_dist_info['Neighbors'])
        # print("  Distances:", controller_nbrs_dist_info['Distances'])
        
        # print("\nSensors with a real direct connection to 'C' (original logic):", original_directly_connected)
        
        return display_sensors_nbrs_dist_info, controller_nbrs_dist_info, original_directly_connected

    def get_reward(self, rt_no):
        if self.reward_method == 'SumReward':
            # Calculate reward using sum mode.
            rewards = [sensor.energy_rem_ratio_w + sensor.sum_dist_hop_p_ack_w_dict[rt_no] for sensor in self.controller.sensors.values()]
            sum_reward = sum(rewards)
            return sum_reward
        elif self.reward_method == 'MinReward':
            # Calculate reward using min mode.
            sum_ratio_arr = np.array([sensor.energy_rem_ratio_w + sensor.sum_dist_hop_p_ack_w_dict[rt_no] for sensor in self.controller.sensors.values() if sensor.id in self.controller.bottlenecks])
            return min(sum_ratio_arr)

    def get_state(self, rt_no):
        # Calculate state by selected state mode.
        if self.state_method == 'max-min':
            e_min = min(
                    (sensor for sensor in self.controller.sensors.values() if sensor.id in self.controller.bottlenecks),
                    key=lambda sensor: sensor.energy_rem_try
                ).energy_rem_try
            e_max = max(
                    (sensor for sensor in self.controller.sensors.values() if sensor.id in self.controller.bottlenecks),
                    key=lambda sensor: sensor.energy_rem_try
                ).energy_rem_try
            state = round((e_max-e_min)*1e-12, self.decimal)
        elif self.state_method == 'min':
            state = round(
                min(
                    (sensor for sensor in self.controller.sensors.values() if sensor.id in self.controller.bottlenecks),
                    key=lambda sensor: sensor.energy_rem_try
                ).energy_rem_try * 1e-12,
                self.decimal
            )
        elif self.state_method == 'rt':
            state = rt_no
        return state

    def step(self, rt_no):
        self.cur_rt = rt_no

        energy_before_step = {sensor.id: sensor.energy for sensor in self.controller.sensors.values()} # NOTE: for logging
        energy_after_step = {} # NOTE: for logging

        round_dies = []

        # Each node Rx, Tx and increase their energy.
        for sensor in self.controller.sensors.values():
            if self.same_rt:
                sensor.energy_rem_try = sensor.energy - sensor.erx_etx_no_downstrm[rt_no]
                sensor.energy = sensor.energy - sensor.erx_etx_no_downstrm[rt_no]
            # If the controller select the difference route from the last round.
            else:
                sensor.energy_rem_try = sensor.energy - sensor.erx_etx_downstrm_upstrm[rt_no]
                sensor.energy = sensor.energy - sensor.erx_etx_downstrm_upstrm[rt_no]
                
                # NOTE: for logging
                if sensor.id in self.controller.bottlenecks:
                    sensor.e_use_rx_downstrm += sensor.erx_downstrm[rt_no]

            sensor.energy_rem_ratio_w = (sensor.energy / sensor.init_energy) * self.controller.energy_rem_w
            # print(f'sensor_id: {sensor.id}')
            # print(f'E: {sensor.energy}')
            if sensor.id in self.controller.bottlenecks:
                sensor.e_use_rx_data += sensor.erx_data[rt_no]
                sensor.e_use_tx_data += sensor.etx_data[rt_no]
                sensor.e_use_rx_upstrm += sensor.erx_upstrm[rt_no]
                sensor.e_use_tx_upstrm += sensor.etx_upstrm[rt_no]
                energy_after_step[sensor.id] = sensor.energy

        # Calculate reward after taking the action (RT)
        reward = self.get_reward(rt_no)
        # Calculate environment's state after using the RT
        state = self.get_state(rt_no)

        # # Check dead nodes (the condition to end the simulation)
        # for sensor in self.controller.sensors.values():
        #     if sensor.energy <= 0.05:
        #         self.die_node.append(sensor.id) # The node that die this round.
        #         round_dies.append(sensor.id) # Collect all node that die.
        #         sensor.energy = 0
        #     # if len(round_dies) > 0:
        #     #     self.done = True
        #     # If all bootle neck node die, simulation end.
        #     if len(set(self.die_node).intersection(set(self.controller.bottlenecks))) == len(self.controller.bottlenecks):
        #         self.done = True
        # # self.done = True

        return state, reward, self.done, self.controller.sensors, self.die_node, round_dies, energy_before_step, energy_after_step, self.controller.bottlenecks

    @timeit_try_step
    def try_step(self, rt_no):
        # Do the same as step() but not reduce energy.
        energy_before_step = {sensor.id: sensor.energy for sensor in self.controller.sensors.values()}
        energy_after_step = {}

        useable_rt = True # This RT make any node use E more than 0?
        # if rt_no == 4097:
        #     print(f'E_cur of 4: {self.controller.sensors['4'].energy}')
        #     print(f'E_use of 4: {self.controller.sensors['4'].erx_etx_no_downstrm[rt_no]}')
        for sensor in self.controller.sensors.values():
            # If the controller select the same route, nodes won't use energy to receive downstream control packet.
            if self.cur_rt == rt_no:
                sensor.energy_rem_try = sensor.energy - sensor.erx_etx_no_downstrm[rt_no]
            # If the controller select the difference route from the last round.
            else:
                sensor.energy_rem_try = sensor.energy - sensor.erx_etx_downstrm_upstrm[rt_no]
            sensor.energy_rem_ratio_w = sensor.energy_rem_try / sensor.init_energy * self.controller.energy_rem_w
            energy_after_step[sensor.id] = sensor.energy_rem_try

            if sensor.energy_rem_try < 0:
                # if sensor.id == '4' and rt_no == 4097:
                #     print(f'E_after of 4: {sensor.energy_rem_try}')
                self.unavailable_rt_dict.setdefault(sensor.id, set()).add(int(rt_no))
                self.unavailable_rt.add(int(rt_no))
                useable_rt = False

        reward = self.get_reward(rt_no)
        state = self.get_state(rt_no)

        return state, reward, self.done, self.controller.sensors, self.die_node, useable_rt
    
    def check_sensor_die(self):
        is_sensor_die = False
        round_dies = []
        unavailable_rt = []
        death_mark = []
        check_unavail_rt = {}
        remain_rt_nos = self.controller.RTs.columns.tolist()

        for sensor_id in self.controller.bottlenecks:
            if sensor_id not in self.die_node:
                sensor = self.controller.sensors[sensor_id]
                if self.unavailable_rt_dict.get(sensor.id, None) != None:
                    print('check num ')
                    if self.controller.RTs.shape[1] == len(self.unavailable_rt_dict[sensor.id]):
                        is_sensor_die = True
                        round_dies.append(sensor.id)
                        death_mark.append(sensor.id)


        print(f'num RTs shape[1]: {self.controller.RTs.shape[1]}')
        print(f'unavailable_rt: {self.unavailable_rt}')
        print(f'num unavailable_rt_dict: {len(self.unavailable_rt_dict.get('2', []))}')
        print(f'unavailable_rt_dict: {self.unavailable_rt_dict.get('2', [])}')
        if round_dies:
            # print(f'node 1:\n{len(self.controller.sensors['2'].die_rt_nos)}')
            # for sensor_id in round_dies:
            #     sensor = self.controller.sensors[sensor_id]
            #     self.unavailable_rt_dict[sensor.id] = set(sensor.die_rt_nos)

            self.die_node.extend(round_dies)
            self.controller.exclude_sensor_rt(self.die_node, round_dies)
            self.RTs = self.controller.RTs
            remain_rt_nos = self.controller.RTs.columns.tolist()
            self.exclude_sensors(round_dies)

            self.unavailable_rt = set()
            self.unavailable_rt_dict = {}
            # self.unavailable_rt = self.unavailable_rt.union(self.unavailable_rt_save)

            # unavailable_rt = set()

            # for s in self.unavailable_rt_dict.values():
            #     unavailable_rt |= s

            # self.unavailable_rt = self.unavailable_rt.union(unavailable_rt)
            self.unavailable_rt_save = self.unavailable_rt_save.union(unavailable_rt)
            # print(len(self.unavailable_rt))

            # remain_rt_nos = self.exclude_sensors(round_dies)


            # print(len(self.unavailable_rt))
            # print(len(self.RTs.columns.tolist()))

            # seen = set()
            # duplicates = set()

            # for item in self.unavailable_rt:
            #     if item in seen:
            #         duplicates.add(item)
            #     else:
            #         seen.add(item)

        # print(f'remain_rt_nos: {len(remain_rt_nos)}')
        # print(f'unavaiable_rt: {len(self.unavailable_rt)}')

        # If all bootle neck node die, simulation end.
        if len(set(self.die_node).intersection(set(self.controller.bottlenecks))) == len(self.controller.bottlenecks):
            self.done = True



        return is_sensor_die, remain_rt_nos, round_dies

    def reset_e_use_count(self):
        # NOTE: for logging.
        for sensor in self.controller.sensors.values():
            sensor.e_use_rx_data = 0
            sensor.e_use_tx_data = 0
            sensor.e_use_rx_upstrm = 0
            sensor.e_use_tx_upstrm = 0
            sensor.e_use_rx_downstrm = 0

    def exclude_sensors(self, round_dies):
        # If any node die, exclude that node and route that need the node to forward packet.
        # self.RTs, remain_rt_nos = self.controller.recalculate_sensor(round_dies)
        for node in round_dies:
            self.controller.sensors.pop(node)
        # self.round_dies = []

        # return remain_rt_nos
    # ===== Operates ===== (T)

    # ===== OLD ===== (H)
    def build_env_rts(self):
        ########## Environment ##########
        #################### main topo ####################
        # C = np.array([
        #     [0, 1, 1, 0, 1, 0, 0, 0, 0], 
        #     [1, 0, 1, 0, 1, 0, 0, 0, 0], 
        #     [1, 1, 0, 1, 1, 0, 1, 0, 0], 
        #     [1, 0, 1, 0, 0, 1, 1, 0, 0], 
        #     [0, 1, 1, 0, 0, 0, 1, 1, 0], 
        #     [0, 0, 0, 1, 0, 0, 1, 0, 1],
        #     [0, 0, 1, 1, 1, 1, 0, 1, 1], 
        #     [0, 0, 0, 0, 1, 0, 1, 0, 1], 
        #     [0, 0, 0, 0, 0, 1, 1, 1, 0]
        # ])

        # sensors_nbrs_dist_info = {
        #     '1': {'Neighbors': ['C', '2', '4'], 'Distances': [75, 55, 37]},
        #     '2': {'Neighbors': ['C', '1', '3', '4', '6'], 'Distances': [50, 55, 50, 62, 37]},
        #     '3': {'Neighbors': ['C', '2', '5', '6'], 'Distances': [90, 50, 62, 62]},
        #     '4': {'Neighbors': ['C', '1', '2', '6', '7'], 'Distances': [100, 37, 62, 55, 37]},
        #     '5': {'Neighbors': ['C', '3', '6', '8'], 'Distances': [137, 62, 74, 65]},
        #     '6': {'Neighbors': ['C', '2', '3', '4', '5', '7', '8'], 'Distances': [85, 37, 62, 55, 74, 67, 74]},
        #     '7': {'Neighbors': ['C', '4', '6', '8'], 'Distances': [125, 37, 67, 65]},
        #     '8': {'Neighbors': ['C', '5', '6', '7'], 'Distances': [155, 65, 74, 65]},
        # }

        # controller_nbrs_dist_info = {
        #     'Neighbors': ['1', '2', '3', '4', '5', '6', '7', '8'], 'Distances': [75, 50, 90, 100, 137, 85, 125, 155]
        #     }
        
        # bottlenecks = ['1', '2', '3']

        # self.sensor_positions = {
        #     '2': (0,  50),   # left middle
        #     '3': (5, 60),   # left top
        #     '1': (0, 40),   # left bottom

        #     '6': (10, 50),  # center
        #     '5': (20, 60),  # right top
        #     '8': (30, 50),  # right middle
        #     '7': (20, 40),  # right bottom
        #     '4': (10, 40),  # bottom center
        # }
        #################### main topo ####################

        ########## main topo (re-label) ##########
        C = np.array([
            [0, 1, 1, 0, 1, 0, 0, 0, 0], 
            [1, 0, 1, 1, 1, 1, 0, 0, 0], 
            [1, 1, 0, 0, 0, 1, 0, 0, 0], 
            [0, 1, 0, 0, 1, 1, 1, 1, 1], 
            [1, 1, 0, 1, 0, 0, 0, 1, 0], 
            [0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 1], 
            [0, 0, 0, 1, 1, 0, 0, 0, 1], 
            [0, 0, 0, 1, 0, 0, 1, 1, 0]
        ])

        sensors_nbrs_dist_info = {
            '1': {'Neighbors': ['C', '2', '3', '4', '5'], 'Distances': [50, 55, 37, 50, 62]},
            '2': {'Neighbors': ['C', '1', '5'], 'Distances': [75, 55, 37]},
            '3': {'Neighbors': ['C', '1', '4', '5', '6', '7', '8'], 'Distances': [85, 37, 62, 55, 67, 74, 74]},
            '4': {'Neighbors': ['C', '1', '3', '7'], 'Distances': [90, 50, 62, 62]},
            '5': {'Neighbors': ['C', '1', '2', '3', '6'], 'Distances': [100, 62, 37, 55, 37]},
            '6': {'Neighbors': ['C', '3', '5', '8'], 'Distances': [125, 67, 37, 65]},
            '7': {'Neighbors': ['C', '3', '4', '8'], 'Distances': [137, 74, 62, 65]},
            '8': {'Neighbors': ['C', '3', '6', '7'], 'Distances': [155, 74, 65, 65]},
        }
        controller_nbrs_dist_info = {
            'Neighbors': ['1', '2', '3', '4', '5', '6', '7', '8'], 'Distances': [55, 67, 45, 50, 105, 130, 107, 102]
            }
        bottlenecks = ['1', '2', '4']

        self.sensor_positions = {
            '1': (0, 50),   # left middle
            '4': (5, 60),   # left top
            '2': (0, 40),   # left bottom

            '3': (10, 50),  # center
            '7': (20, 60),  # right top
            '8': (30, 50),  # right middle
            '6': (20, 40),  # right bottom
            '5': (10, 40),  # bottom center
        }
        # ########## main topo (re-label) ##########

        # sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks, self.sensor_positions = self.load_cache_topo()
        # sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks = self.random_topo()
        C = self.create_connectivity(sensors_nbrs_dist_info, bottlenecks)

        # Initialize the product as a Python integer
        product = 1

        # Loop over each row, starting from row 1 (since row 0 is the controller).
        for i in range(1, C.shape[0]):
            # Count the number of ones in row i for columns 0 to i-1 (the lower triangular part)
            count = int(np.sum(C[i, :i] == 1))
            # print(f"Row {i} has {count} ones in the lower triangle.")
            product *= count
        print("The number of possible routes from C is:", product)
        # Extract the lower triangular part (excluding the main diagonal)
        lower_triangle = np.tril(C, k=-1)

        # Count the number of 1's in the lower triangular part
        count_ones = np.sum(lower_triangle)
        print("The number of links:", count_ones)

        sensors = {}
        for sensor_id, sensor_info in sensors_nbrs_dist_info.items():
            sensor = Sensor(sensor_id) # Create sensor objects
            sensor.neighbors = pd.DataFrame(sensors_nbrs_dist_info[sensor_id])
            sensors[sensor_id] = sensor # Store objects to dict
            print(f'sensor_id: {sensor.id}')
            print(f'E: {sensor.init_energy}')

        self.init_num_nodes = len(sensors)
        self.num_nodes = len(sensors) # Reset
        # Setup controller
        self.controller = Controller('C') # Create controller object and assign attribute controller
        self.controller.neighbors = pd.DataFrame(controller_nbrs_dist_info)
        self.controller.sensors = sensors # This is Observation
        self.controller.bottlenecks = bottlenecks

        ####### for only main use rt ########
        # d = {1: ['C', 'C', '1', 'C','2','5', '4', '3'], 2: ['C', 'C', '1', 'C', '2', '3', '4', '7'],
        #     3: ['C', 'C', '1', 'C', '2', '3', '4', '3'], 4: ['C','C', '1', 'C', '2', '5', '3', '3'],
        #     5: ['C', 'C', '1', 'C', '2', '3', '3', '3'], 6: ['C', 'C', '1', 'C', '1', '5', '4', '7'],
        #     7:['C', 'C', '1', 'C', '1', '5', '4', '3']
        # }
        # RTs = pd.DataFrame(d, index=['1', '2','3','4','5','6','7','8'])

        ####### for only main use rt ########

        if self.expect_rt_num == 'full':
            start_old_get_rt = time.time()
            RTs = self.get_all_RTs()
            # P = self.controller.generate_all_forward_routes(C, self.num_sensors_ctrl)
            # P = self.controller.generate_multiple_matrices_with_connectivity(product, self.num_sensors_ctrl, C)
            # RTs = self.build_RTs(P)
            self.time_old_get_rt = time.time()-start_old_get_rt

        elif self.expect_rt_num == 'fw':
            start_get_all_rt = time.time()
            rts_log = self.get_all_RTs()
            self.time_get_all_rt = time.time()-start_get_all_rt
            print(f'all rt num: {rts_log.shape[1]}')
            start_old_get_rt_fw = time.time()
            # P = self.controller.generate_multiple_matrices_with_connectivity(product, self.num_sensors_ctrl, C)
            P = self.controller.generate_all_forward_routes(C, self.num_sensors_ctrl)
            print(f'num P: {len(P)}')
            RTs = self.build_RTs(P)

            self.time_old_get_rt = time.time()-start_old_get_rt_fw

        RTs.name = 'Node\'s forwarder'
        self.RTs = RTs
        self.total_rt = self.RTs.shape[1]
        # print(self.RTs)
        # print(self.RTs.shape)
        # print(bottlenecks)
        # self.save_rts_topo()

    def generate_edges_combs(self):
        """
        Generate unique edge representations from the current sensor topology.
        - For a sensor connected to the controller 'C', only include an edge if the sensor
        is in self.controller.bottlenecks (as these are the ones directly connected to the controller).
        - For sensor-to-sensor edges, sort the sensor IDs so that the lower one comes first (x-y).
        - Returns a list of edges sorted primarily by the first node; for edges with the same first node,
        the edge with 'C' as the second node comes first.
        """
        edges = set()
        # Loop through sensors stored in the controller.
        for sensor_id, sensor in self.controller.sensors.items():
            # Each sensor's neighbor information is kept in a DataFrame (see build_env_rts())
            # Assume the DataFrame has a column 'Neighbors'
            neighbor_list = sensor.neighbors['Neighbors'].tolist()
            for nbr in neighbor_list:
                if nbr == 'C':
                    # Only add an edge to 'C' if this sensor is a bottleneck (i.e. directly connected)
                    if sensor_id in self.controller.bottlenecks:
                        edges.add(f"{sensor_id}-C")
                else:
                    # For sensor-to-sensor connections, sort the sensor IDs so that the smaller one comes first (x-y)
                    x, y = sorted([sensor_id, nbr], key=int)
                    edges.add(f"{x}-{y}")
        # Sort the final list of edges.
        # For each edge 'x-y', the key is a tuple:
        # - first element is the integer value of x,
        # - second element is 0 if y is 'C' (so 'x-C' comes first) or 1 otherwise,
        # - third element is int(y) if y is numeric (used for sorting sensor-to-sensor edges).
        return sorted(
            list(edges),
            key=lambda edge: (
                int(edge.split('-')[0]),
                0 if edge.split('-')[1] == 'C' else 1,
                0 if edge.split('-')[1] == 'C' else int(edge.split('-')[1])
            )
        )
    
    def get_all_RTs(self):
        # Calculate all possible spanning tree.
        edges_combs = self.generate_edges_combs()

        if self.cache_mode == 'n': # If we not use cache mode..
            combs = combinations(edges_combs, self.num_sensors)
            rt_df = self.controller.get_rts(combs, self.num_sensors_ctrl)
        else: # If we use cache mode.
            sub_dir = 'topo'
            cache_dir = f'{self.cache_dir}/{sub_dir}/{self.name_topo}'
            # Create cache folder.
            if not os.path.exists(f"{self.cache_dir}/{sub_dir}"):
                os.makedirs(f"{self.cache_dir}/{sub_dir}")
            if os.path.exists(cache_dir):
                with open(cache_dir, "rb") as file:
                    rt_df = pickle.load(file)
            else:
                combs = combinations(edges_combs, self.num_sensors)
                rt_df = self.controller.get_rts(combs, self.num_sensors_ctrl)
                with open(cache_dir, "wb") as file:
                    pickle.dump(rt_df, file)

        return rt_df
    
    def gen_RT_direct(self):
        RTs = pd.DataFrame({-1: ['C' for _ in range(len(self.controller.sensors))]}, index=[sensor.id for sensor in self.controller.sensors.values()], dtype='string')
        self.RTs = RTs
        self.controller.RTs = self.RTs
        
        return self.RTs.columns.to_numpy(), self.controller
    # ===== OLD ===== (T)

    # ===== GA ===== (H)
    def build_RTs(self, population):
        forwarders_by_route = [self.controller.extract_forwarders(M, self.num_sensors_ctrl) for M in population]
        RTs = pd.DataFrame(forwarders_by_route).T
        RTs.index = [str(i) for i in range(1, self.num_sensors_ctrl)]
        RTs.columns = [str(i) for i in range(1, len(population)+1)]

        return RTs

    def print_matrix(self, matrix, title="Matrix"):
        print(title + ":")
        for row in matrix:
            print(row)
        print()
            
    def create_connectivity(self, sensors_nbrs_dist_info, bottlenecks):
        """"""
        # Total number of sensors.
        n = len(sensors_nbrs_dist_info)
        # Total nodes: index 0 is the controller, then sensors "1", "2", ..., "n"
        total_nodes = n + 1

        # Initialize connectivity matrix with zeros.
        C = np.zeros((total_nodes, total_nodes), dtype=int)

        # Set controller-sensor connectivity:
        # For each sensor, mark bit 1 in the controller row (and column) only if the sensor's
        # neighbor list (from random_topo()) includes "C".
        for sensor_id, info in sensors_nbrs_dist_info.items():
            # print(sensor_id)
            i = int(sensor_id)  # sensor IDs are strings like '1', '2', etc.
            if "C" in info["Neighbors"] and sensor_id in bottlenecks:
                C[0, i] = 1
                C[i, 0] = 1

        # Set sensor-to-sensor connectivity based on neighbor info.
        for sensor_id, info in sensors_nbrs_dist_info.items():
            i = int(sensor_id)
            for neighbor in info["Neighbors"]:
                # Skip the controller because its connectivity has been handled.
                if neighbor == "C":
                    continue
                j = int(neighbor)
                C[i, j] = 1
                C[j, i] = 1  # Assuming connectivity is bidirectional

        # print("Connectivity matrix C:")
        # print(C)
        return C

    def cal_rt_n_fitness(self, inner_population, population_size):
        """
            inner_population = P or new_population.
        """
        # Recalculate RTs and sensors_info_df from current population P.
        self.RTs = self.build_RTs(inner_population)
        self.controller.RTs = self.RTs
        self.controller.sensors_info_df = self.controller.get_sensors_info_df(self.RTs, self.controller.sensors)

        # Recalculate RTs and sensors_info_df from current population P.
        inner_RTs = self.RTs
        inner_sensor_info_df = self.controller.get_sensors_info_df(inner_RTs, self.controller.sensors)

        # Recalculate energy usage.
        self.controller.cal_E_GA(inner_sensor_info_df, inner_population, self.cache_pop_set, self.cache_fitness)
        # self.controller.cal_E(inner_sensor_info_df)

        # Recalculate fitness.
        sorted_population = self.controller.fitness_function(self.RTs, self.controller.sensors, self.controller.sensors_info_df, n=population_size)
        return sorted_population
    
    def recal_rt_n_fitness(self, inner_population, population_size):
        """
            inner_population = P or new_population.
        """
        
        # Recalculate RTs and sensors_info_df from current population P.
        inner_RTs = self.build_RTs(inner_population[population_size:])
        
        inner_RTs.columns = range(population_size+1, len(inner_population)+1)
        inner_RTs.columns = inner_RTs.columns.astype(str)
        
        inner_sensor_info_df = self.controller.get_sensors_info_df(inner_RTs, self.controller.sensors)

        # Recalculate energy usage.
        self.controller.cal_E_GA(inner_sensor_info_df, inner_population, self.cache_pop_set, self.cache_fitness)
        # self.controller.cal_E(inner_sensor_info_df)

        combined_sensor_info = pd.concat([self.controller.sensors_info_df, inner_sensor_info_df], axis=0, ignore_index=True)
        combined_RTs = pd.concat([self.RTs, inner_RTs], axis=1)

        self.controller.sensors_info_df = combined_sensor_info
        self.RTs = combined_RTs
        self.controller.RTs = combined_RTs
        # Recalculate fitness.
        sorted_population = self.controller.fitness_function(combined_RTs, self.controller.sensors, combined_sensor_info, n=population_size)
        return sorted_population
    
    @timeit
    def genetic_algo(self):
        population_size = 22
        num_generations = 300
        parent_size = 2
        tournament_size = 5
        max_attempts_sel = 100
        crossover_rate = 0.9
        mutation_rate = 0.2
        
        # Check 100% of individual order the same as previous gen.
        check_same_order_indi_percent = 1
        check_same_order_indi_counter = 0

        ## If xx% of indi order the same as prvs yy times, stop GA
        # max_time_same_order_indi_count = num_generations # don't stop GA.
        # max_time_same_order_indi_count = round(num_generations*0.1) # 10% of number of gen.
        max_time_same_order_indi_count = 3 # 3 rounds like ref paper.

        C = np.array([
            [0, 1, 1, 0, 1, 0, 0, 0, 0], 
            [1, 0, 1, 1, 1, 1, 0, 0, 0], 
            [1, 1, 0, 0, 0, 1, 0, 0, 0], 
            [0, 1, 0, 0, 1, 1, 1, 1, 1], 
            [1, 1, 0, 1, 0, 0, 0, 1, 0], 
            [0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 1], 
            [0, 0, 0, 1, 1, 0, 0, 0, 1], 
            [0, 0, 0, 1, 0, 0, 1, 1, 0]
        ])

        sensors_nbrs_dist_info = {
            '1': {'Neighbors': ['C', '2', '3', '4', '5'], 'Distances': [50, 55, 37, 50, 62]},
            '2': {'Neighbors': ['C', '1', '5'], 'Distances': [75, 55, 37]},
            '3': {'Neighbors': ['C', '1', '4', '5', '6', '7', '8'], 'Distances': [85, 37, 62, 55, 67, 74, 74]},
            '4': {'Neighbors': ['C', '1', '3', '7'], 'Distances': [90, 50, 62, 62]},
            '5': {'Neighbors': ['C', '1', '2', '3', '6'], 'Distances': [100, 62, 37, 55, 37]},
            '6': {'Neighbors': ['C', '3', '5', '8'], 'Distances': [125, 67, 37, 65]},
            '7': {'Neighbors': ['C', '3', '4', '8'], 'Distances': [137, 74, 62, 65]},
            '8': {'Neighbors': ['C', '3', '6', '7'], 'Distances': [155, 74, 65, 65]},
        }
        controller_nbrs_dist_info = {
            'Neighbors': ['1', '2', '3', '4', '5', '6', '7', '8'], 'Distances': [55, 67, 45, 50, 105, 130, 107, 102]
            }
        bottlenecks = ['1', '2', '4']

        self.sensor_positions = {
            '1': (0,  50),   # left middle
            '4': (5, 60),   # left top
            '2': (0, 40),   # left bottom

            '3': (10, 50),  # center
            '7': (20, 60),  # right top
            '8': (30, 50),  # right middle
            '6': (20, 40),  # right bottom
            '5': (10, 40),  # bottom center
        }

        start_random_topo = time.time()
        # sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks, self.sensor_positions = self.load_cache_topo()
        # sensors_nbrs_dist_info, controller_nbrs_dist_info, bottlenecks = self.random_topo()
        self.time_random_topo = time.time()-start_random_topo

        C = self.create_connectivity(sensors_nbrs_dist_info, bottlenecks)

        # Initialize the product as a Python integer
        product = 1
        # Loop over each row, starting from row 1 (since row 0 is the controller).
        for i in range(1, C.shape[0]):
            # Count the number of ones in row i for columns 0 to i-1 (the lower triangular part)
            count = int(np.sum(C[i, :i] == 1))
            # print(f"Row {i} has {count} ones in the lower triangle.")
            product *= count
        print("The number of possible routes from C is:", product)
        # Extract the lower triangular part (excluding the main diagonal)
        lower_triangle = np.tril(C, k=-1)

        # Count the number of 1's in the lower triangular part
        count_ones = np.sum(lower_triangle)
        print("The number of links:", count_ones)

        population_size = min(population_size, product)

        print(f'Population size: {population_size}')

        P = self.controller.generate_multiple_matrices_with_connectivity(population_size, self.num_sensors_ctrl, C)
        self.RTs = self.build_RTs(P)
        self.controller.RTs = self.RTs

        sensors = {}
        for sensor_id, sensor_info in sensors_nbrs_dist_info.items():
            sensor = Sensor(sensor_id)
            sensor.neighbors = pd.DataFrame(sensor_info)
            sensors[sensor_id] = sensor
        self.init_num_nodes = len(sensors)
        self.num_nodes = len(sensors) # Reset

        # Setup controller
        self.controller.neighbors = pd.DataFrame(controller_nbrs_dist_info)
        self.controller.sensors = sensors # This is Observation
        self.controller.bottlenecks = bottlenecks

        # Recalculate RTs and fitnesses.
        self.cache_pop_set = set()
        self.cache_fitness = {}
        sorted_population = self.cal_rt_n_fitness(P, population_size)

        for gen in range(1, num_generations + 1):
            # print(f"\nGeneration {gen}")

            if sorted_population.size == 0:
                print(f"Generation {gen}: No valid individuals found in fitness calculation. Retaining previous population.")
                continue

            ## SELECTION ##
            selected_parents = self.controller.GA_selection(population_size, sorted_population, parent_size, tournament_size, max_attempts_sel)
            ## Now we got list of parents of this ganeration to be 'crossover'##

            ## CROSSOVER ##
            gen_childs = self.controller.GA_crossover(P, selected_parents, crossover_rate)
            ## Now we got list of new childs of this genration to be [mutation] or [extend population] ##

            ## MUTATION ##
            gen_mutates = self.controller.GA_mutation(C, P, gen_childs, mutation_rate)
            ## Now we got list of mutates of this genration to be 'extend population' ##
            
            # # Extend population
            old_P = copy.deepcopy(P)

            if gen_childs or gen_mutates:
                # Extend P only when there is offspring
                P = old_P + gen_childs + gen_mutates

                # ── NOW REBUILD RTs & FITNESS ──
                # Recalculate RTs and fitnesses.
                sorted_population = self.recal_rt_n_fitness(P, population_size)

                # Re‑index and collect the top‑N individuals
                selected_indices = [int(ind['rt_no']) for ind in sorted_population[:population_size]]
                indexed_list     = list(enumerate(selected_indices))

                # Build new metric dicts and new_sorted_P
                new_erx_upstrm = {}
                new_etx_upstrm = {}
                new_erx_data   = {}
                new_etx_data   = {}
                new_erx_etx_no_downstrm           = {}
                new_erx_etx_downstrm_upstrm       = {}
                new_erx_etx_no_downstrm_no_upstrm = {}
                new_sorted_P = []
                for index, rt_no in indexed_list:
                    for sensor in self.controller.sensors.values():
                        new_erx_upstrm.setdefault(sensor.id, {})[index+1] = sensor.erx_upstrm[rt_no]
                        new_etx_upstrm.setdefault(sensor.id, {})[index+1] = sensor.etx_upstrm[rt_no]
                        new_erx_data.setdefault(  sensor.id, {})[index+1] = sensor.erx_data[rt_no]
                        new_etx_data.setdefault(  sensor.id, {})[index+1] = sensor.etx_data[rt_no]
                        new_erx_etx_no_downstrm.setdefault(           sensor.id, {})[index+1] = sensor.erx_etx_no_downstrm[rt_no]
                        new_erx_etx_downstrm_upstrm.setdefault(       sensor.id, {})[index+1] = sensor.erx_etx_downstrm_upstrm[rt_no]
                        new_erx_etx_no_downstrm_no_upstrm.setdefault(sensor.id, {})[index+1] = sensor.erx_etx_no_downstrm_no_upstrm[rt_no]
                    new_sorted_P.append(P[rt_no-1])

                # Rebuild RT DataFrame
                self.RTs = self.build_RTs(new_sorted_P)

                # Update sensors_info_df and pull the newly cached metrics into each sensor
                self.controller.sensors_info_df = self.controller.get_sensors_info_df(self.RTs, self.controller.sensors)
                for sensor in self.controller.sensors.values():
                    sensor.erx_upstrm                    = new_erx_upstrm[  sensor.id]
                    sensor.etx_upstrm                    = new_etx_upstrm[  sensor.id]
                    sensor.erx_data                      = new_erx_data[    sensor.id]
                    sensor.etx_data                      = new_etx_data[    sensor.id]
                    sensor.erx_etx_no_downstrm           = new_erx_etx_no_downstrm[           sensor.id]
                    sensor.erx_etx_downstrm_upstrm       = new_erx_etx_downstrm_upstrm[       sensor.id]
                    sensor.erx_etx_no_downstrm_no_upstrm = new_erx_etx_no_downstrm_no_upstrm[ sensor.id]

                # Re‑number the rt_no field so it goes 1…population_size again
                new_arr = sorted_population[:population_size].copy()
                new_arr['rt_no'] = np.arange(1, len(new_arr) + 1)
                sorted_population = new_arr

                # Trim P back to exactly N individuals
                P = new_sorted_P[:population_size]

                # Check “same‑order” stopping criterion
                if old_P[:round(population_size * check_same_order_indi_percent)] == P[:round(population_size * check_same_order_indi_percent)]:
                    check_same_order_indi_counter += 1
                else:
                    check_same_order_indi_counter = 0

                if check_same_order_indi_counter >= max_time_same_order_indi_count:
                    break
        self.e_sorted_data = sorted_population # for log
        # print(self.e_sorted_data)
        # self.save_rts_topo()

        # for idx, individual in enumerate(P, start=1):
        #     self.print_matrix(individual, title=f"Individual {idx}")
        # pd.set_option('display.max_columns', None)
    # ===== GA ===== (T)