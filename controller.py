import numpy as np
import pandas as pd
import pickle
import os
import random
import time
from parent import SimParent


class Controller(SimParent):
    def __init__(self, node_id):
        super().__init__()
        self.id = node_id # The controller id
        self.E_elec = 50000 # Picojoule
        self.E_fs = 10 # Picojoule
        self.E_mp = 0.0013 # Picojoule

        print(f'E_elec: {self.E_elec} pJ')
        print(f'E_fs: {self.E_fs} pJ')
        print(f'E_mp: {self.E_mp} pJ')

        self.d0 = (self.E_fs/self.E_mp)**0.5
        print(f'd0: {self.d0}')
        self.energy_cache = {}

    def timeit(func):  # Decorator function
        def wrapper(*args, **kwargs):  # Wraps the original function
            start = time.perf_counter()
            result = func(*args, **kwargs)  # Calls the original function
            print(f"{func.__name__} took {time.perf_counter() - start:.6f} seconds")
            return result  # Returns original function's output
        return wrapper  # Returns the wrapped function
    
    def cache_find_path(self):
        sensors_info_df = pd.DataFrame()
        cache_path = f"{self.cache_dir}/paths/{self.RTs.shape[1]}_{self.init_num_nodes}_sensors"

        if not os.path.exists(f"{self.cache_dir}/paths"):
            os.makedirs(f"{self.cache_dir}/paths")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as file:
                sensors_info_df = pickle.load(file)
        else:
            for rt_no in self.RTs:
                paths = self.find_paths(rt_no)
                sensors_info_df = pd.concat([sensors_info_df, paths], ignore_index=1)

            with open(cache_path, "wb") as file:
                pickle.dump(sensors_info_df, file)

        self.sensors_info_df = sensors_info_df

    def find_e_use_bottleneck(self, RTs, n):
        "Sort energy use each route [e_max-min] -> [e_sum]"
        e_list = []
        for rt_no in RTs.columns:
            rt_int = rt_no
            energies = []
            for node in self.bottlenecks:
                energy_val = self.sensors[node].erx_etx_no_downstrm.get(rt_int, None)
                if energy_val is not None:
                    energies.append(energy_val)
            if not energies:
                continue
            e_max_min = max(energies) - min(energies)
            e_sum = sum(energies)
            e_list.append((rt_int, e_max_min, e_sum))
        e_data_type = [('rt_no', 'i4'), ('max-min', np.longdouble), ('sum', np.longdouble)]
        e_data = np.array([e_list], dtype=e_data_type)
        e_sorted_data = np.sort(e_data, order=['max-min', 'sum'])
        self.e_sorted_data = e_sorted_data # for log
        self.RTs.columns = self.RTs.columns.astype(int)
        self.RTs = self.RTs.loc[:, self.RTs.columns.isin(e_sorted_data['rt_no'].astype(int).flatten().tolist()[:n])]
        self.sensors_info_df = self.sensors_info_df[self.sensors_info_df['RT-No.'].isin(e_sorted_data['rt_no'].flatten().tolist()[:n])]
        return e_sorted_data['rt_no'].flatten().tolist()[:n]

    # @timeit
    def setup_sim_info(self, all_weights):
        weight_splited = all_weights.split(', ') # Split each weight parameter by ','.
        self.dist_w = float(weight_splited[0]) # Weight of distance to the controller.
        self.hop_w = float(weight_splited[1]) # Weight of hop count to the controller parameter.
        self.energy_rem_w = float(weight_splited[2]) # Weight of energy remain parameter.
        self.p_ack_w = 0 # Weight of packet acknowledgement parameter (P_ack) !ASSUMPTIOIN: we consider their are no loss in the network (perfect communication).

        self.hop_max = len(self.sensors) # The worst path (traval to every node before controller).

        # Calculate informations of each node of each route and store them into sensors_info_df. We can use cache mode with file named 'CACHE/paths'.
        if self.cache_mode == 'n': # If we not use cache mode
            sensors_info_df = pd.DataFrame() # create sensors_info_df as dataframe.
            # Loop into each RT (all possible route).
            for rt_no in self.RTs:
                paths = self.find_paths(rt_no) # Get informations about each node in the route.
                sensors_info_df = pd.concat([sensors_info_df, paths], ignore_index=1) # Store informations [paths] into sensors_info_df.
            self.sensors_info_df = sensors_info_df # Store sensor_info_df into the controller attribute.
        else: # If we use cahce mode
            self.cache_find_path() # Call method cache_find_path() and dump/get data from cache file in folder name paths.
        # Calculate distance, hop, p_ack of each RT
        self.sensors_info_df[['NodeID', 'Paths', 'RT-No.']].apply(lambda x: self.cal_dist_hop_p_ack_ratio(x['NodeID'], x['Paths'], x['RT-No.']), axis=1)
        # Calculate energy use in each RT
        self.cal_E(self.sensors_info_df)

        # Calculate the best effort route (every node send data to the controller directly).
        self.cal_dist_hop_p_ack_ratio_best_effort()
        # Calculate the best effort route's energy use.
        self.cal_erx_etx_best_effort()

    # @timeit
    def cal_dist_hop_p_ack_ratio(self, node_id, paths, rt_no):
        ##################### Distance Ratio Calculation #####################
        max_dist = 0 # Distance from node to the controller.
        nbr_dist = 0 # Distance from neighbor node to the controller.

        # Calculate each node of each route distance ratio.
        if len(paths) == 2: # If the node of the route use single to the controller.
            dist_ratio = 1 # Give single hop's value to 0 or 1 !NOTE ASSUMPTION we give this value to 1.
        else:
            # Add distance to the neighbor
            for d in range(len(paths)-1):
                node = paths[d]
                fwd = paths[d+1]
                dist = np.float64(int(self.sensors[node].neighbors.loc[self.sensors[node].neighbors['Neighbors']==fwd]['Distances'].iloc[0]))
                if d > 0:
                    nbr_dist += dist
                max_dist += dist
            dist_ratio = nbr_dist / max_dist # '1'->[2->C] / ['1'->2->C]
        ########################################################################

        ##################### Hop Count Ratio Calculation #####################
        #  Calculate each node of each route hop ratio.
        if len(paths) == 2: # If the node of the route use single to the controller.
            hop_ratio = 1 # Give single hop's value to 0 or 1 !NOTE ASSUMPTION we give this value to 1.
        else:
            hop_count = len(paths)-2 # Hop from nbr to the controller Ex. '1'->[2->C] = 2.
            # We can use difference hop reward mode 1-H_j or H_j.
            if self.hop_reward_cal == '1-Hj':
                hop_ratio = 1-(hop_count / self.hop_max)
            elif self.hop_reward_cal == 'Hj':
                hop_ratio = hop_count / self.hop_max
        ########################################################################

        ##################### Packet Ack Ratio Calculation #####################
        # !NOTE this part is not really important for now. So, I don't write any commend.
        # p_ack = 0
        # p_send = len(paths) - 1
        # for p in range(len(paths)-1):
        #     node = paths[p]
        #     fwd = paths[p+1]
        #     p_acki = float(self.sensors[node].loss.loc[self.sensors[node].loss['Neighbors']==fwd]['lines'].iloc[0])
        #     p_ack += p_acki
        # p_ack_ratio = p_ack / p_send
        p_ack_ratio = 0 # NOTE for test new topo
        ########################################################################

        # Store calculated data into each node of each route.
        dist_ratio_w = dist_ratio * self.dist_w # Distance ratio
        hop_ratio_w = hop_ratio * self.hop_w # Hop ratio
        p_ack_ratio_w = p_ack_ratio * self.p_ack_w # P_ack ratio
        self.sensors[node_id].dist_ratio_w_dict[rt_no] = dist_ratio_w
        self.sensors[node_id].hop_ratio_w_dict[rt_no] = hop_ratio_w
        self.sensors[node_id].p_ack_ratio_w_dict[rt_no] = p_ack_ratio_w
        self.sensors[node_id].sum_dist_hop_p_ack_w_dict[rt_no] = dist_ratio_w + hop_ratio_w + p_ack_ratio_w # Pre calculate sum of ratios for faster process.
        self.sensors[node_id].sum_dist_hop_p_ack_w_arr = np.append(self.sensors[node_id].sum_dist_hop_p_ack_w_arr, dist_ratio_w + hop_ratio_w + p_ack_ratio_w)

    def cal_E(self, sensors_info_df):
        grouped = sensors_info_df.groupby('RT-No.')
        for _, row in sensors_info_df.iterrows():
            rt_no = row['RT-No.']
            node_id = row['NodeID']
            df_rt = grouped.get_group(rt_no)
            df_node = df_rt[df_rt['NodeID'] == node_id]
            # You can then call a refactored version of cal_erx_etx_core()
            self.cal_erx_etx(row, df_rt, df_node)
    
    # @timeit
    def cal_erx_etx(self, row, df_rt, df_node):
        node_id = row['NodeID']
        rt_no = row['RT-No.']
        erx_downstrm = self.downstrm_ctl_size * self.E_elec
        # Filter where this node appears in the path within its route
        filtered_df = df_rt[df_rt['Paths'].map(lambda path_list: node_id in path_list)]

        count_node = filtered_df['Paths'].count()
        
        # index_node = filtered_df['Paths'].apply(lambda path_list: path_list.index(node_id))
        
        dist2fwd = df_node['Dist2Fwd'].iloc[0]

        erx_upstrm, etx_upstrm = self.cal_e_ctl(count_node, dist2fwd)
        erx_data, etx_data = self.cal_e_data(count_node, dist2fwd)

        self.sensors[node_id].erx_downstrm[rt_no] = erx_downstrm
        self.sensors[node_id].erx_upstrm[rt_no] = erx_upstrm
        self.sensors[node_id].etx_upstrm[rt_no] = etx_upstrm
        self.sensors[node_id].erx_data[rt_no] = erx_data
        self.sensors[node_id].etx_data[rt_no] = etx_data

        self.sensors[node_id].erx_etx_downstrm_upstrm[rt_no] = erx_downstrm + erx_upstrm + etx_upstrm + erx_data + etx_data
        self.sensors[node_id].erx_etx_no_downstrm[rt_no] = erx_upstrm + etx_upstrm + erx_data + etx_data
        self.sensors[node_id].erx_etx_no_downstrm_no_upstrm[rt_no] = erx_data + etx_data

    def cal_e_ctl(self, count_node, dist2fwd, index_node=0):
        # Energy use for Rx for each node each route.
        erx_upstrm = (count_node-1) * self.upstrm_ctl_size * self.E_elec # (index_node.sum()) is the possition of the node. We do [pos * control packet size] (the node may receive control packet from none or multiple nodes).
        
        # Energy use for Rx for each node each route.
        if dist2fwd <= self.d0: # If distance from the node to the neighbor is lower than threashold (d0).
            # index_node.sum(lambda x: x+1) is the possition of the node + 1 (+1 is for transmission). We do [pos+1 * control packet size].
            etx_upstrm = count_node * ((self.upstrm_ctl_size * self.E_fs * (dist2fwd**2)) + (self.upstrm_ctl_size * self.E_elec))
        else: # If distance is higher than threashold (d0)
            etx_upstrm = count_node * ((self.upstrm_ctl_size * self.E_mp * (dist2fwd**4)) + (self.upstrm_ctl_size * self.E_elec))

        return erx_upstrm, etx_upstrm
    
    def cal_e_data(self, count_node, dist2fwd, index_node=0):
        ######### Energy comsumption due to receiving data packet #########
        erx_data = (count_node-1) * self.data_size * self.E_elec

        ######### Energy comsumption due to transmiting data packet #########
        if dist2fwd <= self.d0:
            etx_data = count_node * ((self.data_size * self.E_fs * (dist2fwd**2)) + (self.data_size * self.E_elec)) # Energy for transmission
        else:
            etx_data = count_node * ((self.data_size * self.E_mp * (dist2fwd**4)) + (self.data_size * self.E_elec)) # Energy for transmission
        
        return erx_data, etx_data
   
    # @timeit
    def find_paths(self, rt_no):
        """"""
        paths = list()
        for sensor in self.sensors.values():

            path = list()
            sensor_id = sensor.id
            path.append(sensor_id)
            while True:
                fwd = self.RTs.loc[sensor_id, rt_no]
                if fwd == 'C':
                    break
                else:
                    path.append(fwd)
                    sensor_id = fwd
            path.append('C')
            paths.append([sensor.id, rt_no, path[1], int(sensor.neighbors[sensor.neighbors['Neighbors']==path[1]]['Distances'].iloc[0]), path])

        return pd.DataFrame(paths, columns=['NodeID', 'RT-No.', 'FWD', 'Dist2Fwd', 'Paths'])

    def cal_dist_hop_p_ack_ratio_best_effort(self):
        for sensor in self.sensors.values():
            self.sensors[sensor.id].dist_ratio_w_dict[-1] = 0
            self.sensors[sensor.id].hop_ratio_w_dict[-1] = 0
            self.sensors[sensor.id].p_ack_ratio_w_dict[-1] = 0
            self.sensors[sensor.id].sum_dist_hop_p_ack_w_dict[-1] = 0
            
            self.sensors[sensor.id].sum_dist_hop_p_ack_w_arr = np.append(self.sensors[sensor.id].sum_dist_hop_p_ack_w_arr, 0)

    def cal_erx_etx_best_effort(self):
        for sensor in self.sensors.values():
            # Send own data only
            count_node = 1
            # Distance to the controller
            dist2fwd = int(sensor.neighbors[sensor.neighbors['Neighbors'] == 'C']['Distances'].iloc[0])
            # Start from self and do not forward any packet
            index_node = pd.Series(0)

            erx_downstrm = self.downstrm_ctl_size * self.E_elec
            erx_upstrm, etx_upstrm = self.cal_e_ctl(count_node, dist2fwd, index_node)
            erx_data, etx_data = self.cal_e_data(count_node, dist2fwd, index_node)

            self.sensors[sensor.id].erx_downstrm[-1] = erx_downstrm
            self.sensors[sensor.id].erx_upstrm[-1] = erx_upstrm
            self.sensors[sensor.id].etx_upstrm[-1] = etx_upstrm
            self.sensors[sensor.id].erx_data[-1] = erx_data
            self.sensors[sensor.id].etx_data[-1] = etx_data

            self.sensors[sensor.id].erx_etx_downstrm_upstrm[-1] = erx_downstrm + erx_upstrm + etx_upstrm + erx_data + etx_data
            self.sensors[sensor.id].erx_etx_no_downstrm[-1] = erx_upstrm + etx_upstrm + erx_data + etx_data
            self.sensors[sensor.id].erx_etx_no_downstrm_no_upstrm[-1] = erx_data + etx_data

    def cal_sensor_rt(self):
        """
            Precalculate useable and unuseable RT if any bottleneck node die.
            Inputs: None
            Outputs: None
            Callers: env.py
        """
        df = self.sensors_info_df
        die_rt_nos_all = np.array([])
        remain_rt_nos_all = np.array([])

        for sensor_id in self.bottlenecks:
            sensor = self.sensors[sensor_id]
            # die_rt_nos = df[df['FWD'].isin([sensor.id])]['RT-No.'].unique() # Find RT unuseable.
            die_rt_nos = [int(v) for v in df[df['FWD'].isin([sensor.id])]['RT-No.'].unique()]

            filtered_df = df[~df['RT-No.'].isin(die_rt_nos)]

            filtered_df = filtered_df[~filtered_df['NodeID'].isin([sensor.id])]
            remain_rt_nos = filtered_df['RT-No.'].unique().tolist() # Find RT useable.

            # die_rt_nos_all = np.append(die_rt_nos, die_rt_nos)
            # remain_rt_nos_all = np.append(remain_rt_nos_all, remain_rt_nos)

            # Store them to sensor
            sensor.die_rt_nos = set(die_rt_nos)
            sensor.remain_rt_nos = set(remain_rt_nos)
            # print('check cal_sensor_rt')
            # print(f'sensor_id:\n{sensor_id}')
            # print(f'die_rt_nos:\n{list(die_rt_nos)}')
            # print(f'remain_rt_nos:\n{list(remain_rt_nos)}')
            # print()
        print(f'check cal_sensor_rt node 2')
        print(f'num die_rt_nos: {len(die_rt_nos)}')

    def exclude_sensor_rt(self, die_node=[], round_dies=[]):
        """
            Exclude RT and dead node.
            Inputs: die_node, round_dies
            Outputs: None
            Callers: env.py
        """

        df = self.sensors_info_df
        die_rt_nos_all = np.array([])
        remain_rt_nos_all = np.array([])

        for sensor_id in round_dies:
            print(f'die sensor_id: {sensor_id}')
            sensor = self.sensors[sensor_id]
            die_rt_nos = df[df['FWD'].isin([sensor.id])]['RT-No.'].unique() # Find RT unuseable.
            # die_rt_nos = [int(v) for v in df[df['FWD'].isin([sensor.id])]['RT-No.'].unique()]

            filtered_df = df[~df['RT-No.'].isin(die_rt_nos)]

            filtered_df = filtered_df[~filtered_df['NodeID'].isin([sensor.id])]
            remain_rt_nos = filtered_df['RT-No.'].unique() # Find RT useable.

            die_rt_nos_all = np.append(die_rt_nos, die_rt_nos)
            remain_rt_nos_all = np.append(remain_rt_nos_all, remain_rt_nos)

            # Store them to sensor
            # sensor.die_rt_nos = set(die_rt_nos)
            # sensor.remain_rt_nos = set(remain_rt_nos)
        # print(f'check cal_sensor_rt node 2')
        # print(f'num die_rt_nos: {len(die_rt_nos)}')

        unique_arr = np.unique(remain_rt_nos_all)
        RTs = self.RTs[unique_arr].drop(round_dies)
        self.RTs = RTs
        self.sensors_info_df = filtered_df

    def cal_exclude_rt(self, die_node, unavailable_rt):
        """
            Calculate remaining and to be remove RT, and store them to bottleneck sensors.
            Inputs: None
            Outputs: None
            Callers: env.py
            Ex. prepare which RT can still use or will be remove if node 1 die.
        """
        # print('hi')
        df = self.sensors_info_df
        die_rt_nos_all = np.array([])
        remain_rt_nos_all = np.array([])
        for sensor_id in self.bottlenecks:
            if sensor_id not in die_node:
                sensor = self.sensors[sensor_id]
                die_rt_nos = df[df['FWD'].isin([sensor.id])]['RT-No.'].unique() # Find RT unuseable.
                die_rt_nos = [int(v) for v in df[df['FWD'].isin([sensor.id])]['RT-No.'].unique()]

                filtered_df = df[~df['RT-No.'].isin(die_rt_nos)]
                filtered_df = filtered_df[~filtered_df['NodeID'].isin([sensor.id])]

                # print(f'filtered_df:\n{filtered_df}')

                remain_rt_nos = filtered_df['RT-No.'].unique() # Find RT useable.

                die_rt_nos_all = np.append(die_rt_nos, die_rt_nos)
                remain_rt_nos_all = np.append(remain_rt_nos_all, remain_rt_nos)

                # Store them to sensor
                sensor.die_rt_nos = set(die_rt_nos)
                sensor.remain_rt_nos = set(remain_rt_nos)
        df = self.sensors_info_df
        remain_rt_nos = set(self.RTs.columns.tolist()).difference(unavailable_rt)
        filtered_df = df[df['RT-No.'].isin(remain_rt_nos)]
        unique_arr = np.unique(remain_rt_nos_all)
        RTs = self.RTs[unique_arr]
        self.RTs = RTs
        self.sensors_info_df = filtered_df
        # print(len(self.RTs.columns.tolist()))

    def recalculate_sensor(self, round_dies):
        df = self.sensors_info_df
        die_rt_nos = df[df['FWD'].isin(round_dies)]['RT-No.'].unique()
        filtered_df = df[~df['RT-No.'].isin(die_rt_nos)]

        filtered_df = filtered_df[~filtered_df['NodeID'].isin(round_dies)]
        remain_rt_nos = filtered_df['RT-No.'].unique()
        
        RTs = self.RTs[remain_rt_nos].drop(round_dies)
        self.RTs = RTs
        self.sensors_info_df = filtered_df
        # pd.set_option('display.max_columns', None)

        # pd.reset_option('display.max_columns')
        return RTs, remain_rt_nos
    
    def dfs(self, graph, node, visited, parent, result):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result.append((node, neighbor))
                self.dfs(graph, neighbor, visited, node, result)

    def build_spanning_tree(self, edges, root):
        graph = {}
        for edge in edges:
            u, v = edge.split('-')
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)

        visited = set()
        result = []
        self.dfs(graph, root, visited, None, result)
        formatted_result = []
        for edge in result:
            if edge[0] == root:
                formatted_result.append(edge[1] + '-' + edge[0])
            else:
                formatted_result.append('{}-{}'.format(int(edge[1]), int(edge[0])))
        formatted_result = sorted(formatted_result, key=lambda x: int(x.split('-')[0]))
        return formatted_result

    def find_loop(self, comb):
        nodes_list = []
        edges_list = []
        for edge in comb:
            a, b = edge.split('-')
            if len(nodes_list) == 0:
                nodes_list.append(set([a, b]))
                edges_list.append(edge)
                continue
            is_in_sublist_index = []
            for i, nodes_sublist in enumerate(nodes_list):
                if a in nodes_sublist and b in nodes_sublist:
                    return edges_list, nodes_list, True
                elif a in nodes_sublist or b in nodes_sublist:
                    is_in_sublist_index.append(i)
            is_in_sublist_index.sort(reverse=True)
            if len(is_in_sublist_index) == 0:
                nodes_list.append(set([a, b]))
                edges_list.append(edge)
            elif len(is_in_sublist_index) == 1:
                nodes_list[is_in_sublist_index[0]] = nodes_list[is_in_sublist_index[0]].union(set([a, b]))
                edges_list.append(edge)
            else:
                nodes_list.append(nodes_list[is_in_sublist_index[0]].union(nodes_list[is_in_sublist_index[1]]))
                nodes_list.pop(is_in_sublist_index[0])
                nodes_list.pop(is_in_sublist_index[1])
                edges_list.append(edge)
        return edges_list, nodes_list, False

    def get_rts(self, combs, num_sensors_ctrl):
        l = []
        for i, comb in enumerate(combs):
            edges_list, nodes_list, has_loop = self.find_loop(comb)
            nodes_list = [item for sublist in nodes_list for item in sublist]
            if not has_loop:
                l.append(edges_list)
        rts = []
        root = 'C'
        for i, edges_list in enumerate(l):
            spanning_tree = self.build_spanning_tree(edges_list, root)
            rts.append(spanning_tree)

        result = {i + 1: tuple(element.split('-')[-1] for element in sub_tuple) for i, sub_tuple in enumerate(rts)}
        df = pd.DataFrame(result, index=[str(i) for i in range(1, num_sensors_ctrl)], dtype='string')
        return df

    def find_paths_GA(self, rt_no, RTs):
        """find paths function for genetic algorithm"""
        paths = list()
        for sensor in self.sensors.values():
            path = list()
            sensor_id = sensor.id
            path.append(sensor_id)
            while True:
                fwd = RTs.loc[sensor_id, rt_no]
                if fwd == 'C':
                    break
                else:
                    path.append(fwd)
                    sensor_id = fwd
            path.append('C')
            paths.append([sensor.id, rt_no, path[1], int(sensor.neighbors[sensor.neighbors['Neighbors']==path[1]]['Distances'].iloc[0]), path])

        return pd.DataFrame(paths, columns=['NodeID', 'RT-No.', 'FWD', 'Dist2Fwd', 'Paths'])

    def print_matrix(self, matrix, title="Matrix"):
        """"""
        print(title + ":")
        for row in matrix:
            print(row)
        print()

    def generate_hollow_diagonal_matrix_with_connectivity(self, connectivity, n):
        M = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(1, n):
            valid_columns = [j for j in range(i) if connectivity[i][j] == 1]
            if not valid_columns:
                raise ValueError(f"No valid candidate for row {i}.")
            chosen_col = random.choice(valid_columns)
            M[i][chosen_col] = 1
            M[chosen_col][i] = 1
        return M

    def generate_multiple_matrices_with_connectivity(self, count, num_sensors_ctrl, connectivity):
        if connectivity is None:
            raise ValueError("A connectivity matrix must be provided.")
        population = []
        seen_routes = set()
        while len(population) < count:
            # print(len(population))
            candidate = self.generate_hollow_diagonal_matrix_with_connectivity(connectivity, num_sensors_ctrl)
            route = tuple(self.extract_forwarders(candidate, num_sensors_ctrl))
            if route in seen_routes:
                # print(f"Duplicate route found, not adding candidate: {route}")
                continue
            seen_routes.add(route)
            population.append(candidate)
        return population
    
    # @timeit
    def generate_all_forward_routes(self, connectivity, num_sensors_ctrl):
        """
        Generate all possible route matrices where each node only forwards to a controller or a lower-indexed node,
        respecting the provided connectivity matrix. Returns a list of hollow-diagonal symmetric matrices.
        """
        n = num_sensors_ctrl
        # Initialize an empty matrix template
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        all_routes = []

        def backtrack(i):
            # When i reaches n, we've assigned routes for all nodes
            if i == n:
                # At least one node must connect directly to controller (row 1..n-1 has a 1 in column 0)
                if any(matrix[row][0] == 1 for row in range(1, n)):
                    # Deep copy the matrix
                    all_routes.append([row[:] for row in matrix])
                return

            # For node i, pick exactly one forward connection j < i where connectivity allows
            for j in range(i):
                if connectivity[i][j] == 1:
                    # Set the symmetric link
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                    backtrack(i + 1)
                    # Unset for backtracking
                    matrix[i][j] = 0
                    matrix[j][i] = 0

        # Start backtracking from node 1 (node 0 is the controller)
        backtrack(1)
        return all_routes

    def extract_forwarders(self, matrix, num_sensors_ctrl):
        forwards = []
        for i in range(1, num_sensors_ctrl):
            for j in range(i):
                if matrix[i][j] == 1:
                    forwards.append('C' if j == 0 else str(j))
                    break
        return forwards
     
    def get_sensors_info_df(self, RTs, sensors):
        sensor_info_list = []
        for rt_no in RTs.columns:
            paths_df = self.find_paths_GA(rt_no, RTs)
            sensor_info_list.append(paths_df)
        if sensor_info_list:
            return pd.concat(sensor_info_list, ignore_index=True)
        else:
            # Return an empty DataFrame or handle the case appropriately.
            return pd.DataFrame()

    def fitness_function(self, RTs, sensors, sensors_info_df, n):
        e_list = []
        for rt_no in RTs.columns:
            rt_int = rt_no
            energies = []
            for node in self.bottlenecks:
                energy_val = sensors[node].erx_etx_no_downstrm.get(rt_int, None)
                if energy_val is not None:
                    energies.append(energy_val)
            if not energies:
                continue
            e_max_min = max(energies) - min(energies)
            e_sum = sum(energies)
            e_list.append((rt_int, e_max_min, e_sum))
        e_data_type = [('rt_no', 'i4'), ('max_min', np.longdouble), ('sum', np.longdouble)]
        e_data = np.array(e_list, dtype=e_data_type)
        sorted_population = np.sort(e_data, order=['max_min', 'sum'])
        return sorted_population

    def cal_E_GA(self, sensors_info_df, population, cache_pop_set, cache_fitness):
        grouped = sensors_info_df.groupby('RT-No.')
        for _, row in sensors_info_df.iterrows():
            rt_no   = int(row['RT-No.'])
            node_id = row['NodeID']
            raw_path = row['Paths']
            path_key = tuple(raw_path) if isinstance(raw_path, (list, tuple)) else raw_path
            cache_key = (tuple(map(tuple, population[rt_no-1])), node_id)

            sensor = self.sensors[node_id]
            if cache_key in self.energy_cache:
                # Populate sensor dicts at this rt_no from cache
                m = self.energy_cache[cache_key]
                sensor.erx_downstrm[rt_no]                  = m['erx_downstrm']
                sensor.erx_upstrm[rt_no]                    = m['erx_upstrm']
                sensor.etx_upstrm[rt_no]                    = m['etx_upstrm']
                sensor.erx_data[rt_no]                      = m['erx_data']
                sensor.etx_data[rt_no]                      = m['etx_data']
                sensor.erx_etx_downstrm_upstrm[rt_no]       = m['erx_etx_downstrm_upstrm']
                sensor.erx_etx_no_downstrm[rt_no]           = m['erx_etx_no_downstrm']
                sensor.erx_etx_no_downstrm_no_upstrm[rt_no] = m['erx_etx_no_downstrm_no_upstrm']
                continue
            # Otherwise compute and cache:
            df_rt   = grouped.get_group(str(rt_no))
            df_node = df_rt[df_rt['NodeID'] == node_id]
            self.cal_erx_etx(row, df_rt, df_node)
            self.energy_cache[cache_key] = {
                'erx_downstrm':                  sensor.erx_downstrm[rt_no],
                'erx_upstrm':                    sensor.erx_upstrm[rt_no],
                'etx_upstrm':                    sensor.etx_upstrm[rt_no],
                'erx_data':                      sensor.erx_data[rt_no],
                'etx_data':                      sensor.etx_data[rt_no],
                'erx_etx_downstrm_upstrm':       sensor.erx_etx_downstrm_upstrm[rt_no],
                'erx_etx_no_downstrm':           sensor.erx_etx_no_downstrm[rt_no],
                'erx_etx_no_downstrm_no_upstrm': sensor.erx_etx_no_downstrm_no_upstrm[rt_no],
            }

    def GA_selection(self, population_size, sorted_population, parent_size, tournament_size, max_attempts_sel):
        paired_set = set()
        selected_parents = []
        for _ in range(population_size):
            paired_parents = self.selection_function_temp(sorted_population, parent_size, tournament_size, max_attempts_sel, paired_set) # Parent can't be paired
            # paired_parents = self.selection_function_temp(sorted_population, parent_size, tournament_size, max_attempts_sel) # Parent can be paired
            paired_set.add(paired_set.add(tuple(sorted(parent['rt_no'] for parent in paired_parents))))
            selected_parents.append(paired_parents)
        return selected_parents

    def GA_crossover(self, P, selected_parents, crossover_rate):
        gen_childs = []
        for parents in selected_parents:
            if random.uniform(0, 1) <= crossover_rate:
                child = self.crossover_function_temp(P, parents)
                if child not in gen_childs and child not in P:
                    gen_childs.append(child)
        return gen_childs

    def GA_mutation(self, C, populations, gen_childs, mutation_rate):
        """"""
        gen_mutates = []
        # Perform mutation only in parents.
        for individual in populations:
        # Perform mutation only in childs.
        # for individual in gen_childs:
            mutated_individual = self.mutation_function_temp(C, individual, mutation_rate)
            is_valid_mutated, txt_log = self.check_constraints(mutated_individual, C, populations, gen_childs, gen_mutates)
            if is_valid_mutated:
                gen_mutates.append(mutated_individual)

        return gen_mutates

    def selection_function_temp(self, sorted_population, parent_size, tournament_size, max_attempts, paired_set=set()):
        """
        Selects n candidate individuals from sorted_population using tournament selection.
        
        Args:
            sorted_population: a numpy structured array of individuals (each having fields such as 'rt_no' and 'max_min')
            n: the number of candidate individuals to select
            tournament_size: the number of individuals to compete in each tournament round
            
        Returns:
            A numpy structured array containing the selected candidate individuals.
        """
        attempt = 0
        epsilon = 1e-6  # small constant to avoid division by zero
        while attempt < max_attempts:
            candidate_pool = []
            available = list(sorted_population)  # copy of the population

            # Tournament selection for each parent
            while len(candidate_pool) < parent_size and available:
                current_tournament_size = min(tournament_size, len(available))
                # Tournament Selection
                indices = random.sample(range(len(available)), current_tournament_size)
                tournament_contestants = [available[i] for i in indices]

                # Roulette (Wheel) Selection
                weights = [1.0 / (contestant['max_min'] + epsilon) for contestant in tournament_contestants]
                selected_index = random.choices(range(current_tournament_size), weights=weights, k=1)[0]
                candidate = tournament_contestants[selected_index]
                candidate_pool.append(candidate)
                # Remove the selected candidate from available list to avoid selecting it twice in this round
                available.pop(indices[selected_index])
            # Create a sorted tuple of parent's identifiers (using 'rt_no') for consistency
            candidate_ids = tuple(sorted(parent['rt_no'] for parent in candidate_pool))
            if candidate_ids not in paired_set:
                return np.array(candidate_pool, dtype=sorted_population.dtype)
            attempt += 1

        return np.array(candidate_pool, dtype=sorted_population.dtype)
        
    def crossover_function_temp(self, populations, parents):
        # Retrieve parent IDs from the parents array.
        parent1_id = parents[0][0]
        parent2_id = parents[1][0]
        
        # Convert the candidate matrices from the populations list to numpy arrays.
        parent1_mtrx = np.array(populations[parent1_id - 1])
        parent2_mtrx = np.array(populations[parent2_id - 1])
        
        # Compute the XOR of the two parent matrices.
        parents_xor = np.bitwise_xor(parent1_mtrx, parent2_mtrx)
        
        n = parent1_mtrx.shape[0]
        # Initialize the child matrix with zeros.
        child = np.zeros_like(parent1_mtrx)
        
        # Iterate over each row (starting at 1, ignoring row 0)
        for i in range(1, n):
            # Find candidate columns in row i (only consider j < i)
            candidate_indices = [j for j in range(i) if parents_xor[i, j] == 1]
            if candidate_indices:
                # Randomly choose one candidate index from the positions where parents differ.
                chosen = np.random.choice(candidate_indices)
            else:
                # If parents agree on row i, try to use parent's value from parent1_mtrx.
                indices = np.where(parent1_mtrx[i, :i] == 1)[0]
                if indices.size > 0:
                    chosen = indices[0]
                else:
                    # Fallback: randomly choose an index among j < i.
                    chosen = np.random.choice(range(i))
            # Set the bit in the lower triangle.
            child[i, chosen] = 1
            # Mirror that bit to keep the matrix symmetric.
            child[chosen, i] = 1
            
        return np.array(child).tolist()

    def check_constraints(self, child, C, population, gen_childs, gen_mutates):
        """
        Checks that the candidate 'child' (a hollow-diagonal matrix) satisfies all constraints.
        
        Constraints:
        1. For each row i (from 1 to n-1), exactly one entry among indices 0..i-1 must be 1.
        2. Every row (0 <= i < n) must have at least one 1.
            (Note: For row 0, ones appear from the symmetric mirror of rows with their one in column 0.)
        3. If a row’s unique 1 (in row i, i>=1) is at column 0, then that row (as a string) must be in 
            the allowed set (here, allowed_rows = {'1','2','4'}).
        4. For each row i (from 1 to n-1), the chosen position j must be valid per the connectivity matrix C.
            (I.e. C[i][j] must equal 1.)
        5. At least one row (from 1 to n-1) must have its unique 1 in column 0 (so that row 0 gets at least one 1).
        6. The candidate must not duplicate any individual in the population.
        
        Parameters:
        child: a square matrix (list of lists) representing a candidate.
        C: connectivity matrix (list of lists) where C[i][j]==1 means a link is allowed.
        population: (optional) a list of candidate matrices to check duplication.
        
        Returns:
        A tuple (is_valid, message) where is_valid is True if all constraints pass.
        """
        n = len(child)
        allowed_rows = self.bottlenecks  # Only these rows (by id as strings) are allowed to have their unique 1 in column 0.

        # Constraint 1: For rows 1..n-1, exactly one 1 must appear in columns 0 .. i-1.
        for i in range(1, n):
            ones = [j for j in range(i) if child[i][j] == 1]
            if len(ones) != 1:
                return False, f"Row {i} has {len(ones)} ones (expected exactly 1)."
            j = ones[0]
            # Constraint 4: Check connectivity.
            if C[i][j] != 1:
                return False, f"Row {i}: position {j} is not allowed by connectivity matrix."
            # Constraint 3: If the one is in column 0, only allowed for rows in allowed_rows.
            if j == 0 and str(i) not in allowed_rows:
                return False, f"Row {i} has its one in column 0 but is not allowed (allowed rows: {allowed_rows})."

        # Constraint 2: Check that every row (including row 0) has at least one 1.
        # For row 0, the ones come from the symmetric update. So we check the sum of row 0.
        for i in range(n):
            if sum(child[i]) < 1:
                return False, f"Row {i} does not have at least one 1."

        # Constraint 5: Ensure that at least one row (from 1 to n-1) has its unique 1 in column 0.
        if not any(child[i][0] == 1 for i in range(1, n)):
            return False, "No row (i >= 1) has its one in column 0."

        # Constraint 6: Duplication check.
        if population is not None:
            if child in population or child in gen_childs or child in gen_mutates:
                return False, "Candidate duplicates an individual in the population."

        return True, "Candidate is valid."

    def mutation_function_temp(self, C, origin_individual, mutation_rate=0.01):
        """
        Mutates a hollow-diagonal matrix candidate 'individual'.
        
        For each row i (from 1 to n-1):
        1. Find the current index 'current_one' where the row has a 1.
        2. For each candidate j in range(i) (each bit in that row),
            check with probability mutation_rate whether to consider j.
            Only include j if j is not the current one and if C[i][j] == 1.
        3. If one or more candidates were flagged, pick one at random,
            and update the row by removing the old one and setting that candidate to 1.
            Mirror the change in the symmetric position.
        4. After processing all rows, perform validations:
            • Each row must have exactly one 1.
            • If the one is in column 0, only allowed for rows in a given set (here, {'1','2','4'}).
            • The chosen connection must be valid per C.
            • At least one row (except row 0) must have a 1 in column 0.
            • The candidate must not duplicate an individual in the population.
        
        Returns the mutated child if valid; otherwise, returns None.
        """
        import copy

        individual = copy.deepcopy(origin_individual)
        n = len(individual)
        for i in range(1, n):
            # Find the current one index in row i (only checking j from 0 to i-1)
            current_one = None
            for j in range(i):
                if individual[i][j] == 1:
                    current_one = j
                    break
            if current_one is None:
                return None  # Invalid: each row must have exactly one 1.
            
            if random.random() >= mutation_rate:
                # When current one not mutate
                pass
            else:
                # Check each bit in the row (from j = 0 to i-1)
                mutation_indices = []
                for j in range(i):
                    # Independently, each bit gets a chance to trigger mutation.
                    if random.random() < mutation_rate:
                        # We only consider j if it is not already the current one.
                        if j != current_one and C[i][j] == 1:
                            mutation_indices.append(j)
                # Mutate all index.
                if mutation_indices:
                    mutate_index = random.choice(mutation_indices)
                    # for mutate_index in mutation_indices:
                    #     # Remove the current one and set the new one (mirror the change).
                    individual[i][current_one] = 0
                    individual[current_one][i] = 0

                    individual[i][mutate_index] = 1
                    individual[mutate_index][i] = 1

        return individual
