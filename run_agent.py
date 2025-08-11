import numpy as np
import pandas as pd
from env import SdWsnEnv
from agent_brain import QLearningTable
from parent import SimParent
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime
import statistics
import pickle
from matplotlib.lines import Line2D
import time

# Network hyperparameters
DATA_SIZE = 800 # 100 bytes
UP_STRM_SIZE = DATA_SIZE*0
DOWN_STRM_SIZE = DATA_SIZE*0.5

# Initial INTERVAL
INTERVAL = 1
SIM_INDEX = 1
INT_CONVERTER = 1e12

# Datetime
DATETIME = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

class RunAgent(SimParent):
    def __inti__(self):
        super().__init__()

    # ===== Decorate ===== (H)
    def timeit(func):  # Decorator function
        def wrapper(*args, **kwargs):  # Wraps the original function
            start = time.perf_counter()
            result = func(*args, **kwargs)  # Calls the original function
            print(f"{func.__name__} took {time.perf_counter() - start:.6f} seconds")
            return result  # Returns original function's output
        return wrapper  # Returns the wrapped function
    # ===== Decorate ===== (T)

    # ===== LOG ===== (H)
    def setup_logger(self, log_directory='logs', log_filename='logfile.log', instance_name='sim_log', log_level=logging.DEBUG):
        # Create the directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

        # Create logger with the name '__main__'
        logger = logging.getLogger(instance_name)
        logger.setLevel(log_level)

        # Create file handler and set level to debug with 'w' mode
        log_file_path = os.path.join(log_directory, log_filename)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter('%(message)s')

        # Add formatter to handlers
        fh.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)

        return logger

    def log_main(self, main_logger, interval, state, action_, reward, next_state, env_info, energy_before_step, energy_after_step, q_action, e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm):
        main_logger.info(f'Interval: {interval}')
        main_logger.info(f'Current State: {state}')
        main_logger.info(f'Selected RT: {action_}')
        main_logger.info(f'Q: {q_action}')
        main_logger.info(f'Reward: {reward}')
        main_logger.info(f'State (After use RT): {next_state}\n')
        for sensor in env_info.values():
            main_logger.info(f'ID: {sensor.id}, E: {sensor.energy:,} pJ')

        main_logger.info(f'\nE use Rx data: {e_use_rx_data:,} pJ')
        main_logger.info(f'E use Tx data: {e_use_tx_data:,} pJ')
        main_logger.info(f'E use Rx upstream: {e_use_rx_upstrm:,} pJ')
        main_logger.info(f'E use Tx upstream: {e_use_tx_upstrm:,} pJ')
        main_logger.info(f'E use Rx downstream: {e_use_rx_downstrm:,} pJ')

        main_logger.info('\n#######################################################################')
        main_logger.info('#######################################################################\n')

    def log_rts(self, rts_logger, interval, state, action_, next_state, env_info, energy_before_step, energy_after_step, q_action, e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm):
        rts_logger.info(f'Interval: {interval}')
        rts_logger.info(f'Current State: {state}')
        rts_logger.info(f'Selected RT: {action_}')
        rts_logger.info(f'Q: {q_action}')
        # rts_logger.info(f'\nQ RT 1139: {rt_1139}')
        # rts_logger.info(f'Q RT 2761: {rt_2761}\n')
        rts_logger.info(f'State (After use RT): {next_state} pJ\n')

        rts_logger.info(f'E use Rx data: {e_use_rx_data:,} pJ')
        rts_logger.info(f'E use Tx data: {e_use_tx_data:,} pJ')
        rts_logger.info(f'E use Rx upstream: {e_use_rx_upstrm:,} pJ')
        rts_logger.info(f'E use Tx upstream: {e_use_tx_upstrm:,} pJ')
        rts_logger.info(f'E use Rx downstream: {e_use_rx_downstrm:,} pJ')

        rts_logger.info('\n')

    def log_first_node_die_main(self, main_logger, die_node, interval):
        """"""
        main_logger.info(f'First node die interval: {interval}\n')
        main_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

    def log_first_node_die_rts(self, rts_logger, die_node, interval):
        """"""
        rts_logger.info(f'First node die interval: {interval}\n')
        rts_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

    def log_half_node_die_main(self, main_logger, die_node, interval):
        main_logger.info(f'Half node die interval: {interval}\n')
        main_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

    def log_half_node_die_rts(self, rts_logger, die_node, interval):
        rts_logger.info(f'Half node die interval: {interval}\n')
        rts_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

    def log_first_info_to_rts_logger(self, rts_logger, num_state, interval, die_node, env, info_df, rt_period, num_change_rt):
        rts_logger.info(f'\nRT swap count: {num_change_rt}')

        for sensor in env.controller.sensors.values():
            rts_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

        rts_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
        rts_logger.info(f"List of used RTs{info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
        rts_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
        rts_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}\n')

    def log_half_info_to_rts_logger(self, rts_logger, num_state, interval, die_node, env, info_df, rt_period, num_change_rt):
        rts_logger.info(f'\nRT swap count: {num_change_rt}')

        for sensor in env.controller.sensors.values():
            rts_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

        rts_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
        rts_logger.info(f"List of used RTs{info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
        rts_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
        rts_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}\n')

    def log_end_info_to_rts_logger(self, rts_logger, num_state, interval, die_node, env, info_df, rt_period, first_node_die_interval, half_node_die_interval, last_node_die_interval, total_e_use_rx_data, total_e_use_tx_data, total_e_use_rx_upstrm, total_e_use_tx_upstrm, total_e_use_rx_downstrm, rt_use_interval, num_change_rt):
        rts_logger.info(f'\n#######################################################################')
        rts_logger.info('#######################################################################\n')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            rts_logger.info(f'Data Size: {DATA_SIZE} bytes')
            rts_logger.info(f'DownStream Size: {int(DOWN_STRM_SIZE)} bytes')
            rts_logger.info(f'UpStream Size: {int(UP_STRM_SIZE )} bytes')
            rts_logger.info(f'RT_period: {rt_period}\n')
            rts_logger.info(f'Number of states: {num_state}\n')
            rts_logger.info(f'Last node die interval: {interval}\n')
            rts_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

            rts_logger.info(f'\nRT swap count: {num_change_rt}')

            for sensor in env.controller.sensors.values():
                rts_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

            rts_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
            rts_logger.info(f"List of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
            rts_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
            rts_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}')

            rts_logger.info(f'\nTotal energy use Rx data: {total_e_use_rx_data} pJ')
            rts_logger.info(f'Total energy use Tx data: {total_e_use_tx_data} pJ')
            rts_logger.info(f'Total energy use Rx Upstream: {total_e_use_rx_upstrm} pJ')
            rts_logger.info(f'Total energy use Tx Upstream: {total_e_use_tx_upstrm} pJ')
            rts_logger.info(f'Total energy use Rx Downstream: {total_e_use_rx_downstrm} pJ\n')

            _, each_intervals = self.analyze_sequence(rt_use_interval)
            interval_stats_df = self.cal_interval_statistics(each_intervals)
            interval_stats_df['number'] = pd.Categorical(interval_stats_df['number'], categories=info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist(), ordered=True)
            interval_stats_df = interval_stats_df.sort_values('number')
            interval_stats_df = interval_stats_df.reset_index(drop=True)
            interval_stats_df['count'] = info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)['count'].tolist()

            # rts_logger.info(f"\nIntervals between reuses for each number:\n {each_intervals}\n")
            rts_logger.info(f"Interval statistics for each number:\n {interval_stats_df}")

            rts_logger.info(f'\nFND: {first_node_die_interval}')
            rts_logger.info(f'HND: {half_node_die_interval}')
            rts_logger.info(f'LND: {last_node_die_interval}')

    def log_first_info_to_main_logger(self, main_logger, num_state, interval, die_node, env, info_df, rt_period, num_change_rt):
        main_logger.info(f'\nRT swap count: {num_change_rt}')

        for sensor in env.controller.sensors.values():
            main_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

        main_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
        main_logger.info(f"List of used RTs{info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
        main_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
        main_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}\n')

    def log_half_info_to_main_logger(self, main_logger, num_state, interval, die_node, env, info_df, rt_period, num_change_rt):
        main_logger.info(f'\nRT swap count: {num_change_rt}')

        for sensor in env.controller.sensors.values():
            main_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

        main_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
        main_logger.info(f"List of used RTs{info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
        main_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
        main_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}\n')

    def log_end_info_to_main_logger(self, main_logger, num_state, interval, die_node, env, info_df, rt_period, first_node_die_interval, half_node_die_interval, last_node_die_interval, total_e_use_rx_data, total_e_use_tx_data, total_e_use_rx_upstrm, total_e_use_tx_upstrm, total_e_use_rx_downstrm, rt_use_interval, num_change_rt):
        main_logger.info(f'\n#######################################################################')
        main_logger.info('#######################################################################\n')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            main_logger.info(f'Data Size: {DATA_SIZE} bytes')
            main_logger.info(f'DownStream Size: {int(DOWN_STRM_SIZE)} bytes')
            main_logger.info(f'UpStream Size: {int(UP_STRM_SIZE )} bytes')
            main_logger.info(f'RT_period: {rt_period}\n')
            main_logger.info(f'Number of states: {num_state}\n')
            main_logger.info(f'Last node die interval: {interval}\n')
            main_logger.info(f'Dead node[s]: {", ".join(x for x in die_node)}')

            main_logger.info(f'\nRT swap count: {num_change_rt}')

            for sensor in env.controller.sensors.values():
                main_logger.info(f'Node ID: {sensor.id}, Energy: {sensor.energy} pJ')

            main_logger.info(f"\nNumber of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().shape[0]}")
            main_logger.info(f"List of used RTs: {info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist()}\n")
            main_logger.info(f'Reward Summary:\n{info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)}\n')
            main_logger.info(f'Q Summary:\n{info_df.groupby(by="Slcted_RT")["Q"].describe().sort_values(by="count", ascending=False)}')

            main_logger.info(f'\nTotal energy use Rx data: {total_e_use_rx_data} pJ')
            main_logger.info(f'Total energy use Tx data: {total_e_use_tx_data} pJ')
            main_logger.info(f'Total energy use Rx Upstream: {total_e_use_rx_upstrm} pJ')
            main_logger.info(f'Total energy use Tx Upstream: {total_e_use_tx_upstrm} pJ')
            main_logger.info(f'Total energy use Rx Downstream: {total_e_use_rx_downstrm} pJ\n')

            _, each_intervals = self.analyze_sequence(rt_use_interval)
            interval_stats_df = self.cal_interval_statistics(each_intervals)
            interval_stats_df['number'] = pd.Categorical(interval_stats_df['number'], categories=info_df.groupby(by='Slcted_RT')['Reward'].describe().sort_values(by='count', ascending=False).index.tolist(), ordered=True)
            interval_stats_df = interval_stats_df.sort_values('number')
            interval_stats_df = interval_stats_df.reset_index(drop=True)
            interval_stats_df['count'] = info_df.groupby(by="Slcted_RT")["Reward"].describe().sort_values(by="count", ascending=False)['count'].tolist()

            # main_logger.info(f"\nIntervals between reuses for each number:\n {each_intervals}\n")
            main_logger.info(f"Interval statistics for each number:\n {interval_stats_df}")

            main_logger.info(f'\nFND: {first_node_die_interval}')
            main_logger.info(f'HND: {half_node_die_interval}')
            main_logger.info(f'LND: {last_node_die_interval}')
    # ===== LOG ===== (T)

    # ===== save, show graph ====== (H)
    def shows_graphs(self, info_df, num_state):
        fig1, ax1 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='Slcted_RT', s=1, ax=ax1)
        ax1.set_title(f'Action by Round ({reward_method}, {num_state} states)')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Action')
        ax1.grid(False)
        plt.figure(fig1.number)
        plt.show()
        plt.close()
        
        fig2, ax2 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='State', s=1, ax=ax2)
        ax2.set_title(f'State by Round ({reward_method}, {num_state} states)')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('State')
        ax2.grid(False)
        plt.figure(fig2.number)
        plt.show()
        plt.close()

    def save_graphs(self, info_df, num_state, rt_period, sim_log_dir):
        fig1, ax1 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='Slcted_RT', s=1, ax=ax1)
        ax1.set_title(f'Action by Round ({reward_method}{q}, {num_state}States)')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Action')
        ax1.grid(False)
        plt.figure(fig1.number)
        plt.savefig(f'{sim_log_dir}/{reward_method}, {rt_period}RT Period, {num_state}s, Action_plot.png', bbox_inches='tight')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='State', s=1, ax=ax2)
        ax2.set_title(f'State by Round ({reward_method}{q}, {num_state}States)')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('State')
        ax2.grid(False)
        plt.figure(fig2.number)
        plt.savefig(f'{sim_log_dir}/{reward_method}, {rt_period}RT Period, {num_state}s, State_plot.png', bbox_inches='tight')
        plt.close()

        fig3, ax3 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='Min_E', s=1, ax=ax3)
        ax3.set_title(f'Min Energy by Round')
        ax3.set_xlabel('Min Energy')
        ax3.set_ylabel('State')
        ax3.grid(False)
        plt.figure(fig3.number)
        plt.savefig(f'{sim_log_dir}/Min Energy by Round.png', bbox_inches='tight')
        plt.close()

        fig4, ax4 = plt.subplots()
        info_df.plot(kind='scatter', x='Interval', y='Max_E', s=1, ax=ax4)
        ax4.set_title(f'Max Energy by Round')
        ax4.set_xlabel('Max Energy')
        ax4.set_ylabel('State')
        ax4.grid(False)
        plt.figure(fig4.number)
        plt.savefig(f'{sim_log_dir}/Max Energy by Round.png', bbox_inches='tight')
        plt.close()
    # ===== save, show graph ====== (T)

    # ===== calculate, analyze, summarize statistic ===== (H)
    def store_info(self, info_arr, interval, action_, state_, reward, q, min_e, max_e):
        info_arr = np.append(info_arr, np.array([interval, action_, state_, reward, q, min_e, max_e]))

        return info_arr

    def summarize_info(self, info_arr):
        info_arr = info_arr.reshape(-1, 7)
        info_df = pd.DataFrame(info_arr, columns=['Interval', 'Slcted_RT', 'State', 'Reward', 'Q', 'Min_E', 'Max_E'])
        info_df = info_df.astype({'Interval': np.int64, 'Slcted_RT': np.int64, 'State': np.float64, 'Reward': np.float64, 'Q': np.float64})

        return info_df

    def analyze_sequence(self, seq):
        # Dictionary to store the frequency of each number
        frequency = {}
        # Dictionary to store the last index seen for each number
        last_seen = {}
        # Dictionary to store the time intervals for each number
        intervals = {}
        
        for i, num in enumerate(seq):
            # Update the frequency
            if num not in frequency:
                frequency[num] = 0
                intervals[num] = []
            frequency[num] += 1
            
            # Calculate the interval since last seen
            if num in last_seen:
                interval = i - last_seen[num]
                intervals[num].append(interval)
            
            # Update the last seen index
            last_seen[num] = i
        
        return frequency, intervals

    def cal_e_use_sensors(self, env_info, bottlenecks):
        # for sensor in env_info.values():
        #     if sensor.id in bottlenecks:
        #         print(sensor.id)
        #         print(sensor.e_use_rx_data)
        #         print(sensor.e_use_tx_data)
        #         print(sensor.e_use_rx_downstrm)
        e_use_rx_data = sum([sensor.e_use_rx_data for sensor in env_info.values() if sensor.id in bottlenecks])
        e_use_tx_data = sum([sensor.e_use_tx_data for sensor in env_info.values() if sensor.id in bottlenecks])
        e_use_rx_upstrm = sum([sensor.e_use_rx_upstrm for sensor in env_info.values() if sensor.id in bottlenecks])
        e_use_tx_upstrm = sum([sensor.e_use_tx_upstrm for sensor in env_info.values() if sensor.id in bottlenecks])
        e_use_rx_downstrm = sum([sensor.e_use_rx_downstrm for sensor in env_info.values() if sensor.id in bottlenecks])

        return e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm

    def cal_interval_statistics(self, intervals):
        stats = {
            'number': [],
            'average': [],
            'max': [],
            'min': [],
            'std_dev': []
        }
        
        for num, interval_list in intervals.items():
            stats['number'].append(num)
            if interval_list:
                stats['average'].append(statistics.mean(interval_list))
                stats['max'].append(max(interval_list))
                stats['min'].append(min(interval_list))
                stats['std_dev'].append(statistics.stdev(interval_list) if len(interval_list) > 1 else 0)
            else:
                stats['average'].append(None)
                stats['max'].append(None)
                stats['min'].append(None)
                stats['std_dev'].append(None)
        
        return pd.DataFrame(stats)
    # ===== calculate, analyze, summarize statistic ===== (T)

    # ===== Sim (main function) ===== (H)
    def sim(self, decimal, reward_method, q, is_init_q, is_log, is_chart, rt_period, state_method, weights, log_type, hop_reward_cal, alpha, gamma, rt_num, cache_mode, num_sensors, topo_range_x, topo_range_y, get_rt_method, create_topo, sim_no, sim_log_dir):
        global SIM_INDEX
        info_arr = np.array([])
        interval = INTERVAL
        env = SdWsnEnv(int(decimal), state_method, reward_method, weights, hop_reward_cal, rt_num, cache_mode, DATA_SIZE, UP_STRM_SIZE, DOWN_STRM_SIZE, rt_period, num_sensors, topo_range_x, topo_range_y, get_rt_method, create_topo, sim_no)
        # dtime = datetime.now().strftime("%d-%m-%y-%H:%M:%S")
        # env.dtime = dtime
        env.dtime = DATETIME 
        RL = QLearningTable(decimal, alpha, gamma)
        state, d_ratios, h_ratios, rts_list = env.reset()
        num_state = RL.setup_agent(env.controller, is_init_q, d_ratios, h_ratios, state_method, rts_list)

        # Setup directory and logger
        if is_log or is_chart:
            # # Create log directory
            # if not os.path.exists(sim_log_dir):
            #     os.makedirs(sim_log_dir)

            # Setup logger
            main_logger = self.setup_logger(log_directory=sim_log_dir, log_filename=f'{sim_no}_{get_rt_method}_Full_log', instance_name=f'sim_log_{SIM_INDEX}', log_level=logging.DEBUG)
            rts_logger = self.setup_logger(log_directory=sim_log_dir, log_filename=f'{sim_no}_{get_rt_method}_RT_only_log', instance_name=f'rts_log_{SIM_INDEX}', log_level=logging.DEBUG)
            
            SIM_INDEX += 1
            
            # LOG
            rts_logger.info(f'Q-table:\n{RL.q_table}\n')

        # Print number of state
        # print(f'Number of state: {num_state}')

        is_send_direct = False
        is_first_node_die = False
        is_half_node_die = False

        ##### STATIC (FUTURE WORK)!! #####
        # bottle_necks = ['1', '2', '3']
        # env.bottle_necks = bottle_necks
        ##################################

        total_e_use_rx_data = 0
        total_e_use_tx_data = 0
        total_e_use_rx_upstrm = 0
        total_e_use_tx_upstrm = 0
        total_e_use_rx_downstrm = 0

        rt_use_interval = []
        er_list_1 = []
        er_list_2 = []
        er_list_3 = []
        num_alive_nodes = []

        num_change_rt = 0
        is_node_die = False

        while True:
            print(f'interval: {interval}')
            while True:
                # print(interval)
                # Choose the RT that let sensors send data to the controller directly
                if is_send_direct:
                    action_ = -1
                # Check if this is the first round or is this is RT period round
                elif state_method == 'rt' and interval == 1:
                    action_, q_action, done, die_node, is_node_die, round_dies = RL.choose_action(state, env, interval)
                    state = action_
                elif interval == 1 or interval % rt_period == 0:
                    action_, q_action, done, die_node, is_node_die, round_dies = RL.choose_action(state, env)

                if not is_node_die:

                    break

            # Check if the current action is the same as the previous one
            if interval == 1 or action != action_:
                env.same_rt = False
                num_change_rt += 1
            elif action == action_:
                env.same_rt = True
            
            # If the simulation is end
            if done:
                print(f'lnd: {interval-1}')
                num_alive_nodes.append(0)
                info_df = self.summarize_info(info_arr)
                last_node_die_interval = interval-1
                if is_first_node_die == False:
                    first_node_die_interval = interval-1
                if is_half_node_die == False:
                    half_node_die_interval = interval-1
                # Log simulation results to main and RTs logger
                if is_log:
                    self.log_end_info_to_main_logger(main_logger, num_state, interval-1, die_node, env, info_df, rt_period, first_node_die_interval, half_node_die_interval, last_node_die_interval, total_e_use_rx_data, total_e_use_tx_data, total_e_use_rx_upstrm, total_e_use_tx_upstrm, total_e_use_rx_downstrm, rt_use_interval, num_change_rt)
                if is_log or is_chart:
                    self.log_end_info_to_rts_logger(rts_logger, num_state, interval-1, die_node, env, info_df, rt_period, first_node_die_interval, half_node_die_interval, last_node_die_interval, total_e_use_rx_data, total_e_use_tx_data, total_e_use_rx_upstrm, total_e_use_tx_upstrm, total_e_use_rx_downstrm, rt_use_interval, num_change_rt)
                # print(f'choose_action took {np.mean(np.array([RL.timeit]))} seconds (average)')
                return first_node_die_interval, half_node_die_interval, last_node_die_interval, er_list_1, er_list_2, er_list_3, num_alive_nodes, sim_log_dir, num_change_rt, env, RL
            
            # If their are sensors run out of batt this round
            if len(round_dies) > 0:
                if not is_first_node_die:
                    print(f'fnd: {interval-1}')
                    info_df = self.summarize_info(info_arr)
                    first_node_die_interval = interval-1
                    is_first_node_die = True

                    if is_log:
                        self.log_first_node_die_main(main_logger, die_node, interval-1)
                        self.log_first_info_to_main_logger(main_logger, num_state, interval-1, die_node, env, info_df, rt_period, num_change_rt)
                    if is_log or is_chart:
                        self.log_first_node_die_rts(rts_logger, die_node, interval-1)
                        self.log_first_info_to_rts_logger(rts_logger, num_state, interval-1, die_node, env, info_df, rt_period, num_change_rt)
                    # print(f'First node die interval: {interval}')
                    # print(f'Dead node[s]: {", ".join(x for x in die_node)}\n')
                elif len(set(die_node).intersection(set(bottlenecks))) == 2:
                    print(f'hnd: {interval-1}')
                    is_half_node_die = True
                    info_df = self.summarize_info(info_arr)
                    half_node_die_interval = interval-1
                    if is_log:
                        self.log_half_node_die_main(main_logger, die_node, interval-1)
                        self.log_half_info_to_main_logger(main_logger, num_state, interval-1, die_node, env, info_df, rt_period, num_change_rt)
                    if is_log or is_chart:
                        self.log_half_node_die_rts(rts_logger, die_node, interval-1)
                        self.log_half_info_to_rts_logger(rts_logger, num_state, interval-1, die_node, env, info_df, rt_period, num_change_rt)

            # The environment performs the selected action
            next_state, reward, done, env_info, die_node, round_dies, energy_before_step, energy_after_step, bottlenecks = env.step(action_)

            num_alive_nodes.append(sum(1 for key in ['1', '2', '4'] if key in env.controller.sensors))
            # print(num_alive_nodes)

            # Get Emax and Emin
            min_current_energy_sensor = min(env.controller.sensors.values(), key=lambda sensors: sensors.energy_rem_try).energy_rem_try
            max_current_energy_sensor = max(env.controller.sensors.values(), key=lambda sensors: sensors.energy_rem_try).energy_rem_try
            
            s_1 = env.controller.sensors.get('1', None)
            s_2 = env.controller.sensors.get('2', None)
            s_3 = env.controller.sensors.get('4', None)

            if s_1 is not None:
                er_list_1.append(s_1.energy*1e-12)
            else:
                er_list_1.append(0)  # Or handle it as needed
            if s_2 is not None:
                er_list_2.append(s_2.energy*1e-12)
            else:
                er_list_2.append(0)
            if s_3 is not None:
                er_list_3.append(s_3.energy*1e-12)
            else:
                er_list_3.append(0) 

            # Store info into array
            info_arr = self.store_info(info_arr, interval, action_, state, reward, q_action, min_current_energy_sensor, max_current_energy_sensor)

            # Cal energy used of sensors for logging
            e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm = self.cal_e_use_sensors(env_info, bottlenecks)
            total_e_use_rx_data += e_use_rx_data
            total_e_use_tx_data += e_use_tx_data
            total_e_use_rx_upstrm += e_use_rx_upstrm
            total_e_use_tx_upstrm += e_use_tx_upstrm
            total_e_use_rx_downstrm += e_use_rx_downstrm

            rt_use_interval.append(action_)

            # LOG
            if is_log:
                self.log_main(main_logger, interval, state, action_, reward, next_state, env_info, energy_before_step, energy_after_step, q_action, e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm)
            if is_log or is_chart:
                self.log_rts(rts_logger, interval, state, action_, next_state, env_info, energy_before_step, energy_after_step, q_action, e_use_rx_data, e_use_tx_data, e_use_rx_upstrm, e_use_tx_upstrm, e_use_rx_downstrm)

            # Update the current state
            state = next_state
            # Keep the previous action
            action = action_



                # # Exclude die nodes
                # remain_rt_nos = env.exclude_sensors(round_dies)

                # if len(remain_rt_nos) != 0:
                #     # Update Q-table (exclude actions)
                #     RL.exclude_actions(remain_rt_nos)
                # else:
                #     last_effort_action, controller = env.gen_RT_direct()
                #     RL.add_best_effort_action(last_effort_action, controller)
                #     is_send_direct = True

            env.reset_e_use_count()

            # Increase the interval by 1
            interval +=1
    # ===== Sim (main function) ===== (T)
    
if __name__ == "__main__":
    agent = RunAgent() # Create agent object.
    p = SimParent() # Create parent object

    # Input list
    inputs_list = [
    # [2, 'SumReward', 'rt', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
    [2, 'MinReward', 'min', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
    # [2, 'MinReward', 'max-min', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
    # [2, 'MinReward', 'rt', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
    # [2, 'SumReward', 'min', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
    # [2, 'SumReward', 'max-min', '0, 0, 1', 'y', 'n', 'n', '26-7-2025', 1, '1-Hj', 0.5, 0.5, 'full', 'y', 8, 200, 100, 'old', 'not_create_topo'],
                ] 
    
    # ----- Variables for log, cache ----- (H)
    method_list = []
    e_arrays = []
    alive_arrays = []
    lf_dict = {}
    cache_dir = p.cache_dir
    log_dir = p.log_dir
    result_cache_dir = p.result_cache_dir
    # ----- Variables for log, cache ----- (T)

    # ----- Sim result cache ----- (H)
    if not os.path.exists(f'{result_cache_dir}'):
        os.makedirs(f'{result_cache_dir}')


    if os.path.exists(f'{result_cache_dir}/all_result'):
        with open(f'{result_cache_dir}/all_result', "rb") as file:
            alive_arrays, e_arrays, lf_dict, method_list, num_change_rt, sim_log_dir = pickle.load(file) 
    # ----- Sim result cache ----- (T)
    if False:
        pass
    else:
        # ----- Variables for log ----- (H)
        list_interval = []
        list_interval_old_full = []
        list_interval_old_fw = []
        list_interval_GA = []
        list_time_old_get_rt = []
        list_time_old_cal_info = []
        list_time_GA_get_cal_rt = []
        list_choose_action = []
        list_learn_max_q = []
        list_try_step = []
        list_time_sim = []
        list_init_rts = []
        # ----- Variables for log ----- (T)

        num_repeat_method = 1 # Number of repeat each method.

        # Repeat sim methods. Simulation start here.
        for sim_no in range(num_repeat_method):
            # NOTE: this loop only use with GA method. 
            # NOTE: If you use OLD just edit num_repeat_method = 1
            for input in inputs_list:
                # ----- Inputs ----- (H)
                decimal = input[0]
                reward_method = input[1]
                state_method = input[2]
                weights = input[3]
                log = input[4]
                chart = input[5]
                init_q = input[6]
                log_type = input[7]
                rt_period = input[8]
                hop_reward_cal = input[9]
                alpha = input[10]
                gamma = input[11]
                rt_num = input[12]
                cache_mode = input[13]
                num_sensors = input[14]
                topo_range_x = input[15]
                topo_range_y = input[16]
                get_rt_method = input[17]
                create_topo = input[18]

                is_log = log == 'y'
                is_chart = chart == 'y'
                is_init_q = init_q == 'y'
                q = '+Qinit' if is_init_q else ''

                if state_method == 'rt':
                    method_list.append(r'$s_{t}(route), R_{%s}{(a_t)}$' % (reward_method[:3].lower()))
                else:
                    method_list.append(r'$s_{t}(%s), R_{%s}{(a_t)}$' % (state_method.lower(), reward_method[:3].lower()))
                # ----- Inputs ----- (T)

                # ----- Create cache dir ----- (H)
                if cache_mode == 'y' and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                # ----- Create cache dir ----- (T)

                # ----- Create sim log dir ----- (H)
                sim_log_dir = f'{log_dir}/{log_type}/{num_sensors}_{DATETIME}/{get_rt_method}/'
                if is_log or is_chart:
                    if not os.path.exists(sim_log_dir):
                        os.makedirs(sim_log_dir)
                # ----- Create sim log dir ----- (T)


                # ----- Print sim parameters ----- (H)
                print(f'\nReward Method: {reward_method} {q}')
                print(f'State Method: {state_method}')
                print(f'Weights: {weights}')
                print(f'Data size: {DATA_SIZE} bytes')
                print(f'UpStream size: {UP_STRM_SIZE} bytes')
                print(f'DownStream size: {DOWN_STRM_SIZE} bytes')
                print(f'Alpha: {alpha}')
                print(f'Gamma: {gamma}')
                print(f'Get RT method: {get_rt_method}')
                # ----- Print sim parameters ----- (T)

                # **************************************************************************************************************
                # NOTE: Call the main simulation function.
                start_sim_timer = time.time()
                first_node_die_interval, half_node_die_interval, last_node_die_interval, er_list_1, er_list_2, er_list_3, num_alive_nodes, log_dir, num_change_rt, env, RL = agent.sim(decimal, reward_method, q, is_init_q, is_log, is_chart, rt_period, state_method, weights, log_type, hop_reward_cal, alpha, gamma, rt_num, cache_mode, num_sensors, topo_range_x, topo_range_y, get_rt_method, create_topo, sim_no, sim_log_dir)
                end_sim_timer = time.time() - start_sim_timer
                # **************************************************************************************************************
                
                # ----- Store simulation result informations ----- (H)
                alive_arrays.append(num_alive_nodes)
                e_arrays.append(np.array([er_list_1, er_list_2, er_list_3]))
                lf_dict[f'S({state_method}), R({reward_method[:3]})'.upper()] = [first_node_die_interval, half_node_die_interval, last_node_die_interval]
                list_init_rts.append(env.init_rts)
                list_interval.append(first_node_die_interval)
                list_choose_action.append(np.mean(RL.timeit))
                list_learn_max_q.append(np.mean(RL.learn_max_q_time))
                # print(f'Time choose action {np.mean(RL.timeit)}')
                # print(f'Time learn max Q: {np.mean(RL.learn_max_q_time)}')
                # print(f'len learn max Q: {len(RL.learn_max_q_time)}')
                # print(f'Time try step: {np.mean(env.try_step_time)}')
                if get_rt_method == 'old':
                    list_time_old_get_rt.append(env.time_old_get_rt)
                    list_time_old_cal_info.append(env.time_old_cal_info)
                    list_time_sim.append(end_sim_timer-env.time_get_all_rt-env.time_save_topo)
                    # print(f'time_old_get_rt: {env.time_old_get_rt}')
                    # print(f'time_old_cal_info: {env.time_old_cal_info}')
                    # print(f'Time sim: {end_sim_timer-env.time_get_all_rt-env.time_save_topo}')
                elif get_rt_method == 'GA':
                    list_time_GA_get_cal_rt.append(env.time_GA_get_cal_rt-env.time_random_topo)
                    list_time_sim.append(end_sim_timer-env.time_random_topo-env.time_save_topo)
                    # print(f'time_GA_get_cal_rt: {env.time_GA_get_cal_rt-env.time_random_topo}')
                    # print(f'Time sim: {end_sim_timer-env.time_random_topo-env.time_save_topo}')
                # ----- Store simulation result informations ----- (T)

                # ----- Print simulation result infomations ----- (H)
                print(f'fnd: {first_node_die_interval}')
                print(f'hnd: {half_node_die_interval}')
                print(f'lnd: {last_node_die_interval}')
                print(f'RT swap count: {num_change_rt}\n')
                # ----------------------------------------------- (T)
                
                # ----- Process each array ----- (H)
                for idx, arr in enumerate(e_arrays):
                    # Replace negative values with 0
                    arr[arr < 0] = 0
                # ----- Process each array ----- (T)

                # ----- Dump simulation results into cache file ----- (H)
                with open(f'{result_cache_dir}/all_result', "wb") as file:
                    pickle.dump((alive_arrays, e_arrays, lf_dict, method_list, num_change_rt, sim_log_dir), file)
                # --------------------------------------------------- (T)

                # ----- NOTE: RT position compare ----- (H)
                # print(f'\nRT sorted {get_rt_method}: \n{env.rts_sorted}\n')
                # print(f'\nRT sorted e use {get_rt_method}: \n{env.e_sorted_data}')
                # ----- NOTE: RT position compare ----- (T)

            # ----- remove previous topo cahce ----- (H)
            # if os.path.exists('CACHE/topo'):
            #     os.remove('CACHE/topo/topo.pkl')
            # ----- remove previous topo cahce ----- (T)

        # ----- Print simulation timer in average ----- (H)
        # # print(f'Average FND of sim {num_repeat_method} times: {np.mean(np.array(list_interval))}')
        # # print(f'STD. FND of sim {num_repeat_method} times: {np.std(np.array(list_interval))}')
        # arr_list_interval_GA = np.array(list_interval[0::2])
        # arr_list_interval_old_fw = np.array(list_interval[1::2])

        # arr_time_GA_choose_action = np.array(list_choose_action[0::2])
        # arr_time_old_fw_choose_action = np.array(list_choose_action[1::2])

        # arr_time_GA_learn_max_q = np.array(list_learn_max_q[0::2])
        # arr_time_old_fw_learn_max_q = np.array(list_learn_max_q[1::2])

        # arr_time_GA_sim = np.array(list_time_sim[0::2])
        # arr_time_old_sim = np.array(list_time_sim[1::2])

        # arr_time_old_get_rt_fw = np.array(list_time_old_get_rt)
        # arr_time_old_cal_info_fw = np.array(list_time_old_cal_info)

        # arr_time_GA_get_cal_rt = np.array(list_time_GA_get_cal_rt)

        # print()

        # print(f'arr_list_interval_old_fw: {arr_list_interval_old_fw} {np.mean(arr_list_interval_old_fw)}')
        # print(f'arr_time_old_fw_choose_action: {arr_time_old_fw_choose_action} {np.mean(arr_time_old_fw_choose_action)}')
        # print(f'arr_time_old_fw_learn_max_Q: {arr_time_old_fw_learn_max_q} {np.mean(arr_time_old_fw_learn_max_q)}')
        # print(f'arr_time_old_get_rt_fw: {arr_time_old_get_rt_fw} {np.mean(arr_time_old_get_rt_fw)}')
        # print(f'arr_time_old_cal_info_fw: {arr_time_old_cal_info_fw} {np.mean(arr_time_old_cal_info_fw)}')
        # print(f'arr_time_old_sim: {arr_time_old_sim} {np.mean(arr_time_old_sim)}')
        # print()
        # print(f'sum arr_list_interval_old_fw \n{pd.Series(arr_list_interval_old_fw).describe()}')
        # print(f'sum arr_time_old_fw_choose_action \n{pd.Series(arr_time_old_fw_choose_action).describe()}')
        # print(f'sum arr_time_old_fw_learn_max_Q: \n{pd.Series(arr_time_old_fw_learn_max_q).describe()}')
        # print(f'sum arr_time_old_get_rt_fw \n{pd.Series(arr_time_old_get_rt_fw).describe()}')
        # print(f'sum arr_time_old_cal_info_fw \n{pd.Series(arr_time_old_cal_info_fw).describe()}')
        # print(f'sum arr_time_old_sim \n{pd.Series(arr_time_old_sim).describe()}')
        # print('-----------------------------------------')

        # print(f'arr_list_interval_GA: {arr_list_interval_GA} {np.mean(arr_list_interval_GA)}')
        # print(f'arr_time_GA_choose_action: {arr_time_GA_choose_action} {np.mean(arr_time_GA_choose_action)}')
        # print(f'arr_time_GA_learn_max_Q: {arr_time_GA_learn_max_q} {np.mean(arr_time_GA_learn_max_q)}')
        # print(f'arr_time_GA_get_cal_rt: {arr_time_GA_get_cal_rt} {np.mean(arr_time_GA_get_cal_rt)}')
        # print(f'arr_time_GA_sim: {arr_time_GA_sim} {np.mean(arr_time_GA_sim)}')
        # print()
        # print(f'sum arr_list_interval_GA \n{pd.Series(arr_list_interval_GA).describe()}')
        # print(f'sum arr_time_GA_choose_action \n{pd.Series(arr_time_GA_choose_action).describe()}')
        # print(f'sum arr_time_GA_learn_max_Q \n{pd.Series(arr_time_GA_learn_max_q).describe()}')
        # print(f'sum arr_time_GA_get_cal_rt \n{pd.Series(arr_time_GA_get_cal_rt).describe()}')
        # print(f'sum arr_time_GA_sim \n{pd.Series(arr_time_GA_sim).describe()}')
        # print('-----------------------------------------')
        
        # rt_df_GA = list_init_rts[0]
        # rt_df_old = list_init_rts[1]
        # with pd.option_context(
        #     'display.max_rows', None,
        #     'display.max_columns', None,
        #     'display.width', None,
        #     'display.max_colwidth', None
        # ):
        #     # print("rt_df_GA:")
        #     # print(rt_df_GA)
        #     # print("\nrt_df_old:")
        #     # print(rt_df_old)
        #     # print('GA')
        #     # for rt_no in rt_df_GA:
        #     #     print(f'{rt_no}: {rt_df_GA[rt_no]}')
        #     # print('old')
        #     # for rt_no in rt_df_old:
        #     #     print(f'{rt_no}: {rt_df_old[rt_no]}')
        #     # Store matches
        #     matches = []

        #     for col_rt_df_GA in rt_df_GA.columns:
        #         for col_rt_df_old in rt_df_old.columns:
        #             if rt_df_GA[col_rt_df_GA].equals(rt_df_old[col_rt_df_old]):
        #                 # print(f"rt_df_GA['{col_rt_df_GA}'] matches rt_df_old['{col_rt_df_old}']")
        #                 # print(f'GA: \n{rt_df_GA[col_rt_df_GA]}')
        #                 # print(f'old: \n{rt_df_old[col_rt_df_old]}')
        #                 matches.append((col_rt_df_GA, col_rt_df_old))

        #     # If you want to see the results as a DataFrame
        #     if matches:
        #         match_df = pd.DataFrame(matches, columns=['rt_df_GA_column', 'rt_df_old_column'])
        #         print("\nMatched columns:")
        #         print(match_df)
        #     else:
        #         print("No column sequences matched.")

        #     old_method_rt_index = []
        #     for _, row in match_df.iterrows():
        #         word = row['rt_df_old_column']
        #         if word in env.rts_sorted:
        #             index_in_l1 = env.rts_sorted.index(word)
        #             old_method_rt_index.append(index_in_l1)
        #             print(f"{row['rt_df_GA_column']} ('{word}') is at index {index_in_l1}")

        #     print(f'\nold_method_rt_index (sorted):\n{sorted(old_method_rt_index)}')
        # ----- Print simulation timer in average ----- (T)

    # ----- Chart config, save ----- (H)
    # Cycle through different marker styles for each method
    marker_cycle = ['o', 's', '^', 'v', 'p', 'D']  # Adjust based on the number of methods
    marker_index = 0
    
    # Plot data for each method
    x_values = ['FND', 'HND', 'LND']  # X-values

    for method, values in lf_dict.items():
        # Plot using different marker styles
        plt.plot(x_values, values, label=f"{method}", marker=marker_cycle[marker_index], markersize=3)

        # Increment marker index for the next method
        marker_index = (marker_index + 1) % len(marker_cycle)  # Wrap around if needed

    # Add labels, title, and legend
    plt.ylabel("Rounds")
    plt.title("LF comparision")
    plt.grid(True, linestyle='--')
    plt.legend()
    # Save the plot to a file after all methods are plotted
    plt.savefig(f'{sim_log_dir}/lifetime_round.png')  # You can specify the file format (e.g., .png, .jpg, .pdf)
    # Close the plot
    plt.close()

    # Define colors and markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Different colors for each method
    markers = ['o', 's', '^', 'D', '*', 'x']  # Markers for each method

    # Reorder methods as requested: 
    # S(min), R(sum) --> S(rt), R(sum) --> S(rt), R(min) --> S(max-min), R(sum) --> S(min), R(min) --> S(max-min), R(min)
    custom_order = [
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'rt'),         # S(RT), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'min'),        # S(MIN), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'max-min'),    # S(MAX-MIN), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'rt'),         # S(RT), R(MIN)
        next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'min'),        # S(MIN), R(MIN)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'max-min'),    # S(MAX-MIN), R(MIN)
    ]

    # ---- SINGLE PLOT ----
    plt.figure(figsize=(10, 6))  # Create a single figure

    # Plot in the custom order
    for i, idx in enumerate(custom_order):
        num_alive_nodes = alive_arrays[idx]
        x_markers = []
        y_markers = []

        # Detect changes in the number of alive nodes and plot markers
        previous_value = None
        for x in range(len(num_alive_nodes)):
            y = num_alive_nodes[x]
            if y != previous_value:
                x_markers.append(x + 1)  # 1-based index for rounds
                y_markers.append(y)
            previous_value = y
        
        # Plot line and markers
        plt.plot(range(1, len(num_alive_nodes) + 1), num_alive_nodes, linestyle='-', color=colors[i])
        plt.scatter(x_markers, y_markers, color=colors[i], marker=markers[i], s=80)

    # Customizing y-axis ticks
    all_values = sorted(set(val for data in alive_arrays for val in data))
    plt.yticks(all_values)
    plt.xlim(0, 20000)  # Set x-axis limit to 50 rounds
    plt.xlabel('Rounds')
    plt.ylabel('Number of Alive Nodes')
    # plt.title('Number of Alive Nodes for Custom Ordered Methods (with markers in legend)')
    plt.grid(True, linestyle='--')

    # ---- Custom Legend ----
    legend_elements = []
    for i, idx in enumerate(custom_order):
        line = Line2D([0], [0], color=colors[i], marker=markers[i], linestyle='-', markersize=5, label=method_list[idx])
        legend_elements.append(line)

    plt.legend(handles=legend_elements, loc='lower left')

    # Save the combined plot
    plt.savefig(f'{sim_log_dir}/custom_ordered_alive_nodes_with_custom_legend.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()

    #     # Define colors and markers
    #     colors_sum = ['b', 'c', 'm']  # Colors for R(SUM) methods
    #     colors_min = ['g', 'r', 'y']  # Colors for R(MIN) methods
    #     markers = ['o', 's', '^', 'D', '*', 'x']  # Markers for each method

    #     # Custom order for min_methods (R(MIN))
    #     min_methods_order = [
    #         # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'rt'),         # S(RT), R(MIN)
    #         next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'min'),        # S(MIN), R(MIN)
    #         # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'max-min')     # S(MAX-MIN), R(MIN)
    #     ]

    #     # Custom order for sum_methods (R(SUM))
    #     sum_methods_order = [
    #         # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'rt'),         # S(RT), R(SUM)
    #         # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'min'),        # S(MIN), R(SUM)
    #         # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'max-min')     # S(MAX-MIN), R(SUM)
    #     ]

    # # ---- PLOTTING WITH SUBPLOTS ----
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns for subplots

    #     # ----- PLOT FOR R(SUM) METHODS (Left) -----
    #     for i, idx in enumerate(sum_methods_order):
    #         num_alive_nodes = alive_arrays[idx]
    #         x_markers = []
    #         y_markers = []

    #         # Detect changes in the number of alive nodes and plot markers
    #         previous_value = None
    #         for x in range(len(num_alive_nodes)):
    #             y = num_alive_nodes[x]
    #             if y != previous_value:
    #                 x_markers.append(x + 1)  # 1-based index for rounds
    #                 y_markers.append(y)
    #             previous_value = y
            
    #         # Plot line and markers in the first subplot using colors for R(SUM)
    #         axs[0].plot(range(1, len(num_alive_nodes) + 1), num_alive_nodes, linestyle='-', color=colors_sum[i], label=f"{method_list[idx]}")
    #         axs[0].scatter(x_markers, y_markers, color=colors_sum[i], marker=markers[i], s=40)

    #     # Customizing y-axis ticks for the first subplot (R(SUM))
    #     all_values = sorted(set(val for data in alive_arrays for val in data))
    #     axs[0].set_yticks(all_values)
    #     axs[0].set_xlim(0, 20000)  # Set x-axis limit to 50 rounds
    #     axs[0].set_xlabel('Rounds')
    #     axs[0].set_ylabel('Number of Alive Nodes')
    #     axs[0].legend(loc='lower left')  # Move legend to the left-bottom side
    #     axs[0].set_title('R(SUM) Reward Function')
    #     axs[0].grid(True, linestyle='--')

    #     # ----- PLOT FOR R(MIN) METHODS (Right) -----
    #     for i, idx in enumerate(min_methods_order):
    #         num_alive_nodes = alive_arrays[idx]
    #         x_markers = []
    #         y_markers = []

    #         # Detect changes in the number of alive nodes and plot markers
    #         previous_value = None
    #         for x in range(len(num_alive_nodes)):
    #             y = num_alive_nodes[x]
    #             if y != previous_value:
    #                 x_markers.append(x + 1)  # 1-based index for rounds
    #                 y_markers.append(y)
    #             previous_value = y
            
    #         # Plot line and markers in the second subplot using colors for R(MIN)
    #         axs[1].plot(range(1, len(num_alive_nodes) + 1), num_alive_nodes, linestyle='-', color=colors_min[i], label=f"{method_list[idx]}")
    #         axs[1].scatter(x_markers, y_markers, color=colors_min[i], marker=markers[i], s=40)

    #     # Customizing y-axis ticks for the second subplot (R(MIN))
    #     axs[1].set_yticks(all_values)
    #     axs[1].set_xlim(0, 20000)  # Set x-axis limit to 50 rounds
    #     axs[1].set_xlabel('Rounds')
    #     axs[1].set_ylabel('Number of Alive Nodes')
    #     axs[1].legend(loc='lower left')  # Move legend to the left-bottom side
    #     axs[1].set_title('R(MIN) Reward Function')
    #     axs[1].grid(True, linestyle='--')
    #     # ----- Custom Legend with Line2D objects for both subplots -----
    #     legend_elements_sum = [Line2D([0], [0], color=colors_sum[i], marker=markers[i], linestyle='-', markersize=5,
    #                                 label=method_list[sum_methods_order[i]]) for i in range(len(sum_methods_order))]

    #     legend_elements_min = [Line2D([0], [0], color=colors_min[i], marker=markers[i], linestyle='-', markersize=5,
    #                                 label=method_list[min_methods_order[i]]) for i in range(len(min_methods_order))]

    #     # Add legends to each subplot
    #     axs[0].legend(handles=legend_elements_sum, loc='lower left')
    #     axs[1].legend(handles=legend_elements_min, loc='lower left')

    #     # Adjust layout for subplots
    #     plt.tight_layout()

    #     # Save the combined plot with subplots
    #     plt.savefig(f'{sim_log_dir}/combined_subplots_alive_nodes_RSUM_left_RMIN_right_ordered_colors.png', dpi=300, format='png', bbox_inches='tight')

    #     plt.close()

    # Define colors and labels for the plot
    colors = ['b', 'g', 'r']  # Colors for Node 1, Node 2, Node 3
    labels = ['Node 1', 'Node 2', 'Node 3']  # Labels for nodes within each method
    markers = ['o', 's', '^']  # Different markers for each node (circle, square, triangle)

    # Group methods into R(SUM) and R(MIN)
    sum_methods = [0, 4, 5]  # Indices corresponding to R(SUM) methods
    min_methods = [1, 2, 3]  # Indices corresponding to R(MIN) methods

    # Create subplots: 3x3 grid for six methods (3 rows and 3 columns)
    num_methods = 6
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # 3 rows, 3 columns, adjusting figure size

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Labels like (a), (b), (c), ... for each subplot
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    # Adjust the custom order for subplots as per your desired order:
    custom_order = [
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'rt'),         # S(RT), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'min'),        # S(MIN), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'SumReward' and option[2] == 'max-min'),    # S(MAX-MIN), R(SUM)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'rt'),         # S(RT), R(MIN)
        next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'min'),        # S(MIN), R(MIN)
        # next(i for i, option in enumerate(inputs_list) if option[1] == 'MinReward' and option[2] == 'max-min')     # S(MAX-MIN), R(MIN)
    ]

    # Iterate through each method and plot using the reordered indices
    for i, (method_idx, ax) in enumerate(zip(custom_order, axes[:num_methods])):
        # Set title based on the current method
        if method_idx == custom_order[0]:
            met = r'$s_{t}(route), R_{sum}{(a_t)}$'
        elif method_idx == custom_order[1]:
            met = r'$s_{t}(min), R_{sum}{(a_t)}$'
        elif method_idx == custom_order[2]:
            met = r'$s_{t}(max-min), R_{sum}{(a_t)}$'
        elif method_idx == custom_order[3]:
            met = r'$s_{t}(route), R_{min}{(a_t)}$'
        elif method_idx == custom_order[4]:
            met = r'$s_{t}(min), R_{min}{(a_t)}$'
        elif method_idx == custom_order[5]:
            met = r'$s_{t}(max-min), R_{min}{(a_t)}$'
        
        # x-values for this method
        x_values = range(1, len(alive_arrays[method_idx]) + 1)

        # Plotting each node (1, 2, 3) separately with different markers for each label
        for j, (node_data, color, label, marker) in enumerate(zip(e_arrays[method_idx], colors, labels, markers)):
            if marker == 'o':
                markersize = 8  # Larger size for the triangle marker
                scatter_size = 60 # Larger size for the final point
            elif marker == 's':
                markersize = 6
                scatter_size = 50
            else:
                markersize = 4  # Default size for other markers
                scatter_size = 40  # Default size for final point

            # Plot the line with different markers
            ax.plot(x_values[:len(node_data)], node_data, linestyle='-', color=color, label=label, 
                    marker=marker, markersize=markersize, markevery=(200, 200))  # Each node uses a different marker
            
            # Additionally mark the end of the simulation (the last point)
            ax.scatter(x_values[-1], node_data[-1], color=color, marker=marker, s=scatter_size)  # Larger triangle marker at the end

        # Set title and labels for the subplot
        ax.set_title(met, pad=5)  # Title with some padding
        ax.set_xlabel('Rounds', labelpad=10)  # Add padding to xlabel
        ax.set_ylabel('Energy of each bottle neck node (J)')
        ax.legend(loc='upper right')

        # Set x-axis limit
        ax.set_xlim(0, 4000)  # Adjust to the appropriate limit

        # Add the label like (a), (b), (c), ... to the bottom of the subplot
        ax.text(0.5, -0.15, subplot_labels[i], transform=ax.transAxes, fontsize=12, 
                va='top', ha='center')  # Position below the plot (negative y-value for bottom)

        # Add grid to the plot
        ax.grid(True, linestyle='--')

    # Hide any extra subplots (if fewer than 9 methods)
    for j in range(num_methods, len(axes)):
        axes[j].axis('off')  # Turn off the axis for unused subplots

    # Adjust layout to prevent overlap and add padding between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Add more space between the subplots


    # Optionally, save the plot after displaying it
    plt.savefig(f'{sim_log_dir}/Energy_comparison_3x3_grid_with_group_methods_diff_markers.png', dpi=300, format='png', bbox_inches='tight')

    # Close the plot
    plt.close()
    # ----- Chart config, save ----- (T)
