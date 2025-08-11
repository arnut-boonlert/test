import numpy as np
import time
import random
from parent import SimParent

class QLearningTable(SimParent):
    def __init__(self, decimal, alpha=0.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha # Alpha value (Learning rate)
        self.gamma = gamma # Gamma value (Discount factor)
        self.decimal = decimal
        self.step = 10**-self.decimal
        self.timeit = []
        self.learn_max_q_time = []

    @staticmethod
    def timeit_avg(func):
        def wrapper(self, *args, **kwargs):  # Ensure 'self' is passed for methods
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            self.timeit.append(elapsed)  # ✅ Now self.timeit is accessible
            # print(f"{func.__name__} took {elapsed:.6f}  seconds")
            return result
        return wrapper
    
    @staticmethod
    def timeit_learn_max_q_avg(func):
        def wrapper(self, *args, **kwargs):  # Ensure 'self' is passed for methods
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            self.learn_max_q_time.append(elapsed)  # ✅ Now self.timeit is accessible
            # print(f"{func.__name__} took {elapsed:.6f}  seconds")
            return result
        return wrapper
    
    def setup_agent(self, controller, is_init_q, d_ratios, h_ratios, state_method, rts_list):
        # All STP as actions (All possible route, RT)
        self.rts_list = rts_list
        self.actions = controller.RTs.columns.to_numpy()
        # States
        if state_method == 'max-min' or state_method == 'min':
            # Use energy as state
            self.states = np.round(np.arange(0, float((controller.sensors[random.choice(list(controller.sensors.keys()))].init_energy)*1e-12)+self.step, self.step), decimals=self.decimal)
            self.states = np.round(np.arange(0, 0.5+self.step, self.step), decimals=self.decimal)
        elif state_method == 'rt':
            # Use All STP as state
            self.states = controller.RTs.columns.to_numpy()

        # Map action and state into dict for working with numpy array
        self.actions_dict = {int(action): i for i, action in enumerate(self.actions)} # Key = RT No. and Value = index
        self.states_dict = {float(state): i for i, state in enumerate(self.states)} # Key = state and Value = index
        self.states_map_index_to_rt = {i: state for i, state in enumerate(self.states)}

        # Create Q-table
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        # Q-init
        if is_init_q:
            d_list = [[v for k, v in value.items() if k != -1] for value in d_ratios.values()]
            d_array = np.array(d_list)
            d_sums = np.sum(d_array, axis=0)
            h_list = [[v for k, v in value.items() if k != -1] for value in h_ratios.values()]
            h_array = np.array(h_list)
            h_sums = np.sum(h_array, axis=0)

            self.q_table[:] = d_sums + h_sums
        return self.q_table.shape[0]

    @timeit_avg
    def choose_action(self, state, env, interval=np.inf):
        q_list, q_dict, is_node_die, remain_rt_nos, round_dies = self.learn_max_q(env, state)
        # print(state)
        if interval == 1:
            state = max(q_dict, key=q_dict.get)
        self.q_table[self.states_dict[state], :] = q_list
        # print(list(self.q_table[self.states_dict[state], :]))
        action = self.actions[np.argmax(self.q_table[self.states_dict[state], :])]
        q_action = self.q_table[self.states_dict[state], self.actions_dict[action]]

        mask = np.isfinite(self.q_table[self.states_dict[state], :]) 
        result = self.q_table[self.states_dict[state], :][mask]
        row = self.q_table[self.states_dict[state], :]
        indices = np.where(~np.isneginf(row))[0]
        print(f'actions_dict: {self.actions_dict}')
        print(f'check -inf: {result}')
        print(f'indix of non -inf: {indices}')
        print(f'check q_table: {self.q_table[self.states_dict[state], :]}')

        if is_node_die:
            self.exclude_actions(remain_rt_nos)
            self.q_table[:] = 0

        # print(self.q_table[self.states_dict[state]])

        return action, q_action, env.done, env.die_node, is_node_die, round_dies
    
    @timeit_learn_max_q_avg
    def learn_max_q(self, env, current_state):
        q_list_1 = []
        q_dict_1 = {}
        counter_pick = 0
        for action_1 in self.actions:
            counter_pick += 1
            if action_1 == 4097:
                print(f'action: {action_1}')
            if action_1 in env.unavailable_rt:
                if action_1 == 4097:
                    print('unavailable_rt')
                # print(f'action: {action_1}')
                # print(f'if {action_1} in env.unavailable_rt.keys()')
                q_list_1.append(float('-inf')) # Store -inf.
                q_dict_1[action_1] = float('-inf') # Store -inf.
            else:
                if action_1 == 4097:
                    print('available_rt')
                # print(f'else {action_1} in env.unavailable_rt.keys()')
                state_1, reward_1, done_1, env_info_1, die_node_1, useable_rt = env.try_step(action_1)
                # If RT is useable (don't make any node E < 0).
                if useable_rt:
                    if action_1 == 4097:
                        print('usable_rt')
                    q_1 = ((1-self.alpha) * self.q_table[self.states_dict[current_state], self.actions_dict[action_1]]) + (self.alpha * (reward_1 + (self.gamma * self.q_table[self.states_dict[state_1], :].max())))
                    # print(f'action: {action_1}')
                    # print('useable')
                    # print(f'current_state: {current_state}')
                    # print(f'Old Q: {self.q_table[self.states_dict[current_state], self.actions_dict[action_1]]}')
                    # print(f'Q: {q_1}\n')
                    q_list_1.append(q_1)
                    q_dict_1[action_1] = q_1
                # If RT is unuseable (make any node E < 0).
                else:
                    if action_1 == 4097:
                        print('unuseable_rt')
                    # print(f'action: {action_1}')
                    # print('unuseable')
                    # print(f'current_state: {current_state}')
                    # print(f'Old Q: {self.q_table[self.states_dict[current_state], self.actions_dict[action_1]]}\n')
                    # print(f'{action_1}: unuseable_rt')
                    q_list_1.append(float('-inf')) # Store -inf.
                    q_dict_1[action_1] = float('-inf') # Store -inf.

        print(f'counter_pick: {counter_pick}')
        is_node_die, remain_rt_nos, round_dies = env.check_sensor_die()

        return q_list_1, q_dict_1, is_node_die, remain_rt_nos, round_dies
    
    def exclude_actions(self, remain_rt_nos):
        self.actions = remain_rt_nos
        action_index_map = np.array([self.actions_dict[action] for action in remain_rt_nos])

        # Create an array of column indices to keep
        columns_to_keep = np.array([col for col in range(self.q_table.shape[1]) if col in action_index_map])

        # Use fancy indexing to select columns to keep
        self.q_table = self.q_table[:, columns_to_keep]
        self.actions_dict = {action: i for i, action in enumerate(self.actions)}
        # print(self.states_dict)

    def add_best_effort_action(self, best_effort_action, controller):
        self.actions = best_effort_action
        self.states = np.round(np.arange(0, controller.sensors[random.choice(list(controller.sensors.keys()))].init_energy+self.step, self.step), decimals=self.decimal)
        self.actions_dict = {action: i for i, action in enumerate(self.actions)}
        self.states_dict = {state: i for i, state in enumerate(self.states)}
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        