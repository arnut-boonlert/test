from decimal import Decimal, getcontext

# getcontext().prec = 28
class SimParent:
    def __init__(self):
        self.sim_dir = f'SIM' # /SIM
        self.cache_dir = f'{self.sim_dir}/CACHE' # /SIM/CACHE
        self.log_dir = f'{self.sim_dir}/LOG'
        self.result_cache_dir = f'{self.cache_dir}/sim_result' # /SIM/CACHE/sim_result
        self.cache_ga_dir = f'{self.cache_dir}/ga'