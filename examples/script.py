# script.py

import os
import sys
from memory_profiler import profile
from memory_profiler import memory_usage
@profile
def run_ddpg():
    os.system('python examples/ddpg_env.py --env_name ClothMove --headless 1')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        log_file = sys.argv[1]
        with open(log_file, 'w') as f:
            run_ddpg()
            f.write('\n\nMemory profile:\n')
            f.write('\n'.join(str(x) for x in memory_usage()))
    else:
        print('Usage: python script.py <log_file>')
