# script.py

import os
import argparse

def run_ddpg():
    os.system('python examples/ddpg_env.py --env_name ClothMove --headless 1 --test_depth 1 --max_episode 50')
    
def test_ddpg():
    os.system('python examples/ddpg_test.py --env_name ClothMove --headless 1 --test_depth 1 --max_episode 10')
    
def main():
    parser = argparse.ArgumentParser(description='MultiDDPG ')
    parser.add_argument('--train', type=int, default=1)
    args = parser.parse_args()
    
    if args.train == 1:
        run_ddpg()
    else:
        test_ddpg()



if __name__ == '__main__':
    main()