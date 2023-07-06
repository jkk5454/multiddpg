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
    folder_path = "./data"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Directory " , folder_path ,  " Created ")
    else:
        print("Directory " , folder_path ,  " already exists")
    
    if args.train == 1:
        folder_path = "./data/train"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Directory " , folder_path ,  " Created ")
        else:
            print("Directory " , folder_path ,  " already exists")
        folder_path = "./data/train/test"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Directory " , folder_path ,  " Created ")
        else:
            print("Directory " , folder_path ,  " already exists")
        run_ddpg()
    else:
        folder_path = "./data/test"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Directory " , folder_path ,  " Created ")
        else:
            print("Directory " , folder_path ,  " already exists")
        test_ddpg()



if __name__ == '__main__':
    main()