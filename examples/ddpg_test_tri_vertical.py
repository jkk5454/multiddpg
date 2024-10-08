import os.path as osp
import argparse
import numpy as np
import gc
import sys
# sys.path.append('/home/clothsim/multiddpg')
import softgym

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt

import time
import tensorflow as tf
import tensorlayer as tl

from triddpg import DDPG




RANDOMSEED = 1              # random seed
MEMORY_CAPACITY = 2400     # size of replay buffer
BATCH_SIZE = 256             # update batchsize
MAX_EPISODES = 500         # total number of episodes for training
MAX_EP_STEPS = 150          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 0.00005                    # control exploration


min_xyz = np.array([-0.25, 0., -0.35])
max_xyz = np.array([0.25, 1.0, 0.35])

log_file = './data/test/reward.txt'

def show_depth(savename=None):
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    #plt.show()
    plt.savefig(savename)
    #print(savename)
    #plt.pause(1)
    plt.close(fig)

    
def initial_state(env, frames, img_size=720):
    env.reset()
    #for i in range(env.horizon):
    # for i in range(50):
    #     if i < 20:
    #         action = np.array([[-0.001, 0.000, 0.000, 0.001],
    #                       [-0.001, 0.000, 0.000, 0.001]])
    #     elif 19<i<50:
    #         action = np.array([[-0.00, 0.0018, 0.000, 0.001],
    #                       [-0.00, 0.0018, 0.000, 0.001]])
    for i in range(104):
        if i < 20:
            action = np.array([[-0.0010, 0.000, 0.000, 0.001],
                          [-0.0010, 0.000, 0.000, 0.001]])
        elif 19<i<54:
            action = np.array([[-0.00, 0.00313, -0.0003, 0.001],
                          [-0.00, 0.00283, -0.0003, 0.001]])
            
        elif 53<i<104:
            if i == 54:
                draw=[0.00, 0.00, 8.0, 60.0, 240.0]
                env.env_update(draw)
            action = np.array([[0.0011, -0.000, 0.000, 0.001],
                          [0.0011, -0.000, 0.000, 0.001]])
        
        #action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        
        next_obs, rewards, done, info = env.step(action, record_continuous_video=True, img_size=img_size)
        #print('intial obs',next_obs)
        del action
        gc.collect()
        frames.extend(info['flex_env_recorded_frames'])
    return next_obs, rewards, done, info
    
def position_and_wrinkle_inf():
    show_depth()
    wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
    center_x, center_y=pyflex.center_inf()
    print('wrinkle desity:',wrinkle_density,'   wrinkle averange depth:', wrinkle_avedepth, '   center_x:', center_x,'  ceneter_y:',center_y)

def normalize_function(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def denormalize_function(value, min_value, max_value):
    return value * (max_value - min_value) + min_value
    
    
def net_test(env, action_base , frames, img_size=720,ddpg=None, max_episodes=MAX_EPISODES):
    print('load...')
    ddpg.load_ckpt()
    print('load success')
    for n in range(max_episodes):
        frames = [env.get_image(img_size, img_size)]
        s, r, done, info = initial_state(env, frames, img_size) # initial state
        a1 = np.random.uniform(-0.00005, 0.00005)
        a2 = np.random.uniform(-0.00005, 0.00005)
        a3 = np.random.uniform(-0.00005, 0.00005)
        a4 = np.random.uniform(-0.00005, 0.00005)
        a5 = np.random.uniform(-0.00005, 0.00005)
        a6 = np.random.uniform(-0.00005, 0.00005)
        action_random = np.array([[a1, a2, a3, 0.001],
                 [a4, a5, a6, 0.001]])
        for i in range(20):
             s, r, done, info = env.step(action_random, record_continuous_video=True, img_size=img_size)
             gc.collect()
             frames.extend(info['flex_env_recorded_frames'])
        #random initial state
        
        for j in range(MAX_EP_STEPS):
            s0_normalize = normalize_function(s[:3], min_xyz, max_xyz)
            s1_normalize = normalize_function(s[3:], min_xyz, max_xyz)
            s_normalize = np.concatenate([s0_normalize, s1_normalize])
            #print('s_normalize',s_normalize)
            a_normalize = ddpg.choose_action(s_normalize)
            a0 = a_normalize[:3]*(max_xyz-min_xyz)
            a1 = a_normalize[3:]*(max_xyz-min_xyz)
            a = np.concatenate([a0, a1])
            action1 = np.append(a[:3],0)
            action2 =np.append(a[3:],0)
            
            
            if j<40:
                action = [action_base[0][0]+action1, action_base[0][1]+action2]
                env._wrapped_env.state_flag = 0
                
            if 39<j<80:
                action = [action_base[1][0]+action1, action_base[1][1]+action2]
                env._wrapped_env.state_flag = 1
            if 79<j<MAX_EP_STEPS:
                if i == 130:
                    draw = [0.7, 0.7, 4.0, 30.0, 120.0]
                    env.env_update(draw)
                action = [action_base[2][0]+action1, action_base[2][1]+action2]
                env._wrapped_env.state_flag = 1
                
            
            #action = [action_base[0]+action1, action_base[1]+action2]
            # action from DDPG
            s_, r, done, info = env.step(action, record_continuous_video=True, img_size=img_size)
            print('s_',s_)
            frames.extend(info['flex_env_recorded_frames'])

            if j == MAX_EP_STEPS - 1:
                #release steps
                action_release = np.array([[0.000, 0.0000, 0.000, 0.00],
                        [0.000, 0.0000, 0.000, 0.00]])
                for k in range(20):
                    s_relase, r, done_release, info_release = env.step(action_release, record_continuous_video=True, img_size=img_size)
                    frames.extend(info_release['flex_env_recorded_frames'])
                env._wrapped_env.state_flag = 2
                
                savename='./data/test/Test_{}_top.png'.format(n)
                show_depth(savename)
                center_x, center_y=pyflex.center_inf()
                wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
                with open(log_file, 'a') as f:
                    f.write('\rwrinkle desity: {:.4f} |  wrinkle averange depth: {:.4f}  | center_x_top: {:.4f} | ceneter_y: {:.4f}'.format(wrinkle_density, wrinkle_avedepth, center_x,center_y)
                        )
                print('\rcenter_x_top: {:.4f} | ceneter_y_top: {:.4f}'.format(center_x,center_y))
                print('\rwrinkle desity: {:.4f} |  wrinkle averange depth: {:.4f}'.format(wrinkle_density, wrinkle_avedepth))
                
                save_name = osp.join('./data/test/', 'ClothMove{}.gif'.format(n))
                save_numpy_as_gif(np.array(frames), save_name)
                print('Video generated and save to {}'.format(save_name))
                    
            
            s = s_
                
            env._wrapped_env.is_final_state = 0
            
        del frames
        
    

    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--test', dest='test', action='store_false',default=True)
    parser.add_argument('--max_episode', type=int, default=100)


    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    
    max_episodes = args.max_episode

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)
    env.reset()
    
    
    frames = [env.get_image(args.img_size, args.img_size)]  
    
    
    initial_obs, rewards, _, info = initial_state(env, frames, args.img_size)
    a_bound = np.array([0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005])
    action_0 = np.array([[0.0011, -0.000, 0.000, 0.001],
                          [0.0011, -0.000, 0.000, 0.001]])
    action_1 = np.array([[-0.000, -0.0006, 0.0008, 0.001],
                            [-0.000, -0.0009, 0.0008, 0.001]])
    action_2 = np.array([[0.000, -0.0003, -0.0003, 0.001],
                        [0.000, -0.0000, -0.0003, 0.001]])
    action_3 = np.array([[0.000, -0.0005, 0.000, 0.001],
                        [0.000, -0.0005, 0.000, 0.001]])
    
    action = [action_1, action_2, action_3]
    
    a0_bound = a_bound[:3]/(max_xyz-min_xyz)
    a1_bound = a_bound[3:]/(max_xyz-min_xyz)
    a_bound = np.concatenate([a0_bound, a1_bound])
    
    print('initial_obs_dim:',initial_obs.shape[0],'a_dim:',a_bound.shape[0])
    
    tf.config.list_physical_devices('GPU')
    
    
    with tf.device('/GPU:0'):
        ddpg = DDPG(a_dim=a_bound.shape[0], s_dim=initial_obs.shape[0], a_bound=a_bound)
        with open(log_file, 'w') as f:
                    f.write('\rDDPG\n')
            
        if args.test:
            print('start test')
            net_test(env,action, frames, args.img_size,ddpg, max_episodes)
                
    
                
    
    


if __name__ == '__main__':
    main()
