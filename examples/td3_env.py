import os.path as osp
import argparse
import numpy as np
import gc

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt

import time
import tensorflow as tf
import tensorlayer as tl

from td3 import TD3



RANDOMSEED = 1              # random seed
MEMORY_CAPACITY = 600     # size of replay buffer
BATCH_SIZE = 64             # update batchsize
MAX_EPISODES = 200         # total number of episodes for training
MAX_EP_STEPS = 60          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 0.00015                    # control exploration

log_file = './data/train_td3_0811/reward.txt'

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

def learn_step(i,s,r, ep_reward,t1, env, action_base , frames, img_size=720,ddpg=None, max_episodes=MAX_EPISODES):
     for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)       # action from DDPG
            
            a = np.clip(np.random.normal(a, VAR), -0.0002, 0.0002)

            # 添加随机噪声
            '''
            a[1] += np.random.uniform(-0.00005, 0.00005)
            a[2] += np.random.uniform(-0.00005, 0.00005)
            a[4] += np.random.uniform(-0.00005, 0.00005)
            a[5] += np.random.uniform(-0.00005, 0.00005)
            '''

            #print('action:',a) 
            action1 = np.append(a[:3],0)
            action2 = np.append(a[3:],0)
            action = [action_base[0]+action1, action_base[1]+action2]
            # get acton
            s_, r, done, info = env.step(action, record_continuous_video=True, img_size=img_size)
            frames.extend(info['flex_env_recorded_frames'])
            if(np.any(np.isnan(s_))):
                print(f'nan in states')
                save_name = osp.join('./data/train_td3_0811/', 'ClothMove{}.gif'.format(i))
                save_numpy_as_gif(np.array(frames), save_name)
                print('Video generated and save to {}'.format(save_name))
                break
            
            #print('episode:', i, 'step:', j,'r',r)
            print('r',r)
            print('s',s[0],s[3],s[1],s[4] , s[2], s[5])

           
            if j == MAX_EP_STEPS-1:
                #release steps
                action_release = np.array([[0.000, 0.0000, 0.000, 0.00],
                        [0.000, 0.0000, 0.000, 0.00]])
                print('pick position:', s[0],s[3])
                env._wrapped_env.is_final_state = 1
                for k in range(20):
                    s_relase, r, done_release, info_release = env.step(action_release, record_continuous_video=True, img_size=img_size)
                    frames.extend(info_release['flex_env_recorded_frames'])
                gc.collect()
                print('r_release',r)
                #ep_reward += r
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        i, max_episodes, ep_reward+r,
                        time.time() - t1
                    )
                )
                
                with open(log_file, 'a') as f:
                    f.write('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, max_episodes, ep_reward+r,
                            time.time() - t1
                        ))
                #reward_buffer.append(ep_reward)
                #top_view
                savename='./data/train_td3_0811/train{}_top.png'.format(i)
                show_depth(savename)
                center_x, center_y=pyflex.center_inf()
                wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
                with open(log_file, 'a') as f:
                    f.write('\rwrinkle desity: {:.4f}  | wrinkle averange depth: {:.4f}  | center_x: {:.4f} | ceneter_y: {:.4f}'.format(wrinkle_density, wrinkle_avedepth, center_x,center_y)
                            )
                print('\rcenter_x_top: {:.4f} | ceneter_y_top: {:.4f}'.format(center_x,center_y))
                
                
                
                cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
                pyflex.set_camera_params(
                    np.array([*cam_pos2,*cam_angle2,720,720]))
                #side view
                savename='./data/train_td3_0811/train{}_side.png'.format(i)
                show_depth(savename)
                center_x, center_y=pyflex.center_inf()
                with open(log_file, 'a') as f:
                    f.write( '\rcenter_x_side: {:.4f} | ceneter_y_side: {:.4f}'.format(center_x,center_y)
                            )
                print('\rcenter_x_side: {:.4f} | ceneter_y_side: {:.4f}'.format(center_x,center_y),end='')
                cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
                pyflex.set_camera_params(
                    np.array([*cam_pos1,*cam_angle1,720,720]))
                
        
                
                # save_name = osp.join('./data/train_td3_0811/', 'ClothMove{}.gif'.format(i))
                # save_numpy_as_gif(np.array(frames), save_name)
                # print('Video generated and save to {}'.format(save_name))
                
                
             #  store s，a，r，s_
            #print('r:',r,'is_final_state:',env._wrapped_env.is_final_state)
            ddpg.store_transition(s, a, r, s_, env._wrapped_env.is_final_state)
            

            # update state
            s = s_  
            ep_reward += r  # accumulate reward
            
            env._wrapped_env.is_final_state = 0
            
            # learn
            if ddpg.pointer > MEMORY_CAPACITY:
                #ddpg.learn()
                learn_function(ddpg)
                
            #plt.show()
            
            #print('episode',i,'step:',j)
  
def learn_function(ddpg = None):
    ddpg.learn()
    
def initial_state(env, frames, img_size=720):
    env.reset()
    #for i in range(env.horizon):
    for i in range(50):
        if i < 20:
            action = np.array([[-0.001, 0.000, 0.000, 0.001],
                          [-0.001, 0.000, 0.000, 0.001]])
        elif 19<i<50:
            action = np.array([[-0.00, 0.0018, 0.000, 0.001],
                          [-0.00, 0.0018, 0.000, 0.001]])
        
        #action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        
        next_obs, rewards, done, info = env.step(action, record_continuous_video=True, img_size=img_size)
        del action
        gc.collect()
        frames.extend(info['flex_env_recorded_frames'])
    return next_obs, rewards, done, info
    
def position_and_wrinkle_inf():
    #top vision
    cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
    pyflex.set_camera_params(
        np.array([*cam_pos1,*cam_angle1,720,720]))
    show_depth()
    wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
    center_x, center_y=pyflex.center_inf()
    print('wrinkle desity:',wrinkle_density,'   wrinkle averange depth:', wrinkle_avedepth, '   center_x:', center_x,'  ceneter_y:',center_y)
    #side vision
    cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
    pyflex.set_camera_params(
        np.array([*cam_pos2,*cam_angle2,720,720]))
    show_depth()
    center_x, center_y=pyflex.center_inf()
    print('wrinkle desity:',wrinkle_density,'   wrinkle averange depth:', wrinkle_avedepth, '   center_x:', center_x,'  ceneter_y:',center_y)


def net_train(env, action_base , frames, img_size=720,ddpg=None, max_episodes=MAX_EPISODES):
    reward_buffer = []      
    t0 = time.time()
    for i in range(max_episodes):
        frames = [env.get_image(img_size, img_size)]
        t1 = time.time()
        s, r, done, info = initial_state(env, frames, img_size) # initial state
        ep_reward = 0
        #print('ep_reward:',ep_reward)
        
        learn_step(i,s,r, ep_reward,t1, env, action_base, frames, img_size, ddpg, max_episodes)       
        
               
        # test
        if i and not i % TEST_PER_EPISODES:
            t1 = time.time()
            s, r, done, info = initial_state(env, frames, img_size) # initial state
            ep_reward = 0
            for j in range(MAX_EP_STEPS):

                a = ddpg.choose_action(s)
                action1 = np.append(a[:3],0)
                action2 =np.append(a[3:],0)
                action = [action_base[0]+action1, action_base[1]+action2]
                # action from DDPG
                s_, r, done, info = env.step(action, record_continuous_video=True, img_size=img_size)
                frames.extend(info['flex_env_recorded_frames'])

                if j == MAX_EP_STEPS - 1:
                    #release steps
                    action_release = np.array([[0.000, 0.0000, 0.000, 0.00],
                            [0.000, 0.0000, 0.000, 0.00]])
                    env._wrapped_env.is_final_state = 1
                    for k in range(20):
                        s_relase, r, done_release, info_release = env.step(action_release, record_continuous_video=True, img_size=img_size)
                        frames.extend(info_release['flex_env_recorded_frames'])
                    ep_reward += r
                    
                    print(
                        '\rTest Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, max_episodes, ep_reward,
                            time.time() - t1
                        )
                    )
                    with open(log_file, 'a') as f:
                        f.write('\rTest Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, max_episodes, ep_reward,
                            time.time() - t1
                        )
                                )
                        
                    
                    #top vision
                    savename='./data/train_td3_0811/test/Test_{}_top.png'.format(i)
                    show_depth(savename)
                    center_x, center_y=pyflex.center_inf()
                    wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
                    with open(log_file, 'a') as f:
                        f.write('\rwrinkle desity: {:.4f} |  wrinkle averange depth: {:.4f}  | center_x_top: {:.4f} | ceneter_y: {:.4f}'.format(wrinkle_density, wrinkle_avedepth, center_x,center_y)
                            )
                    print('\rcenter_x_top: {:.4f} | ceneter_y_top: {:.4f}'.format(center_x,center_y))
                    
                    
                    
                    cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
                    pyflex.set_camera_params(
                        np.array([*cam_pos2,*cam_angle2,720,720]))
                    #side vision
                    savename='./data/train_td3_0811/test/Test_{}_side.png'.format(i)
                    show_depth(savename)
                    center_x, center_y=pyflex.center_inf()
                    mean_half_front, mean_half_back=pyflex.sidecam_inf()
                    diff = abs(mean_half_front-mean_half_back)
                    with open(log_file, 'a') as f:
                        f.write( '\rcenter_x_side: {:.4f} | ceneter_y_side: {:.4f} | Mirror Diff:{:.4f}'.format(center_x,center_y, diff)
                            )
                    print('\rcenter_x_side: {:.4f} | ceneter_y_side: {:.4f}| Mirror Diff:{:.4f}'.format(center_x,center_y, diff),end='')
                    cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
                    pyflex.set_camera_params(
                        np.array([*cam_pos1,*cam_angle1,720,720]))
                    
                    '''
                    save_name = osp.join('./data/train_td3_0811/test/', 'ClothMove{}.gif'.format(i))
                    save_numpy_as_gif(np.array(frames), save_name)
                    print('Video generated and save to {}'.format(save_name))
                    '''
                    
                    reward_buffer.append(ep_reward)
                s = s_
                if j!=MAX_EP_STEPS-1:
                    ep_reward += r
                    
                env._wrapped_env.is_final_state = 0
            
        #print('frames:',sys.getsizeof(frames))
        del frames
    '''
        if reward_buffer:
            plt.ion()
            plt.cla()
            plt.title('DDPG')
            plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
            plt.xlabel('episode steps')
            plt.ylabel('normalized state-action value')
            #plt.ylim(-2000, 0)
            plt.show()
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    '''
    print('\nRunning time: ', time.time() - t0)
    ddpg.save_ckpt()
    
    

    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='test', action='store_false')
    parser.add_argument('--max_episode', type=int, default=200)

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
    a_bound = np.array([0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002])
    action_0 = np.array([[0.0009, 0.00005, -0.00015, 0.001],
                          [0.0009, -0.00005, 0.00015, 0.001]])
                       
    
    print('initial_obs_dim:',initial_obs.shape[0],'a_dim:',a_bound.shape[0])
    
    tf.config.list_physical_devices('GPU')
    
    
    with tf.device('/GPU:0'):
        ddpg = TD3(a_dim=a_bound.shape[0], s_dim=initial_obs.shape[0], a_bound=a_bound)
        with open(log_file, 'w') as f:
                    f.write('\rDDPG\n')
        if args.train:
            net_train(env,action_0, frames, args.img_size,ddpg, max_episodes)
                
    
    


if __name__ == '__main__':
    main()
