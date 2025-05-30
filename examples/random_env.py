import os.path as osp
import argparse
import numpy as np

import sys
sys.path.append('/home/clothsim/softgym')

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt




def show_depth():
    # render rgb and depth
    print('rendering')
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
    depth_plot = axes[1].imshow(depth)
    #plt.show()
    fig.colorbar(depth_plot, ax=axes[1], orientation='vertical')
    plt.show()
    
        
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()
    
    
    frames = [env.get_image(args.img_size, args.img_size)]  
    #######################################################################################
    ##########################Vertical Pulling###########################################
    #######################################################################################
    # for i in range(env.horizon):
    # #for i in range(80):
    #     if i < 20:
    #         action = np.array([[-0.0010, 0.000, 0.000, 0.001],
    #                       [-0.0010, 0.000, 0.000, 0.001]])
    #     elif 19<i<54:
    #         action = np.array([[-0.00, 0.00313, -0.0003, 0.001],
    #                       [-0.00, 0.00283, -0.0003, 0.001]])
    #     elif 53 <i<104:
    #         if i == 54:
    #             draw=[0.00, 0.00, 8.0, 60.0, 240.0]
    #             env.env_update(draw)
    #         action = np.array([[0.0011, -0.000, 0.000, 0.001],
    #                       [0.0011, -0.000, 0.000, 0.001]])
            
    #     # elif 104<i<115:
    #     #     if i==105:
    #     #         draw=[0.00, 0.00, 6.0, 60.0, 240.0]
    #     #         env.env_update(draw)
    #     #     action = np.array([[0.000, -0.000, 0.00, 0.001],
    #     #                     [0.000, -0.00, 0.00, 0.001]])
    #     elif 103<i<144:
    #         action = np.array([[-0.000, -0.0006, 0.0008, 0.001],
    #                         [-0.000, -0.0009, 0.0008, 0.001]])
    #     elif 143<i<184:
    #         action = np.array([[0.000, -0.0002, -0.0003, 0.001],
    #                     [0.000, -0.0000, -0.0003, 0.001]])
    #     elif 183<i<240:
    #         if i == 179:
    #             draw = [0.7, 0.7, 4.0, 30.0, 120.0]
    #             env.env_update(draw)
    #         action = np.array([[0.000, -0.0005, 0.000, 0.001],
    #                     [0.000, -0.0005, 0.000, 0.001]])
            
    #     # elif 49<i<101:
    #     #     action = np.array([[0.0012, -0.00012, 0.0000, 0.001],
    #     #                   [0.0012, -0.00012, -0.0000, 0.001]])
    #     #     if i==78 or i==95:
    #     #         draw=1
    #     # elif 100<i<151:
    #     #     action = np.array([[0.000, 0.0000, 0.000, 0.001],
    #     #                 [0.000, 0.0000, 0.000, 0.001]])
    #     # elif 150<i<env.horizon:
    #     #     env._wrapped_env.is_final_state = 1
    #     #     action = np.array([[0.000, 0.0000, 0.000, 0.00],
    #     #                 [0.000, 0.0000, 0.000, 0.00]])
    #     #     if i == env.horizon-1:
    #     #         env._wrapped_env.is_final_state = 1
    #     else:
    #         action = np.array([[0.000, 0.0000, 0.000, 0.000],
    #                     [0.000, 0.0000, 0.000, 0.000]])
        
    #     #action = env.action_space.sample()
    #     # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
    #     # intermediate frames. Only use this option for visualization as it increases computation.
    
    for i in range(env.horizon):
        draw=0
    #for i in range(80):
        if i < 20:
            action = np.array([[-0.001, 0.000, 0.000, 0.001],
                          [-0.001, 0.000, 0.000, 0.001]])
        elif 19<i<50:
            action = np.array([[-0.00, 0.0018, 0.000, 0.001],
                          [-0.00, 0.0018, 0.000, 0.001]])
        elif 49<i<101:
            action = np.array([[0.00120, -0.00015, 0.000, 0.001],
                          [0.00120, -0.00015, -0.000, 0.001]])
            if i==78 or i==95:
                draw=1
        elif 100<i<115:
            action = np.array([[0.000, 0.0000, 0.000, 0.001],
                        [0.000, 0.0000, 0.000, 0.001]])
        elif 114<i<env.horizon:
            action = np.array([[0.000, 0.0000, 0.000, 0.00],
                        [0.000, 0.0000, 0.000, 0.00]])
            if i == env.horizon-1:
                env._wrapped_env.is_final_state = 1
        #action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        

        obs, reward, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])
        print('i',i,'obs',obs)
    
    #env.get_elongation_gif()
    
    if args.test_depth:
        #top vision
        # cam_pos1, cam_angle1 = np.array([0.1,1.6, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
        # pyflex.set_camera_params(
        #     np.array([*cam_pos1,*cam_angle1,720,720]))
        show_depth()
        wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
        center_x, center_y=pyflex.center_inf()
        print('wrinkle desity:',wrinkle_density,'   wrinkle averange depth:', wrinkle_avedepth, '   center_x:', center_x,'  ceneter_y:',center_y)
        #side vision
        # cam_pos2, cam_angle2 = np.array([-1.5,0.20, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
        # pyflex.set_camera_params(
        #     np.array([*cam_pos2,*cam_angle2,720,720]))
        # show_depth()
        center_x, center_y=pyflex.center_inf()
        print('wrinkle desity:',wrinkle_density,'   wrinkle averange depth:', wrinkle_avedepth, '   center_x:', center_x,'  ceneter_y:',center_y)
        # mean_half_front, mean_half_back=pyflex.sidecam_inf()
        # print('mean_half_front:',mean_half_front,'   mean_half_back:', mean_half_back)

    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
