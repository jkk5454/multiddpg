from turtle import shape
import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
from softgym.utils.misc import quatFromAxisAngle
from pathlib import Path
import trimesh
import random
import scipy.spatial
import matplotlib.pyplot as plt
import imageio
import math


class ClothMoveEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_move_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        self.picked_particles = None
        self.fig = plt.figure(figsize=(12, 5))
        self.fig.cbar = [None]*2
        self.elongnation_images=[]
        self.is_final_state=0
        self.wall_num = 1  # number of obstacle's wall
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        
        self.normorlize_elongations=None
        

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)
        


    def get_default_config(self):
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            cam_pos, cam_angle = np.array([0.1, 0.8, 0.8]), np.array([0, -45 / 180. * np.pi, 0.])
            #cam_pos, cam_angle = np.array([-0.0, 0.3, 0.5]), np.array([0, -0, 90 / 180 * np.pi])
            #cam_pos, cam_angle = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
            #cam_pos, cam_angle = np.array([-0.5,0.20, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
            #cam_pos, cam_angle = np.array([-0.5,0.3, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
        config = {
            #'ClothPos': [-1.6, 2.0, -0.8],
            'ClothPos': [-0.2, 0.005, 0], # for T-shirt
            #'ClothPos': [-0.5, 0.005, -0.2], #for garment
            'ClothSize': [int(0.3 / particle_radius), int(0.3 / particle_radius)],
            #'ClothStiff': [2.0, 1, 0.9],  # Stretch, Bend and Shear
            'ClothStiff': [2.0, 2.3, 0.2],  # Stretch, Bend and Shear standard
            #'ClothStiff': [1.5, 1.8, 0.4],  # Stretch, Bend and Shear  Test for different material
            'glass': {
                'glass_border': 0.015,
                'glass_length': 0.20,
                'glass_width':0.06,
            },
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}
                                    },
            'flip_mesh': 0
        }

        return config

    def load_cloth(self,path):
        """Load .obj of cloth mesh. Only quad-mesh is acceptable!
        Return:
            - vertices: ndarray, (N, 3)
            - triangle_faces: ndarray, (S, 3)
            - stretch_edges: ndarray, (M1, 2)
            - bend_edges: ndarray, (M2, 2)
            - shear_edges: ndarray, (M3, 2)
        This function was written by Zhenjia Xu
        email: xuzhenjia [at] cs (dot) columbia (dot) edu
        website: https://www.zhenjiaxu.com/
        """
        vertices, faces = [], []
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # 3D vertex
            if line.startswith('v '):
                vertices.append([float(n)
                                for n in line.replace('v ', '').split(' ')])
            # Face
            elif line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
                face = [int(n[0]) - 1 for n in idx]
                assert(len(face) == 4)
                faces.append(face)

        triangle_faces = []
        for face in faces:
            triangle_faces.append([face[0], face[1], face[2]])
            triangle_faces.append([face[0], face[2], face[3]])

        stretch_edges, shear_edges, bend_edges = set(), set(), set()

        # Stretch & Shear
        for face in faces:
            stretch_edges.add(tuple(sorted([face[0], face[1]])))
            stretch_edges.add(tuple(sorted([face[1], face[2]])))
            stretch_edges.add(tuple(sorted([face[2], face[3]])))
            stretch_edges.add(tuple(sorted([face[3], face[0]])))

            shear_edges.add(tuple(sorted([face[0], face[2]])))
            shear_edges.add(tuple(sorted([face[1], face[3]])))

        # Bend
        neighbours = dict()
        for vid in range(len(vertices)):
            neighbours[vid] = set()
        for edge in stretch_edges:
            neighbours[edge[0]].add(edge[1])
            neighbours[edge[1]].add(edge[0])
        for vid in range(len(vertices)):
            neighbour_list = list(neighbours[vid])
            N = len(neighbour_list)
            for i in range(N - 1):
                for j in range(i+1, N):
                    bend_edge = tuple(
                        sorted([neighbour_list[i], neighbour_list[j]]))
                    if bend_edge not in shear_edges:
                        bend_edges.add(bend_edge)

        return np.array(vertices), np.array(triangle_faces),\
                np.array(list(stretch_edges)), np.array(
                list(bend_edges)), np.array(list(shear_edges))

    
    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1
        
        mesh_verts = np.array([])
        mesh_stretch_edges = np.array([])
        mesh_bend_edges = np.array([])
        mesh_shear_edges = np.array([])
        mesh_faces = np.array([])
    
        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            
            cloth_dimx, cloth_dimy = -1, -1
            # sample random mesh
            path = "/home/clothsim/softgym/cloth3d/val/Tshirt_processed.obj"
            retval = self.load_cloth(path)
            mesh_verts = retval[0]
            mesh_faces = retval[1]
            mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]
            num_particle = mesh_verts.shape[0]//3
            #flattened_area = trimesh.load(path).area/2
            config.update({
                'ClothSize':[cloth_dimx,cloth_dimy],
                'mesh_verts': mesh_verts.reshape(-1),
                'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
                'mesh_bend_edges': mesh_bend_edges.reshape(-1),
                'mesh_shear_edges': mesh_shear_edges.reshape(-1),
                'mesh_faces': mesh_faces.reshape(-1),
            })
            self.set_scene(config)
            #self.action_tool.reset([0., -1., 0.])
            pyflex.step()
            #print(pyflex.get_positions().reshape(-1,4))
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states
    '''
    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1
        
        mesh_verts = np.array([])
        mesh_stretch_edges = np.array([])
        mesh_bend_edges = np.array([])
        mesh_shear_edges = np.array([])
        mesh_faces = np.array([])

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            config.update({
                'ClothSize':[cloth_dimx,cloth_dimy],
                'mesh_verts': mesh_verts.reshape(-1),
                'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
                'mesh_bend_edges': mesh_bend_edges.reshape(-1),
                'mesh_shear_edges': mesh_shear_edges.reshape(-1),
                'mesh_faces': mesh_faces.reshape(-1),
            })
            self.set_scene(config)

            pyflex.step()

           
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states
    '''
    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)
        
    def _get_obs(self):
        """ Get observation. """
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        obs = np.zeros(self.action_tool.num_picker*3+2)
        rgb, depth = pyflex.render_cloth()
        center_x, center_y=pyflex.center_inf()
        if math.isnan(center_x) and math.isnan(center_y):
            center_x, center_y=0,0
        if self.picked_particles is not None:
            for i in range(len(self.picked_particles)):
                if self.picked_particles[i] is not None:
                    obs[i*3:i*3+3]=particle_pos[self.picked_particles[i],:3]
        obs[-2:]=np.array([center_x,center_y])
        return obs
        
        
    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3] # Was changed from from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            #self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            self.action_tool.reset([0., -2., 0.])
            #reset glass
            shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
            # create glass
            self.create_glass(self.glass_length, self.glass_width, self.glass_border)
            # move glass to be at initial position
            self.glass_states = self.init_glass_state(0.1, 0.3,self.glass_length, self.glass_width, self.glass_border)


            self.set_glass_shape_states(self.glass_states,shape_states)

            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        #num_particles = np.prod(config['ClothSize'], dtype=int)
        mesh_verts = config['mesh_verts']
        num_particles = mesh_verts.shape[0]//3
        #particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        #cloth_dimx = config['ClothSize'][0]
        #x_split = cloth_dimx // 2
        #self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        #self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()



        colors = np.zeros(num_particles)
        #colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        #pos_a = self.init_pos[self.fold_group_a, :]
        #pos_b = self.init_pos[self.fold_group_b, :]
        #self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        #print('reset',pyflex.get_positions().reshape(-1,4))
        return self._get_obs()

    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        camera_params = deepcopy(self.camera_params)
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase, 'camera_params': camera_params,
                'glass_states': self.glass_states,'glass_params': self.glass_params,
                'config_id': self.current_config_id}

    def set_state(self, state_dic):
        pyflex.set_positions(state_dic['particle_pos'])
        pyflex.set_velocities(state_dic['particle_vel'])
        pyflex.set_shape_states(state_dic['shape_pos'])
        pyflex.set_phases(state_dic['phase'])
        self.glass_states = state_dic['glass_states']
        self.camera_params = deepcopy(state_dic['camera_params'])
        self.update_camera(self.camera_name)

    def _step(self, action):
        pick_flag=self.action_tool.step(action)
        if self.picked_particles is None:
            self.picked_particles=self.action_tool.picked_id_info()
        #self.set_glass_shape_states(self.glass_states,shape_states)
        if self.action_mode in ['sawyer', 'franka']:
             print(self.action_tool.next_action)
             pyflex.step(self.action_tool.next_action)
        else:
            if pick_flag[0]:
               pyflex.step()
               self.dragging_detection()
            else:
               test_param=np.array([0.1,0.2,0.3,0.4])
               pyflex.step(update_params=test_param)


    def set_glass_params(self, config):
        params = config
        self.glass_border = params['glass_border']
        self.glass_length = params['glass_length']
        self.glass_width = params['glass_width']
        
        self.glass_params = params


    def set_scene(self, config, state=None):
        # create cloth
        #super().set_scene(config)
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 0 if 'env_idx' not in config else config['env_idx']
        mass = config['mass'] if 'mass' in config else 0.5
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], mass,
                                 config['flip_mesh']])
        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(
                scene_idx=env_idx,
                scene_params=scene_params,
                vertices=config['mesh_verts'],
                stretch_edges=config['mesh_stretch_edges'],
                bend_edges=config['mesh_bend_edges'],
                shear_edges=config['mesh_shear_edges'],
                faces=config['mesh_faces'],
                thread_idx=0)

        # compute glass params
        if state is None:
            self.action_tool.reset([0., -1., 0.])
            shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
            self.set_glass_params(config["glass"])
            # create glass
            self.create_glass(self.glass_length, self.glass_width, self.glass_border)
            # move glass to be at initial position
            self.glass_states = self.init_glass_state(0.1, 0.3,self.glass_length, self.glass_width, self.glass_border)


            self.set_glass_shape_states(self.glass_states,shape_states)
        else:
            glass_params = state['glass_params']
            self.glass_border = glass_params['glass_border']
            self.glass_length = glass_params['glass_length']
            self.glass_width = glass_params['glass_width']
            self.glass_params = glass_params
            self.set_state(state)


        self.current_config = deepcopy(config)

    def create_glass(self, glass_length, glass_width, glass_border):
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, 1.0], 0)
        boxes = []
        
        #bench
        halfEdge = np.array([glass_length / 2. + glass_border, glass_border / 4., glass_width / 2. + glass_border])
        boxes.append([halfEdge, center, quat])

        #capsule
        halfEdge = np.array([glass_border / 2., glass_width / 2. + glass_border-0.002])
        quat = quatFromAxisAngle([0, -1., 0], np.pi/2.)
        boxes.append([halfEdge, center, quat])

        
        halfEdge = boxes[0][0]
        center = boxes[0][1]
        quat = boxes[0][2]
        pyflex.add_box(halfEdge, center, quat)

        halfEdge = boxes[1][0]
        center = boxes[1][1]
        quat = boxes[1][2]
        pyflex.add_capsule(halfEdge, center, quat)
        #pyflex.add_sphere(0.08, center, quat)
        
        return boxes

    def init_glass_state(self, x, y,glass_length, glass_width, glass_border):
        x_center, y_curr, y_last = x, y, 0.
        quat = quatFromAxisAngle([0, 0, 1.0], 0)
        quat_cap = quatFromAxisAngle([0, -1., 0], np.pi/2.)
        
        # states of 1 walls
        states = np.zeros((2, 14))
        #states = np.zeros((1, 14))
        
        states[0, :3] = np.array([x_center, y_curr, 0.])
        states[0, 3:6] = np.array([x_center, y_last, 0.])

        #states[1, :3] = np.array([x_center - (glass_length  + 2*glass_border) / 2., (glass_border) / 2. + y_curr, 0.])
        #states[1, 3:6] = np.array([x_center - (glass_length  + 2*glass_border) / 2., (glass_border) / 2. + y_last, 0.])

        states[1, :3] = np.array([x_center - (glass_length  + 2*glass_border) / 2., 0. + y_curr-glass_border/4+0.002, 0.])
        states[1, 3:6] = np.array([x_center - (glass_length  + 2*glass_border) / 2., 0. + y_last-glass_border/4+0.002, 0.])

        states[0, 6:10] = quat
        states[0, 10:] = quat

        states[1, 6:10] = quat_cap
        states[1, 10:] = quat_cap
        
        return states
  
    def set_glass_shape_states(self, glass_states,shape_states):
 
        all_states = np.concatenate((shape_states,glass_states), axis=0)
        pyflex.set_shape_states(all_states)
        
    def final_state(self):
        return self.is_final_state
    '''
    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        # Compute elongation penalty/reward
        elongation_reward = 0
        if self.normorlize_elongations is not None:
            if np.max(self.normorlize_elongations)<0.8:
                elongation_reward = 1
            else:
                elongation_reward = -1
        
        # Compute picker position reward
        picker_reward = 0
        position_reward = 0
        s = self._get_obs()
        if self.is_final_state == 0:
            picker_reward = -math.sqrt((s[0] - 0.27)**2 + (s[3] - 0.27)**2)
            #add side camera center
            cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
            pyflex.set_camera_params(
            np.array([*cam_pos2,*cam_angle2,720,720]))
            rgb, depth = pyflex.render_cloth()
            #Mirror symmetry
            center_x, center_y=pyflex.center_inf()
            if np.isnan(center_x):
                center_x=100
            position_reward = -10*abs(center_x-0.5)
            cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
            pyflex.set_camera_params(np.array([*cam_pos1,*cam_angle1,720,720])) # reset camera to original position
            
        
        # Compute wrinkle density/depth reward, only in final state
        wrinkle_reward = 0
        center_reward = 0
        diff_reward = 0
        if self.is_final_state:
            cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
            pyflex.set_camera_params(np.array([*cam_pos1,*cam_angle1,720,720]))
            rgb, depth = pyflex.render_cloth()
            wrinkle_density, wrinkle_avedepth=pyflex.wrinkle_inf()
            wrinkle_reward = -(wrinkle_density - 10*wrinkle_avedepth)
            
            center_x, center_y=pyflex.center_inf()
            if np.isnan(center_x):
                center_x=0
            if np.isnan(center_y):
                center_y=0
            center_reward_top = -math.sqrt((center_y - 0.5)**2+(center_x - 0.5)**2)
            if abs(center_x-0.5) > 0.3:
                center_reward_top = -1000
            if np.isnan(wrinkle_avedepth):
                wrinkle_reward = -1000
                print('wrinkle_avedepth is nan')
            
            cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
            pyflex.set_camera_params(
            np.array([*cam_pos2,*cam_angle2,720,720]))
            rgb, depth = pyflex.render_cloth()
            #Mirror symmetry
            center_x, center_y=pyflex.center_inf()
            mean_half_front, mean_half_back=pyflex.sidecam_inf()           
            #print('center_x_side, center_y_side',center_x, center_y)
            if np.isnan(center_x):
                center_x=0
            diff_reward = -100*abs(mean_half_front-mean_half_back)
            center_reward_side = -math.sqrt((center_x - 0.5)**2)
            if center_reward_side < -0.3:
                center_reward_side = -1000
            center_reward_side = center_reward_side
            
            center_reward = 10*(0.5*center_reward_top + 0.5*center_reward_side)
            
            cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
            pyflex.set_camera_params(np.array([*cam_pos1,*cam_angle1,720,720])) # reset camera to original position

        reward = 0.1*(0.4 * elongation_reward + 0.4 * picker_reward+0.2*position_reward) + 0.9*(0.6*center_reward + 0.2 * wrinkle_reward+0.2*diff_reward)
        if np.isnan(reward):
            reward = -100
        
        return reward
    
    '''
    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        picker_reward = 0
        position_reward = 0
        s = self._get_obs()
        '''
        if self.is_final_state == 0:
            picker_reward = -math.sqrt((s[0] - 0.27)**2 + (s[3] - 0.27)**2)
            #add side camera center
            cam_pos2, cam_angle2 = np.array([-0.5,0.15, 0.0]), np.array([-+90 / 180 * np.pi, 0, 0.])
            pyflex.set_camera_params(
            np.array([*cam_pos2,*cam_angle2,720,720]))
            rgb, depth = pyflex.render_cloth()
            #Mirror symmetry
            center_x, center_y=pyflex.center_inf()
            if np.isnan(center_x):
                center_x=100
            position_reward = -10*abs(center_x-0.5)
            cam_pos1, cam_angle1 = np.array([0.1,0.7, 0.0]), np.array([0, -90 / 180 * np.pi, 0.])
            pyflex.set_camera_params(np.array([*cam_pos1,*cam_angle1,720,720])) # reset camera to original position
        
        reward = 0.6 * picker_reward+0.4*position_reward
        if np.isnan(reward):
            reward = -100
        '''
        reward = 0
        return reward
    

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        performance = self.normorlize_elongations
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'max_normorlize_elongations': np.max(self.normorlize_elongations),
            'min_normorlize_elongations': np.min(self.normorlize_elongations)
        }
        return info
    
    #get the impact point with glass and check the impact points are not dragging the particles too far away that violates the actual physicals constraints.
    
    def dragging_detection(self):
        impact_threshold=0.005
        particle_radius=0.00625
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        center_capsule = self.glass_states[1,:3]
        impact_point=np.zeros((100,3))
        impact_particles=[None]*100
        impact_point[0] = center_capsule-[0,0,self.glass_width/2]
        #get the impact particles id
        for i in range(100):
            impact_point[i]=impact_point[0]+[0,0,self.glass_width*i/100]
            dists = scipy.spatial.distance.cdist(impact_point[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
            idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
            mask = dists.flatten() <= impact_threshold + self.glass_border/2. + particle_radius
            idx_dists = idx_dists[mask, :].reshape((-1, 2))
            if idx_dists.shape[0] > 0:
                impact_id, impact_dist = None, None
                for j in range(idx_dists.shape[0]):
                    if idx_dists[j, 0] not in impact_particles and (impact_id is None or idx_dists[j, 1] < impact_dist):
                        impact_id = idx_dists[j, 0]
                        impact_dist = idx_dists[j, 1]
                    if impact_id is not None:
                        impact_particles[i] = int(impact_id)
            
        if impact_particles is not None:
            impact_particle_idices = []
            active_impact_indices = []
            for i in range(100):
                if impact_particles[i] is not None:
                    impact_particle_idices.append(impact_particles[i])
                    active_impact_indices.append(i)

            l = len(impact_particle_idices)
            elongations = [[] for _ in range(len(self.picked_particles))]
            #print('len of l is:',l)
            for i in range(len(self.picked_particles)):
                draw_particles_x=[]
                draw_particles_y=[]
                for j in range(l):
                    init_distance = np.linalg.norm(self.init_pos[self.picked_particles[i], :3] -
                                                   self.init_pos[impact_particle_idices[j], :3])
                    now_distance = np.linalg.norm(particle_pos[self.picked_particles[i], :3] -
                                                  particle_pos[impact_particle_idices[j], :3])
                    elongations[i].append(now_distance / init_distance)          
                    #print('elongation:',now_distance/init_distance)
                    draw_particles_x.append(particle_pos[impact_particle_idices[j], 0])
                    draw_particles_y.append(particle_pos[impact_particle_idices[j], 2])
                    
                #colors set
            if l:
                self.normorlize_elongations = [[] for _ in range(len(self.picked_particles))]
                for i in range(len(self.picked_particles)):
                    #normalized_elongations = (elongations - min_elongations) / (max_elongations - min_elongations)
                    normalized_elongations= (elongations[i] - np.array(1.0)) / (np.array(1.8) - np.array(1.0))
                    self.normorlize_elongations[i]=normalized_elongations
                '''
                    ax = self.fig.add_subplot(1,2,i+1)
                    colors = plt.cm.rainbow(normalized_elongations)
                    ax.cla()
                    
                    im=ax.scatter(draw_particles_x, draw_particles_y, c=colors,cmap='rainbow')
                    for j in range(l):
                        ax.plot([particle_pos[self.picked_particles[i]][0], particle_pos[impact_particle_idices[j]][0]]
                            ,[particle_pos[self.picked_particles[i]][2], particle_pos[impact_particle_idices[j]][2]],color=colors[j])
                    ax.set_ylim(-0.12,0.12)
                    ax.set_xlim(-0.05,0.3)
                    if self.fig.cbar[i] is None:
                        cb = self.fig.colorbar(im, ax=ax)
                        cb.ax.yaxis.set_label_position('left')
                        cb.ax.set_ylabel('elongation',rotation=270,fontsize=14,fontweight='bold')
                        self.fig.cbar[i]=cb
                    self.fig.cbar[i].update_normal(im)
                
                self.fig.canvas.draw()
                #argb_image = self.fig.canvas.tostring_rgb()
                #print('argb_image:',argb_image) 
                #image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                #image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                buf = self.fig.canvas.tostring_rgb()
                ncols, nrows = self.fig.canvas.get_width_height()
                image = np.fromstring(buf, dtype=np.uint8).reshape(nrows*2, ncols*2, 3) # 2 for 4k monitor
                self.elongnation_images.append(image)
                #print('max elongation:',np.max(self.normorlize_elongations))
                if self.headless is False:
                    plt.draw()
                    plt.pause(0.1)
                #plt.close()
                '''  
                
    # get the elonganation of the object
    def get_elongation_gif(self):
        imageio.mimsave('./data/elongation.gif', self.elongnation_images,fps=5)

        
               
        
        
        
        
