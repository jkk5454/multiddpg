import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl


#####################  hyper parameters  ####################
LR_A = 0.0002                # learning rate for actor
LR_C = 0.0002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 600     # size of replay buffer
BATCH_SIZE = 32             # update batchsize

MAX_EPISODES = 1000         # total number of episodes for training
MAX_EP_STEPS = 60          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 0.0003                     # control exploration

POLICY_NOISE = 0.00005
NOISE_CLIP = 0.0001
POLICY_UPDATE_FREQUENCY = 2

decay_steps = 500  # Set the decay steps as per your requirement
decay_rate = 0.96  # Set the decay rate as per your requirement


###############################  DDPG  ####################################


import numpy as np
import tensorflow as tf
import tensorlayer as tl

class TD3(object):
    """
    TD3 class
    """
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)  # state, next_state, action, reward, final_state_flag
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # create actor network
        def get_actor(input_state_shape, name=''):
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound, dtype=np.float32) * x)(x)            
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        # create critic network
        def get_critic(input_state_shape, input_action_shape, name=''):
            s = tl.layers.Input(input_state_shape, name='C_s_input', dtype=tf.float32)  # 指定输入为float32
            a = tl.layers.Input(input_action_shape, name='C_a_input', dtype=tf.float32)  # 指定输入为float32
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic_1 = get_critic([None, s_dim], [None, a_dim], name='1')
        self.critic_2 = get_critic([None, s_dim], [None, a_dim], name='2')

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)
                
        
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        
        self.critic_target_1 = get_critic([None, s_dim], [None, a_dim], name='_target_1')
        copy_para(self.critic_1, self.critic_target_1)
        
        self.critic_target_2 = get_critic([None, s_dim], [None, a_dim], name='_target_2')
        copy_para(self.critic_2, self.critic_target_2)
        
        self.actor_target.eval()
        self.critic_target_1.eval()
        self.critic_target_2.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        lr_schedule_actor = tf.keras.optimizers.schedules.ExponentialDecay(
            LR_A,
            decay_steps=10000,
            decay_rate=0.99,
            staircase=True)

        lr_schedule_critic = tf.keras.optimizers.schedules.ExponentialDecay(
            LR_C,
            decay_steps=10000,
            decay_rate=0.99,
            staircase=True)

        self.actor_opt = tf.optimizers.Adam(lr_schedule_actor)
        self.critic_opt = tf.optimizers.Adam(lr_schedule_critic)
        
        self.learn_step_counter = 0
        
        print('initiate TD3 model')


    def ema_update(self):

        paras = self.actor.trainable_weights + self.critic_1.trainable_weights + self.critic_2.trainable_weights    
        self.ema.apply(paras)                                                 
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target_1.trainable_weights + self.critic_target_2.trainable_weights, paras):
            i.assign(self.ema.average(j))                                     
            
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def add_noise(self, bt):
        bt_noise = np.tile(bt, (8, 1))
        for i in range(len(bt)):
           bt_noise[i+len(bt),self.s_dim+self.a_dim+1] = bt[i,self.s_dim+self.a_dim+1] + 0.3*np.random.normal(0, 0.01)
           bt_noise[i+len(bt)*2,self.s_dim+self.a_dim+1] = bt[i,self.s_dim+self.a_dim+1] + 0.5*np.random.normal(0, 0.01)
           bt_noise[i+len(bt)*3,self.s_dim+self.a_dim+1] = bt[i,self.s_dim+self.a_dim+1] + 0.8*np.random.normal(0, 0.01)
           bt_noise[i+len(bt)*4,self.s_dim+self.a_dim+1] = bt[i,self.s_dim+self.a_dim+1] + np.random.normal(0, 0.01)
           lam = bt[i][self.s_dim+self.a_dim+1]
           lam = np.nan_to_num(lam)
           lam = np.clip(lam, 0, None)
           poisson_noise = np.random.poisson(lam, 3)
           bt_noise[i+len(bt)*5,self.s_dim+self.a_dim+1] = poisson_noise[0]
           bt_noise[i+len(bt)*6,self.s_dim+self.a_dim+1] = poisson_noise[1]
           bt_noise[i+len(bt)*7,self.s_dim+self.a_dim+1] = poisson_noise[2]
        return bt_noise 
    
    def learn(self):
        
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)   
        bt = self.memory[indices, :]   
        bs = bt[:, :self.s_dim].astype(np.float32)
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim].astype(np.float32)
        br = bt[:, self.s_dim + self.a_dim].astype(np.float32)
        bs_ = bt[:, self.s_dim + self.a_dim + 1:self.s_dim*2 + self.a_dim + 1].astype(np.float32)
        final_state_flag = bt[:, -1]

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            noise = np.clip(np.random.normal(0, POLICY_NOISE, size=ba.shape), -NOISE_CLIP, NOISE_CLIP).astype(np.float32)
            a_ = self.actor_target(bs_) + noise
            a_ = np.clip(a_, -self.a_bound, self.a_bound).astype(np.float32)

            q1 = self.critic_target_1([bs_, a_])
            q2 = self.critic_target_2([bs_, a_])
            q_ = tf.minimum(q1, q2)
            y = br + (1 - final_state_flag) * GAMMA * q_
            q1 = self.critic_1([bs, ba])
            q2 = self.critic_2([bs, ba])
            td_error1 = tf.losses.mean_squared_error(y, q1)
            td_error2 = tf.losses.mean_squared_error(y, q2)

        c_grads1 = tape1.gradient(td_error1, self.critic_1.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads1, self.critic_1.trainable_weights))

        c_grads2 = tape2.gradient(td_error2, self.critic_2.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads2, self.critic_2.trainable_weights))

        if self.learn_step_counter % POLICY_UPDATE_FREQUENCY == 0:
            with tf.GradientTape() as tape:
                a = self.actor(bs)
                q = self.critic_1([bs, a])
                a_loss = -tf.reduce_mean(q)
            a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
            self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

            self.ema_update()

        self.learn_step_counter += 1
        
    def store_transition(self, s, a, r, s_, final_state_flag):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :param final_state_flag: flag indicating if it's the final state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        transition = np.hstack((s, a, [r], s_, [final_state_flag]))

        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        Save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/td3_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/td3_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/td3_critic_1.hdf5', self.critic_1)
        tl.files.save_weights_to_hdf5('model/td3_critic_target_1.hdf5', self.critic_target_1)
        tl.files.save_weights_to_hdf5('model/td3_critic_2.hdf5', self.critic_2)
        tl.files.save_weights_to_hdf5('model/td3_critic_target_2.hdf5', self.critic_target_2)

    def load_ckpt(self):
        """
        Load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/td3_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/td3_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/td3_critic_1.hdf5', self.critic_1)
        tl.files.load_hdf5_to_weights_in_order('model/td3_critic_target_1.hdf5', self.critic_target_1)
        tl.files.load_hdf5_to_weights_in_order('model/td3_critic_2.hdf5', self.critic_2)
        tl.files.load_hdf5_to_weights_in_order('model/td3_critic_target_2.hdf5', self.critic_target_2)
