import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl


#####################  hyper parameters  ####################
LR_A = 0.0001                # learning rate for actor
LR_C = 0.0001                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 2400     # size of replay buffer
BATCH_SIZE = 256             # update batchsize

MAX_EPISODES = 200        # total number of episodes for training
MAX_EP_STEPS = 60          # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 0.0001                     # control exploration

decay_steps = 500  # Set the decay steps as per your requirement
decay_rate = 0.96  # Set the decay rate as per your requirement

###############################  DDPG  ####################################


class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, a_dim, s_dim, a_bound):
        #self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 +1), dtype=np.float32) # state, next_state, action, reward, final_state_flag
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # create actor network
        '''
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            # scale output to -action_bound to action_bound
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)
        '''
        
        def get_actor(input_state_shape, name=''):
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=40, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(x)
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l3')(x)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)
        
        # create critic network
        '''
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)
        '''
        
        
        def get_critic(input_state_shape, input_action_shape, name=''):
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=70, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l3')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)
        
        
        self.actor = get_actor([None, s_dim])
        self.critic_prepare = get_critic([None, s_dim], [None, a_dim],name='_prepare')
        self.critic_process = get_critic([None, s_dim], [None, a_dim],name='_process')
        self.critic_final = get_critic([None, s_dim], [None, a_dim],name='_final')
        self.actor.train()
        self.critic_prepare.train()
        self.critic_process.train()
        self.critic_final.train()

        # copy weights from actor to actor_target
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
        self.actor_target.eval()

    
        self.critic_prepare_target = get_critic([None, s_dim], [None, a_dim], name='_prepare_target')
        copy_para(self.critic_prepare, self.critic_prepare_target)
        self.critic_prepare_target.eval()

        self.critic_process_target = get_critic([None, s_dim], [None, a_dim], name='_process_target')
        copy_para(self.critic_process, self.critic_process_target)
        self.critic_process_target.eval()
        
        self.critic_final_target = get_critic([None, s_dim], [None, a_dim], name='_final_target')
        copy_para(self.critic_final, self.critic_final_target)
        self.critic_final_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement
        
        lr_schedule_actor = tf.keras.optimizers.schedules.ExponentialDecay(
            LR_A,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)

        lr_schedule_critic = tf.keras.optimizers.schedules.ExponentialDecay(
            LR_C,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)

        self.actor_opt = tf.optimizers.Adam(lr_schedule_actor)
        self.critic_opt = tf.optimizers.Adam(lr_schedule_critic)
        
        
        print('initiate DDPG model')
    '''    
    def call(self, state_inputs, action_inputs, final_state_flag):
        if final_state_flag:
            return self.critic_final([state_inputs, action_inputs])
        else:
            return self.critic_process([state_inputs, action_inputs])
    '''
    
    def emu_update_prepare(self):
        paras = self.actor.trainable_weights + self.critic_prepare.trainable_weights   
        self.ema.apply(paras)                                                   
        for i, j in zip(self.actor_target.trainable_weights + self.critic_prepare_target.trainable_weights, paras):
            i.assign(self.ema.average(j))
        

    def ema_update_process(self):
        paras = self.actor.trainable_weights + self.critic_process.trainable_weights   
        self.ema.apply(paras)                                                   
        for i, j in zip(self.actor_target.trainable_weights + self.critic_process_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       

    def ema_update_final(self):
        pars = self.critic_final.trainable_weights+ self.actor.trainable_weights
        self.ema.apply(pars)
        for i, j in zip(self.critic_final_target.trainable_weights + self.actor_target.trainable_weights, pars):
            i.assign(self.ema.average(j))
            
    # store transition
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]
    
    def add_noise(self, bt):
        bt_noise = np.tile(bt, (8, 1))
        #print(bt_noise.shape)
        for i in range(len(bt)):
           bt_noise[i+len(bt),self.s_dim+self.a_dim+1-1] = bt[i,self.s_dim+self.a_dim+1-1] + 0.3*np.random.normal(0, 0.01) # add alpha = 0.3 gaussian noise
           bt_noise[i+len(bt)*2,self.s_dim+self.a_dim+1-1] = bt[i,self.s_dim+self.a_dim+1-1] + 0.5*np.random.normal(0, 0.01) # add alpha = 0.5 gaussian noise
           bt_noise[i+len(bt)*3,self.s_dim+self.a_dim+1-1] = bt[i,self.s_dim+self.a_dim+1-1] + 0.8*np.random.normal(0, 0.01) # add alpha = 0.8 gaussian noise
           bt_noise[i+len(bt)*4,self.s_dim+self.a_dim+1-1] = bt[i,self.s_dim+self.a_dim+1-1] + np.random.normal(0, 0.01) # add alpha = 1 gaussian noise
           lam = bt[i][self.s_dim+self.a_dim+1-1]
           lam = np.nan_to_num(lam)
           lam = np.clip(lam, 0, None) # clip negative values to 0
           poisson_noise = np.random.poisson(lam, 3) # add poisson noise
           bt_noise[i+len(bt)*5,self.s_dim+self.a_dim+1-1] = poisson_noise[0] 
           bt_noise[i+len(bt)*6,self.s_dim+self.a_dim+1-1] = poisson_noise[1] 
           bt_noise[i+len(bt)*7,self.s_dim+self.a_dim+1-1] = poisson_noise[2]
        return bt_noise 
    
    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices_prepare = np.where(self.memory[:,-1]==0)
        indices_prepare = np.random.choice(indices_prepare[0], size=BATCH_SIZE)
        #bt = self.memory[indices, :]
        #bs = bt[:, :self.s_dim]
        # Prepare：
        bt_prepare = self.memory[indices_prepare,:]
        bs_prepare = bt_prepare[:, :self.s_dim]
        ba_prepare = bt_prepare[:, self.s_dim:self.s_dim + self.a_dim]
        #br_prepare = bt_prepare[:, -self.s_dim - 1:-self.s_dim]
        #bs__prepare = bt_prepare[:, -self.s_dim:]
        
        br_prepare = bt_prepare[:, self.s_dim + self.a_dim]
        bs__prepare = bt_prepare[:, self.s_dim + self.a_dim + 1:self.s_dim*2 + self.a_dim + 1]
        
        # Critic prepare：
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs__prepare)
            q__prepare = self.critic_prepare_target([bs__prepare, a_])
            y = br_prepare + GAMMA * q__prepare
            q_prepare = self.critic_prepare([bs_prepare, ba_prepare])
            td_error = tf.losses.mean_squared_error(y, q_prepare)
        c_grads = tape.gradient(td_error, self.critic_prepare.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic_prepare.trainable_weights))
        
        indices = np.where(self.memory[:,-1]==1)
        indices_process = np.random.choice(indices[0], size=BATCH_SIZE)    
        #bt = self.memory[indices, :]
        #bs = bt[:, :self.s_dim]
        # Process：
        #indices_process = np.where(bt[:,-1]==0)
        bt_process = self.memory[indices_process,:]                    
        bs_process = bt_process[:, :self.s_dim]                         
        ba_process = bt_process[:, self.s_dim:self.s_dim + self.a_dim]  
        #br_process = bt_process[:, -self.s_dim - 1:-self.s_dim]         
        #bs__process = bt_process[:, -self.s_dim:]
        
        br_process = bt_process[:, self.s_dim + self.a_dim]        
        bs__process = bt_process[:, self.s_dim + self.a_dim + 1:self.s_dim*2 + self.a_dim + 1]                       

        # Critic process：
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs__process)
            q__process = self.critic_process_target([bs__process, a_])
            y = br_process + GAMMA * q__process
            q_process = self.critic_process([bs_process, ba_process])
            td_error = tf.losses.mean_squared_error(y, q_process)
        c_grads = tape.gradient(td_error, self.critic_process.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic_process.trainable_weights))
        
        #q_process = self.critic_process([bs_process, ba_process])
        '''
        # Actor：
        # -q
        with tf.GradientTape() as tape:
            a = self.actor(bs_process)
            q = self.critic_process([bs_process, a])
            a_loss = -tf.reduce_mean(q)  # maximize the q
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        #print('a_grads',a_grads)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        '''
        # Final：
        indices_final = np.where(self.memory[:,-1]==2)
        #indices_final = np.random.choice(indices_final[0], size=4)
        if len(indices_final[0]) == 0:
            print("No matching indices found. Skipping this iteration.")
            return
        else:
            indices_final = np.random.choice(indices_final[0], size=4)

        #print('indices_final',indices_final)
        bt_final = self.memory[indices_final, :]
        
        bt_final_noise = self.add_noise(bt_final)
                          
        bs_final_noise = bt_final_noise[:, :self.s_dim]                         
        ba_final_noise = bt_final_noise[:, self.s_dim:self.s_dim + self.a_dim]  
        #br_final_noise = bt_final_noise[:, -self.s_dim - 1:-self.s_dim]         
        #bs__final_noise = bt_final_noise[:, -self.s_dim:] 
        br_final_noise = bt_final_noise[:, self.s_dim + self.a_dim]         
        bs__final_noise = bt_final_noise[:, self.s_dim + self.a_dim + 1:self.s_dim*2 + self.a_dim + 1]                      
            
        # Critic final：
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs__final_noise)
            q__final = self.critic_final_target([bs__final_noise, a_])
            y = br_final_noise + GAMMA * q__final
            q_final = self.critic_final([bs_final_noise, ba_final_noise])
            td_error = tf.losses.mean_squared_error(y, q_final)
        c_grads = tape.gradient(td_error, self.critic_final.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic_final.trainable_weights))
        
        #get q_final
        bs_final = bt_final[:, :self.s_dim]
        #ba_final = bt_final[:, self.s_dim:self.s_dim + self.a_dim]
        #q_final = self.critic_final([bs_final, ba_final])

        '''
        # Actor：
        # -q
        with tf.GradientTape() as tape:
            a = self.actor(bs_final)
            q = self.critic_final([bs_final, a])
            a_loss = -tf.reduce_mean(q)  # maximize the q
            print('a_loss',a_loss)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        print('a_grads',a_grads)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        '''
        
        # Actor：
        # -q
        with tf.GradientTape() as tape:
            # 假设 bs_prepare, bs_process, bs_final 均已定义并有相同的维度
            bs_combined = tf.concat([bs_prepare, bs_process, bs_final], axis=0)

            # 生成动作
            a = self.actor(bs_combined)

            # 获取各个 critic 网络的输出
            q_prepare = self.critic_prepare([bs_combined, a])
            q_process = self.critic_process([bs_combined, a])
            q_final = self.critic_final([bs_combined, a])

            # 对各个 critic 网络的输出进行加权组合
            q_weighted = tf.add_n([
                tf.multiply(q_prepare, 0.2),  # 假设为三个 critic 网络分配相同的权重
                tf.multiply(q_process, 0.4),
                tf.multiply(q_final, 0.4)
            ]) 
            a_loss = -tf.reduce_mean(q_weighted)
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        
        self.emu_update_prepare()
        self.ema_update_process()
        self.ema_update_final()                       


    # store transition
    def store_transition(self, s, a, r, s_, final_state_flag):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        transition = np.hstack((s, a, [r], s_, final_state_flag))

        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_prepare.hdf5', self.critic_prepare)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_prepare_target.hdf5', self.critic_prepare_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_process.hdf5', self.critic_process)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_process_target.hdf5', self.critic_process_target)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_final.hdf5', self.critic_final)
        tl.files.save_weights_to_hdf5('model/ddpg_critic_final_target.hdf5', self.critic_final_target)
        

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        print(tl.__version__)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_prepare.hdf5', self.critic_prepare)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_prepare_target.hdf5', self.critic_prepare_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_process.hdf5', self.critic_process)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_process_target.hdf5', self.critic_process_target)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_final.hdf5', self.critic_final)
        tl.files.load_hdf5_to_weights_in_order('model/ddpg_critic_final_target.hdf5', self.critic_final_target)