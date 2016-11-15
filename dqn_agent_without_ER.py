# -*- coding: utf-8 -*-

import argparse
import copy

import pickle
import numpy as np
import scipy.misc as spm
import dnn_6_f
import dnn_6_BN
from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F



    
class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 100#10**4  # Initial exploratoin. original: 5x10^4
    #replay_size = 1000  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    
    def __init__(self, state_dimention, enable_controller,batchsize = 1):
        
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.replay_size = batchsize
        
        self.state_dimention = state_dimention
        print "Initializing DQN..."
        print "Model Building"
        self.model = dnn_6_BN.Q_DNN(self.state_dimention,200,self.num_of_actions)
        #self.model.to_gpu()
        
        
        self.model_target = copy.deepcopy(self.model)

        print "Initizlizing Optimizer"
        self.reset_optimizer()
        
    def forward(self, state, action, Reward, state_dash, episode_end):

        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.model.Q_func(s,train=True)  # Get Q-value

        # Generate Target Signals
        tmp = self.model_target.Q_func(s_dash,train=True)  # Q(s',*)
        tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data, dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end:
                tmp_ = Reward + self.gamma * max_Q_dash[i]
            else:
                tmp_ = Reward
            #print action
            action_index = self.action_to_index(action)
            target[i, action_index] = tmp_

        # TD-error clipping
        td = Variable(target) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, Q
        
    def online_update(self, time,state, action, reward, state_dash,episode_end_flag):
    
        self.model.cleargrads()
        self.model_target.cleargrads()#これをやる必要あるかは不明
        loss, _ = self.forward(state.astype(np.float32), action, reward, state_dash.astype(np.float32), episode_end_flag)
        loss.backward()
        self.optimizer.update()
        

    
    def e_greedy(self, state, epsilon):
        
        s = Variable(state)
        Q = self.model.Q_func(s,train=False)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)

        else:
            index_action = np.argmax(Q)

        return self.index_to_action(index_action), Q

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)
    
    def save_model(self, folder_name, epoch):
        print 'save model'
        self.model.to_cpu()
        with open(folder_name+'model'+str(epoch),'wb') as o:
            pickle.dump(self.model,o)
        self.model.to_gpu()
        self.optimizer.setup(self.model)

    def load_model(self, model):
        with open(model, 'rb') as m:
            print "open " + model
            self.model = pickle.load(m)
            print 'load model'
            self.model.to_gpu()
            
    def get_model_copy(self):
        return copy.deepcopy(self.model)
        
    def model_to_gpu(self):
        self.model.to_gpu()
        
    def reset_optimizer(self):
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model)
        
class dqn_agent():  # RL-glue Process
    #lastAction = Action()
    policyFrozen = False
    learning_freq = 1#何日ごとに学習するか
    
    def __init__(self,enable_controller,state_dimention=0,epsilon_discount_size=0):
        
        self.enable_controller = enable_controller
        self.state_dimention = state_dimention
        self.epsilon_discount_size = epsilon_discount_size

    def agent_init(self):
        # Some initializations for rlglue
        #self.lastAction = Action()

        self.time = 0
        self.learned_time = 0
        self.epsilon = 0.05  # Initial exploratoin rate
        self.max_Q_list = []
        self.reward_list = []
        self.Q_recent = 0
        
        # Pick a DQN from DQN_class
        self.DQN = DQN_class(state_dimention=self.state_dimention,enable_controller=self.enable_controller)  # default is for "Pong".

    def agent_start(self, observation):

        
        # Initialize State
        self.state = observation
        state_ = np.asanyarray(self.state, dtype=np.float32)

        # Generate an Action e-greedy
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)
        self.Q_recent = Q_now[0]
        # Update for next step
        self.lastAction = action
        self.last_state = self.state.copy()
        self.last_observation = observation.copy()
        self.max_Q_list.append(np.max(self.Q_recent))
        
        return action
        
    def agent_step(self, reward, observation):

        self.state = observation
        state_ = np.asanyarray(self.state, dtype=np.float32)


        # Generate an Action by e-greedy action selection
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)
        self.Q_recent = Q_now[0]

        self.max_Q_list.append(np.max(self.Q_recent))
        self.reward_list.append(reward)

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            if (self.time % self.learning_freq) == 0:
                
                self.DQN.online_update(self.learned_time, self.last_state, self.lastAction, reward, self.state, False)
                self.learned_time += 1

        # Target model update
        if self.DQN.initial_exploration < self.learned_time and np.mod(self.learned_time, self.DQN.target_model_update_freq) == 0:
            #print "########### MODEL UPDATED ######################"
            self.DQN.target_model_update()
            
        # Simple text based visualization
        #print ' Time Step %d /   ACTION  %d  /   REWARD %.4f   / EPSILON  %.6f  /   Q_max  %3f' % (self.time, action, reward, eps, np.max(Q_now.get()))
        #print Q_now.get()

        # Updates for next step
        self.last_observation = observation.copy()

        if self.policyFrozen is False:
            self.lastAction = action
            self.last_state = self.state.copy()
            self.time += 1

        return action

    def agent_end(self, reward):  # Episode Terminated

        self.reward_list.append(reward)
        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            if (self.time % self.learning_freq) == 0:
                
                #self.DQN.online_update()
                self.learned_time += 1

        # Target model update
        if self.DQN.initial_exploration < self.learned_time and np.mod(self.learned_time, self.DQN.target_model_update_freq) == 0:
            #print "########### MODEL UPDATED ######################"
            self.DQN.target_model_update()
            
            
        # Simple text based visualization
        #print '  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon)

        # Time count
        if self.policyFrozen is False:
            self.time += 1
    
    def init_max_Q_list(self):
        self.max_Q_list = []
        
    def init_reward_list(self):
        self.reward_list = []
        
    def get_average_Q(self):
        return sum(self.max_Q_list)/len(self.max_Q_list)
        
    def get_average_reward(self):
        return sum(self.reward_list)/len(self.reward_list)
    
    def get_variance_Q(self):
        return np.var(np.array(self.max_Q_list))
        
    def get_varance_reward(self):
        return np.var(np.array(self.reward_list))
        
    def get_learned_time(self):
        return self.learned_time
        
    def agent_cleanup(self):
        pass




