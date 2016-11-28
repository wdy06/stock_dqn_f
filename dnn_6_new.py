# -*- coding: utf-8 -*-



import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable

class Q_DNN(chainer.Chain):
    
    modelname = 'dnn6_new'
    layer_num = 6
    
    
    def __init__(self, input_num, hidden_num,num_of_actions):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_of_actions = num_of_actions
        self.agent_state_dim = 4
        self.market_state_dim = input_num - self.agent_state_dim
        assert self.market_state_dim > 0
        
        super(Q_DNN, self).__init__(
            a1=L.Linear(self.agent_state_dim, 2),
            a2=L.Linear(2, 2),
            a3=L.Linear(2, 2),
            s1=L.Linear(self.market_state_dim, self.hidden_num),
            s2=L.Linear(self.hidden_num, self.hidden_num),
            s3=L.Linear(self.hidden_num, self.hidden_num),
            fc4=L.Linear(self.hidden_num + 2, self.hidden_num),
            fc5=L.Linear(self.hidden_num, self.hidden_num),
            q_value=L.Linear(self.hidden_num, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, self.hidden_num),
                                               dtype=np.float32))
            
        )
        
    def Q_func(self, state):
        if state.ndim == 2:
            agent_state = state[:, - self.agent_state_dim :]
            market_state = state[:,:self.market_state_dim]

        elif state.ndim == 3:
            agent_state = state[:, :,- self.agent_state_dim :]
            market_state = state[:,:,:self.market_state_dim]
        
        a_state = Variable(agent_state)
        m_state = Variable(market_state)
        a = F.tanh(self.a1(a_state))
        a = F.tanh(self.a2(a))
        a = F.tanh(self.a3(a))
        m = F.tanh(self.s1(m_state))
        m = F.tanh(self.s2(m))
        m = F.tanh(self.s3(m))
        new_state = F.concat((a, m), axis=1)

        h = F.tanh(self.fc4(new_state))
        h = F.tanh(self.fc5(h))
        Q = self.q_value(h)

        return Q
        
    

        