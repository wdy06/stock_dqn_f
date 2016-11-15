# -*- coding: utf-8 -*-



import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class Q_DNN(chainer.Chain):
    
    modelname = 'dnn6_BN'
    layer_num = 6

    
    def __init__(self, input_num, hidden_num,num_of_actions):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_of_actions = num_of_actions
        
        super(Q_DNN, self).__init__(
            fc1=L.Linear(self.input_num, self.hidden_num),
            bn1=L.BatchNormalization(self.hidden_num),
            fc2=L.Linear(self.hidden_num, self.hidden_num),
            bn2=L.BatchNormalization(self.hidden_num),
            fc3=L.Linear(self.hidden_num, self.hidden_num),
            bn3=L.BatchNormalization(self.hidden_num),
            fc4=L.Linear(self.hidden_num, self.hidden_num),
            bn4=L.BatchNormalization(self.hidden_num),
            fc5=L.Linear(self.hidden_num, self.hidden_num),
            bn5=L.BatchNormalization(self.hidden_num),
            q_value=L.Linear(self.hidden_num, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, self.hidden_num),
                                               dtype=np.float32))
            
        )
        
    def Q_func(self, state, train=True):
        
        test = not train
        
        h = F.tanh(self.bn1(self.fc1(state),test=test))
        h = F.tanh(self.bn2(self.fc2(h),test=test))  
        h = F.tanh(self.bn3(self.fc3(h),test=test))
        h = F.tanh(self.bn4(self.fc4(h),test=test))
        h = F.tanh(self.bn5(self.fc5(h),test=test))
        Q = self.q_value(h)
        
        return Q
        
    

        