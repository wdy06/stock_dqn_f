# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle
import env_stockmarket
import dqn_agent_nature
import dqn_agent_without_ER
import copy
import tools
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda

def trading_files(files):
    pass
    

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('model',help='path of using agent')
parser.add_argument('--input_num', '-in', default=60, type=int,
                    help='input node number')
parser.add_argument('--channel', '-c', default=8, type=int,
                    help='data channel')
parser.add_argument('--experiment_name', '-n', default='experiment',type=str,help='experiment name')
parser.add_argument('--action_split_number', '-asn', type=int, default=2,help='how many split action')
parser.add_argument('--online_update', '-ou', type=int, default=0,help='not use online update:0,use online update:1')
parser.add_argument('--u_vol', '-vol',type=int,default=1,
                    help='use vol or no')
parser.add_argument('--u_ema', '-ema',type=int,default=1,
                    help='use ema or no')
parser.add_argument('--u_rsi', '-rsi',type=int,default=1,
                    help='use rsi or no')
parser.add_argument('--u_macd', '-macd',type=int,default=0,
                    help='use macd or no')
parser.add_argument('--u_stoch', '-stoch',type=int,default=1,
                    help='use stoch or no')
parser.add_argument('--u_wil', '-wil',type=int,default=1,
                    help='use wil or no')
                    
args = parser.parse_args()

if args.u_vol == 0: u_vol = False
elif args.u_vol == 1: u_vol = True
if args.u_ema == 0: u_ema = False
elif args.u_ema == 1: u_ema = True
if args.u_rsi == 0: u_rsi = False
elif args.u_rsi == 1: u_rsi = True
if args.u_macd == 0: u_macd = False
elif args.u_macd == 1: u_macd = True
if args.u_stoch == 0: u_stoch = False
elif args.u_stoch == 1: u_stoch = True
if args.u_wil == 0: u_wil = False
elif args.u_wil == 1: u_wil = True


if args.gpu >= 0:
    cuda.check_cuda_available()
    print "use gpu"

folder = './test_result/' + args.experiment_name + '/'
if os.path.isdir(folder) == True:
    print 'this experiment name is existed'
    print 'please change experiment name'
    raw_input()
else:
    print 'make experiment folder'
    os.makedirs(folder)

#コントローラの設定
enable_controller = range( - args.action_split_number,args.action_split_number + 1)
print 'enable_controller:',enable_controller
    
END_TRAIN_DAY = 20081230
START_TEST_DAY = 20090105
#START_TEST_DAY = 20100104


org_model = 0
#モデルの読み込み
#not use online update
if args.online_update == 0:
    Agent = dqn_agent_nature.dqn_agent(gpu_id = args.gpu,state_dimention=1,enable_controller=enable_controller)
    Agent.agent_init()
    Agent.DQN.load_model(args.model)
    Agent.policyFrozen = True

#use online update
elif args.online_update == 1:
    Agent = dqn_agent_without_ER.dqn_agent(state_dimention=1,enable_controller=enable_controller)
    Agent.agent_init()
    #オリジナルを改変しないようにコピー
    with open(args.model, 'rb') as m:
        print "open " + args.model
        org_model = pickle.load(m)
    Agent.DQN.model = copy.deepcopy(org_model)
    Agent.policyFrozen = False
    
market = env_stockmarket.StockMarket(END_TRAIN_DAY,START_TEST_DAY,u_vol=u_vol,u_ema=u_ema,u_rsi=u_rsi,u_macd=u_macd,u_stoch=u_stoch,u_wil=u_wil)

files = os.listdir("./nikkei100")

Agent.init_max_Q_list()
Agent.init_reward_list()
profit_list = []


for f in files:
    print f
    if args.online_update == 1:
        #銘柄ごとに初期化
        Agent.DQN.model = copy.deepcopy(org_model)
        Agent.DQN.model_target = copy.deepcopy(org_model)
        Agent.DQN.reset_optimizer()
        
    stock_agent = env_stockmarket.Stock_agent(Agent,args.action_split_number)
    
    try:
        testdata,testprice = market.get_testData(f,args.input_num)
        #testdata, testprice = market.get_trainData(f,END_TRAIN_DAY,args.input_num)
    except:
        print 'skip',f
        continue
        
    profit_ratio, proper, order, stocks, price, Q_list, ave_buyprice_list ,reward_list= stock_agent.trading_test(args.input_num,testprice,testdata)
    profit_list.append(profit_ratio)
    
    
    tools.listToCsv(folder+str(f).replace(".CSV", "")+'.csv', price, proper, order,stocks,ave_buyprice_list,reward_list)
    
    buy_order, sell_order = tools.order2buysell(order,price)

    #2軸使用
    
    fig, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.set_ylabel('price')
    axis1.set_ylabel('buy')
    axis1.set_ylabel('sell')
    axis2.set_ylabel('property')
    axis1.plot(price, label = "price")
    axis1.plot(buy_order,'o',label='buy point')
    axis1.plot(sell_order,'^',label='sell point')
    axis1.legend(loc = 'upper left')
    axis2.plot(proper, label = 'property', color = 'g')
    axis2.legend()
    filename = folder + str(f).replace(".CSV", "") + ".png"
    plt.savefig(filename)
    plt.close()
    
    plt.subplot(2, 1, 1)
    plt.plot(stocks,label='stocks')
    plt.legend()
    plt.subplot(2, 1, 2)
    Q_list = np.array(Q_list)
    for i in range(len(Q_list[0])):
        plt.plot(Q_list[:,i],label=str(enable_controller[i]))
        
    plt.legend()
    filename = folder + str(f).replace(".CSV", "") + "_sub.png"
    plt.savefig(filename)
    plt.close()
    
print 'average profit:', sum(profit_list)/len(profit_list)