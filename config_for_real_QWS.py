#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# Network
net_arg = add_argument_group('Network')
# mase -- 隐藏层神经元 默认值为30个 （用于LSTM）
net_arg.add_argument('--hidden_dim', type=int, default=30, help='actor LSTM num_neurons') #128

# Data
data_arg = add_argument_group('Data')
# mase -- node节点 默认10个 （应该是state节点） 04-11mase--换成100了    mase 05-07 回归真实QWS数据集
data_arg.add_argument('--node_num', type=int, default=90, help='node num')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=500, help='nb epoch')
train_arg.add_argument('--pre_epoch', type=int, default=2000, help='pre-train epoch')
# mase -- 初始学习率
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')#
# mase -- 学习率 多少个Step 衰减一次
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
# mase -- 学习率 衰减的比例或速率 是多少
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')
# mase -- 最大 贪婪率
train_arg.add_argument('--max_epsilon', type=float, default=0.9, help='max_epsilon')#
# mase -- 最小 贪婪率      03-27 -- 根据multi-D3QN论文，将最小设定为0.01
train_arg.add_argument('--min_epsilon', type=float, default=0.1, help='min_epsilon')#
# mase -- 贪婪增加率      #    0.000001
train_arg.add_argument('--epsilon_increment', type=float, default=0.0001, help='epsilon')#
# mase -- 记忆区容量
train_arg.add_argument('--memory_capacity', type=int, default=300, help='memory_capacity')#
# mase -- 可能是  每迭代多少次 置换target
train_arg.add_argument('--target_replace_iter', type=int, default=50, help='target_replace_iter')#
# Misc
misc_arg = add_argument_group('User options') #####################################################
misc_arg.add_argument('--train_from', type=str, default='data', help='train data position')#test



def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
