#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:59:15 2016

@author: joe

simple_bind example with mnist dataset
"""

import mxnet as mx
import logging
import numpy as np
import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)                              
                                
def onehot(data):
    nb_classes = len(np.unique(data))
    onehot_vec = np.zeros((data.shape[0], nb_classes))
    for i in range(onehot_vec.shape[0]):
        onehot_vec[i, data.astype("int32")[i]] = 1
    return onehot_vec

def handle_last_batch(data, batch_size, mode='discard'):
    num_data = data.shape[0]
    if mode=='pad':
        pad = batch_size - num_data%batch_size
        data = np.concatenate((data, data[:pad]), axis=0)
    else:  # discard tail
        discard_size = num_data%batch_size
        if discard_size>0:
            data = data[:-discard_size,:]
    return data

def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
    name.endswith("gamma") or name.endswith("beta")

def weight_intilize(args, initializer):
    for name in args:
        if is_param_name(name):
            initializer(name, args[name])
            
# Network 
def build_mlp():
    data_mnist = mx.symbol.Variable('data_mnist')  
    label_mnist = mx.symbol.Variable('label_mnist')
    fc1  = mx.symbol.FullyConnected(data = data_mnist, name='fc1', num_hidden=128)    
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")    
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)    
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")    
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, label=label_mnist, name = 'softmax')
    return mlp
    

#------------ Load data  -----------------
with open('mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)

dataX_train = mnist[0][0]
dataX_train = dataX_train.reshape((dataX_train.shape[0], -1))
dataY_train = mnist[0][1]

dataX_val = mnist[1][0]
dataX_val = dataX_val.reshape((dataX_val.shape[0], -1))
dataY_val = mnist[1][1]

# dataY_train = onehot(dataY_train)
# dataY_val = onehot(dataY_val)

print dataX_train.shape
print dataY_train.shape
print dataX_val.shape
print dataY_val.shape

batch_size = 100

 
network = build_mlp()
executor = network.simple_bind(ctx=mx.gpu(0), **{"data_mnist":(batch_size,784)})
args = executor.arg_dict

# Weight initializer
weight_intilize(args, mx.initializer.Xavier())

# Gradients & Optimizer
grads = executor.grad_dict
optimizer = mx.optimizer.SGD(momentum=0.9)
optimizer.lr = 0.01
updater  = mx.optimizer.get_updater(optimizer) # avoid handling state

nb_epoch = 10

# network layer names
keys = network.list_arguments()

pred_prob = mx.nd.zeros(executor.outputs[0].shape)

dataX_train = dataX_train / 255.
dataX_val = dataX_val / 255.

for epoch_i in range(nb_epoch):
    train_acc = 0.
    val_acc = 0.
    
    ## Shuffle
    idx = np.arange(dataX_train.shape[0])
    np.random.shuffle(idx)
    dataX_train = dataX_train[idx]
    dataY_train = dataY_train[idx]

    # Train
    batch_round = dataX_train.shape[0] / batch_size
    for batch_i in range(batch_round):
        mnist_data = dataX_train[batch_i*batch_size:(batch_i+1)*batch_size, :]
        label = dataY_train[batch_i*batch_size:(batch_i+1)*batch_size]
                             
        mx.nd.array(mnist_data).copyto(args["data_mnist"])
        mx.nd.array(label).copyto(args["label_mnist"])
        
        executor.forward(is_train=True)
        pred_prob[:] = executor.outputs[0]
        # print executor.outputs[0].asnumpy()
        
        executor.backward()
        # Update gradient
        for idx, key in enumerate(args):
            if is_param_name(key):
                ### !!! Remember to divide 'batch_size' !!!
                updater(idx, grads[key] / batch_size, args[key])
            
        train_acc += np.sum(pred_prob.asnumpy().argmax(axis=1)==label)*1.0 / batch_size
        # logging.info("Finish training iteration %d" % batch_i)
    train_acc /= batch_round
    logging.info("Epoch {} Training Acc {:.4f}".format(epoch_i, train_acc))
    
    
    ###### eval
    batch_round = dataX_val.shape[0] / batch_size
    for batch_i in range(batch_round):
        mnist_data = dataX_val[batch_i*batch_size:(batch_i+1)*batch_size, :]
        label = dataY_val[batch_i*batch_size:(batch_i+1)*batch_size]
                             
        mx.nd.array(mnist_data).copyto(args["data_mnist"])
        mx.nd.array(label).copyto(args["label_mnist"])
        
        executor.forward(is_train=False)
        pred_prob[:] = executor.outputs[0]
        # print pred_prob.asnumpy().argmax(axis=1) 
        val_acc += np.sum(pred_prob.asnumpy().argmax(axis=1)==label)*1.0 / batch_size
        # logging.info("Finish training iteration %d" % batch_i)
    val_acc /= batch_round
    logging.info("Epoch {} Validation Acc {:.4f}".format(epoch_i, val_acc))