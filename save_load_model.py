"""
Created on Mon Nov 28 15:15:19 2016

@author: zy

Two methods to save and load model
"""

import mxnet as mx
import logging
import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

with open('mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)

dataX_train = mnist[0][0]
dataX_train = dataX_train.reshape((dataX_train.shape[0], -1))
dataY_train = mnist[0][1]
dataX_val = mnist[1][0]
dataX_val = dataX_val.reshape((dataX_val.shape[0], -1))
dataY_val = mnist[1][1]

print dataX_train.shape
print dataY_train.shape
print dataX_val.shape
print dataY_val.shape

batch_size = 100
train_iter = mx.io.NDArrayIter(dataX_train/255., dataY_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(dataX_val/255., dataY_val, batch_size=batch_size)

# Network 
data = mx.symbol.Variable('data')

fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)

act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)

act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")

fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)

mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')


model = mx.model.FeedForward(
    ctx = mx.cpu(),      # Run on GPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularizatio

model_prefix = 'models/mnist'    ## you need to create the folder 'model' manually  
model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter, 
    eval_metric=['acc'],  # to diaplsy custom accuracy
    batch_end_callback=[mx.callback.Speedometer(batch_size, 600)],  # To display training accuracy, 600 means 600*batch_size
    epoch_end_callback=mx.callback.do_checkpoint(model_prefix, period=5)  ## 1. It will save parameters to a model file every period epoch
)

## 1-1. Save model after fit
model.save(model_prefix)

## 1-2. Manually Save model when using simple_bind (executor=network.simple_bind(xxx))
mx.callback.save_checkpoint('model/officebox', 0, network, executor.arg_dict, executor.aux_dict)

## 2-1. load model
# Load the pre-trained model (param and json)
prefix = "models/mnist"
num_epoch = 10
model = mx.model.FeedForward.load(prefix, num_epoch, ctx=mx.gpu(), numpy_batch_size=1)
print('model {} loaded'.format(prefix+str(num_epoch)))

## 2-2. load checkpoint (param and json)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_epoch)

## 2-3. load checkpoint (only *.param file)
save_dict = mx.nd.load('vgg/%s-%04d.params' % ('vgg16', 1))
arg_params = {}
aux_params = {}
for k, v in save_dict.items():
    tp, name = k.split(':', 1)
    if tp == 'arg':
        arg_params[name] = v
    if tp == 'aux':
        aux_params[name] = v
'''
mod.fit(train, eval_data=val, optimizer_params=optim_args,
            eval_metric=eval_metrics, num_epoch=args.num_epochs,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch,
            )
'''
