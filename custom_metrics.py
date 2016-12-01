"""
Created on Mon Nov 28 15:15:19 2016

@author: zy

Two methods to define your own metric
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

# 1. Custom accuracy class
class AccuracyCustom(mx.metric.EvalMetric):

    def __init__(self):
        super(AccuracyCustom, self).__init__('AccuracyCustom')

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            pred_label = mx.ndarray.argmax_channel(pred_label).asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

# 2. Custom accuracy callback function (as parameter of mx.metric.CustomMetric)
def customEval(label, pred):
    sum_metric = (label == pred.argmax(axis=1)).sum()
    num_inst = len(label)
    return (sum_metric, num_inst)
    
model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter, 
    eval_metric=['acc', mx.metric.CustomMetric(customEval), AccuracyCustom()],  # to diaplsy custom accuracy
    batch_end_callback=[mx.callback.Speedometer(batch_size, 600)]  # To display training accuracy, 600 means 600*batch_size
)
