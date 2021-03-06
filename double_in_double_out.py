'''

#-------concat_layer = act1 + act_f1

INFO:root:Epoch 1468 Training Acc 0.8980
INFO:root:Epoch 1468 Validation Acc_bingo 43.7143
INFO:root:Epoch 1468 Validation Acc_1away 78.2857
INFO:root:Epoch 1468 bestAcc:0.4757, bestBingo:47.5714, best1Away:82.0000

#-------concat_layer = act1 * act_f1
INFO:root:Epoch 1999 Training Acc 0.1617
INFO:root:Epoch 1999 Validation Acc_bingo 18.0000
INFO:root:Epoch 1999 Validation Acc_1away 54.0000
INFO:root:Epoch 1999 bestAcc:0.4329, bestBingo:43.2857, best1Away:85.1429

#-------concat_layer = Concate(act1,  act_f1)
INFO:root:Epoch 1999 Training Acc 0.9240
INFO:root:Epoch 1999 Validation Acc_bingo 42.4286
INFO:root:Epoch 1999 Validation Acc_1away 76.5714
INFO:root:Epoch 1999 bestAcc:0.4629, bestBingo:46.2857, best1Away:81.8571
'''


import mxnet as mx
import numpy as np
import logging

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.INFO, filename='output.log')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def handle_last_batch(data, batch_size, mode='discard'):
    num_data = data.shape[0]
    if mode=='pad':
        pad = batch_size - num_data%batch_size
        data = np.concatenate((data, data[:pad]), axis=0)
    else:  # discard tail
        discard_size = num_data%batch_size
        if discard_size>0:
            data = data[:-discard_size]
    return data
    
def getFeatures(data, feature):
    # 38 in total. [0:1] (order) [1:2](gross) [2:24](22 genres)  [24:38](properties) [38:39](label)  
    # Extract property columns and normalize
    data_x = data[:, 2:38]  # genres and properties
    data_y = data[:, 38:39]
    data_gross = np.squeeze(data[:, 1:2])
    
    data_id = np.squeeze(data[:, 0:1].astype(np.uint16))
    feature = feature[data_id, :]
    
    return data_x, data_y, feature, data_gross  


def dense_10(input_sym, dim_list):

    x  = mx.symbol.FullyConnected(data=input_sym, num_hidden=dim1)
    x = mx.symbol.Activation(data=x, act_type='relu')
    x = mx.symbol.Dropout(data=x, p=0.1)
    
    x  = mx.symbol.FullyConnected(data=x, num_hidden=dim2)
    x = mx.symbol.Activation(data=x, act_type='relu')
    x = mx.symbol.Dropout(data=x, p=0.1)
    
    x  = mx.symbol.FullyConnected(data=x, num_hidden=dim3)
    x = mx.symbol.Activation(data=x, act_type='relu')
    x = mx.symbol.Dropout(data=x, p=0.1)
    
    out = x + input_sym
    out = mx.symbol.Activation(data=out, act_type='relu')
    
    return out

def build_mlp():
    properties = mx.symbol.Variable('properties')
    features = mx.symbol.Variable('features')
    label_output = mx.symbol.Variable('label')
    gross_output = mx.symbol.Variable('gross')
    
    
    fc1  = mx.symbol.FullyConnected(data = properties, name='fc1_input', num_hidden=72)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
#    act1 = mx.symbol.Dropout(data=act1, p=0.2)
    
    fc_feature1  = mx.symbol.FullyConnected(data = features, name = 'fc_feature1', num_hidden=8)
    act_f1 = mx.symbol.Activation(data=fc_feature1, name='relu3', act_type="relu")
#    act_f1 = mx.symbol.Dropout(data=act_f1, p=0.2)

    act1 = mx.symbol.Concat(*[act1, act_f1], dim = 1, name = "concat")

#    net2 = mx.symbol.Concat(*[act1, act_f1], dim = 1, name = "concat")

    hidden_nums = [72,]*10
    funcs = ["relu"] * len(hidden_nums)
    for idx, (num, func) in enumerate(zip(hidden_nums, funcs)):
        act1  = mx.symbol.FullyConnected(data=act1, name='fc{}'.format(idx), num_hidden=num)
        act1 = mx.symbol.Activation(data=act1, name='relu{}'.format(idx), act_type=func)
        act1 = mx.symbol.Dropout(data=act1, p=0.1)
    
    cls_net = mx.symbol.FullyConnected(data = act1, num_hidden = 6, name = "fc25")
#    cls_net = mx.symbol.Activation(data = cls_net, act_type="sigmoid")
    cls_net = mx.symbol.SoftmaxOutput(data = cls_net, label=label_output, name="sf")
    
    #----------------   Regression
#    reg_net_128 = mx.symbol.FullyConnected(data = cls_net_128, num_hidden = 1, name = "fc1_reg")
#    reg_net_128 = mx.symbol.Activation(data = reg_net_128, act_type="sigmoid")
#    
#    reg_net_256 = mx.symbol.FullyConnected(data = cls_net_256, num_hidden = 1, name = "fc2_reg")
#    reg_net_256 = mx.symbol.Activation(data = reg_net_256, act_type="sigmoid")
#    
#    reg_net_128_s = mx.symbol.FullyConnected(data = cls_net_128_s, num_hidden = 1, name = "fc3_reg")
#    reg_net_128_s = mx.symbol.Activation(data = reg_net_128_s, act_type="sigmoid")
#    
#    reg_net_72 = mx.symbol.FullyConnected(data = cls_net_72, num_hidden = 1, name = "fc4_reg")
#    reg_net_72 = mx.symbol.Activation(data = reg_net_72, act_type="sigmoid")
#    
##    reg_fuse = mx.symbol.Concat(*[reg_net_128, reg_net_256, reg_net_128_s, reg_net_72], dim = 1, name = "concat")
#    # reg_fuse = reg_net_128 * reg_net_256 * reg_net_128_s * reg_net_72
#    reg_fuse = reg_net_128 + reg_net_256 + reg_net_128_s + reg_net_72
    reg_fuse = mx.symbol.FullyConnected(data = act1, num_hidden = 1, name = "reg_fuse")
#    reg_fuse = mx.symbol.Activation(data = reg_fuse, act_type="sigmoid")
    reg_net = mx.symbol.LinearRegressionOutput(data = reg_fuse, label=gross_output, name = "lro")
    
    return mx.symbol.Group([cls_net, reg_net]) 

    
def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
    name.endswith("gamma") or name.endswith("beta")

def weight_intilize(args, initializer):
    for name in args:
        if is_param_name(name):
            initializer(name, args[name])
            
#-------------------------Load Data-------------------------------       
data_train = np.load('./data/data_train.npy')
data_val = np.load('./data/data_val.npy')
bottleneck_features = np.load('./data/features4096.npy')
bottleneck_features_flat = bottleneck_features.reshape((bottleneck_features.shape[0], -1))

data_x_train, data_y_train, features_train, data_gross_train = getFeatures(data_train, bottleneck_features_flat)
data_x_val, data_y_val, features_val, data_gross_val = getFeatures(data_val, bottleneck_features_flat)
data_y_train = np.squeeze(data_y_train)
data_y_val = np.squeeze(data_y_val)

#-------------------------Build Network-------------------------------  
batch_size = 100
network = build_mlp()
executor = network.simple_bind(ctx=mx.gpu(0), 
                               properties=(batch_size, 36), 
                                features=(batch_size, 4096))
args = executor.arg_dict
# Weight initializer
weight_intilize(args, mx.initializer.Xavier())

#weight_intilize(args, mx.initializer.MSRAPrelu())

grads = executor.grad_dict

# Optimizer
#optimizer = mx.optimizer.SGD(momentum=0.9)
optimizer = mx.optimizer.Adam()
optimizer.lr = 0.005
updater  = mx.optimizer.get_updater(optimizer) # avoid handling state

optimizer_reg = mx.optimizer.Adam()
optimizer_reg.lr = 0.0005 
updater_reg  = mx.optimizer.get_updater(optimizer_reg) # avoid handling state
nb_epoch = 20000


keys = network.list_arguments()
pred_cls_prob = mx.nd.zeros(executor.outputs[0].shape)
pred_reg = mx.nd.zeros(executor.outputs[1].shape)

bestAcc = 0.
bestBingo = 0.
best1Away = 0.
bestMse = 100000.

for epoch_i in range(nb_epoch):
    train_acc = 0.
    val_acc = 0.
    val_acc_bingo = 0.
    val_acc_1away = 0.
    train_mse = 0.
    val_mse = 0.
    
    ## Shuffle
    idx = np.arange(data_x_train.shape[0])
    np.random.shuffle(idx)
    data_x_train = data_x_train[idx]
    features_train = features_train[idx]
    data_y_train = data_y_train[idx]
    data_gross_train = data_gross_train[idx]
    
    # handle the last batch
    data_x_train_handled = handle_last_batch(data_x_train, batch_size)
    data_y_train_handled = handle_last_batch(data_y_train, batch_size)
    features_train_handled = handle_last_batch(features_train, batch_size)
    data_gross_train_handled = handle_last_batch(data_gross_train, batch_size)
    
    data_x_val_handled = handle_last_batch(data_x_val, batch_size)
    data_y_val_handled = handle_last_batch(data_y_val, batch_size)
    features_val_handled = handle_last_batch(features_val, batch_size)
    data_gross_val_handled = handle_last_batch(data_gross_val, batch_size) 
    
    batch_round = data_x_train_handled.shape[0] / batch_size
    for batch_i in range(batch_round):
        properties = data_x_train_handled[batch_i*batch_size:(batch_i+1)*batch_size, :]
        features = features_train_handled[batch_i*batch_size:(batch_i+1)*batch_size, :]
        label = data_y_train_handled[batch_i*batch_size:(batch_i+1)*batch_size]
        gross = data_gross_train_handled[batch_i*batch_size:(batch_i+1)*batch_size]
                             
        mx.nd.array(properties).copyto(args["properties"])
        mx.nd.array(features).copyto(args["features"])
        mx.nd.array(label).copyto(args["label"])
        mx.nd.array(gross).copyto(args["gross"])
        
        executor.forward(is_train=True)
        pred_cls_prob[:] = executor.outputs[0]
        pred_reg[:] = executor.outputs[1]
        pred_reg_sqz = np.squeeze(pred_reg.asnumpy())
        # print pred_prob.asnumpy().argmax(axis=1)
        executor.backward()
        # Update gradient
        for idx, key in enumerate(args):
            if is_param_name(key):
#                updater(idx, grads[key]/batch_size, args[key])
                if key.find('reg')==-1:
                    #### remember to divide 'batch_size' !!!
                    updater(idx, grads[key]/batch_size, args[key])
                else:
                    pass
#                    bestBingo:42.8571, best1Away:84.0000
#                    updater_reg(idx, grads[key]/batch_size, args[key])
            
        train_acc += np.sum(pred_cls_prob.asnumpy().argmax(axis=1)==label)*1.0 / batch_size
        train_mse += np.sum((pred_reg_sqz-gross)**2)*1.0 / batch_size
        # logging.info("Finish training iteration %d" % batch_i)
    train_acc /= batch_round
    logging.info("Epoch {}                     Training Acc {:.4f}".format(epoch_i, train_acc))
    train_mse /= batch_round
    logging.info("Epoch {} ######## Training mse {:.4f}".format(epoch_i, train_mse))
    
    
    ###### eval
    batch_round = data_x_val_handled.shape[0] / batch_size
    for batch_i in range(batch_round):
        properties = data_x_val_handled[batch_i*batch_size:(batch_i+1)*batch_size, :]
        features = features_val_handled[batch_i*batch_size:(batch_i+1)*batch_size, :]
        label = data_y_val_handled[batch_i*batch_size:(batch_i+1)*batch_size]
        gross = data_gross_val_handled[batch_i*batch_size:(batch_i+1)*batch_size]
                             
        mx.nd.array(properties).copyto(args["properties"])
        mx.nd.array(features).copyto(args["features"])
        mx.nd.array(label).copyto(args["label"])
        mx.nd.array(gross).copyto(args["gross"])
        
        executor.forward(is_train=False)
        pred_cls_prob[:] = executor.outputs[0]
        pred_reg[:] = executor.outputs[1]
        pred_reg_sqz = np.squeeze(pred_reg.asnumpy())
        
        # print pred_prob.asnumpy().argmax(axis=1) 
        pred_diff = np.abs(pred_cls_prob.asnumpy().argmax(axis=1)-label)
        val_acc_bingo += np.sum(pred_diff==0)
        val_acc_1away += (np.sum(pred_diff==1) + np.sum(pred_diff==0))
        val_acc += np.sum(pred_cls_prob.asnumpy().argmax(axis=1)==label)*1.0 / batch_size
        
        val_mse += np.sum((pred_reg_sqz-gross)**2)*1.0 / batch_size
        # logging.info("Finish training iteration %d" % batch_i)
    val_acc /= batch_round
    val_acc_bingo /= batch_round
    val_acc_1away /= batch_round
    # logging.info("Epoch {} Validation Acc {:.4f}".format(epoch_i, val_acc))
    logging.info("Epoch {} Validation Acc_bingo {:.4f}".format(epoch_i, val_acc_bingo))
    logging.info("Epoch {} Validation Acc_1away {:.4f}".format(epoch_i, val_acc_1away))
    
    val_mse /= batch_round
    logging.info("Epoch {} ######## Validation mse {:.4f}".format(epoch_i, val_mse))
    
    if bestAcc<val_acc:
        bestAcc = val_acc
    if bestBingo<val_acc_bingo:
        bestBingo = val_acc_bingo
        # Save model
#        mx.callback.save_checkpoint('model/officebox', 0, network, executor.arg_dict, executor.aux_dict)
    if best1Away<val_acc_1away:
        best1Away = val_acc_1away
        # Save model
        mx.callback.save_checkpoint('model/officebox', 0, network, executor.arg_dict, executor.aux_dict)
    if bestMse>val_mse:
        bestMse = val_mse
    logging.info("Epoch {} bestMse:{:.4f}, bestBingo:{:.4f}, best1Away:{:.4f}".format(epoch_i, bestMse, val_acc_bingo, best1Away))

import matplotlib.pyplot as plt    
plt.plot(gross)
plt.plot(pred_reg_sqz)
