"""
CIFAR-10 Image classification dataset
Data available from and described at:
http://www.cs.toronto.edu/~kriz/cifar.html
If you use this dataset, please cite "Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
"""


import cPickle as pickle
import numpy as np

def load_cifar(path):
	train_data_x = np.zeros((60000, 32, 32, 3), dtype=np.uint8)
	train_data_y = np.zeros((60000, ), dtype=np.uint8)
	val_data_x = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
	val_data_y = np.zeros((10000, ), dtype=np.uint8)

	TrainFileList = ['data_batch_{}'.format(i) for i in range(1,6)]
	TestFileList = ['test_batch']

	def RT(X):
		return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

	for idx, fileName in enumerate(TrainFileList):
		fileNamePath = path+fileName
		print fileNamePath
		with open(fileNamePath, 'rb') as f:
			batch = pickle.load(f)
		train_data_x[idx*10000:(idx+1)*10000] = RT(batch['data'])
		train_data_y[idx*10000:(idx+1)*10000] = batch['labels']


	for idx,fileName in enumerate(TestFileList):
		fileNamePath = path + fileName
		print fileNamePath
		with open(fileNamePath, 'rb') as f:
			batch = pickle.load(f)
		val_data_x[idx*10000:(idx+1)*10000] = RT(batch['data'])
		val_data_y[idx*10000:(idx+1)*10000] = batch['labels']
	return (train_data_x, train_data_y, val_data_x, val_data_y)

if __name__=='__main__':
     cifar10 = load_cifar('/home/zy/datasets/cifar-10-batches-py/')
     
#	fileName = 'cifar10.pkl'
#	with open(fileName, 'wb') as f:
#			pickle.dump((train_data_x, train_data_y, val_data_x, val_data_y), f)

     print('finished')

