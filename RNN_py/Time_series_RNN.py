#stock price prediction using RNN
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

def MinMaxScalar(data):
	"""
	Min Max Normalization
	
	shape: [Batch_size, dimension]
	"""
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
	#noise term prevents the zero division
	return numerator / (denominator + 1e-7)

seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
iterations = 1000
lr = 0.01

#Stock price format (Open, High, Low, Close, Volume)
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1] #reverse the order (chronologically ordered)
xy = MinMaxScalar(xy) # to normalize the fluctuating values 
x = xy
y = xy[:, [-1]]

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
	_x = x[i:i + seq_length]
	_y = y[i + seq_length] #Next close price
	dataX.append(_x)
	dataY.append(_y)

#split to train and test data
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

#input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim]) #Output_dim = 1 

cell = rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#We use the last cell's output (since output_dim = 1) of the fully-connected layer
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)

#cost
loss = tf.reduce_sum(tf.square(Y_pred - Y))

#optimizer
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

for i in range(iterations):
	_, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
	print(i, ": ", "loss: ", l)

testPredict = sess.run(Y_pred, feed_dict={X: testX})
print('test prediction value: ', testPredict[:-1])

#plt.plot(testY)
#plt.plot(testPredict)
#plt.show() 
