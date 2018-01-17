from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

tf.reset_default_graph()

sentence = open('test.txt', 'r').read()
char_set = list(set(sentence))
char_dic = {w:i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10
lr = 0.1

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i+1 : i + sequence_length + 1]
    
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    
    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

#One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)


cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)


X_for_softmax = tf.reshape(outputs, [-1,hidden_size])

softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])

outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

mean_loss = tf.reduce_mean(sequence_loss)

train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(mean_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
	if i % 50 == 0:
        	print(i, j, ''.join([char_set[t] for t in index]), l)


#Let's print the last char of each result to check if it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0: #print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='') #print the last char
