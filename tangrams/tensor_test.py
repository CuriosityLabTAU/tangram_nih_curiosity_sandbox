import tensorflow as tf
import math
import numpy as np

input_size = 100
num_hidden = 20
output_size = 100
learning_rate = 0.1


def fill_feed_dict(inputlabel, outlabel):
    feed_diction = {inp:inputlabel, label:outlabel}
    # feed_diction = {'input': [1]*100, 'label': [0]*100}
    return feed_diction


inp = tf.placeholder(tf.float32, shape=(1, input_size), name='input')
label = tf.placeholder(tf.float32, shape=(1, output_size), name='label')

weights_1 = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_1')
biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')

weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_2')
# biases_2 = tf.Variable(tf.zeros([num_hidden]), name='biases_2')

pre_act = tf.matmul(inp, weights_1)
preactivation = pre_act + biases_1
activation = tf.nn.sigmoid(preactivation)

output = tf.matmul(activation, weights_2)


loss = tf.reduce_mean(tf.square(output - label))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for step in range(200):
        temp = np.array([[1]*100])
        feed_dict = fill_feed_dict(temp, temp)
        maor, loss_value, W = sess.run([train_op, loss, weights_1], feed_dict=feed_dict)
        print('maor loast {0}'.format(loss_value))






