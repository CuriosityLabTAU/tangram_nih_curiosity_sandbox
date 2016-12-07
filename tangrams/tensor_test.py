from tangrams.training_set import *
import tensorflow as tf
import math
import numpy as np
import random


def run_test():
    input_size = 289
    num_hidden = 312 # 40
    output_size = 312
    learning_rate = 0.05

    np.random.seed(1)
    random.seed(1)
    sol, training_set_task, training_set_input, training_set_output = create_training_set(set_size = 1000, number_pieces = 3)
    # for k in range(len(training_set_output)):
    #     disp_training_data(training_set_input[k*3], training_set_input[k*3+1], training_set_input[k*3+2], training_set_output[k], sol,  'test '+str(k))


    def fill_feed_dict(inputlabel, outlabel):
        feed_diction = {inp:inputlabel, label:outlabel}
        # feed_diction = {'input': [1]*100, 'label': [0]*100}
        return feed_diction


    inp = tf.placeholder(tf.float32, shape=(None, input_size), name='input')
    label = tf.placeholder(tf.float32, shape=(None, output_size), name='label')

    weights_1 = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_1')
    biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')

    # weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_2')
    # biases_2 = tf.Variable(tf.zeros([num_hidden]), name='biases_2')

    pre_act = tf.matmul(inp, weights_1)
    preactivation = pre_act + biases_1
    activation = tf.nn.sigmoid(preactivation)
    output = activation

    # output_ = tf.matmul(activation, weights_2)
    # output = tf.nn.sigmoid(output_)

    # output = tf.nn.sigmoid(output_)
    # output = tf.maximum(tf.sign(output_ - 0.5),0)
    # decision = tf.maximum(tf.sign(output - 0.5),0)
    # tf.nn.top_k(output, k=3, sorted=True, name=None)
    # sorted = np.sort(output)
    # decision = tf.maximum(tf.sign(output-sorted[-4]),0)

    loss = tf.reduce_mean(tf.square(output - label))
    # loss = -tf.reduce_mean(output * tf.log(label))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver()
    path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/model.ckpt"


    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for epoch in range(50000*1):
            for batch_num in range(1):
                mini_batch_inp = np.array(training_set_input[0:100+batch_num*100])
                mini_batch_out = np.array(training_set_output[0:100+batch_num*100])
                feed_dict = fill_feed_dict(mini_batch_inp, mini_batch_out)
                maor, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                #if step % 100 == 0:

            print('loss {0}'.format(loss_value))
            print epoch

        save_path = saver.save(sess, path)
            #
            # for step in range(2):
            #     temp_inp = np.array([training_set_input[step]])
            #     #temp = np.array([[1]*100])
            #     temp_out = np.array([training_set_output[step]])
            #     feed_dict = fill_feed_dict(temp_inp, temp_out)
            #     maor, loss_value, W = sess.run([train_op, loss, weights_1], feed_dict=feed_dict)
            #     if step % 100 == 0:
            #         print('loss {0}'.format(loss_value))

        # save_path = saver.save(sess, path)


    print '                                                                                       '

    #sess = tf.Session()

    with tf.Session() as sess:
        # Restore variables from disk.
        # saver.restore(sess, path)
        print("Model restored.")
        step=0
        for step in range(100):
            saver.restore(sess, path)
            temp_inp = np.array([training_set_input[step]])
            temp_out = np.array([training_set_output[step]])

            out, loss_val = sess.run([output, loss], feed_dict={inp: temp_inp, label: temp_out})
            desc = np.maximum(np.sign(out[0] - np.sort(out[0])[-4]), 0)
            print out
            print loss_val
            disp_training_data(training_set_input[step].reshape(17,17), desc, sol, 'test ' + str(step)) # ADD OUTPUT OF LEARNING

#            disp_training_data(training_set_input[step * 3], out[0]>0.4*max(out[0]), training_set_input[step * 3 + 2], training_set_output[step], sol, 'test ' + str(step)) # ADD OUTPUT OF LEARNING
# def run_test_with_rotation():
#     input_size = 890
#     num_hidden = 200
#     output_size = 312
#     learning_rate = 0.01
#
#     np.random.seed(1)
#     random.seed(1)
#     sol, training_set_input, training_set_output = create_training_set()
#     for k in range(len(training_set_output)):
#         disp_training_data(training_set_input[k*3], training_set_input[k*3+1], training_set_input[k*3+2], training_set_output[k], sol,  'test '+str(k))
#
#
#     def fill_feed_dict(inputlabel, outlabel):
#         feed_diction = {inp:inputlabel, label:outlabel}
#         # feed_diction = {'input': [1]*100, 'label': [0]*100}
#         return feed_diction
#
#
#     inp = tf.placeholder(tf.float32, shape=(1, input_size), name='input')
#     label = tf.placeholder(tf.float32, shape=(1, output_size), name='label')
#
#     weights_1 = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_1')
#     biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')
#
#     weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_2')
#     # biases_2 = tf.Variable(tf.zeros([num_hidden]), name='biases_2')
#
#     pre_act = tf.matmul(inp, weights_1)
#     preactivation = pre_act + biases_1
#     activation = tf.nn.sigmoid(preactivation)
#
#     output_ = tf.matmul(activation, weights_2)
#     output = tf.maximum(tf.sign(output_ - 0.5),0)
#
#     loss = tf.reduce_mean(tf.square(output - label))
#     # loss = -tf.reduce_mean(output * tf.log(label))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     train_op = optimizer.minimize(loss)
#     saver = tf.train.Saver()
#     path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/model.ckpt"
#
#
#     with tf.Session() as sess:
#         init = tf.initialize_all_variables()
#         sess.run(init)
#         for epoch in range(6):
#
#             for step in range(10000):
#                 temp_inp = np.concatenate((training_set_input[step*3].flatten(), training_set_input[step*3+1], training_set_input[step*3+2].flatten()))
#                 temp_inp = np.array([temp_inp])
#                 #temp = np.array([[1]*100])
#                 temp_out = np.array([training_set_output[step]])
#                 feed_dict = fill_feed_dict(temp_inp, temp_out)
#                 maor, loss_value, W = sess.run([train_op, loss, weights_1], feed_dict=feed_dict)
#                 print('loss {0}'.format(loss_value))
#
#         save_path = saver.save(sess, path)
#
#
#     print '                                                                                       '
#
#     with tf.Session() as sess:
#         # Restore variables from disk.
#         saver.restore(sess, path)
#         print("Model restored.")
#         step=0
#         temp_inp = np.concatenate((training_set_input[step * 3].flatten(), training_set_input[step * 3 + 1],
#                                    training_set_input[step * 3 + 2].flatten()))
#         temp_inp = np.array([temp_inp])
#         temp_out = np.array([training_set_output[step]])
#         output, loss_val = sess.run([output, loss], feed_dict={inp: temp_inp, label: temp_out})
#         print output
#         print loss_val


# run_test()
