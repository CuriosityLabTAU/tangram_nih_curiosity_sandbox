from tangrams import *
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
import random


#
# task = Task()
# task.create_from_json('{"pieces": [["square", "0", "1 0"], ["small triangle1", "0", "0 0"]], "size": "5 5"}')

def create_all_tasks_from_cmpnd(json_str):

    task = Task()
    task.create_from_json(json_str)
    p1 = task.solution[0]
    p2 = task.solution[1]
    # p1.create('small triangle', '0', [0,0])
    # p2.create('square', '0', [1,0])
    q3 = p1.unite(p2)

    print q3.x
    piece_list = [q3]

    for p in piece_list:
        p.x = p.base()
    p_base = piece_list
    # Piece.p_base = p_base
    nodes = []
    for p in p_base:
        p_list = p.rotate()
        for q in p_list:
            nodes.extend(q.translate(task.I, task.J))

    task_list = []
    for k in range(len(nodes)):
        p_list = nodes[k].decompose()
        task = Task()
        task.set_shape(p_list)
        task_list.append(task)
        plt.imshow(task.x, interpolation='none')
        plt.pause(0.1)

    return task_list

def create_training_set_cmpnd(task_list):
    sol = Solver()
    task = Task()
    task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
    sol.set_initial_task(task)

    # dictionary: key = name of node, value = number of node
    dic = {}
    for n in range(len(sol.networks[0].nodes)):
        # print sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]
        dic[sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]] = n

    training_set_task = []
    training_set_input = []
    training_set_output = []

    for task in task_list:
        training_set_task.append(task)
        training_set_input.append((np.minimum(task.x, 1)).flatten()) # only 0/1 (not 1,2,5)


        # solve the orignial task using the solution
        activation = np.zeros_like(sol.networks[0].a)
        for piece in task.solution:
            node_num = dic[piece.name[0] + ' ' + piece.name[1] + ' ' + piece.name[2]]
            activation[node_num] = 1
        training_set_output.append(activation)

    return sol, training_set_task, training_set_input, training_set_output

def run_training(task_list):
    input_size = 289
    num_hidden = 312 # 40
    output_size = 312
    learning_rate = 0.05
    #
    # np.random.seed(1)
    # random.seed(1)

    # task_list = create_all_tasks_from_cmpnd('{"pieces": [["square", "0", "1 0"], ["small triangle1", "0", "0 0"]], "size": "5 5"}')
    sol, training_set_task, training_set_input, training_set_output = create_training_set_cmpnd(task_list)
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

    weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),  name='weights_2')
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
    path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/model_cmpnd.ckpt"


    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for epoch in range(5000*1):
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
        for step in range(len(training_set_output)):
            saver.restore(sess, path)
            temp_inp = np.array([training_set_input[step]])
            temp_out = np.array([training_set_output[step]])

            out, loss_val = sess.run([output, loss], feed_dict={inp: temp_inp, label: temp_out})
            desc = np.maximum(np.sign(out[0] - np.sort(out[0])[-3]), 0)
            print out
            print loss_val
            disp_training_data(training_set_input[step].reshape(17,17), desc, sol, 'test ' + str(step)) # ADD OUTPUT OF LEARNING


task_list = create_all_tasks_from_cmpnd('{"pieces": [["square", "0", "1 0"], ["small triangle1", "0", "0 0"]], "size": "5 5"}')
run_training(task_list)

# task_list = create_all_tasks_from_cmpnd('{"pieces": [["square", "0", "1 0"], ["small triangle1", "0", "0 0"]], "size": "5 5"}')
# sol, training_set_task, training_set_input, training_set_output = create_training_set_cmpnd(task_list)

# for task in task_list:
#     plt.imshow(task.x, interpolation='none')
#     plt.pause(0.1)