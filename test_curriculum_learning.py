from tangrams import *
import tensorflow as tf

#run_test()
def create_curriculum_set():
    sol = Solver()
    task = Task()
    task.create_from_json('{"pieces": [["large triangle2", "180", "1 1"], ["medium triangle", "180", "0 1"], ["square", "0", "2 0"],["small triangle2", "90", "1 0"], ["small triangle1", "0", "2 3"], ["large triangle1", "0", "1 1"],["parrallelogram", "180", "1 3"]], "size": "5 5"}')

    sol.set_initial_task(task)
    jsons_list = ['{"pieces": [["large triangle2", "180", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["square", "0", "2 0"],["large triangle1", "0", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["square", "0", "2 0"],["small triangle2", "90", "1 0"], ["large triangle1", "0", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["medium triangle", "180", "0 1"],["square", "0", "2 0"], ["small triangle2", "90", "1 0"],["large triangle1", "0", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["medium triangle", "180", "0 1"],["square", "0", "2 0"], ["small triangle2", "90", "1 0"], ["small triangle1", "0", "2 3"],["large triangle1", "0", "1 1"]], "size": "5 5"}',
                  '{"pieces": [["large triangle2", "180", "1 1"], ["medium triangle", "180", "0 1"],["square", "0", "2 0"], ["small triangle2", "90", "1 0"], ["small triangle1", "0", "2 3"],["large triangle1", "0", "1 1"], ["parrallelogram", "180", "1 3"]], "size": "5 5"}'
                  ]

    # dictionary: key = name of node, value = number of node
    dic = {}
    for n in range(len(sol.networks[0].nodes)):
        # print sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]
        dic[sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[
            2]] = n

    training_set_task = []
    training_set_input = []
    training_set_output = []

    for i in range(len(jsons_list)):

        # generate a random tangram with N pieces
        # task.random_task(sol.networks[0], number_pieces=number_pieces)
        task.create_from_json(jsons_list[i])
        training_set_task.append(task)
        training_set_input.append((np.minimum(task.x, 1)).flatten())  # only 0/1 (not 1,2,5)

        # solve the orignial task using the solution
        activation = np.zeros_like(sol.networks[0].a)
        for piece in task.solution:
            node_num = dic[piece.name[0] + ' ' + piece.name[1] + ' ' + piece.name[2]]
            activation[node_num] = 1
        training_set_output.append(activation)

    return sol, training_set_task, training_set_input, training_set_output

def test_curric():
    input_size = 289
    num_hidden = 312  # 40
    output_size = 312
    learning_rate = 0.05

    def fill_feed_dict(inputlabel, outlabel):
        feed_diction = {inp: inputlabel, label: outlabel}
        # feed_diction = {'input': [1]*100, 'label': [0]*100}
        return feed_diction

    sol = Solver()
    task = Task()
    task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
    sol.set_initial_task(task)

    sol, training_set_task, training_set_input, training_set_output = create_curriculum_set()

    inp = tf.placeholder(tf.float32, shape=(None, input_size), name='input')
    label = tf.placeholder(tf.float32, shape=(None, output_size), name='label')

    weights_1 = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden))),
                            name='weights_1')
    biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')

    #weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),name='weights_2')
    # biases_2 = tf.Variable(tf.zeros([num_hidden]), name='biases_2')

    pre_act = tf.matmul(inp, weights_1)
    preactivation = pre_act + biases_1
    activation = tf.nn.sigmoid(preactivation)
    output = activation
    loss = tf.reduce_mean(tf.square(output - label))
    # loss = -tf.reduce_mean(output * tf.log(label))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/curric.ckpt"

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for epoch in range(5000 * 1):
            for batch_num in range(2):
                mini_batch_inp = np.array(training_set_input[0:6])
                mini_batch_out = np.array(training_set_output[0:6])
                feed_dict = fill_feed_dict(mini_batch_inp, mini_batch_out)
                maor, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                # if step % 100 == 0:

            print('loss {0}'.format(loss_value))
            print epoch

        save_path = saver.save(sess, path)

    neutral_duration = []
    cond_duration = []

    with tf.Session() as sess:
        # Restore variables from disk.
        # saver.restore(sess, path)
        print("Model restored.")
        step = 0
        for step in range(6):
            saver.restore(sess, path)
            temp_inp = np.array([training_set_input[step]])
            temp_out = np.array([training_set_output[step]])

            out, loss_val = sess.run([output, loss], feed_dict={inp: temp_inp, label: temp_out})
            H = np.sum((out - 1) * np.log2(1 - out) - out * np.log2(out))/len(out[0])
            print ('H',H)
            # desc = np.maximum(np.sign(out[0] - np.sort(out[0])[-4]), 0)
            desc = np.maximum(np.sign(out[0]-0.5),0)
            print out
            print loss_val
            disp_training_data(training_set_input[step].reshape(17, 17), desc, sol,
                               'test ' + str(step))  # ADD OUTPUT OF LEARNING
            plt.pause(2)
            # net, duration = sol.run_task(training_set_task[0], duration=20, stop=True, init_network=False)
            # neutral_duration.append(duration)
            # print ('neutral_duration', duration)
            # sol.set_activation(desc)
            # net, duration = sol.run_task(training_set_task[0], duration=20, stop=True, init_network=False)
            # cond_duration.append(duration)
            # print ('cond_duration', duration)

test_curric()