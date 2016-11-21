from tangrams import *
import tensorflow as tf

#run_test()

input_size = 289
num_hidden = 312  # 40
output_size = 312
learning_rate = 0.05

sol = Solver()
task = Task()
task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
sol.set_initial_task(task)

sol, training_set_task, training_set_input, training_set_output = create_training_set(set_size = 100, number_pieces = 3)

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
path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/model.ckpt"

neutral_duration = []
cond_duration = []

with tf.Session() as sess:
    # Restore variables from disk.
    # saver.restore(sess, path)
    print("Model restored.")
    step = 0
    for step in range(100):
        saver.restore(sess, path)
        temp_inp = np.array([training_set_input[step]])
        temp_out = np.array([training_set_output[step]])

        out, loss_val = sess.run([output, loss], feed_dict={inp: temp_inp, label: temp_out})
        desc = np.maximum(np.sign(out[0] - np.sort(out[0])[-4]), 0)
        # print out
        print loss_val
        disp_training_data(training_set_input[step].reshape(17, 17), desc, sol,
                           'test ' + str(step))  # ADD OUTPUT OF LEARNING

        net, duration = sol.run_task(training_set_task[0], duration=200, stop=True, init_network=False)
        neutral_duration.append(duration)
        print ('neutral_duration', duration)
        sol.set_activation(desc)
        net, duration = sol.run_task(training_set_task[0], duration=200, stop=True, init_network=False)
        cond_duration.append(duration)
        print ('cond_duration', duration)

