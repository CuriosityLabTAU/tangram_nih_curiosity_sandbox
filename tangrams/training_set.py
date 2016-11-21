

# generate training set tuple
from tangrams import *
import random
import matplotlib.pyplot as plt

def disp_training_data(board_before, activation_before, solver, title):
#    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(board_before, interpolation='none')
    plt.title(title+' board_before')

    x_before = np.zeros_like(solver.networks[0].nodes[0].x)
    for n in range(len(activation_before)):
        if activation_before[n] == 1:
            x_before += solver.networks[0].nodes[n].x
    #plt.figure()
    plt.subplot(2, 1, 2)
    plt.imshow(x_before, interpolation='none')
    plt.title(title+' activation_before')

    #plt.figure()
    # plt.subplot(1, 4, 3)
    # plt.imshow(board_after, interpolation='none')
    # plt.title(title + ' board_after')
    #
    # x_after = np.zeros_like(solver.networks[0].nodes[0].x)
    # for n in range(len(activation_after)):
    #     if activation_after[n] == 1:
    #         x_after += solver.networks[0].nodes[n].x
    # #plt.figure()
    # plt.subplot(1, 4, 4)
    # plt.imshow(x_after, interpolation='none')
    # plt.title(title + ' activation_after')
    plt.pause(1)


def create_training_set(set_size = 100, number_pieces = 6):
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

    for i in range(set_size):

        # generate a random tangram with N pieces
        task.random_task(sol.networks[0], number_pieces=number_pieces)
        training_set_task.append(task)
        training_set_input.append((np.minimum(task.x, 1)).flatten()) # only 0/1 (not 1,2,5)


        # solve the orignial task using the solution
        activation = np.zeros_like(sol.networks[0].a)
        for piece in task.solution:
            node_num = dic[piece.name[0] + ' ' + piece.name[1] + ' ' + piece.name[2]]
            activation[node_num] = 1
        training_set_output.append(activation)

        # sol.set_initial_task(task)
        # sol.run_task(task, duration=100, stop=True)
        # training_set_input.append(sol.networks[0].a)

# # rotate 90 TODO: don't rotate by this, but rotate_compound
        # task.x = np.rot90(task.x)
        # training_set_input.append(np.minimum(task.x, 1))
        #
        # # solve the rotated task without solver
        #
        # activation = np.zeros_like(sol.networks[0].a)
        # rot_ind = 1  #  rotation index: 0->0 1->90 2->180 3->270
        # [i_max, j_max] = task.x.shape
        # i_max = (i_max - 1) / Piece.JUMP - 1
        # j_max = (j_max - 1) / Piece.JUMP - 1
        #
        #
        # for piece in task.solution:
        #     rot = piece.name[1]
        #
        #     if 'parrallelogram' in piece.name[0]:
        #         if rot == '0' or rot == '180':
        #             width = 1
        #             height = 2
        #             new_rot = str(int(rot) + 90)
        #         else:
        #             width = 2
        #             height = 1
        #             new_rot = str(int(rot) - 90)
        #         #new_rot = str((int(rot) + rot_ind * 90) % 360)  # TODO: correct later
        #     elif 'square' in piece.name[0] in piece.name[0]:
        #         width = 1
        #         height = 1
        #         # new_rot = str((int(rot) + rot_ind * 90) % 360)
        #         new_rot = 0 # square has only 0 rotation
        #     elif 'small triangle' in piece.name[0]:
        #         width = 1
        #         height = 1
        #         new_rot = str((int(rot) + rot_ind * 90) % 360)
        #     elif 'large triangle' in piece.name[0]:
        #         width = 2
        #         height = 2
        #         new_rot = str((int(rot) + rot_ind * 90) % 360)
        #     elif 'medium triangle' in piece.name[0]:
        #         if rot == '0' or rot == '180':
        #             width = 2
        #             height = 1
        #         else:
        #             width = 1
        #             height = 2
        #         new_rot = str((int(rot) + rot_ind * 90) % 360)
        #     i_old_pos = int(piece.name[2].split(' ')[0])
        #     j_old_pos = int(piece.name[2].split(' ')[1])
        #
        #     i_new = j_max - j_old_pos - (width - 1)
        #     j_new = i_old_pos
        #
        #     node_num = dic[piece.name[0] + ' ' + str(new_rot) + ' ' + str(i_new)+ ' ' + str(j_new)]
        #     activation[node_num] = 1
        #
        #
        # # sol.set_initial_task(task)
        # # sol.run_task(task, duration=100, stop=True)
        # training_set_output.append(activation)
        #
        # # print(training_set_input)
        # # print(training_set_output)
    return sol, training_set_task, training_set_input, training_set_output

# np.random.seed(1)
# random.seed(1)
# sol, training_set_input, training_set_output = create_training_set()
# for k in range(len(training_set_output)):
#     disp_training_data(training_set_input[k*3], training_set_input[k*3+1], training_set_input[k*3+2], training_set_output[k], sol,  'test')


# TODO test set: take this from the tangram list