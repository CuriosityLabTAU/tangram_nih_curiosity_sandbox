

# generate training set tuple
from tangrams import *
import random
import matplotlib.pyplot as plt

np.random.seed(1)
random.seed(1)


sol = Solver()
task = Task()
task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
sol.set_initial_task(task)

# dictionary: key = name of node, value = number of node
dic = {}
for n in range(len(sol.networks[0].nodes)):
    dic[sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]] = n

training_set_input = []
training_set_output = []

for i in range(2):

    # generate a random tangram with N pieces
    task.random_task(sol.networks[0], number_pieces=2)
    training_set_input.append(np.minimum(task.x, 1)) # only 0/1 (not 1,2,5)


    # solve the orignial task using the solution
    activation = np.zeros_like(sol.networks[0].a)
    for piece in task.solution:
        node_num = dic[piece.name[0] + ' ' + piece.name[1] + ' ' + piece.name[2]]
        activation[node_num] = 1
    training_set_input.append(activation)

    # sol.set_initial_task(task)
    # sol.run_task(task, duration=100, stop=True)
    # training_set_input.append(sol.networks[0].a)

    # rotate 90 TODO: don't rotate by this, but rotate_compound
    task.x = np.rot90(task.x)
    training_set_input.append(np.minimum(task.x, 1))

    # solve the rotated task without solver

    activation = np.zeros_like(sol.networks[0].a)
    rot_ind = 1  #  rotation index: 0->0 1->90 2->180 3->270
    [i_max, j_max] = task.x.shape
    i_max = (i_max - 1) / Piece.JUMP - 1
    j_max = (j_max - 1) / Piece.JUMP - 1


    for piece in task.solution:
        rot = piece.name[1]

        if 'parrallelogram' in piece.name[0]:
            if rot == '0' or rot == '180':
                width = 1
                height = 2
            else:
                width = 2
                height = 1
            new_rot = str((int(rot) + rot_ind * 90) % 360)  # TODO: correct later
        elif 'square' in piece.name[0] or 'small triangle' in piece.name[0]:
            width = 1
            height = 1
            new_rot = str((int(rot) + rot_ind * 90) % 360)
        elif 'large triangle' in piece.name[0]:
            width = 2
            height = 2
            new_rot = str((int(rot) + rot_ind * 90) % 360)
        elif 'medium triangle' in piece.name[0]:
            if rot == '0' or rot == '180':
                width = 2
                height = 1
            else:
                width = 1
                height = 2
            new_rot = str((int(rot) + rot_ind * 90) % 360)
        i_old_pos = int(piece.name[2].split(' ')[0])
        j_old_pos = int(piece.name[2].split(' ')[1])

        i_new = j_max - j_old_pos - (width - 1)
        j_new = i_old_pos

        node_num = dic[piece.name[0] + ' ' + str(new_rot) + ' ' + str(i_new)+ ' ' + str(j_new)]
        activation[node_num] = 1


    # sol.set_initial_task(task)
    # sol.run_task(task, duration=100, stop=True)
    training_set_output.append(activation)

    print(training_set_input)
    print(training_set_output)


# TODO test set: take this from the tangram list