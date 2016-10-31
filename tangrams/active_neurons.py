from tangrams import *
from tangrams.Piece import Piece
import matplotlib.pyplot as plt
import random
import copy


np.random.seed(2)
random.seed(2)

sol = Solver()
task = Task()
# task.create_from_json('{"pieces": [["large triangle2", "0", "1 0"], ["small triangle2", "90", "0 1"]], "size": "5 5"}')
# task.create_from_json('{"pieces": [["square", "180", "1 1"], ["large triangle1", "90", "0 2"], ["parrallelogram", "180", "2 1"]], "size": "5 5"}')
task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')


sol.set_available_pieces(task)
sol.run_task(task,duration=100, stop=True)

m = []
v = []
sol_max = []
chosen_moves = []
seq_jsons = []

temp_json = sol.available_pieces.export_to_json()
init_pos_json = sol.available_pieces.transfer_json_to_json_initial_pos(temp_json)
seq_jsons.append(init_pos_json)

task_dict = json.loads(init_pos_json)
pieces_vec = task_dict['pieces']
size = task_dict['size']

for n in range(len(sol.solutions[0])):
    v = np.add(np.dot(sol.networks[0].w, sol.solutions[0][n]), sol.networks[0].input)
    # ADD exp stuff
    temp = copy.deepcopy(sol.solutions[0][n])
    ind_active = np.where(sol.solutions[0][n])
    #ind_max = np.where(v[ind_active] == max(v[ind_active]))
    ind_max_sorted = [x for (y,x) in sorted(zip(v[ind_active],ind_active[0]), reverse=True)] # sort the active indexes according to v()


    most_active_piece = sol.networks[0].nodes[ind_max_sorted[0]].name
    for k in range(len(ind_max_sorted)):
        most_active_piece = sol.networks[0].nodes[ind_max_sorted[k]].name
        if most_active_piece not in pieces_vec:
            for piece_iter in range(len(pieces_vec)):
                if pieces_vec[piece_iter][0] == most_active_piece[0]:  # the name is the same.
                    pieces_vec[piece_iter] = most_active_piece
            break
    print pieces_vec
    print ind_max_sorted
    print v[ind_max_sorted]
    #next_move = ind_active[0][ind_max[0]][0]
    # if next_move in chosen_moves[-1]:
    #     next_move = ind_active[0][ind_max[0]][1]
    #chosen_moves.append(next_move)
    #temp[ind_active[0][ind_max_sorted[0]]]=10
    temp[ind_max_sorted[0]] = 10
    sol_max.append(temp)
    m.append(v)



plt.figure()
plt.imshow(sol.solutions[0], interpolation='none')
plt.figure()
plt.imshow(m, interpolation='none')
plt.figure()
plt.imshow(sol_max, interpolation='none')
plt.pause(20)
