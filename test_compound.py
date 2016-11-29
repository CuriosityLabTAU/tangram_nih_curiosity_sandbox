from tangrams import *
import matplotlib.pyplot as plt


task = Task()
task.create_from_json('{"pieces": [["square", "0", "1 0"], ["small triangle1", "0", "0 0"]], "size": "5 5"}')

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

l = nodes[0].decompose()


for k in range(len(nodes)):
    plt.imshow(nodes[k].x, interpolation='none')
    plt.pause(0.1)

