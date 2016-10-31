from tangrams import *
import numpy as np
import copy


class Piece:
    JUMP = 4

    def __init__(self):
        self.x = []
        self.name = ['', '0', '']
        self.G = 0

    def print_me(self):
        print(self.name[0])
        print(self.name[1])
        print(self.name[2])
        print(self.x)

    def save_png(self):
        color = {
            'small triangle1': [255,0,0],
            'small triangle2': [255,255,0],
            'large triangle1': [255,0,255],
            'large triangle2': [0,255,0],
            'medium triangle': [0,0,255],
            'square': [0, 255, 255],
            'parrallelogram': [125, 0, 200]
        }

        filename = self.name[0] + "_" + self.name[1] + "_" + self.name[2] + ".png"
        bare = np.uint8(copy.deepcopy(self.x))
        bare[bare>0] = 1
        s = bare.shape
        img = np.zeros((9, 9, 4), 'uint8')
        img[0:s[0],0:s[1],0] = bare * color[self.name[0]][0]
        img[0:s[0],0:s[1],1] = bare * color[self.name[0]][1]
        img[0:s[0],0:s[1],2] = bare * color[self.name[0]][2]
        img[0:s[0],0:s[1],3] = bare * 255
        # self.to_image(img).save(filename, 'png')

    # @staticmethod
    # def to_image(img):
    #     s = np.array(img.shape)
    #     im = Image.fromarray(np.uint8(img))
    #     for k in range(0,3):
    #         s[0] *= 2
    #         s[1] *= 2
    #         im = im.resize([s[0], s[1]], Image.BILINEAR)
    #         s[0] *= 2
    #         s[1] *= 2
    #         im = im.resize([s[0], s[1]], Image.AFFINE)
    #
    #     return im

    def copy(self, target):
        target.name = self.name
        target.x = copy.deepcopy(self.x)
        target.G = self.G
        return target

    def rotate_compound(self, i):
        #  rotate a compound piece with i*90 degrees counter clockwise
        name_list = self.name[0].split('+')
        rot_list = self.name[1].split('+')
        new_rot_list = []
        pos_list = self.name[2].split('+')
        new_pos_list = []
        [i_max, j_max] = self.x.shape
        i_max = (i_max - 1)/Piece.JUMP - 1
        j_max = (j_max - 1)/Piece.JUMP - 1
        if len(name_list) != len(rot_list) or len(name_list) != len(pos_list):
            print("Piece: illegal compound name", self.name)
        for n in range(len(name_list)):
            if 'parrallelogram' in name_list[n]:
                if rot_list[n] == '0' or rot_list[n] == '180':
                    width = 1
                    height = 2
                else:
                    width = 2
                    height = 1
                new_rot = str((int(rot_list[n]) + i * 90) % 360)  # correct later
            elif 'square' in name_list[n] or 'small triangle' in name_list[n]:
                width = 1
                height = 1
                new_rot = str((int(rot_list[n]) + i*90) % 360)
            elif 'large triangle' in name_list[n]:
                width = 2
                height = 2
                new_rot = str((int(rot_list[n]) + i * 90) % 360)
            elif 'medium triangle' in name_list[n]:
                if rot_list[n] == '0' or rot_list[n] == '180':
                    width = 2
                    height = 1
                else:
                    width = 1
                    height = 2
                new_rot = str((int(rot_list[n]) + i * 90) % 360)

            i_old = int(pos_list[n].split(' ')[0])
            j_old = int(pos_list[n].split(' ')[1])
            if i==0:
                i_new = i_old
                j_new = j_old
            elif i==1:
                i_new = j_max-j_old - (width - 1)
                j_new = i_old
            elif i==2:
                i_new = i_max - i_old - (height - 1)
                j_new = j_max - j_old - (width - 1)
            elif i==3:
                i_new = j_old
                j_new = i_max - i_old - (height - 1)
            new_pos = str(i_new)+' '+str(j_new)

            new_rot_list.append(new_rot)
            new_pos_list.append(new_pos)

        new_rot_str = new_rot_list[0]
        for rot in new_rot_list[1:]:
            new_rot_str = new_rot_str+'+'+rot

        new_pos_str = new_pos_list[0]
        for pos in new_pos_list[1:]:
            new_pos_str = new_pos_str+'+'+pos

        self.name[1] = new_rot_str
        self.name[2] = new_pos_str
        self.x = np.rot90(self.x, i)
        #return [self.name[0], new_rot_str, new_pos_str]


    def rotate(self):
        p_list = [self]

        for i in range(1, 4):
            p_new = copy.deepcopy(self)
            if '+' in p_new.name[0]:
                p_new.rotate_compound(i)
            else:
                p_new.x = np.rot90(self.x, i)
                #p_new.name = [self.name[0], str(i * 90), '']
            found = False
            for q in p_list:
                if np.array_equal(p_new.x, q.x):
                    found = True
                    break
            if not found:
                p_new.name = [self.name[0], str(i * 90), '']
                p_list.append(p_new)

        for r in p_list:
            p_new = copy.deepcopy(r)
            p_new.x = np.fliplr(r.x)
            found = False
            for q in p_list:
                if np.array_equal(p_new.x, q.x):
                    found = True
                    break
            if not found:
                p_new.name = [self.name[0], str(int(r.name[1])+180), '']
                p_list.append(p_new)

        return p_list

    def translate(self, I, J):
        t = copy.deepcopy(self)
        t.x = np.zeros([I,J])
        t_list = []
        for i in range(0,I-self.x.shape[0]+1, Piece.JUMP):
            for j in range(0, J-self.x.shape[1]+1, Piece.JUMP):
                t_new = copy.deepcopy(t)
                if '+' in t.name[0]:  # in case of a compound piece, add the translation to each sub piece's position
                    temp_pos = t.name[2].split('+')
                    temp_pos = [str(int(pair.split(' ')[0])+i/Piece.JUMP)+' '+str(int(pair.split(' ')[1])+j/Piece.JUMP) for pair in temp_pos]
                    new_pos = temp_pos[0] # create a string with '+' from the list
                    for pair in temp_pos[1:]:
                        new_pos = new_pos+'+'+pair
                    t_new.name = [t.name[0], t.name[1], new_pos]
                else:
                    t_new.name = [t.name[0], t.name[1], str(i / Piece.JUMP) + " " + str(j / Piece.JUMP)]

                t_new.x[i:i+self.x.shape[0], j:j+self.x.shape[1]] = self.x
                t_list.append(t_new)

        return t_list

    def overlap(self, p):
        x = self.x + p.x
        return np.amax(x) > 5.01

    def unite(self, p):
        q = Piece()
        q.name[0] = self.name[0] + "+" + p.name[0]
        q.name[1] = self.name[1] + "+" + p.name[1]
        q.name[2] = self.name[2] + "+" + p.name[2]
        q.x = self.x + p.x
        return q

    def touch(self, p):
        x = self.x + p.x
        return len(np.where((x > 1) & (x < 5))[0]) > 0 and len(np.where(x > 5)[0]) == 0

    def base(self):
        a = np.transpose(np.argwhere(self.x > 0))
        x = self.x[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
        return x

    def compare(self, p):
        return np.array_equal(self.x, p.x)

    def create(self, name, rot, pos):
        # create a piece according to parameters:
        # name is a string - 'small triangle1'
        # rot is a string - '90'
        # pos is a vector - [1, 2]

        self.name = [ name , rot, str(pos[0])+" "+str(pos[1]) ]
        if 'small triangle' in name:
            self.x = np.array([[1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 5, 1, 0, 0],
                               [1, 5, 5, 1, 0],
                               [1, 1, 1, 1, 1]])
            if rot == '90':
                self.x = np.rot90(self.x, 1)
            elif rot == '180':
                self.x = np.rot90(self.x,2)
            elif rot == '270':
                self.x = np.rot90(self.x,3)
        elif 'medium triangle' in name:
            self.x = np.array([[1,1,1,1,1,1,1,1,1],
                               [0,1,5,5,5,5,5,1,0],
                               [0,0,1,5,5,5,1,0,0],
                               [0,0,0,1,5,1,0,0,0],
                               [0,0,0,0,1,0,0,0,0]])
            if rot == '90':
                self.x = np.rot90(self.x, 1)
            elif rot == '180':
                self.x = np.rot90(self.x, 2)
            elif rot == '270':
                self.x = np.rot90(self.x, 3)
        elif 'large triangle' in name:
            self.x = np.array([[1,1,1,1,1,1,1,1,1],
                               [1,5,5,5,5,5,5,1,0],
                               [1,5,5,5,5,5,1,0,0],
                               [1,5,5,5,5,1,0,0,0],
                               [1,5,5,5,1,0,0,0,0],
                               [1,5,5,1,0,0,0,0,0],
                               [1,5,1,0,0,0,0,0,0],
                               [1,1,0,0,0,0,0,0,0],
                               [1,0,0,0,0,0,0,0,0]])
            if rot == '90':
                self.x = np.rot90(self.x, 1)
            elif rot == '180':
                self.x = np.rot90(self.x, 2)
            elif rot == '270':
                self.x = np.rot90(self.x, 3)
        elif 'square' in name:
            self.x = np.array([[1,1,1,1,1],
                               [1,5,5,5,1],
                               [1,5,5,5,1],
                               [1,5,5,5,1],
                               [1,1,1,1,1]])
        elif 'parrallelogram' in name:
            self.x = np.array([[0,0,0,0,1],
                               [0,0,0,1,1],
                               [0,0,1,5,1],
                               [0,1,5,5,1],
                               [1,5,5,5,1],
                               [1,5,5,1,0],
                               [1,5,1,0,0],
                               [1,1,0,0,0],
                               [1,0,0,0,0]])
            if rot == '90':
                self.x = np.rot90(self.x, 1)
            elif rot == '180':
                self.x = np.fliplr(self.x)
            elif rot == '270':
                self.x = np.fliplr(np.rot90(self.x, 1))

        temp_x = np.zeros([pos[0]*self.JUMP+self.x.shape[0], pos[1]*self.JUMP+self.x.shape[1]])
        temp_x[pos[0] * self.JUMP:pos[0]*self.JUMP+self.x.shape[0],pos[1] * self.JUMP:pos[1]*self.JUMP+self.x.shape[1]] = self.x
        self.x = temp_x


