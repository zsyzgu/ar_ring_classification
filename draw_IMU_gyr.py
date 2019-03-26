import matplotlib.pyplot as plt
import utils
import numpy as np
import os

# [status, name, ring, label, length, key]

def draw_label(label, key):
    plt.figure('IMU_acc', figsize=(12,5))

    names = ['gyz', 'wxy', 'lzp', 'fjy']

    for col in range(4):
        for row in range(2):
            print col, row
            for i in range(0, len(info)):
                if (info[i][1] == names[col] and info[i][2] == 'index1' and info[i][3] == 0):
                    ax = plt.subplot(2, 4, row * 4 + col + 1)
                    gyr_x = data[i][:,2]
                    gyr_y = data[i][:,3]
                    gyr_z = data[i][:,4]
                    acc_x = data[i][:,5]
                    acc_y = data[i][:,6]
                    acc_z = data[i][:,7]
                    length = np.size(data[i], 0)
                    gra_x = np.zeros(length)
                    gra_y = np.zeros(length)
                    gra_z = np.zeros(length)
                    for j in range(length):
                        gra = utils.qua_to_vec(data[i][j, 8:12])
                        gra_x[j] = gra[0]
                        gra_y[j] = gra[1]
                        gra_z[j] = gra[2]
                    t = [(i - key) * 5 for i in range(len(acc_x))]
                    if (row == 0):
                        plt.title('User' + str(col + 1))
                    if (row == 1 and col == 0):
                        plt.xlabel('Delay (ms)', fontsize=20)
                        plt.ylabel('Gravity (g)', fontsize=20)
                    elif (row == 0 and col == 0):
                        plt.ylabel('Gryo (rad/s)', fontsize=20)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if (row == 0):
                        plt.vlines(0, -2.5, 2.5, colors='r', linestyles = "dashed")
                        plt.ylim(-5, 5)
                        plt.plot(t, gyr_x, color='#FB9100')
                        plt.plot(t, gyr_y, color='#37ACCB')
                        plt.plot(t, gyr_z, color='#B0C915')
                    else:
                        plt.vlines(0, -1, 1, colors='r', linestyles = "dashed")
                        plt.ylim(-2, 2)
                        plt.plot(t, gra_x, color='#FB9100', label='x')
                        plt.plot(t, gra_y, color='#37ACCB', label='y')
                        plt.plot(t, gra_z, color='#B0C915', label='z')
                    if (row == 1 and col == 0):
                        plt.legend(loc='lower left', fontsize=12)
                    break

rootdir = './data/'
list = os.listdir(rootdir)
path_list = []
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'ext':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'horizontal' and ring == 'index1'):
            path_list.append(path)

key = 20
info, data = utils.input(path_list, key, 20)

font = {'family':'Times New Roman','weight' : 'normal','size': 16}
plt.rc('font',**font)
plt.rcParams['savefig.dpi'] = 240
plt.rcParams['figure.dpi'] = 240
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
draw_label(0, key)

plt.savefig('IMU_gyr.png')
