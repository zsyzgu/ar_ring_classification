import matplotlib.pyplot as plt
import utils
import numpy as np
import os

fig = None
forbiden = None
label = 0
info = None
data = None
ids = None

def getSubPlotNr(event):
    i = 1
    axisNr = None
    global fig
    for axis in fig.axes:
        if axis == event.inaxes:
            axisNr = i		
            break
        i += 1
    return axisNr

def onClick(event):	
    i = getSubPlotNr(event)
    if (i != None):
        global ids
        i = ids[i]
        global forbiden
        forbiden[i] = 1 - forbiden[i]
        global fig
        plt.close(fig)
        draw_label()

def draw_label():
    global label
    global info
    global data

    figure_name = str(info[0][0]) + ' ' + str(info[0][1]) + ' ' + str(info[0][2]) + ' ' + str(label)
    plt.figure(figure_name, figsize=(15,8))
    global fig
    fig = plt.get_current_fig_manager().canvas.figure
    fig.canvas.mpl_connect('button_press_event', onClick)

    print 'drawing'

    global ids
    ids = [0] * 50
    cnt = 0
    for i in range(0, len(info)):
        if (info[i][3] == label):
            cnt = cnt + 1
            if (cnt > 30):
                forbiden[i] = 1
                break
            ids[cnt] = i
            plt.subplot(6, 5, cnt)
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
            
            if (forbiden[i] == 0):
                plt.plot(acc_x)
                plt.plot(acc_y)
                plt.plot(acc_z)
                plt.vlines(info[i][5], -5, 5, colors='r')
    
    plt.show()

def illustration():
    global forbiden
    global info
    global data
    forbiden = [0] * len(info)

    for i in range(0, 10):
        global label
        label = i
        draw_label()

def output():
    global info
    global data
    global forbiden

    path = './data/' + str(info[0][0]) + '_' + str(info[0][1]) + '_' + str(info[0][2])

    fout_ext = open(path + '.ext', 'w')
    fin = open(path + '.txt', 'r')
    lines = fin.readlines()
    fout_txt = open(path + '.txt', 'w')

    for label in range(0, 10):
        for i in range(len(info)):
            if (info[i][3] == label and forbiden[i] == 0):
                fout_txt.write(lines[i])
                fout_ext.write(str(info[i][3]) + ' ' + str(info[i][4]) + ' ' + str(info[i][5]) + '\n')
                row = np.size(data[i], 0)
                col = np.size(data[i], 1)
                for r in range(row):
                    fout_ext.write(str(int(data[i][r, 0])) + ' ' + str(int(data[i][r, 1])))
                    for c in range(2, col - 1):
                        fout_ext.write(' ')
                        fout_ext.write(str(data[i][r, c]))
                    fout_ext.write(' ')
                    fout_ext.write(str(int(data[i][r, -1])))
                    fout_ext.write('\n')

rootdir = './data/'
list = os.listdir(rootdir)
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'ext':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'horizontal' and ring == 'index1' and name == 'gyz'):
            info, data = utils.input([path])
            illustration()
            output()
