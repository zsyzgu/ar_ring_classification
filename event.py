import numpy as np
from sklearn.svm import SVC
from point import Point
import math
import random
import os
import matplotlib.pyplot as plt
import utils

def evaluate(file_set):
    info = []
    data = []

    for file in file_set:
        fin = open(file, 'r')
        lines = fin.readlines()
        lines = [line.strip('\n') for line in lines]
        
        status, name, ring = utils.get_file_info(file[7:])

        i = 0
        while (i < len(lines)):
            tags = lines[i].split()
            label, length, key = int(tags[0]), int(tags[1]), int(tags[2])
            info.append([status, name, ring, label, length, key])
            
            i = i + 1
            frames = []
            for j in range(i, i + length):
                tags = lines[j].split()
                frame = [float(v) for v in tags]
                frames.append(frame)
            frames = np.array(frames).reshape(length, -1)
            data.append(frames)

            i = i + length
    
    for label in range(0, 1):
        plt.figure(str(label), figsize=(15,8))
        cnt = 0
        for i in range(0, len(info)):
            if (info[i][3] == label):
                cnt = cnt + 1
                if (cnt > 25):
                    break
                plt.subplot(5, 5, cnt)
                gyr_x = data[i][:,2]
                gyr_y = data[i][:,3]
                gyr_z = data[i][:,4]
                acc_x = data[i][:,5]
                acc_y = data[i][:,6]
                acc_z = data[i][:,7]

                plt.vlines(info[i][5], -1, 1, colors='r')
                plt.plot(acc_x)
                plt.plot(acc_y)
                plt.plot(acc_z)
        
        plt.show()

rootdir = './data/'
list = os.listdir(rootdir)
file_set = []
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'ext':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'vertical' and name == 'gyz' and ring == 'index1'):
            file_set.append(path)

evaluate(file_set)
