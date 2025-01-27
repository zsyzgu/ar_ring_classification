import os
import numpy as np
import math
import utils
import random

def parseFrames(imu, frames, j):
    factors_list = []
    left_gap = 500
    right_gap = 100
    i = j
    while (i > 0 and int(imu[j][0]) - int(imu[i][0]) < left_gap):
        i = i - 1
    begin_j = i
    i = j
    while (i + 1 < len(imu) and int(imu[i][0]) - int(imu[j][0]) < right_gap):
        i = i + 1
    end_j = i
    k = 1
    for i in range(begin_j, end_j + 1):
        for v in imu[i]:
            if (math.isnan(float(v))):
                print 'no imu data'
                return [], -1
        factors = [v for v in imu[i]]
        while (k + 1 < len(frames) and int(frames[k][0]) < int(imu[i][0])):
            k = k + 1
        duration = (float)(int(frames[k][0]) - int(frames[k - 1][0]))
        if (duration > 50):
            print 'no hand data'
            return [], -1
        t = (float)(int(frames[k][0]) - int(imu[i][0])) / duration
        v0 = np.array([float(v) for v in frames[k - 1]])
        v1 = np.array([float(v) for v in frames[k]])
        v = v0 * t + v1 * (1.0 - t)
        factors.extend(v)
        factors_list.append(factors)
    
    return factors_list, j - begin_j

def parseDir(dir):
    list = os.listdir(dir)
    vision = [None] * 10

    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if (path[-3:] == 'txt'):
            input = open(path, 'r')

            if (path[-5] == 'U'):
                IMU = input.readlines()
            else:
                gesture = int(path[-5]) - int('0')
                vision[gesture] = input.readlines()
            input.close()

    imu = []
    for line in IMU:
        line = line.strip('\n')
        tags = line.split()
        imu.append(tags)

    fout = open(dir[2:] + '.txt', 'w')
    fout_ext = open(dir[2:] + '.ext', 'w')

    print dir[2:]

    for i in range(10):
        frames = []
        for line in vision[i]:
            line = line.strip('\n')
            tags = line.split()
            frames.append(tags)
        start = int(frames[0][0])
        end = int(frames[-1][0])
        candidate = []
        for j in range(1, len(imu) - 1):
            t = int(imu[j][0])
            if (start <= t and t <= end):
                if (int(imu[j - 1][-1]) == 0 and int(imu[j][-1]) == 1 and int(imu[j + 1][-1]) == 1):
                    candidate.append(j)
        tap = [v for v in candidate]
        for j in range(len(candidate) - 1):
            j0 = candidate[j]
            j1 = candidate[j + 1]
            t0 = int(imu[j0][0])
            t1 = int(imu[j1][0])
            if (t1 - t0 < 100): # tapping too close
                if (j0 in tap):
                    tap.remove(j0)
                if (j1 in tap):
                    tap.remove(j1)
        
        cnt = 0
        for j in candidate:
            factors_list, key = parseFrames(imu, frames, j)
            if (key == -1): # no hand data
                continue

            key = min(key, len(factors_list))

            factors = factors_list[key]

            palm_z = float(factors[18])
            if (palm_z < 0): # wrong marker direction
                continue
            
            cnt = cnt + 1

            fout.write(str(i))
            for v in factors:
                fout.write(' ')
                fout.write(str(v))
            fout.write('\n')
            
            fout_ext.write(str(i) + ' ' + str(len(factors_list)) + ' ' + str(key) + '\n')
            for factors in factors_list:
                fout_ext.write(str(factors[0]))
                for k in range(1, len(imu[j])):
                    fout_ext.write(' ')
                    fout_ext.write(str(factors[k]))
                fout_ext.write('\n')

        print i, cnt

def positive_parser():
    rootdir = './data/'
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            if ((list[i] + '.txt') in list and (list[i] + '.ext') in list):
                continue
            status, name, ring = utils.get_file_info(list[i])
            if (name == 'xcy' and ring == 'ring1' and status == 'vertical'):
                parseDir(path)

def parseNegative(path):
    input = open(path, 'r')
    lines = input.readlines()
    fout_ext = open(path[:-3] + 'ext', 'w')

    imu = []
    for line in lines:
        line = line.strip('\n')
        tags = line.split()
        imu.append(tags)

    length = len(imu)

    left_gap = 500
    right_gap = 100
    samples = 170
    s = 0

    while (s < samples):
        j = random.randint(left_gap, length - right_gap - 1)

        i = j
        while (i > 0 and int(imu[j][0]) - int(imu[i][0]) < left_gap):
            i = i - 1
        begin_j = i
        i = j
        while (i + 1 < len(imu) and int(imu[i][0]) - int(imu[j][0]) < right_gap):
            i = i + 1
        end_j = i
        
        factors_list = []
        key = j - begin_j
        flag = True
        for i in range(begin_j, end_j + 1):
            for v in imu[i]:
                if (math.isnan(float(v))):
                    print 'no imu data'
                    flag = False
            factors = [v for v in imu[i]]
            factors_list.append(factors)
        
        tmp = np.array(factors_list).reshape(len(factors_list), -1)
        for i in range(2, np.size(tmp, 1) - 1):
            x = [float(v) for v in tmp[:,i]]
            if (np.var(x) == 0):
                flag = False

        if (flag == True):
            s = s + 1

            fout_ext.write(str(-1) + ' ' + str(len(factors_list)) + ' ' + str(key) + '\n')
            for factors in factors_list:
                fout_ext.write(str(factors[0]))
                for k in range(1, len(factors)):
                    fout_ext.write(' ')
                    fout_ext.write(str(factors[k]))
                fout_ext.write('\n')

def negative_parser():
    rootdir = './negative'
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if (os.path.isfile(path) and list[i][-3:] == 'txt'):
            if ((list[i][:-3] + '.ext') in list):
                continue
            parseNegative(path)

#positive_parser()
negative_parser()
