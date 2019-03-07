import os
import numpy as np

def parseDir(dir):
    list = os.listdir(dir)
    vision = [None] * 10
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
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
            if (t1 - t0 < 100):
                if (j0 in tap):
                    tap.remove(j0)
                if (j1 in tap):
                    tap.remove(j1)
        
        for j in candidate:
            factors = imu[j]
            for k in range(1, len(frames)):
                if (int(frames[k][0]) >= int(imu[j][0])):
                    t = (float)(int(frames[k][0]) - int(imu[j][0])) / (float)(int(frames[k][0]) - int(frames[k - 1][0]))
                    v0 = np.array([float(v) for v in frames[k - 1]])
                    v1 = np.array([float(v) for v in frames[k]])
                    v = v0 * t + v1 * (1.0 - t)
                    factors.extend(v)
                    break
            
            fout.write(str(i))
            for v in factors:
                fout.write(' ')
                fout.write(str(v))
            fout.write('\n')

rootdir = '.'
list = os.listdir(rootdir)
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isdir(path):
        parseDir(path)
