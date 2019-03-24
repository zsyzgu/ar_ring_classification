import numpy as np

def get_file_info(file):
    status = file.split('.')[0].split('_')[0]
    name = file.split('.')[0].split('_')[1]
    ring = file.split('.')[0].split('_')[2]
    return status, name, ring

def qua_to_vec(q):
    return [float(2*(q[1]*q[3] - q[0]*q[2])), float(2*(q[0]*q[1] + q[2]*q[3])), float(2*(0.5 - q[1]**2 - q[2]**2))]

def input(file_set, cut_l = -1, cut_r = -1):
    info = []
    data = []

    for file in file_set:
        fin = open(file, 'r')
        lines = fin.readlines()
        lines = [line.strip('\n') for line in lines]
        
        status, name, ring = get_file_info(file.split('/')[-1])

        i = 0
        while (i < len(lines)):
            tags = lines[i].split()
            #print name, tags
            label, length, key = int(tags[0]), int(tags[1]), int(tags[2])
            info.append([status, name, ring, label, length, key])
            
            i = i + 1
            frames = []
            for j in range(i, i + length):
                tags = lines[j].split()
                frame = [float(v) for v in tags]
                if (cut_l == -1 or (-cut_l <= (j - i) - key and (j - i) - key <= cut_r)):
                    frames.append(frame)
            frames_length = len(frames)
            frames = np.array(frames).reshape(frames_length, -1)
            data.append(frames)

            i = i + length
    
    return info, data
