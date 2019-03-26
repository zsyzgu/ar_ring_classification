import numpy as np
from sklearn.svm import SVC
from point import Point
import math
import random
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import utils
from sklearn.metrics import precision_score, recall_score
import event

def caln_Xy(X_imu, line):
    line = line.strip('\n')
    tags = line.split()
    tags = [float(v) for v in tags]

    y = int(tags[0])
    #y_trans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 10 classes
    y_trans = [0, 1, 2, 3, -1, -1, 6, 7, -1, 9] # 7 classes
    #y_trans = [0, 1, -1, 3, -1, -1, -1, 7, -1, -1] # 4 classes
    y = y_trans[y]
    
    #X = [v for v in X_imu]
    X = []
    
    palm = Point(tags[17], tags[18], tags[19])
    palm_normal = Point(tags[20], tags[21], tags[22])
    palm_direction = Point(tags[23], tags[24], tags[25])
    fingers = []
    for i in range(5):
        fingers.append(Point(tags[40 + i * 17], tags[41 + i * 17], tags[42 + i * 17]))
    palm_size = Point(tags[62], tags[63], tags[64]).dist(palm)
    X.extend(caln_factors(palm, palm_direction, palm_normal, fingers, palm_size))
    X = np.array(X)

    return X, y

accs = []

def evaluate(file):
    fin = open(file, 'r')
    lines = fin.readlines()

    file_acc = []
    for label in range(0, 10):
        min_zs = []
        for line in lines:
            tags = line.strip('\n').split()
            tags = [float(v) for v in tags]

            if (int(tags[0]) == label):
                palm = Point(tags[17], tags[18], tags[19])
                min_z = 1e9
                fingers = []
                for i in range(5):
                    fingers.append(Point(tags[40 + i * 17], tags[41 + i * 17], tags[42 + i * 17]))
                    min_z = min(min_z, fingers[i].z)
                knuckle = Point(tags[54], tags[55], tags[56])
                min_z = min(min_z, knuckle.z)
                min_zs.append(min_z)
        
        min_zs = np.array(min_zs)
        acc = 0
        for i in min_zs:
            zs = min_zs - i
            acc = max(acc, float(len([1 for v in zs if 0 < v and v < 10])) / len(zs))
        file_acc.append(acc)

        print file, label, acc
    
    file_acc = np.array(file_acc)
    accs.append(np.mean(file_acc))

rootdir = './data/'
list = os.listdir(rootdir)
file_set = []
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'txt':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'vertical' and ring == 'index1'):
            evaluate(path)
        
accs = np.array(accs)
print("Precision: %0.3f(%0.3f)" % (accs.mean(), accs.std()))
