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
from sklearn.metrics import precision_score, recall_score, accuracy_score
import event

def caln_factors(palm, palm_direction, palm_normal, fingers, palm_size):
    factors = np.zeros(26)

    palm_direction = palm_direction.unit()
    palm_normal = palm_normal.unit()
    palm_right = palm_direction.mul(palm_normal)

    factors[0] = palm_normal.z
    factors[1] = palm_direction.z
    
    for i in range(0, 5):
        factors[2 + i] = palm.dist(fingers[i]) / palm_size
    for i in range(0, 4):
        factors[7 + i] = fingers[i].dist(fingers[i + 1]) / palm_size
    
    for i in range(0, 5):
        finger = fingers[i] - palm
        alpha = finger.dot(palm_direction)
        beta = finger.dot(palm_normal)
        theta = finger.dot(palm_right)
        factors[11 + i] = math.atan2(beta, alpha) / math.pi
        factors[16 + i] = math.atan2(theta, alpha) / math.pi
    
    for i in range(0, 5):
        factors[21 + i] = fingers[i].z

    return factors

def caln_Xy(X_imu, line):
    line = line.strip('\n')
    tags = line.split()
    tags = [float(v) for v in tags]

    y = int(tags[0])
    #y_trans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 10 classes
    #y_trans = [0, 1, 2, 3, -1, -1, 6, 7, -1, 9] # 7 classes
    y_trans = [0, 1, -1, 3, -1, -1, -1, 7, -1, -1] # 4 classes
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

def evaluate(file_set, features):
    lines = []
    names = []
    name_set = []
    for file in file_set:
        status, name, ring = utils.get_file_info(file[7:])
        fin = open(file, 'r')
        curr_line = fin.readlines()
        lines.extend(curr_line)
        names.extend([name] * len(curr_line))
        if (name not in name_set):
            name_set.append(name)

    accuracy_list = []

    for leave in range(len(name_set)):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for i in range(len(lines)):
            X, y = caln_Xy(features[i], lines[i])
            if (y == -1):
                continue
            if (info[i][1] != name_set[leave]):
                X_train.append(X)
                y_train.append(y)
            else:
                X_test.append(X)
                y_test.append(y)

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
        print name_set[leave], accuracy
        #print confusion_matrix(y_test, y_pred)
    
    accuracy_list.sort()
    accuracy_list = np.array(accuracy_list[1:])
    print("Accuracy: %0.3f (+/- %0.3f)" % (accuracy_list.mean(), accuracy_list.std() * 2))

rootdir = './data/'
list = os.listdir(rootdir)
file_set = []
ext_set = []
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'txt':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'horizontal'):# and ring == 'index1'):
            file_set.append(path)
            ext_set.append(path[:-3] + 'ext')

info, data = utils.input(ext_set, 5, 4) # 10 frames
features = event.caln_features(info, data)
evaluate(file_set, features)

        #clf = SVC(gamma='auto')
        #clf = DecisionTreeClassifier(random_state=0)
        #scores = cross_val_score(clf, X, y, cv=10)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))