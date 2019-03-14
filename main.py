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

def caln_factors(palm, palm_direction, palm_normal, fingers, palm_size):
    factors = np.zeros(25)

    palm_direction = palm_direction.unit()
    palm_normal = palm_normal.unit()
    palm_right = palm_direction.mul(palm_normal)

    factors[0] = palm_normal.z
    factors[1] = palm.z
    
    for i in range(0, 5):
        factors[2 + i] = palm.dist(fingers[i]) / palm_size
    for i in range(0, 4):
        factors[7 + i] = fingers[i].dist(fingers[i + 1]) / palm_size
    for i in range(0, 5):
        factors[11 + i] = fingers[i].z
    
    for i in range(0, 5):
        finger = fingers[i] - palm
        alpha = finger.dot(palm_direction)
        beta = finger.dot(palm_normal)
        theta = finger.dot(palm_right)
        factors[15 + i] = math.atan2(beta, alpha) / math.pi
        factors[20 + i] = math.atan2(theta, alpha) / math.pi

    return factors

def evaluate(file_set):
    lines = []
    for file in file_set:
        fin = open(file, 'r')
        lines.extend(fin.readlines())

    X = []
    y = []

    for line in lines:
        line = line.strip('\n')
        tags = line.split()
        tags = [float(v) for v in tags]
        y_result = int(tags[0])
        if (y_result == 4): y_result = 0 # cancel finger tips
        if (y_result == 5): y_result = 1
        if (y_result == 8): y_result = 2
        y.append(y_result)
        factors = tags[3: 9]
        factors.extend(utils.qua_to_vec(tags[9: 13]))
        palm = Point(tags[17], tags[18], tags[19])
        palm_normal = Point(tags[20], tags[21], tags[22])
        palm_direction = Point(tags[23], tags[24], tags[25])
        fingers = []
        for i in range(5):
            fingers.append(Point(tags[40 + i * 17], tags[41 + i * 17], tags[42 + i * 17]))
        palm_size = Point(tags[62], tags[63], tags[64]).dist(palm)
        factors.extend(caln_factors(palm, palm_direction, palm_normal, fingers, palm_size))
        factors = np.array(factors)
        X.append(factors)

    X = np.array(X)
    y = np.array(y)
    #clf = SVC(gamma='auto')
    clf = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(clf, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.3)
    X_train, X_test, y_train, y_test = X[len(X) / 3:], X[:len(X) / 3], y[len(X) / 3:], y[:len(X) / 3]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))

rootdir = './data/'
list = os.listdir(rootdir)
file_set = []
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and list[i].split('.')[-1] == 'txt':
        status, name, ring = utils.get_file_info(list[i])
        if (status == 'vertical' and ring == 'index1'):
            file_set.append(path)

evaluate(file_set)
