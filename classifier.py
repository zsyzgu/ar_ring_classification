import numpy as np
from sklearn.svm import SVC
from point import Point
import math
import random
import os
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.externals import joblib

def caln_sequence(X):
    X = np.array(X)
    X_std = np.std(X)
    X_min = np.min(X)
    X_max = np.max(X)
    X_mean = np.mean(X)
    X_var = np.var(X)
    X_sc = np.mean((X - X_mean) ** 3) / pow(X_std, 3)
    X_ku = np.mean((X - X_mean) ** 4) / pow(X_std, 4)
    if (math.isnan(X_ku)):
        print 'data error'
        X_ku = 0
    return [X_mean, X_min, X_max, X_sc, X_ku]

def caln_features(info, data):
    features = []
    for i in range(len(info)):
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
        feature = []
        #gyr = np.array([(gyr_x ** 2 + gyr_y ** 2 + gyr_z ** 2) ** 0.5])
        #acc = np.array([(acc_x ** 2 + acc_y ** 2 + acc_z ** 2) ** 0.5])
        #feature.extend(caln_sequence(gyr))
        #feature.extend(caln_sequence(acc))
        feature.extend(caln_sequence(acc_x))
        feature.extend(caln_sequence(acc_y))
        feature.extend(caln_sequence(acc_z))
        feature.extend(caln_sequence(gra_x))
        feature.extend(caln_sequence(gra_y))
        feature.extend(caln_sequence(gra_z))
        feature.extend(caln_sequence(gyr_x))
        feature.extend(caln_sequence(gyr_y))
        feature.extend(caln_sequence(gyr_z))
        features.append(feature)
    
    return features

def caln_1_precision_recall(y_test, y_pred):
    pre_a = 0
    pre_b = 0
    rec_a = 0
    rec_b = 0
    for i in range(len(y_test)):
        if (y_test[i] == 1):
            rec_b = rec_b + 1
            if (y_test[i] == y_pred[i]):
                rec_a = rec_a + 1
        if (y_pred[i] == 1):
            pre_b = pre_b + 1
            if (y_test[i] == y_pred[i]):
                pre_a = pre_a + 1
    if (pre_b == 0):
        precision = 1
    else:
        precision = float(pre_a) / pre_b
    if (rec_b == 0):
        recall = 1
    else:
        recall = float(rec_a) / rec_b
    return precision, recall

def normalize_X(X_train, X_test):
    len_train = len(X_train)
    len_test = len(X_test)
    X_train = np.array(X_train).reshape(len_train, -1)
    X_test = np.array(X_test).reshape(len_test, -1)
    len_features = np.size(X_train, 1)
    for i in range(len_features):
        X_mean = np.mean(X_train[:,i])
        X_std = np.std(X_train[:,i])
        X_train[:,i] = (X_train[:,i] - X_mean) / X_std
        X_test[:,i] = (X_test[:,i] - X_mean) / X_std
    return X_train, X_test

def input(frames_delay, ring_in):
    file_set = []
    name_set = []

    rootdir = './data/'
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path) and list[i].split('.')[-1] == 'ext':
            status, name, ring = utils.get_file_info(list[i])
            if (name not in name_set):
                name_set.append(name)
            if (status == 'vertical' and ring == ring_in):
                file_set.append(path)

    info_neg, data_neg = utils.input(file_set, 14, -5)

    rootdir = './negative/'
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path) and list[i].split('.')[-1] == 'ext':
            status, name, ring = utils.get_file_info(list[i])
            if (name not in name_set):
                name_set.append(name)
            if (status == 'negative' and ring == ring_in):
                file_set.append(path)

    info, data = utils.input(file_set, 9 - frames_delay, frames_delay) # 10 frames

    # add neg from positive
    for i in range(len(info_neg)):
        info_neg[i][3] = -1
    info.extend(info_neg)
    data.extend(data_neg)

    features = caln_features(info, data)

    return info, data, name_set, features

def train(info, data):

    X_train = []
    y_train = []
    for i in range(len(info)):
        label = 0
        if (info[i][3] != -1):
            label = 1
        X_train.append(features[i])
        y_train.append(label)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'clf.model')

    return features

def test(info, data, name_set, features):
    clf = joblib.load('clf.model')

    precision_list = []
    recall_list = []
    f1_list = []

    for leave in range(len(name_set)):
        X_test = []
        y_test = []

        for i in range(len(info)):
            label = 0
            if (info[i][3] != -1):
                label = 1

            if (info[i][1] == name_set[leave]):
                X_test.append(features[i])
                y_test.append(label)

        if (len(X_test) == 0):
            print 'no data'
        else:
            y_pred = clf.predict(X_test)
            precision, recall = caln_1_precision_recall(y_test, y_pred)
            f1 = 2 * precision * recall / (precision + recall)
            #print name_set[leave], precision, recall, f1
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)

    print("Precision: %0.3f(%0.3f)" % (precision_list.mean(), precision_list.std()))
    print("Recall: %0.3f(%0.3f)" % (recall_list.mean(), recall_list.std()))
    print("F1: %0.3f(%0.3f)" % (f1_list.mean(), f1_list.std()))
    #print("Label=0: %d Label=1: %d" % (label_0, label_1))

imu_list = []

def predict(clf, imu):
    global imu_list

    if (len(imu_list) >= 10):
        imu_list.pop(0)
        imu_list.append(imu)
        data = []
        n = len(imu_list)
        for i in range(n):
            tags = imu_list[i].split()
            data.append([float(v) for v in tags])
        data = np.array(data).reshape(n, -1)
        
        gyr_x = data[:,2]
        gyr_y = data[:,3]
        gyr_z = data[:,4]
        acc_x = data[:,5]
        acc_y = data[:,6]
        acc_z = data[:,7]
        gra_x = np.zeros(n)
        gra_y = np.zeros(n)
        gra_z = np.zeros(n)
        for j in range(n):
            gra = utils.qua_to_vec(data[j, 8:12])
            gra_x[j] = gra[0]
            gra_y[j] = gra[1]
            gra_z[j] = gra[2]
        feature = []
        feature.extend(caln_sequence(acc_x))
        feature.extend(caln_sequence(acc_y))
        feature.extend(caln_sequence(acc_z))
        feature.extend(caln_sequence(gra_x))
        feature.extend(caln_sequence(gra_y))
        feature.extend(caln_sequence(gra_z))
        feature.extend(caln_sequence(gyr_x))
        feature.extend(caln_sequence(gyr_y))
        feature.extend(caln_sequence(gyr_z))
        for v in feature:
            if math.isnan(v):
                return 0
        return clf.predict([feature])[0]
    else:
        imu_list.append(imu)
        return 0

if __name__ == "__main__":
    #info, data, name_set, features = input(4, 'index1')
    #train(info, data)
    #test(info, data, name_set, features)

    '''
    clf = joblib.load('clf.model')
    fin = open('./data/horizontal_gyz_index1/horizontal_gyz_index1_IMU.txt')
    lines = fin.readlines()
    cnt = 0
    for line in lines:
        line = line.strip('\n')
        if predict(clf, line) == 1:
            print line
    '''

    clf = joblib.load('clf.model')
    imu = 'current IMU data'
    result = predict(clf, imu)
