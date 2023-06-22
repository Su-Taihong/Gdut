import os
from math import sqrt
from PIL import Image
import numpy as np
import sklearn
from numpy import mean
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import graphviz

# 读取face文件（faceDR与faceDS的合并）
data_dr = []
with open(r'E:\权\模式识别\人脸图像识别\face\face', 'r') as file_dr:
    for line_dr in file_dr:
            line_dr = line_dr.replace('(', '')
            line_dr = line_dr.replace(')', '')
            line_dr_1 = line_dr.split('_')
            data_dr.append(line_dr_1)

x, y = [], []
path = r'E:\权\模式识别\人脸图像识别\face\rawdata_ori'
i = -1
a = 0
exp_dict = {'sex  male ': 0, 'sex  female ': 1}
for label in os.listdir(path):
    i = i + 1
    # 先判断该编号的图片是否有缺失数据
    while True:
        if len(data_dr[i]) == 6:
            # 如果无缺失，则调出whlie
            break
        else:
            i = i + 1
    # 判断rawdata取出的图片与face取出的图片是否为同一张图片
    if label == data_dr[i][0].strip(' '):
        image_path = os.path.join(path, label)
        f = open(image_path, mode='rb')
        x1 = np.fromfile(f, dtype=np.ubyte)
        # 剔除全黑或全白的图片
        if mean(x1) <= 5 or mean(x1) >= 220:
            print('全黑或全白：', label)
            # 记录下数量，最终发现有50张全黑或全白
            a = a + 1
            continue
        else:
            # 如果图片是有效数据，则分别存入x、y
            y.append(exp_dict[data_dr[i][1]])
            x1 = x1 / 225.0
            if len(x1) == (128 * 128):
                x2 = x1.reshape([128, 128])
            elif sqrt(len(x1)) % 1 == 0.0:
                n = int(sqrt(len(x1)))
                x1 = x1.reshape([n, n])
                im = Image.fromarray(x1)
                im = im.resize((128, 128))
                x2 = np.array(im)
            x.append(x2)
print('len(x):', len(x),'len(y):',  len(y), '剔除的错误图像数量:', a)

# 提取图片hog特征
def extract_hog_features(X):
    image_descriptors = []
    for i in range(len(X)):
        fd, _ = hog(X[i], orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm ='L2-Hys', visualize=True)
        image_descriptors.append(fd)
    # 返回的是训练部分所有图像的hog特征
    return image_descriptors

X_features = extract_hog_features(x)
# 检查每个hog向量的长度
print(X_features[0].shape)
#切割训练集与测试集，这里设置训练集占0.8，测试集占0.2
X_train, X_test, Y_train, Y_test = train_test_split(X_features, y, test_size=0.114, random_state=10)



# 决策树分类
clf = tree.DecisionTreeClassifier(criterion="entropy",splitter='best',max_depth=5,min_samples_split=15,min_samples_leaf=200)
clf = clf.fit(X_train ,Y_train)
score = clf.score(X_test,Y_test)
print('score_1:%s'%score)

# 交叉验证过程
clf = tree.DecisionTreeClassifier(criterion="entropy",splitter='best',max_depth=5,min_samples_split=15,min_samples_leaf=200)
scores = cross_val_score(clf, X_features, y, cv=10)
print(scores)