import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
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

# 查看单个图片信息
# f = open(r'E:\权\模式识别\人脸图像识别\face\rawdata\1240','rb')
# x = np.fromfile(f,dtype=np.ubyte)
# x = x.reshape(128,-1)
# print(x.shape)
# plt.imshow(x,cmap=plt.cm.gray)
# plt.show()
# f.close()

# f = open(r'E:\权\模式识别\人脸图像识别\face\rawdata\2099','rb')
# x = np.fromfile(f,dtype=np.ubyte)
# x = x.reshape(128,-1)
# print(x.shape)
# plt.imshow(x,cmap=plt.cm.gray)
# plt.show()
# f.close()

# **********************源数据处理**********************
all_file = os.listdir(r'E:\权\模式识别\人脸图像识别\face\rawdata')
# 读取所有data_x数据,删除有问题的数据
data_x = np.zeros((len(all_file),128,128))
data_y = np.zeros(len(all_file))
# 将两个数据标签整理再一起
f = open(r'E:\权\模式识别\人脸图像识别\face\faceDR_1','r')
r1 = f.readlines()  # 存储每一行的标签
f.close()
f = open(r'E:\权\模式识别\人脸图像识别\face\faceDS_1','r')
r2 = f.readlines()  # 存储每一行的标签
f.close()
for line in r2:
    r1.append(line)  # 最终将所有标签存到r1里

# 整理出data_x,和data_y
man = 0
woman = 0
all_file = os.listdir(r'E:\权\模式识别\人脸图像识别\face\rawdata')
for i in range(len(all_file)):
    with open('E:/权/模式识别/人脸图像识别/face/rawdata/'+all_file[i],'rb') as f:
        x = np.fromfile(f,dtype=np.ubyte)
        x = x.reshape(128,-1)
        data_x[i] = x
        for j in r1:
            if all_file[i] in j:
                if 'female' in j:
                    data_y[i] = 0
                    woman = woman + 1
                elif 'male' in j:
                    data_y[i] = 1
                    man = man + 1
                else:
                    data_y[i] = None
                break
face_data=data_x.reshape(data_x.shape[0],-1)
face_target = data_y
#print(face_data.shape)
print("number of man is %s,number of woman is %s"%(man,woman))
X = face_data
Y = face_target

print(X.shape)
np.unique(Y) #去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表。
print(Y)
data = pd.DataFrame(X)  #DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
# print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
print(data.describe().T)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.25,random_state=1)
print("Ytrain:")
valuec = pd.Series(Ytrain).value_counts()
print(valuec)
print(valuec[0]/valuec.sum())
print("Ytest:")
valuec = pd.Series(Ytest).value_counts()
print(valuec)
print(valuec[0]/valuec.sum())


#pca = PCA(n_components=76, svd_solver='randomized',whiten=True).fit(X)
pca = PCA(n_components=0.9).fit(X)  #设置n_components=0.95就表示希望保留95%的信息量，那么PCA就会自动选使得信息量>=95%的特征数量。
x_dr = pca.transform(X)
print(x_dr.shape)
X_1 = x_dr

pca = PCA(svd_solver='randomized',n_components=X_1.shape[1], whiten=True).fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# # plot the gallery of the most significative eigenfaces
eigenfaces = pca.components_.reshape((X_1.shape[1], 128, 128))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 128, 128)

plt.show()


data = pd.DataFrame(X_1)
# print(data.describe([0.01,0.1,0.3,0.6,0.9,0.99]).T)
print(data.describe().T)

X_2 = StandardScaler().fit_transform(X_1)   # X_2是X_1标准化后的数据
data = pd.DataFrame(X_2)
# print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
print(data.describe().T)


# **********************对未降维数据进行训练**********************
Xtrain_1, Xtest_1, Ytrain_1, Ytest_1 = train_test_split(X,Y,test_size=0.15,random_state=1,stratify=Y)
tr_man_1 = 0
tr_woman_1 = 0
# 训练集是否均衡
for i in range(len(Ytrain_1)):
    if Ytrain_1[i] == 1.0:
        tr_man_1 = tr_man_1 + 1
    if Ytrain_1[i] == 0.0:
        tr_woman_1 = tr_woman_1 + 1
rate_1 = tr_man_1 / tr_woman_1
print("man:%s"%tr_man_1)
print("woman:%s"%tr_woman_1)
print("rate:%f"%rate_1)

# 决策树分类
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain_1 ,Ytrain_1)
score = clf.score(Xtest_1,Ytest_1)
print('score_1:%s'%score)

# 绘制决策树图
# dot_data=tree.export_graphviz(clf
#                               ,class_names=["男","女"]
#                               ,filled=True
#                               ,rounded=True
#                              )
# graph=graphviz.Source(dot_data)
# print(graph)

# K折交叉验证过程
clf = tree.DecisionTreeClassifier(criterion="entropy")
scores = cross_val_score(clf, X, Y, cv=10)
print(scores)



# **********************对降维后数据进行训练**********************
Xtrain_2, Xtest_2, Ytrain_2, Ytest_2 = train_test_split(X_2,Y,test_size=0.15,random_state=1,stratify=Y)
tr_man_2 = 0
tr_woman_2 = 0
# 训练集是否均衡
for i in range(len(Ytrain_2)):
    if Ytrain_2[i] == 1.0:
        tr_man_2 = tr_man_2 + 1
    if Ytrain_2[i] == 0.0:
        tr_woman_2 = tr_woman_2 + 1
rate_2 = tr_man_2 / tr_woman_2
print("man:%s"%tr_man_2)
print("woman:%s"%tr_woman_2)
print("rate:%f"%rate_2)

# 决策树分类
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain_2,Ytrain_2)
score = clf.score(Xtest_2,Ytest_2)
print('score_2:%s'%score)

# 绘制决策树图
# dot_data=tree.export_graphviz(clf
#                               ,class_names=["男","女"]
#                               ,filled=True
#                               ,rounded=True
#                              )
# graph=graphviz.Source(dot_data)
# print(graph)

# K折交叉验证过程
clf = tree.DecisionTreeClassifier(criterion="entropy")
scores = cross_val_score(clf, X_2, Y, cv=10)
print(scores)