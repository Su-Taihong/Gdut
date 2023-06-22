from sklearn.decomposition import PCA
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
# from skimage.feature import hog

# 人脸图像文件路径
all_file = os.listdir(r'D:\..\face\rawdata')
# 初始化产生全零数组，读取所有 data_x数据
data_x = np.zeros((len(all_file),128,128))   # 存放图像维度信息
data_y = np.zeros(len(all_file))             # 存放图像性别标签

# 将两个数据标签整理再一起
f = open(r'D:\..\face\faceDR','r')
r1 = f.readlines()  # 存储每一行的标签
f.close()
f = open(r'D:\..\face\faceDS','r')
r2 = f.readlines()  # 存储每一行的标签
f.close()
for line in r2:
    r1.append(line)  # 最终将所有标签存到 r1里

# 整理出 data_x和 data_y
man = 0
woman = 0
all_file = os.listdir(r'D:\..\face\rawdata')
for i in range(len(all_file)):
    with open('D:/../face/rawdata/'+all_file[i],'rb') as f:
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
np.unique(Y) # 去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表。
print(Y)
data = pd.DataFrame(X)  # DataFrame是 Python中 Pandas库中的一种数据结构，它类似 excel，是一种二维表。
print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)

# random_state=420

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3) # test_size：测试集占完整数据集的比例为 0.3   random_state：随机数种子，应用于分割前对数据的洗牌。
print("Ytrain:")
valuec = pd.Series(Ytrain).value_counts()  # 在 pandas中，value_counts常用于数据表的计数及排序，它可以用来查看数据表中，指定列里有多少个不同的数据值，并计算每个不同值有在该列中的个数，同时还能根据需要进行排序。
print(valuec)
print(valuec[0]/valuec.sum())
print("Ytest:")
valuec = pd.Series(Ytest).value_counts()
print(valuec)
print(valuec[0]/valuec.sum())


#pca = PCA(n_components=76, svd_solver='randomized',whiten=True).fit(X)
pca = PCA(n_components=0.95).fit(X)  # 设置 n_components=0.95就表示希望保留 95%的信息量，那么 PCA就会自动选使得信息量 >=95%的特征数量。
x_dr = pca.transform(X)   # transform：在 fit的基础上，进行标准化，降维，归一化等操作
print(x_dr.shape)
X_1 = x_dr

pca = PCA(svd_solver='randomized',n_components=X_1.shape[1], whiten=True).fit(Xtrain)
Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

# # 提取图片hog特征
# def extract_hog_features(X):
#     image_descriptors = []
#     for i in range(len(X)):
#         fd, _ = hog(X[i], orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm ='L2-Hys', visualize=True)
#         image_descriptors.append(fd)
#     # 返回的是训练部分所有图像的hog特征
#     return image_descriptors

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


data = pd.DataFrame(X_1)   # 一个Datarame表示一个表格，类似电子表格的数据结构，包含一个经过排序的列表集，它的每一列都可以有不同的类型值（数字，字符串，布尔等等）。
print(data.describe([0.01,0.1,0.3,0.6,0.9,0.99]).T)

X_2 = StandardScaler().fit_transform(X_1)
data = pd.DataFrame(X_2)
print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)


# 调用 KNN算法
k_range = range(3, 31, 2)
k_score = []
for k in k_range:
    KNN = KNeighborsClassifier(algorithm='auto',   # 在KNN中使用的算法，其他选项还有ball_tree，kd_tree和 brute
                               leaf_size=30,   # 当使用和树有关的算法时的叶子数量
                               metric='minkowski',p=2,  # 使用的是明可夫斯基距离中的欧式距离，1代表曼哈顿距离，2代表欧氏距离
                               metric_params=None,
                               n_jobs=1,  # 并行计算的线程数量
                               n_neighbors=k,  # K值的选取
                               weights='distance'  # 距离计算中使用的权重，distance表示按照距离的倒数加权，uniform表示各样本权重相同
    )
    # 将 KNN算法应用在训练集上
    KNN.fit(Xtrain,Ytrain)
    # 将结果应用于测试集中
    predict = KNN.predict(Xtest)
    score = KNN.score(Xtest,Ytest)
    k_score.append(score)
# print(predict)
# 计算模型的正确率
print(KNN.score(Xtest,Ytest))


# 画图
plt.plot(k_range, k_score)
plt.xlabel('Value of K for KNN')
plt.ylabel('score')
plt.show()


# 定义可视化函数
def visualize_cv(cv, face_data, face_target):
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (tr, tt) in enumerate(cv.split(face_data, face_target)):   # 枚举
        p1 = ax.scatter(tr, [i] * len(tr), c="#221f1f", marker="_", lw=8)
        p2 = ax.scatter(tt, [i] * len(tt), c="#b20710", marker="_", lw=8)
        ax.set(
            title=cv.__class__.__name__,
            xlabel="Data Index",
            ylabel="CV Iteration",
            ylim=[cv.n_splits, -1],
        )
        ax.legend([p1, p2], ["Training", "Validation"])

    plt.show()

cv = KFold(n_splits=10)   # 分割为 10个 K子集
visualize_cv(cv, face_data, face_target)
scores = cross_val_score(KNN, Xtrain, Ytrain, cv=10)
print(scores)
average = 0
for i in range(len(scores)):
    average = average + scores[i]
print("The average score is %f" % (average / len(scores)))


# 学习曲线 &交叉验证
if __name__ == '__main__':
    face_data = data_x.reshape(data_x.shape[0],-1)
    face_target = data_y
    # print(face_data.shape)
    X = face_data
    y = face_target
    scores = []
    ks = []
    for k in range(3,31,2):
        KNN = KNeighborsClassifier(n_neighbors=k)
        cross_score = cross_val_score(KNN,Xtrain,Ytrain,cv=10).mean()
        scores.append(cross_score)
        ks.append(k)
    ks_arr = np.array(ks)
    scores_arr = np.array(scores)
    plt.plot(ks_arr, scores_arr)
    plt.xlabel('k_value')
    plt.ylabel('score')
    plt.show()
    # 取最大值的下标
    max_idx = scores_arr.argmax()
    # 最大值对应的 k值
    max_k = ks[max_idx]
    # 最大值下标：
    print('最大值下标：',max_idx)
    # 最大值对应的 k值：
    print('最大值对应的k值：',max_k)