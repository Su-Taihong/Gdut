import os
from math import sqrt
from PIL import Image
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from numpy import mean
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 读取face文件（faceDR与faceDS的合并）
data_dr = []
with open('face', 'r') as file_dr:
    for line_dr in file_dr:
            line_dr = line_dr.replace('(', '')
            line_dr = line_dr.replace(')', '')
            line_dr_1 = line_dr.split('_')
            data_dr.append(line_dr_1)

x, y = [], []
path = 'rawdata'
i = -1
a = 0
exp_dict = {'sex  male ': 0, 'sex  female ': 1}
for label in os.listdir(path):
    i = i + 1
    # 先判断该编号的图片是否有缺失数据
    while True:
        if len(data_dr[i]) == 6:
            # 如果无缺失，则跳出whlie
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

#切割训练集与测试集，这里设置训练集占0.8，测试集占0.2
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = extract_hog_features(X_train_raw)
X_test = extract_hog_features(X_test_raw)
# 检查每个hog向量的长度
print(X_train[0].shape)

# # 选择支持向量机作为分类器
# svm = sklearn.svm.SVC(C=5.1, kernel='rbf')
# svm.fit(X_train, Y_train)
# Y_predict = svm.predict(X_test)
# acc = accuracy_score(Y_test, Y_predict)
# print('accuracy:', acc)

#选择贝叶斯分类器
gnb = GaussianNB(priors=[0.33, 0.67], var_smoothing=1e-6)
gnb.fit(X_train, Y_train)
Y_predict = gnb.predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (Y_test != Y_predict).sum()))
acc = accuracy_score(Y_test, Y_predict)
print('accuracy:',acc)

# 展示部分预测结果（100张图，0是男性、1是女性）
i = 0
plt.figure(figsize=(10, 10))
for img in X_test_raw[:100]:
    plt.subplot(10, 10, i + 1)
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(Y_predict[i], labelpad= -2, fontsize=15).set_size(8)
    i = i + 1
plt.show()

# #绘制混淆矩阵
# def plot_confusion_matrix(cm, labels_name, title):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
#     plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
#     plt.title(title)    # 图像标题
#     plt.colorbar()
#     num_local = np.array(range(len(labels_name)))
#     plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
#     plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# labels_name = ['male', 'female']
# c = confusion_matrix(y_true=Y_test, y_pred=Y_predict)
# print(c)
# plot_confusion_matrix(c, labels_name, "HAR Confusion Matrix")
# plt.show()

# #绘制ROC曲线
# def plot_roc_curve(roc_auc, fpr, tpr):
#     lw = 2
#     plt.figure(figsize=(10, 10))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
# fpr, tpr, threshold = roc_curve(Y_test, Y_predict)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
# plt.show()