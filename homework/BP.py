import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
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
# 加载数据集
# print(x)
# print(y)
# 标签编码
le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)
y_one_hot = np_utils.to_categorical(y_encoded)
def extract_hog_features(X):
    image_descriptors = []
    for i in range(len(X)):
        fd, _ = hog(X[i], orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm ='L2-Hys', visualize=True)
        image_descriptors.append(fd)
    # 返回的是训练部分所有图像的hog特征
    return image_descriptors
X_features = extract_hog_features(x)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_features, y_one_hot, test_size=0.15, random_state=40)
X_train = np.array(X_train)
y_train= np.array(y_train)
X_test= np.array(X_test)
y_test= np.array(y_test)

print(len(y_test),len(y_train),len(X_test),len(X_train))

model = Sequential()
model.add(Dense(units=256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# 编译模型
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=60, batch_size=10, validation_data=(X_test, y_test))

def predict(X, model):
    prediction = np.argmax(model.predict(X), axis=1)
    gender = le.inverse_transform(prediction)
    return gender

from sklearn.metrics import accuracy_score

y_pred = predict(X_test, model)
accuracy = accuracy_score(y_test.argmax(axis=1), le.transform(y_pred))
print(f"Accuracy: {accuracy}")

y_pred= np.array(y_pred)
y_test= np.array(y_test)
print(len(y_pred))
print(y_pred)
print(y_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# 准确率
acc = accuracy_score(y_test.argmax(axis=1), le.transform(y_pred))
print("准确率:", acc)

# 精确率
precision = precision_score(y_test.argmax(axis=1), le.transform(y_pred))
print("精确率:", precision)

# 召回率
recall = recall_score(y_test.argmax(axis=1), le.transform(y_pred))
print("召回率:", recall)

# F1-score
f1 = f1_score(y_test.argmax(axis=1), le.transform(y_pred))
print("F1-score:", f1)

# 混淆矩阵
cm = confusion_matrix(y_test.argmax(axis=1), le.transform(y_pred))
print("混淆矩阵:\n", cm)
#问题可能是数组数量不一样导致的问题。