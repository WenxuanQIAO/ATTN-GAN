# -*- coding: utf-8 -*-
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN, Flatten, Dense, Reshape
import numpy as np
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM
from keras.models import Sequential, Model
all_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
Y = np.repeat(all_label, 80)
Y1= np.repeat(Y, 3)
file_path = 'D:/BaiduNetdiskDownload/SEED_IV/SEED_IV/eegdata/'
people_name = ['1_1', '1_2', '1_3']
short_name = ['cz', 'cz', 'cz']
tmp_trial_signal=np.zeros((72,62,8000))
# 导入对抗数据
GAN_data=np.load('D:/BaiduNetdiskDownload/SEED_IV/SEED_IV/eegdata/666.npy')

Z=np.zeros((80,62,98))
sum_z = np.zeros((72,80,62))  # 初始化一个二维数组
for i in range(len(people_name)):
    file_name = file_path + people_name[i]
    data = loadmat(file_name)
    for trial in range(24):
        tmp_trial_signal[i*24+trial,:,:] = data[short_name[i] + '_eeg' + str(trial + 1)][:,:8000]
        diff_matrix=np.diff(tmp_trial_signal[i*24+trial,:,:],axis=1)    
        ########### 提取图形特征
        for j in range(80):
            row_data = diff_matrix[:,j*100:((j+1)*100)-1]
            # 提取 x 和 y 轴的数据
            x_axis = row_data[:,:-1]  
            y_axis = row_data[:,1:]   
            Z[j,:,:] = np.sqrt(x_axis**2 + y_axis**2)# 计算SDC
            sum_z[i*24+trial,j,:]=np.sum(Z[j,:,:],axis=1)

data_3d=sum_z
labels=Y1

# 将标签转换为 one-hot 编码
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=4)

# 构建4D数据 
x_train = data_3d
X89 = np.zeros((72, 80, 72))
j = 0
for i in range(72):
    if i in {0, 1, 6, 7, 56, 63, 64, 65, 70, 71}:
        j += 1
    else:
        X89[:, :, i] = x_train[:,:, min(i - j, x_train.shape[0] - 1)]
   
x_train_reshaped = X89.reshape((5760, 8, 9, 1))

y_train = one_hot_labels

#  CRNN模型

img_size = (8, 9, 1)
def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
    seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
    seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
    seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
    seq.add(MaxPooling2D(2, 2, name='pool1'))
    seq.add(Flatten(name='fla1'))
    seq.add(Dense(512, activation='relu', name='dense1'))
    seq.add(Reshape((1, 512), name='reshape'))

    return seq

base_network = create_base_network(img_size)
input = Input(shape=img_size)

out_all = Concatenate(axis=1)([base_network(input)])
lstm_layer = LSTM(128, name='lstm')(out_all)
out_layer = Dense(4, activation='softmax', name='out')(lstm_layer)
model = Model([input], out_layer)

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history =model.fit([x_train_reshaped[:, :]], y_train,
          epochs=300, batch_size=128,validation_split=0.2)
# 转换数据为 NumPy 数组

# 获取训练过程中的损失值和准确度
loss = history.history['loss']
accuracy = history.history['accuracy']

# 获取验证集上的损失值和准确度
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# 打印最后一次的准确度和验证集准确度
print(f'Training Accuracy: {accuracy[-1]:.4f}')
print(f'Validation Accuracy: {val_accuracy[-1]:.4f}')

