# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import scipy.io
from scipy import interpolate
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

folder_path = 'D:\\BaiduNetdiskDownload\\SEED_IV\\SEED_IV\\eeg_feature_smooth\\a\\'
file_name = '1_20160518.mat'
folder_path2 = 'D:\\BaiduNetdiskDownload\\SEED_IV\\SEED_IV\\eeg_feature_smooth\\2\\'
file_name2 = '1_20161125.mat'
folder_path3 = 'D:\\BaiduNetdiskDownload\\SEED_IV\\SEED_IV\\eeg_feature_smooth\\3\\'
file_name3 = '1_20161126.mat'
# 加载.mat文件
data = scipy.io.loadmat(folder_path + file_name)
data2 = scipy.io.loadmat(folder_path2 + file_name2)
data3 = scipy.io.loadmat(folder_path3 + file_name3)

c = np.zeros((62,20,24))  # 11111111111111
resized_array=np.zeros((62,20))
text={}
for i in range(24):
    text = 'de_LDS' + str(i + 1)
    array = data[text][:, :, 1]
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    f = interp2d(x, y, array, kind='linear')  # 使用转置的array，匹配RectBivariateSpline的x和y顺序
    x_new = np.linspace(0, array.shape[1], 20)
    y_new = np.linspace(0, array.shape[0], 62)
    resized_array = f(x_new, y_new)
    # 将插值后的数据保存到c数组中
    c[:, :, i] = resized_array
    
c2 = np.zeros((62,20,24))  # 11111111111111
# 获取变量a的值并赋给aaa
for i in range(24):
    text='de_LDS'+str(i+1)
    array = data[text][:,:,1]
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    f = interp2d(x, y, array, kind='linear')  # 使用转置的array，匹配RectBivariateSpline的x和y顺序
    x_new = np.linspace(0, array.shape[1], 20)
    y_new = np.linspace(0, array.shape[0], 62)
    resized_array = f(x_new, y_new)
    
    c2[:,:,i]= resized_array
    
c3 = np.zeros((62,20,24))  # 11111111111111
# 获取变量a的值并赋给aaa
for i in range(24):
    text='de_LDS'+str(i+1)
    array = data[text][:,:,1]
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    f = interp2d(x, y, array, kind='linear')  # 使用转置的array，匹配RectBivariateSpline的x和y顺序
    x_new = np.linspace(0, array.shape[1], 20)
    y_new = np.linspace(0, array.shape[0], 62)
    resized_array = f(x_new, y_new)
    
    c3[:,:,i]= resized_array

d = np.concatenate((c, c2, c3), axis=2)
#####################################################   CBAM
def cbam_module(input_tensor):
    cbam_module_url = "https://tfhub.dev/google/cbam/v1/1"
    cbam_model = hub.load(cbam_module_url)
    enhanced_data = cbam_model(input_tensor)
    return enhanced_data

# 转换为 TensorFlow 张量
input_tensor = tf.constant(d)

# 将输入数据传递给CBAM模块进行增强
output_tensor = cbam_module(input_tensor)

#############################################################################
# 加载数据集
transposed_data = np.transpose(output_tensor, (2, 0, 1))
train_images=transposed_data
train_images = train_images / 255.0  # 归一化到[0, 1]范围

# 定义生成器模型
generator = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(1240, activation='sigmoid'),
    Reshape((62, 20))
])

# 定义判别器模型
discriminator = Sequential([
    Flatten(input_shape=(62, 20)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译判别器模型
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 定义GAN模型
discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# 训练GAN
batch_size = 72
epochs = 20000
d_losses = []
g_losses = []
log_d=[]
for epoch in range(epochs):
    # 训练判别器
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_imgs = train_images[idx]
    fake_imgs = generator.predict(np.random.randn(batch_size, 100))
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 训练生成器
    noise = np.random.randn(batch_size, 100)
    g_loss = gan.train_on_batch(noise, real_labels)

    # 打印损失信息
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        # 保存生成的图像
        if epoch % 1000 == 0:
            generated_images = generator.predict(np.random.randn(16, 100))
            fig, axs = plt.subplots(4, 4)
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i,j].imshow(generated_images[count, :, :], cmap='gray')
                    axs[i,j].axis('off')
                    count += 1
            plt.show()

plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()   
