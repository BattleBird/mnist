#coding: utf-8
""" 打印前20张图片并保存 """
import os

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc


# 读取MNIST数据集。如果不存在会事先下载。
MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 我们把原始图片保存在MNIST_data/raw/文件夹下
# 如果没有这个文件夹会自动创建
SAVE_DIR = 'MNIST_data/raw/'
if os.path.exists(SAVE_DIR) is False:
    os.makedirs(SAVE_DIR)

# 保存前20张图片
for i in range(20):
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = MNIST.train.images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = SAVE_DIR + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print 'Please check: %s ' % SAVE_DIR
