# coding:utf-8
""" 查看变量形状大小
    打印第0副图片的向量表示
    打印第0副图片的标签 """
# 从tensorflow.examples.tutorials.mnist引入模块。这是TensorFlow为了教学MNIST而提前编制的程序
from tensorflow.examples.tutorials.mnist import input_data
# 从MNIST_data/中读取MNIST数据。这条语句在数据不存在时，会自动执行下载
MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 查看训练数据的大小
print MNIST.train.images.shape # (55000, 784)
print MNIST.train.labels.shape  # (55000, 10)

# 查看验证数据的大小
print MNIST.validation.images.shape  # (5000, 784)
print MNIST.validation.labels.shape  # (5000, 10)

# 查看测试数据的大小
print MNIST.test.images.shape  # (10000, 784)
print MNIST.test.labels.shape  # (10000, 10)

# 打印出第0幅图片的向量表示
print MNIST.train.images[0, :]

# 打印出第0幅图片的标签
print MNIST.train.labels[0, :]
