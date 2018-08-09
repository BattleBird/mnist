# coding: utf-8

""" 用卷积神经网络识别MNIST手写字符 """

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    """ 返回一个给定形状的变量并自动以截断正态分布初始化 """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ 返回一个给定形状的变量 """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_image, w_conv):
    """ 卷积计算 """
    return tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(h_conv):
    """ 池化操作 """
    return tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    # 读入数据
    MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # x为训练图像的占位符、y_为训练图像标签的占位符
    X = tf.placeholder(tf.float32, [None, 784])
    Y_ = tf.placeholder(tf.float32, [None, 10])

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    X_IMAGE = tf.reshape(X, [-1, 28, 28, 1])

    # 第一层卷积层
    W_CONV1 = weight_variable([5, 5, 1, 32])
    B_CONV1 = bias_variable([32])
    H_CONV1 = tf.nn.relu(conv2d(X_IMAGE, W_CONV1) + B_CONV1)
    H_POOL1 = max_pool_2x2(H_CONV1)

    # 第二层卷积层
    W_CONV2 = weight_variable([5, 5, 32, 64])
    B_CONV2 = bias_variable([64])
    H_CONV2 = tf.nn.relu(conv2d(H_POOL1, W_CONV2) + B_CONV2)
    H_POOL2 = max_pool_2x2(H_CONV2)

    # 全连接层，输出为1024维的向量
    W_FC1 = weight_variable([7 * 7 * 64, 1024])
    B_FC1 = bias_variable([1024])
    H_POOL2_FLAT = tf.reshape(H_POOL2, [-1, 7 * 7 * 64])
    H_FC1 = tf.nn.relu(tf.matmul(H_POOL2_FLAT, W_FC1) + B_FC1)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    KEEP_PROB = tf.placeholder(tf.float32)
    H_FC1_DROP = tf.nn.dropout(H_FC1, KEEP_PROB)

    # 把1024维的向量转换成10维，对应10个类别
    W_FC2 = weight_variable([1024, 10])
    B_FC2 = bias_variable([10])
    Y_CONV = tf.matmul(H_FC1_DROP, W_FC2) + B_FC2

    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    CROSS_ENTROPY = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_CONV))
    # 同样定义train_step
    TRAIN_STEP = tf.train.AdamOptimizer(1e-4).minimize(CROSS_ENTROPY)

    # 定义测试的准确率
    CORRECT_PREDICTION = tf.equal(tf.argmax(Y_CONV, 1), tf.argmax(Y_, 1))
    ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PREDICTION, tf.float32))

    # 创建Session和变量初始化
    SESS = tf.InteractiveSession()
    SESS.run(tf.global_variables_initializer())

    # 训练20000步
    for i in range(20000):
        batch = MNIST.train.next_batch(50)
        # 每100步报告一次在验证集上的准确度
        if i % 100 == 0:
            train_accuracy = ACCURACY.eval(feed_dict={
                X: batch[0], Y_: batch[1], KEEP_PROB: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        TRAIN_STEP.run(feed_dict={X: batch[0], Y_: batch[1], KEEP_PROB: 0.5})

    # 训练结束后报告在测试集上的准确度
    print "test accuracy %g" % ACCURACY.eval(feed_dict={
        X: MNIST.test.images, Y_: MNIST.test.labels, KEEP_PROB: 1.0})
