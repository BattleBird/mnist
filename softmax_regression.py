# coding:utf-8
""" Softmax模型 实现了MNIST数据机的分类 """

# 导入tensorflow。
# 这句import tensorflow as tf是导入TensorFlow约定俗成的做法，请大家记住。
import tensorflow as tf
# 导入MNIST教学的模块
from tensorflow.examples.tutorials.mnist import input_data
# 与之前一样，读入MNIST数据
MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x，x是一个占位符（placeholder），代表待识别的图片
X = tf.placeholder(tf.float32, [None, 784])

# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在TensorFlow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))
# b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）。
B = tf.Variable(tf.zeros([10]))

# y=softmax(Wx + b)，y表示模型的输出
Y = tf.nn.softmax(tf.matmul(X, W) + B)

# y_是实际的图像标签，同样以占位符表示。
Y_ = tf.placeholder(tf.float32, [None, 10])

# 至此，我们得到了两个重要的Tensor：y和y_。
# y是模型的输出，y_是实际的图像标签，不要忘了y_是独热表示的
# 下面我们就会根据y和y_构造损失

# 根据y, y_构造交叉熵损失
CROSS_ENTROPY = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y)))

# 有了损失，我们就可以用随机梯度下降针对模型的参数（W和b）进行优化
TRAIN_STEP = tf.train.GradientDescentOptimizer(0.01).minimize(CROSS_ENTROPY)

# 创建一个session。只有在session中才能运行优化步骤train_step。
SESS = tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存。
tf.global_variables_initializer().run()
print 'start training...'

# 进行1000步梯度下降
# Python中对于无需关注其实际含义的变量可以用_代替，这就和for i in range(5)一样，因为这里我们对i并不关心，所以用_代替仅获取值而已
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
    # batch_xs, batch_ys对应着两个占位符x和y_
    batch_xs, batch_ys = MNIST.train.next_batch(100)
    # 在session中运行train_step，运行时要传入占位符的值
    SESS.run(TRAIN_STEP, feed_dict={X: batch_xs, Y_: batch_ys})

# 正确的预测结果
CORRECT_PREDICTION = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
# 计算预测准确率，它们都是Tensor
ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PREDICTION, tf.float32))
# 在session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的正确率
print SESS.run(ACCURACY, feed_dict={X: MNIST.test.images, Y_: MNIST.test.labels})  # 0.9185
