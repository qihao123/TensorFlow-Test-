from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
sess = tf.InteractiveSession()
#开一个session
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
#输入数据第一个是数据类型第二个是tensor的shape
w = tf.Variable(tf.zeros([784,10]))
#权重
b = tf.Variable(tf.zeros([10]))
#y=wx+b中的b
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None,10])
cross_enteropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#信息熵做Loss Function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_enteropy)
#利用SGD随即梯度下降法做优化函数，学习速率选0.5
tf.global_variables_initializer().run()
#全局参数初始化，并run
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_: batch_ys})
#选择训练数据的100条样本迭代训练1000次
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#计算分类是否正确
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#计算acu
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

