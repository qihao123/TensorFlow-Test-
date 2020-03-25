import numpy as np
import pandas as pd
from pandas import Series,DataFrame

from matplotlib import pyplot as plt

#导入tensorflow
import tensorflow as tf

#导入MNIST(手写数字数据集)
from tensorflow.examples.tutorials.mnist import input_data
#获取训练数据与测试数据
import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets('./TensorFlow',one_hot=True)

test = mnist.test
test_images = test.images

train = mnist.train
images = train.images
#模拟线性方程
#创建占矩阵位符X,Y
X = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])

#随机生成斜率W和截距b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#根据模拟线性方程得出预测值
y_pre = tf.matmul(X,W)+b

#将预测值结果概率化
y_pre_r = tf.nn.softmax(y_pre)
#构建损失函数
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_pre_r),axis=1))
#实现梯度下降
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#TensorFlow初始化，并开始训练
#定义相关参数

#训练循环次数
training_epochs = 25
#batch 一批，每次训练给算法10个数据
batch_size = 10
#每隔5次，打印输出运算的结果
display_step = 5


#预定义初始化
init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    #初始化
    sess.run(init)
    #保存模型
    saver = tf.train.Saver()
    model_path = "./model/logistic/logistic.ckpt"
    save_path = saver.save(sess,model_path)

    #循环训练次数
    for epoch in range(training_epochs):
        avg_cost = 0.
        #总训练批次total_batch =训练总样本量/每批次样本数量
        total_batch = int(train.num_examples/batch_size)
        for i in range(total_batch):
            #每次取出100个数据作为训练数据
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost +=c/total_batch
        if(epoch+1)%display_step == 0:
            print(batch_xs.shape,batch_ys.shape)
            print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
    print('Optimization Finished!')

    #7.评估效果
    # Test model
    correct_prediction = tf.equal(tf.argmax(y_pre_r,1),tf.argmax(Y,1))
    # Calculate accuracy for 3000 examples
    # tf.cast类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({X: mnist.test.images[:3000], Y: mnist.test.labels[:3000]}))
'''
(10, 784) (10, 10)
epoch: 0005 cost= 0.311371713
(10, 784) (10, 10)
epoch: 0010 cost= 0.287550523
(10, 784) (10, 10)
epoch: 0015 cost= 0.277123527
(10, 784) (10, 10)
epoch: 0020 cost= 0.270665784
(10, 784) (10, 10)
epoch: 0025 cost= 0.266008329
Optimization Finished!
Accuracy: 0.899
'''

