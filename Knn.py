# encoding:utf-8
import numpy as np
import tensorflow as tf

# 加载mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./TensorFlow", one_hot=True)

# 这里从数据集中抽取5000个样本数据作为训练集，200个样本做为测试集,这里Ytr是一个10维ndarray
Xtr, Ytr = mnist.train.next_batch(100)
Xte, Yte = mnist.test.next_batch(4)

# tf Graph的输入
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# 使用L1距离计算最近邻
# tf.neg()给xte中的每个元素取反，这样可以使用tf.add()
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)  # 这里reduction_indices=1是计算每一行的和
# 从distance tensor中寻找最小的距离索引
pred = tf.argmin(distance, 0)

accuracy = 0.

# 初始化所有 variables
init = tf.initialize_all_variables()

# Launch  graph
with tf.Session() as sess:
    sess.run(init)

    # 对每个测试样本，计算它的分类类别
    for i in range(len(Xte)):
        # 获得最近邻
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print ("nn_indext:", nn_index)
        print ("Ytr[nn_index]:", Ytr[nn_index])
        print ("np.argmax(Ytr[nn_index]):", np.argmax(Ytr[nn_index]))
        # 获得测试样本的最近邻类别，并将它与真实类别做比较
        print ("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
        # 计算 accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print ("Done!")
    print ("Accuracy:", accuracy)
