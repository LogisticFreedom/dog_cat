import tensorflow as tf

# 卷积操作
def conv2d(name, input, w, b, pad = "SAME", step=1):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, step, step, 1], padding=pad),b), name=name)


# 最大下采样操作
def maxPool(name, input, size, step):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, step, step, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, input, lsize=0):
    return tf.nn.lrn(input, depth_radius=lsize, name=name)

def createCNN(inputBatch, classNum = 2, dropout = 0.1):
    # 要求图像尺寸为128*128*3
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wd1': tf.Variable(tf.random_normal([256*16*16, 1024])),
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, classNum]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bc3': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([classNum]))
    }

    with tf.name_scope('layer1'):
        # 卷积层
        conv1 = conv2d('conv1', inputBatch, weights['wc1'], biases['bc1'])
        # 下采样层
        pool1 = maxPool('pool1', conv1, size=2, step=2)
        # 归一化层
        norm1 = norm('norm1', pool1, lsize=5)
        # Dropout
        norm1 = tf.nn.dropout(norm1, dropout)

    with tf.name_scope('layer2'):
        # 卷积
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        # 下采样
        pool2 = maxPool('pool1', conv2, size=2, step=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=0)
        # Dropout
        norm2 = tf.nn.dropout(norm2, dropout)

    with tf.name_scope('layer3'):
        # 卷积
        conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        # 下采样
        pool3 = maxPool('pool1', conv3, size=2, step=2)
        # 归一化
        norm3 = norm('norm3', pool3, lsize=4)
        # Dropout
        norm3 = tf.nn.dropout(norm3, dropout)


    with tf.name_scope('fullconnect'):
        # 全连接层，先把特征图转为向量
        dense1 = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1')
        # 全连接层
        dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')  # Relu activation

        # 网络输出层
        out = tf.matmul(dense2, weights['out']) + biases['out']

    return out




