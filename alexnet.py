import tensorflow as tf

class AlexNet(object):

    def __init__(self, classNum, keepProb):
        self.KEEP_PROB = keepProb
        self.NUM_CLASSES = classNum


    def conv(self, x, wH, wW, numFilters, xstep, ystep,  name, padding="SAME", groups = 1):

        channelNum = int(x.get_shape()[-1])

        convolve = lambda i, k: tf.nn.conv2d(i, k,strides=[1, ystep, xstep, 1],padding=padding)

        with tf.name_scope(name) as scope:
            W = tf.Variable(tf.truncated_normal([wH, wW, int(channelNum/groups), numFilters], mean=0,  stddev=0.1))
            b = tf.Variable(tf.truncated_normal([numFilters], mean= 0, stddev=0.1))
            if groups == 1:
                conv = convolve(x, W)
            else:
                inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weightGroups = tf.split(axis=3, num_or_size_splits=groups, value=W)
                outputGroups = [convolve(i, k) for i, k in zip(inputGroups, weightGroups)]

                conv = tf.concat(axis=3, values=outputGroups)

            bias = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape().as_list())

            relu = tf.nn.relu(bias, name=name)

            return relu

    def fc(self,x,inNum, outNum, name, relu = True):

        with tf.variable_scope(name) as scope:
            weights =  tf.Variable(tf.truncated_normal([inNum, outNum], mean=0.1, stddev=0.1))
            biases = tf.Variable(tf.truncated_normal([outNum], mean=0.1, stddev=0.1))

            act = tf.nn.xw_plus_b(x, weights, biases, name=name)

            if relu == True:
                relu = tf.nn.relu(act)
                return relu
            else:
                return act

    def max_pool(self, x, wH, wW, xstep, ystep, name, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, wH, wW, 1],
                              strides=[1, ystep, xstep, 1],
                              padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    # def createNetwork(self, x, testFlag = False):
    #
    #     with tf.name_scope("layer1"):
    #         conv1 = self.conv(x, 11, 11, 96, 4, 4, padding='SAME', name='conv1')
    #         pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='SAME', name='pool1')
    #         norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')
    #         tf.summary.histogram('norm1', norm1)
    #
    #     with tf.name_scope("layer2"):
    #         # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    #         conv2 = self.conv(norm1, 5, 5, 128, 1, 1, groups=1, name='conv2')
    #         pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='SAME', name='pool2')
    #         norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')
    #         tf.summary.histogram('norm2', norm2)
    #
    #     with tf.name_scope("layer3"):
    #         # 3rd Layer: Conv (w ReLu)
    #         conv3 = self.conv(norm2, 3, 3, 256, 1, 1, name='conv3')
    #         conv3 =  self.lrn(conv3, 2, 2e-05, 0.75, name='norm3')
    #         tf.summary.histogram('conv3', conv3)
    #
    #     with tf.name_scope("layer4"):
    #         # 4th Layer: Conv (w ReLu) splitted into two groups
    #         conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=1, name='conv4')
    #         conv4 = self.lrn(conv4, 2, 2e-05, 0.75, name='norm4')
    #         tf.summary.histogram('conv4', conv4)
    #
    #     with tf.name_scope("layer5"):
    #         # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    #         conv5 = self.conv(conv4, 3, 3, 512, 1, 1, groups=1, name='conv5')
    #         pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='SAME', name='pool5')
    #         pool5 = self.lrn(pool5, 2, 2e-05, 0.75, name='norm5')
    #         tf.summary.histogram('pool5', pool5)
    #
    #     with tf.name_scope("fclayer1"):
    #         # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    #         veclen = pool5.get_shape()[1]*pool5.get_shape()[2]*pool5.get_shape()[3]
    #         print(veclen)
    #         flattened = tf.reshape(pool5, [-1, 8192])
    #         fc6 = self.fc(flattened,8192, 2048, name='fc6')
    #         dropout6 = self.dropout(fc6, self.KEEP_PROB)
    #         #dropout6 = self.lrn(dropout6, 2, 2e-05, 0.75, name='norm6')
    #         tf.summary.histogram('dropout6', dropout6)
    #
    #     with tf.name_scope("fclayer2"):
    #         # 7th Layer: FC (w ReLu) -> Dropout
    #         fc7 = self.fc(dropout6, 2048, 1024, name='fc7')
    #         dropout7 = self.dropout(fc7, self.KEEP_PROB)
    #         #dropout7 = self.lrn(dropout7, 2, 2e-05, 0.75, name='norm7')
    #         tf.summary.histogram('dropout7',dropout7)
    #
    #     # 8th Layer: FC and return unscaled activations
    #     # (for tf.nn.softmax_cross_entropy_with_logits)
    #     with tf.name_scope("fclayer3"):
    #         out = self.fc(dropout7, 1024, self.NUM_CLASSES, relu=False, name='fc8')
    #         tf.summary.histogram('out1', out)
    #
    #     # with tf.name_scope("softmax"):
    #     #     out = tf.nn.softmax(out)
    #
    #     if testFlag:
    #
    #         with tf.name_scope("softmax"):
    #             out = tf.nn.softmax(out)
    #
    #     return out

    def createNetwork2(self, x, testFlag = False):

        with tf.name_scope("layer1"):
            conv1 = self.conv(x, 11, 11, 16, 4, 4, padding='SAME', name='conv1')
            pool1 = self.max_pool(conv1, 4, 4, 2, 2, padding='SAME', name='pool1')
            norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')
            tf.summary.histogram('norm1', norm1)

        with tf.name_scope("layer2"):
            # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
            conv2 = self.conv(norm1, 5, 5, 16, 1, 1, groups=1, name='conv2')
            pool2 = self.max_pool(conv2, 4, 4, 2, 2, padding='SAME', name='pool2')
            norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')
            tf.summary.histogram('norm2', norm2)

        with tf.name_scope("layer3"):
            # 3rd Layer: Conv (w ReLu)
            conv3 = self.conv(norm2, 3, 3, 32, 1, 1, name='conv3')
            pool3 = self.max_pool(conv3, 2, 2, 2, 2, padding='SAME', name='pool3')
            #pool3 =  self.lrn(pool3, 2, 2e-05, 0.75, name='norm3')
            tf.summary.histogram('pool3', pool3)

        with tf.name_scope("layer4"):
            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = self.conv(pool3, 3, 3, 32, 1, 1, groups=1, name='conv4')
            pool4 = self.max_pool(conv4, 2, 2, 2, 2, padding='SAME', name='pool4')
            #norm4 = self.lrn(pool4, 2, 2e-05, 0.75, name='norm4')
            tf.summary.histogram('norm4', pool4)

        # with tf.name_scope("layer5"):
        #     # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        #     conv5 = self.conv(conv4, 3, 3, 512, 1, 1, groups=1, name='conv5')
        #     pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='SAME', name='pool5')
        #     pool5 = self.lrn(pool5, 2, 2e-05, 0.75, name='norm5')
        #     tf.summary.histogram('pool5', pool5)

        with tf.name_scope("fclayer1"):
            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            veclen = pool4.get_shape()[1]*pool4.get_shape()[2]*pool4.get_shape()[3]
            print(veclen)
            flattened = tf.reshape(pool4, [-1, 128])
            fc6 = self.fc(flattened, 128, 64, name='fc6')
            dropout6 = self.dropout(fc6, self.KEEP_PROB)
            #dropout6 = self.lrn(dropout6, 2, 2e-05, 0.75, name='norm6')
            tf.summary.histogram('dropout6', dropout6)


        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        with tf.name_scope("fclayer2"):
            out = self.fc(dropout6, 64, self.NUM_CLASSES, relu=False, name='fc8')
            tf.summary.histogram('out1', out)

        if testFlag:

            with tf.name_scope("softmax"):
                out = tf.nn.softmax(out)

        return out

