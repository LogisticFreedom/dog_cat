
import tensorflow as tf
from PIL import Image
import numpy as np

def inputData(fileNameQue, size, trainFlag = True):
    '''
    :param fileNameQue: 文件名队列
    :param size: 图片尺寸
    :param trainFlag: 读取的是否是训练集
    :return: 训练集返回图像和标签，测试集返回图像
    '''

    #fileNameQue = tf.train.string_input_producer(["./data/faceTF.tfrecords"])

    # 创建tfrecorder阅读器
    reader = tf.TFRecordReader()
    key,value = reader.read(fileNameQue)

    if trainFlag:
        # 这里以数据类型的固定长度解析数据，要和封装时的数据对应
        features = tf.parse_single_example(value,features={ 'label': tf.FixedLenFeature([], tf.int64),
                                               'img' : tf.FixedLenFeature([], tf.string),})

        img = tf.decode_raw(features["img"], tf.uint8) # 解码图像
        img = tf.reshape(img, [size, size, 3]) # 恢复图像
        img = tf.cast(img, dtype=tf.float32) # 转化图像数据格式
        label = tf.cast(features['label'], dtype=tf.int64) # 转换标签数据格式
        #label = tf.cast(label, dtype=tf.float32)

        return img, label
    else:
        features = tf.parse_single_example(value, features={'img': tf.FixedLenFeature([], tf.string), })

        img = tf.decode_raw(features["img"], tf.uint8)
        img = tf.reshape(img, [size, size, 3])
        img = tf.cast(img, dtype=tf.float32)
        return img

def inputBatch(filename, batchSize, imgSize, dequeue=10000):
    '''

    :param filename: 文件名
    :param batchSize: batch大小
    :param imgSize: 图像尺寸
    :param dequeue: 缓冲队列大小
    :return: 图像batch，标签batch
    '''
    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    example, label = inputData(fileNameQue,imgSize)
    min_after_dequeue = dequeue   # 样本池调整的大一些随机效果好
    capacity = min_after_dequeue + 3 * batchSize

    # 上一个函数生成的样本会在这里积蓄并打乱成batch输出
    exampleBatch, labelBatch = tf.train.shuffle_batch([example, label], batch_size=batchSize, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return exampleBatch, labelBatch

def inputNoShuffle(filename,imgSize):
    '''

    :param filename: 文件名
    :param imgSize: 图像大小
    :return: 图像
    '''
    fileNameQue = tf.train.string_input_producer([filename], shuffle=False)
    example = inputData(fileNameQue, imgSize,trainFlag=False)
    print(type(int(example.get_shape()[0])))
    ch1, ch2, ch3 = int(example.get_shape()[0]),int(example.get_shape()[1]),int(example.get_shape()[2])
    example = tf.reshape(example,[1,ch1,ch2,ch3])
    return example



if __name__ == "__main__":

    SIZE = 128
    with tf.Session() as sess:

        #fileNameQue = tf.train.string_input_producer(["./data/train.tfrecords"])
        #img, label = inputData(fileNameQue, SIZE)

        imgBatch, labelBatch = inputBatch("./data/train.tfrecords", 200, SIZE)
        #img = inputNoShuffle("./data/test.tfrecords",SIZE)

        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            imgArr = sess.run(imgBatch)
            labelArr = sess.run(labelBatch)
            print(imgArr.shape, labelArr)
            # im = Image.fromarray(np.uint8(imgArr))
            # im.save("./data/img.jpg")

        coord.request_stop()
        coord.join(threads)