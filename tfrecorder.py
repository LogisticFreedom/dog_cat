import tensorflow as tf
import numpy as np
from PIL import Image

import os

cwd = os.getcwd()
print(cwd)

# 数据路径
trainRoot = cwd+"/train/train"
testRoot = cwd+"/test/test"

def createTFRecorder(rootPath, TFwriter,  size = 128, trainFlag = True):
    '''
    :param rootPath: 数据文件根目录
    :param TFwriter: TFrecorder写入器
    :param size:  图片归一化尺寸
    :param trainFlag: 是否是训练集，如果是训练集，要有加label操作
    :return: 无
    '''
    for parent, dirnames, filenames in os.walk(rootPath):
        label = 0
        if not trainFlag:
            filenames = [str(i+1)+".jpg" for i in range(12500)]
            print(filenames)
        for filename in filenames:
            imgPath = rootPath+"/"+filename
            print (imgPath)
            rawImg = Image.open(imgPath)
            print (rawImg.size,rawImg.mode)
            img = rawImg.resize((size, size)) # 归一化尺寸，只是用了简单的resize
            imgRaw = img.tobytes()
            print(img.size, img.mode)
            # 训练集获取标签，并写入，测试集只写入数据
            if trainFlag:
                strLabel = filename[:3]
                if strLabel == "cat":
                    label = 0
                if strLabel == "dog":
                    label = 1
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                    "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
                }) )
                TFwriter.write(example.SerializeToString())
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                }))
                # 写入
                TFwriter.write(example.SerializeToString())
    TFwriter.close()


if __name__ == "__main__":

    # 归一化尺寸大小
    SIZE = 128
    # 训练数据生成
    TFwriter_train = tf.python_io.TFRecordWriter("./data/train.tfrecords")
    createTFRecorder(trainRoot, TFwriter_train, SIZE)
    # 测试数据生成
    TFwriter_test = tf.python_io.TFRecordWriter("./data/test.tfrecords")
    createTFRecorder(testRoot, TFwriter_test, SIZE, trainFlag=False)
