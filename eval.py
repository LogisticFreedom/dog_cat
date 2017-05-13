import tensorflow as tf
from alexnet import AlexNet
from read_data import inputNoShuffle
import csv



SIZE = 128
batchSize = 100
classNum = 2
keepprob = 0.95
pictureNum = 12500

csvfile = open('./result/csv_test3.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['id', 'label'])

with tf.name_scope("model"):

    img = inputNoShuffle("./data/test.tfrecords", SIZE)
    alexnet = AlexNet(classNum, keepprob)

    pred = alexnet.createNetwork2(img, testFlag=True)
    ans = tf.argmax(pred, axis=1)


with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess, "./model_save/model5.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(pictureNum):
        ansArr, predArr = sess.run([ans, pred])
        print(ansArr, predArr)
        writer.writerow([str(i+1), float(predArr[0, 1])])

    csvfile.close()
    coord.request_stop()
    coord.join(threads)


