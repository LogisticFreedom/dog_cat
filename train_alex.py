import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from read_data import inputBatch

with tf.Session() as sess:

    SIZE = 128
    learningRate = 0.001
    epoch = 10000
    disp = 10
    batchSize = 200
    classNum = 2
    logs_path = "C:/tf_log/dog_cat/"
    keepprob = 0.90


    with tf.name_scope("model"):
        imgBatch, labelBatch = inputBatch("./data/train.tfrecords", batchSize, SIZE)
        labelBatch = tf.one_hot(labelBatch, depth=2, on_value=1, off_value=0)
        alexnet = AlexNet(classNum, keepprob)
        pred = alexnet.createNetwork2(imgBatch)

    with tf.name_scope("loss"):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0001
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labelBatch))+lossL2
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
        tf.summary.scalar("loss", cost)

    with tf.name_scope("eval"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labelBatch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("acc", accuracy)

    merged_summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epoch):
        _, costArr, summary, predArr = sess.run([optimizer, cost, merged_summary_op, pred])
        summary_writer.add_summary(summary, i)
        if i % disp == 0:
            acc = sess.run(accuracy)
            print(i, acc, costArr)
            print(np.argmax(predArr, axis=1))

    saver_path = saver.save(sess, "./model_save/model5.ckpt")

    coord.request_stop()
    coord.join(threads)