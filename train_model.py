import tensorflow as tf
from read_data import inputBatch
from model import createCNN


with tf.Session() as sess:

    SIZE = 227
    learningRate = 0.001
    epoch = 10000
    disp = 10
    batchSize = 50
    classNum = 2
    logs_path = "C:/tf_log/dog_cat/"

    with tf.name_scope("model"):
        imgBatch, labelBatch = inputBatch("./data/train.tfrecords", batchSize, SIZE)
        labelBatch = tf.one_hot(labelBatch, depth=2, on_value=1, off_value=0)
        pred = createCNN(imgBatch)
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labelBatch))
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
        tf.summary.scalar("loss", cost)
    with tf.name_scope("eval"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labelBatch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("loss", accuracy)

    merged_summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epoch):
        _, costArr, summary = sess.run([optimizer, cost, merged_summary_op])
        summary_writer.add_summary(summary, i)
        if i % disp == 0:
            acc = sess.run(accuracy)
            print(i, acc, costArr)


    coord.request_stop()
    coord.join(threads)

#tensorboard --logdir=c:/tf_log/dog_cat/



