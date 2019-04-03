import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
pb_file_path = 'mnist.pb'
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.int32, [None, ], name='y')

dense1 = tf.layers.dense(inputs=x,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.nn.l2_loss)
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.nn.l2_loss)
logits = tf.layers.dense(inputs=dense2,
                         units=10,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.nn.l2_loss, name='logit')

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
pre_label = tf.argmax(logits, 1, name='output')
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
# saver = tf.train.Saver(max_to_keep=1)
max_acc = 0
for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
    val_loss, val_acc = sess.run([loss, acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('epoch:%d, val_loss:%f, val_acc:%f' % (i, val_loss, val_acc))
    if val_acc > max_acc:
        # max_acc = val_acc
        # saver.save(sess, 'Model/mnist.ckpt', global_step=i + 1)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
