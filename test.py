import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
pb_file_path = 'mnist.pb'

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        pre_label = sess.graph.get_tensor_by_name("output:0")
        print(pre_label)
        for i in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            a = sess.run(pre_label, feed_dict={'x:0': batch_xs})
            prediction = np.array(a)
            print(prediction)
            # print(prediction.argmax(axis=1))
