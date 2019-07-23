import tensorflow as tf
from tensorflow.python.platform import gfile
import sys

with tf.Session() as sess:
    tf.train.import_meta_graph(sys.argv[1])
    print(tf.get_default_graph())
    nodes = [n for n in tf.get_default_graph().as_graph_def(add_shapes=True).node]
    for node in nodes:
        print('%s -< %s (%s)'%(node.name, node.input, node.attr['_input_shapes']))

