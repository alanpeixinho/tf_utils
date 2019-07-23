import tensorflow as tf
from tensorflow.python.platform import gfile
import sys

with tf.Session() as sess:
    with gfile.FastGFile(sys.argv[1], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        print(tf.get_default_graph())
        nodes = [n for n in tf.get_default_graph().as_graph_def(add_shapes=True).node]

        import pdb; pdb.set_trace()

        for node in nodes:
            print('%s -< %s (%s)'%(node.name, node.input, node.attr['_input_shapes']))

        print('Total of %d nodes.'%(len(nodes)))
