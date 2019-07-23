import sys
import tensorflow as tf
from tensorflow.python.framework import graph_io
import tensorflow.contrib.tensorrt as trt
import tf2onnx


INPUT_NODE = 'worker_0/validation/IteratorGetNext:0' # ADJUST
OUTPUT_NODE = 'worker_0/post_processing/Softmax' # ADJUST
MAX_BATCH_SIZE = 1
DATA_TYPE = 'FP32' # ADJUST # 'FP16' | 'FP32'
MAX_WORKSPACE = int(0.1*(1<<32))
#MAX_WORKSPACE = 1 << 16

graphdef_frozen = tf.GraphDef()
with tf.gfile.GFile(sys.argv[1], "rb") as f:
    graphdef_frozen.ParseFromString(f.read())

with tf.Session() as sess:
    graphdef_trt = trt.create_inference_graph(
    input_graph_def=graphdef_frozen,
    outputs=[OUTPUT_NODE],
    max_batch_size=MAX_BATCH_SIZE,
    max_workspace_size_bytes=MAX_WORKSPACE,
    precision_mode=DATA_TYPE,
    is_dynamic_op=True)

    tf.import_graph_def(graphdef_trt)

    #pb
    graph_io.write_graph(graphdef_trt, './', sys.argv[2], as_text=False)

    #import pdb; pdb.set_trace() 

    #onnx
    #onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=[INPUT_NODE], output_names=[OUTPUT_NODE])
    #model_proto = onnx_graph.make_model("test")
    #with open(sys.argv[2], "wb") as f:
    #    f.write(model_proto.SerializeToString())

    #uff
    #uff.from_tensorflow_frozen_model(graphdef_trt, output_filename=sys.argv[2], output_nodes=[OUTPUT_NODE], text=False)
