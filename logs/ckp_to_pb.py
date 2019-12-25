import tensorflow as tf
#import horovod.tensorflow as hvd
from tensorflow.python.tools import optimize_for_inference_lib

print (tf.__version__)
meta_path = './ckpt_mv2/alphabet_try.ckpt.meta'
#meta_path = './ckpt_old/try1.ckpt.meta'
output_node_names = ['y2/Reshape', 'y3/Reshape']
#input_node_name = ['IteratorGetNext']
input_node_name = ['Reshape']
#input_node_name = ['predict_image']
with tf.Session() as sess:
    # Restore the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)


    # Load weights
    saver.restore(sess, tf.train.latest_checkpoint('./ckpt_mv2/'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('model_mv2.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())




    outputGraph = optimize_for_inference_lib.optimize_for_inference(
                frozen_graph_def,
                input_node_name, # an array of the input node(s)
                output_node_names, # an array of output nodes
                tf.float32.as_datatype_enum
                )
    with open('model_mv2_opt.pb', 'wb') as f:
      f.write(outputGraph.SerializeToString())

