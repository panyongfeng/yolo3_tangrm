import tensorflow as tf
#saved_model_dir = "./jsp_saved_model"
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#tflite_quant_model = converter.convert()
print (tf.__version__)
#graph_def_file = "./ckpt/direct.pb"
graph_def_file = "model_mv2_opt.pb"

output_node_names = ['y2/Reshape', 'y3/Reshape']

input_node_name = ['Reshape']
#input_node_name = ['predict_image']




converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_node_name, output_node_names, input_shapes={"Reshape":[1, 416,416,3]})

#converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_node_name, output_node_names, input_shapes={"predict_image":[None,416,416,3]})
#converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_node_name, output_node_names, input_shapes={"ForwardPass/w2l_encoder/conv11/conv1d/ExpandDims":[1,1,1024,64]})

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

#converter.post_training_quantize=True


tflite_model = converter.convert()

open("mobilenetv2.tflite", "wb").write(tflite_model)
