import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wave
import python_speech_features as psf
import time
from PIL import Image



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_ep7.tflite")

#print(interpreter)
#

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
details_all = interpreter.get_tensor_details()

print(input_details)
print(output_details)
#print(details_all)


def get_pic_data_zoom(filename):
	im = Image.open(filename).convert('L')
	scale = im.size[1] * 1.0 / 32
	w = im.size[0] / scale
	w = int(w)
	im = im.resize((w, 32))
	img = np.array(im).astype(np.float32) / 255.0
	X = img.reshape((32, w, 1))
	#X = np.array([X])
	#print(np.shape(X))
	return X

pic_data = get_pic_data_zoom("../svt1/cropped_img/0014_ORIGINAL_123.jpg")

feature_shape = np.shape(pic_data)
print(feature_shape)

feature_length = feature_shape[1]


interpreter.resize_tensor_input(input_details[0]['index'], (1, 32, feature_length,1))
interpreter.allocate_tensors()

print(np.shape(pic_data))
input_data = np.array(pic_data, dtype=np.float32)
input_data = np.reshape(input_data, (1,32, feature_length,1))

interpreter.set_tensor(input_details[0]['index'], input_data)
print(time.time())
interpreter.invoke()
print(time.time())

output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)
print("input data shape:")
print(np.shape(input_data))
print("output data shape:")
print(np.shape(output_data))
#print(output_data)
prev_idx = 0
alphabet = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','0','1','2','3','4','5','6','7','8','9','']
rc = ""
for i, log_prob in enumerate(output_data):
	idx = np.argmax(log_prob)
	#print(log_prob)
	#print(idx)
	#print str(log_prob[idx])

	#print(log_prob)
	if idx not in (36, prev_idx, 0):

		rc += " "+alphabet[idx]

		print(str(i)+"\t"+alphabet[idx]+"\t"+str(log_prob[idx]))
	prev_idx = idx

print(rc)