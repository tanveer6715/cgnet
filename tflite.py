

import tensorflow as tf
print("Tensorflow version: {}".format(tf.__version__))
from tensorflow import lite
import numpy as np
import time
import cv2
# model = tf.keras.models.load_model(
#     "/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.24 deeplab_net/")
#model.summary()
converter = tf.lite.TFLiteConverter.from_saved_model(
    "/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.27 ESNet_cityscapes")

converter.experimental_new_converter = True

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                               tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

open("esnet_1.tflite", "wb").write(tflite_model) 


interpreter = tf.lite.Interpreter(model_path = "/home/soojin/cgnet/esnet_1.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
print(input_size)
print("input shape:", input_details[0]['shape'])
print("input shape:", input_details[0]['dtype'])
print("output shape", output_details[0]['shape'])
print("output shape", output_details[0]['dtype'])

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

start = time.time()   
interpreter.invoke()
end = time.time()
print("--- %s miliseconds3 ---" % ((end - start)*1000)) 

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)






