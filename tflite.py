

import tensorflow as tf
print("Tensorflow version: {}".format(tf.__version__))
from tensorflow import lite

# model = tf.keras.models.load_model(
#     "/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.01 tflite_testing/")
# model.summary()
converter = tf.lite.TFLiteConverter.from_saved_model(
    "/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.01 tflite_testing/")

converter.experimental_new_converter = True

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                               tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

open("check.tflite", "wb").write(tflite_model) 


interpreter = tf.lite.Interpreter(model_path = "/home/soojin/cgnet/check.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input shape:", input_details[0]['shape'])
print("input shape:", input_details[0]['dtype'])
print("output shape", output_details[0]['shape'])
print("output shape", output_details[0]['dtype'])
