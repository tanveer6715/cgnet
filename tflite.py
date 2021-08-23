import tensorflow as tf
from tensorflow import lite

#model=tf.load_model("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.18 cityscapes/")
# converter = tf.lite.TFLiteConverter.from_saved_model("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.18 cityscapes/")
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)  

model = tf.keras.models.load_model("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.21 for test/",compile = False)
# converter = tf.lite.TFLiteConverter.from_saved_model("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.21 for test/")
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model) 
