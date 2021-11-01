import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from time import time
from matplotlib import pyplot as plt
import os
from datasets.cityscapes import CityscapesDatset
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from pipeline import load_image_test
import cv2
# #Parasitized image
# #image = load_img('cell_images/Parasitized/C39P4thinF_original_IMG_20150622_111206_cell_99.png', target_size=(150,150))

# #Uninfected image
# image = load_img('/home/soojin/cgnet/org_img.png', target_size=(680,680))

# image = img_to_array(image)
# # reshape data for the model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# print(image.shape)

# from tensorflow.keras.utils import normalize
# image = normalize(image, axis=1)

# ###############################################
# ###PREDICT USING REGULAR KERAS TRAINED MODEL FILE (h5). 
# ##########################################################
# keras_model_size = os.path.getsize("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.10.09 ESNet_cityscapes/")  #/1048576  #Convert to MB
# print("Keras Model size is: ", keras_model_size, "MB")
# #Using regular keral model
# model = tf.keras.models.load_model("/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.10.09 ESNet_cityscapes/")

# time_before=time()
# keras_prediction = model(image)
# time_after=time()
# total_keras_time = time_after - time_before
# print("Total prediction time for keras model is: ", total_keras_time)





##################################################################################
#### PREDICT USING tflite ###
############################################################################

tflite_size = os.path.getsize("/home/soojin/cgnet/esnet_1.tflite")/1048576  #Convert to MB
print("tflite Model without opt. size is: ", tflite_size, "MB")
#Not optimized (file size = 540MB). Taking about 0.5 seconds for inference
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
#tf.executing_eagerly()

# choose 'val' for validation or 'test' for test 
cityscapes_dataset = CityscapesDatset(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
tflite_model_path = "/home/soojin/cgnet/esnet_1.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Load image
#input_data = image
img_idx = 256

image, label = load_image_test(cityscapes_dataset[img_idx])

img = tf.stack([image])
interpreter.set_tensor(input_details[0]['index'], img)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_time = time_after - time_before
print("Total prediction time for tflite without opt model is: ", total_tflite_time)

output_data_tflite = interpreter.get_tensor(output_details[0]['index'])

argmax_predictions = tf.math.argmax(output_data_tflite, 3)

image, label = load_image_test(cityscapes_dataset[img_idx], is_normalize =False)

print(argmax_predictions.shape)

np_predictions = argmax_predictions.numpy()
np_predictions = np.squeeze(np_predictions)

prediction_map = np.zeros_like(image)
ground_truth = np.zeros_like(image)
#pred_iou = tf.keras.metrics.MeanIoU(num_classes=5, name='pred_miou')

for idx in range(19):
    prediction_map[np_predictions == idx] = cityscapes_dataset.palette[idx]
    label_map = label == idx
    label_map = np.squeeze(label_map)
    ground_truth[label_map, :] = cityscapes_dataset.palette[idx]

cv2.imwrite('save_img.png', np_predictions)
cv2.imwrite('prediction_img.png', prediction_map)

image_save = image.numpy()
print(image_save.dtype)
cv2.imwrite('org_img.png', image_save)
cv2.imwrite('ground_truth.png', ground_truth)
# num_classes = 5
# pred_iou.update_state(label, argmax_predictions)
# #print(pred_iou.result())
# values = np.array(pred_iou.get_weights()).reshape(num_classes, num_classes)
# #print(values)
# class0_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0] + values[2,0] + values[3,0] + values[4,0])
# class1_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1] + values[2,1] + values[3,1] + values[4,1])
# class2_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2] + values[1,2] + values[3,2] + values[4,2])
# class3_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3] + values[1,3] + values[2,3] + values[4,3])
# class4_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[0,4] + values[1,4] + values[2,4] + values[3,4])
# sum_classes = (class1_IoU + class2_IoU + class3_IoU + class4_IoU)/4
# print(class0_IoU)
# print(class1_IoU)
# print(class2_IoU)
# print(class3_IoU)
# print(class4_IoU)
# print(sum_classes)
# #################################################################
#### PREDICT USING tflite with optimization###
#################################################################
# tflite_optimized_size = os.path.getsize("models/malaria_model_100epochs_optimized.tflite")/1048576  #Convert to MB
# print("tflite Model with optimization size is: ", tflite_optimized_size, "MB")
# #Optimized using default optimization strategy (file size = 135MB). Taking about 39 seconds for prediction
# tflite_optimized_model_path = "models/malaria_model_100epochs_optimized.tflite"


# # Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=tflite_optimized_model_path)
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Test the model on input data.
# input_shape = input_details[0]['shape']
# print(input_shape)

# Load image
# input_data = image

# interpreter.set_tensor(input_details[0]['index'], input_data)

# time_before=time()
# interpreter.invoke()
# time_after=time()
# total_tflite_opt_time = time_after - time_before
# print("Total prediction time for tflite model with opt is: ", total_tflite_opt_time)

# output_data_tflite_opt = interpreter.get_tensor(output_details[0]['index'])
# print("The tflite with opt prediction for this image is: ", output_data_tflite_opt, " 0=Uninfected, 1=Parasited")

#############################################

#Summary
print("###############################################")
#print("Keras Model size is: ", keras_model_size)
print("tflite Model without opt. size is: ", tflite_size)

print("________________________________________________")
#print("Total prediction time for keras model is: ", total_keras_time)
print("Total prediction time for tflite without opt model is: ", total_tflite_time)
