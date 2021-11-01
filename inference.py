from models.Deeplabv3 import deeplabv3_plus
from models.ESPNet import ESPNet
from models.ddrnet_23_slim import ddrnet_23_slim
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from models.ESNet import ESNet
from models.CGNet import CGNet
import tensorflow as tf 
import numpy as np
from datasets.cityscapes import CityscapesDatset
from models.Enet import Enet
from pipeline import batch_generator
from tqdm import tqdm 

from pipeline import load_image_test
from numpy import matlib
import cv2
from time import time
from tensorflow import keras
tf.random.set_seed(42)
tf.executing_eagerly()
model = ESNet()
## TODO we need to make argument input from command line 

#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
#tf.executing_eagerly()

# choose 'val' for validation or 'test' for test 
cityscapes_dataset = Concrete_Damage_Dataset_as_Cityscapes(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.13 es_net/epoch_325.h5'
#model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.10.09 ESNet_cityscapes/'

#model =keras.models.load_model(model_weight_path)
model.build((1, 680, 680, 3))
model.load_weights(model_weight_path)
pred_iou = tf.keras.metrics.MeanIoU(num_classes=5, name='pred_miou')

img_idx = 256

image, label = load_image_test(cityscapes_dataset[img_idx])

img = tf.stack([image])

start = time()

predictions = model(img, training=False)
end = time()
print("--- %s miliseconds ---" % ((end - start)*1000)) 
#print(predictions)
argmax_predictions = tf.math.argmax(predictions, 3)

image, label = load_image_test(cityscapes_dataset[img_idx], is_normalize =False)

print(argmax_predictions.shape)

np_predictions = argmax_predictions.numpy()
np_predictions = np.squeeze(np_predictions)

prediction_map = np.zeros_like(image)
ground_truth = np.zeros_like(image)

num_classes = 5

for idx in range(5):
    prediction_map[np_predictions == idx] = cityscapes_dataset.palette[idx]
    label_map = label == idx
    label_map = np.squeeze(label_map)
    ground_truth[label_map, :] = cityscapes_dataset.palette[idx]

pred_iou.update_state(label, argmax_predictions)
#print(pred_iou.result())
values = np.array(pred_iou.get_weights()).reshape(num_classes, num_classes)
#print(values)
class0_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0] + values[2,0] + values[3,0] + values[4,0])
class1_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1] + values[2,1] + values[3,1] + values[4,1])
class2_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2] + values[1,2] + values[3,2] + values[4,2])
class3_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3] + values[1,3] + values[2,3] + values[4,3])
class4_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[0,4] + values[1,4] + values[2,4] + values[3,4])
sum_classes = (class1_IoU + class2_IoU + class3_IoU + class4_IoU)/4
print(class0_IoU)
print(class1_IoU)
print(class2_IoU)
print(class3_IoU)
print(class4_IoU)
print(sum_classes)
cv2.imwrite('save_img.png', np_predictions)
cv2.imwrite('prediction_img.png', prediction_map)

image_save = image.numpy()
print(image_save.dtype)
cv2.imwrite('org_img.png', image_save)
cv2.imwrite('ground_truth.png', ground_truth)