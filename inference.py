

import tensorflow as tf 
import numpy as np

from cityscapes import CityscapesDatset
from model import CGNet
from pipelines import batch_generator
from tqdm import tqdm 

from pipelines import load_image_test

import cv2


model = CGNet(classes=20)


## TODO we need to make argument input from command line 

DATA_DIR = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

tf.executing_eagerly()

# choose 'val' for validation or 'test' for test 
cityscapes_dataset = CityscapesDatset(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
model_weight_path = 'first_chk_point.h5'

model.build((1, 680, 680, 3))
model.load_weights(model_weight_path)

image, label = load_image_test(cityscapes_dataset[1])

img = tf.stack([image])
predictions = model(img)

argmax_predictions = tf.math.argmax(predictions, 3)

image, label = load_image_test(cityscapes_dataset[1], is_normalize =False)

print(argmax_predictions.shape)

np_predictions = argmax_predictions.numpy()
np_predictions = np.squeeze(np_predictions)

prediction_map = np.zeros_like(image)

for idx in range(19):
    prediction_map[np_predictions == idx] = cityscapes_dataset.palette[idx]


cv2.imwrite('save_img.png', np_predictions)
cv2.imwrite('prediction_img.png', prediction_map)

image_save = image.numpy()
print(image_save.dtype)
cv2.imwrite('org_img.png', image_save)