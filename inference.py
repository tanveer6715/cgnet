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
import time

tf.random.set_seed(42)
model = CGNet(num_classes=5)


## TODO we need to make argument input from command line 

#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
#tf.executing_eagerly()

# choose 'val' for validation or 'test' for test 
cityscapes_dataset = Concrete_Damage_Dataset_as_Cityscapes(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.31 concrete/epoch_286.h5'

model.build((1, 680, 680, 3))
model.load_weights(model_weight_path)

img_idx = 294

image, label = load_image_test(cityscapes_dataset[img_idx])

img = tf.stack([image])

start = time.time()

predictions = model(img)

end = time.time()
print("--- %s miliseconds ---" % ((end - start)*1000)) 
argmax_predictions = tf.math.argmax(predictions, 3)

image, label = load_image_test(cityscapes_dataset[img_idx], is_normalize =False)

print(argmax_predictions.shape)

np_predictions = argmax_predictions.numpy()
np_predictions = np.squeeze(np_predictions)

prediction_map = np.zeros_like(image)
ground_truth = np.zeros_like(image)


for idx in range(5):
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