import tensorflow as tf 
import numpy as np

from cityscapes import CityscapesDatset
from model import CGNet
from pipelines import batch_generator
from tqdm import tqdm 

import cv2


model = CGNet(classes=19)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_iou = tf.keras.metrics.MeanIoU(num_classes=19, name='test_miou')

## TODO we need to make argument input from command line 

DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

# choose 'val' for validation or 'test' for test 
cityscapes_dataset = CityscapesDatset(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.07.20 changing sep2d/epoch_276.h5'


model.build((1, 680, 680, 3))
model.load_weights(model_weight_path)

@tf.function
def test_step(model, images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    argmax_predictions = tf.math.argmax(predictions, 3)
    
    test_iou.update_state(labels, argmax_predictions)

    """
    TODO 
    print validation result of each epoch
    """

def test():
    cityscapes_generator = batch_generator(cityscapes_dataset, 1)


    "TODO: add progress bar to training loop"
    for images, labels in tqdm(cityscapes_generator):
        

        test_step(model, images, labels)

        template = 'Loss: {}, Accuracy: {} , MeanIoU: {}' 
        print (template.format(test_loss.result(),
                                test_accuracy.result()*100,
                                test_iou.result()*100
                                ))

if __name__ == "__main__" : 
     test()
