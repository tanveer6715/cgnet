import tensorflow as tf 
import numpy as np
from tensorflow import keras

from model import CGNet
from pipeline import batch_generator
from tqdm import tqdm 
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from datasets.cityscapes import CityscapesDatset
import cv2

model = CGNet(num_classes=5)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_iou = tf.keras.metrics.MeanIoU(num_classes=5, name='test_miou')
class_weight=[  2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
                10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
                10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
                10.396974, 10.055647   ]
## TODO we need to make argument input from command line 

DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'

# choose 'val' for validation or 'test' for test 
cityscapes_dataset =Concrete_Damage_Dataset_as_Cityscapes(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))
model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.21 for test/'


#model.build((1, 680, 680, 3))
#model.load_weights(model_weight_path)
keras.models.load_model(model_weight_path)

@tf.function
def test_step(model, images, labels):
    ignore_class = 255
    
    idx_to_ignore = labels!= ignore_class
    tf.print(tf.math.reduce_max(labels))
    labels = tf.where(idx_to_ignore, labels, 0)
    
    

    predictions = model(images)
    
    per_example_loss = loss_object(labels, predictions)

    weight_map = tf.ones_like(per_example_loss)

    weight_map = tf.where(tf.squeeze(idx_to_ignore), weight_map, 0)
    
    t_loss = loss_object(labels, predictions)


    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    argmax_predictions = tf.math.argmax(predictions, 3)

    test_iou.update_state(labels, argmax_predictions)#, sample_weight = weight_map)


    
    """
    TODO 
    print validation result of each epoch
    """

def test():
    cityscapes_generator = batch_generator(cityscapes_dataset, 1)#, ignore_class=None)


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
