import numpy
from models.ESNet import ESNet
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from models.CGNet import CGNet
from models.ESPNet import ESPNet
from models.ddrnet_23_slim import ddrnet_23_slim
from models.Enet import Enet
from pipeline import batch_generator
from tqdm import tqdm 
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from datasets.cityscapes import CityscapesDatset
import os
import time
from timeit import default_timer as timer
tf.random.set_seed(42)
model = CGNet(num_classes=5)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_iou = tf.keras.metrics.MeanIoU(num_classes=5, name='test_miou')

## TODO we need to make argument input from command line 

#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
# choose 'val' for validation or 'test' for test 
cityscapes_dataset =Concrete_Damage_Dataset_as_Cityscapes(DATA_DIR, data_type = 'val')
TEST_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TEST_LENGTH))

model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.31 concrete/epoch_286.h5'


model.build((1, 680, 680, 3))
model.load_weights(model_weight_path)

@tf.function
def test_step(model, images, labels):
    start = time.time()
    predictions = model(images)
    end = time.time()
    print("--- %s miliseconds ---" % ((end - start)*1000)) 
   
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    argmax_predictions = tf.math.argmax(predictions, 3)

    test_iou.update_state(labels, argmax_predictions)#, sample_weight = weight_map)

    
    """
    TODO 
    print validation result of each epoch
    """
num_classes = 5
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
        values = np.array(test_iou.get_weights()).reshape(num_classes, num_classes)
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
if __name__ == "__main__" : 
     test()
