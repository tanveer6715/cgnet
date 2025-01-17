from re import VERBOSE
from sys import path
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from datasets.bridge_bearing import Bridge_Bearing_Dataset_as_Cityscapes
from datasets.cityscapes import CityscapesDatset
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
from models.CGNet import CGNet
from models.Deeplabv3 import deeplabv3_plus
from models.ESNet import ESNet
from models.Enet import Enet
from pipeline import batch_generator
from tqdm import tqdm

def compute_loss(labels, predictions, ignore_class = 255):

    idx_to_ignore = labels!= ignore_class
    labels = tf.where(idx_to_ignore, labels, 0)
    
    per_example_loss = loss_object(labels, predictions)
    weight_map = tf.ones_like(per_example_loss)

    #detach labels with 255 
    for idx in range(3):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(labels), idx)
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])

    weight_map = tf.where(tf.squeeze(idx_to_ignore), weight_map, 0)
    #tf.print(weight_map)
    
    per_example_loss = tf.math.multiply(per_example_loss, weight_map)
    per_example_loss = tf.reduce_mean(tf.boolean_mask(per_example_loss,tf.math.not_equal(per_example_loss, 0)))

    return per_example_loss


def compute_iou_loss(labels, predictions, num_classes = 3):

    argmax_predictions = tf.math.argmax(predictions, 3)

    iou = tf.keras.metrics.MeanIoU(num_classes)

    iou.update_state(labels, argmax_predictions)

    return -tf.math.log(iou.result())


@tf.function
def train_step(inputs):
    images, labels = inputs
    
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    argmax_predictions = tf.math.argmax(predictions, 3)
    
    train_iou.update_state(labels, argmax_predictions)


    return loss


def distributed_train_step(dist_inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    return mirrored_strategy.reduce("MEAN", per_replica_losses,
                        axis=None)
    


def gen():
    for images, labels in cityscapes_generator:
        images = tf.squeeze(images)
        labels = tf.squeeze(labels, axis = 0)
        yield images, labels

loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction=tf.keras.losses.Reduction.NONE)
    
mirrored_strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
    model = CGNet(num_classes=3)
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.001, 80000, 
                                                                end_learning_rate=0.0001, power=0.9,
                                                                cycle=False, name=None)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08)

    # model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2022.01.17 CGNet_highresolution/epoch_1170.h5'

    # model.build((4, 680, 680, 3))
    # model.load_weights(model_weight_path, skip_mismatch=True, by_name=True)
#class_weight = np.load('class_weight_cityscapes_210711.npy', 'r')

# class_weight=[  2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
#                 10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
#                 10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
#                 10.396974, 10.055647   ]

#class_weight = [ 2.5959933, 8.7415504, 4.5354059, 4.8663225, 4.690899 ]

class_weight = [ 10.5959933, 5.7415504, 10.5959933]

#class_weight = [ 1.432, 25.1701, 10.3584, 15.7912]
#class_weight  = [1.007 ,10.558, 3.564, 7.365]
# class_weight = [0.00018367134648961431, 9.509244322546513, 0.12194520067309561, 0.11083528782832379]

print(class_weight)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou = tf.keras.metrics.MeanIoU(num_classes=3, name='train_miou')


## TODO we need to make argument input from command line 

EPOCHS = 1600

#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/01. Under_Process/2021.05_서울시설공단 교량받침 데이터/high resolution/psd/JM'
#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/01. Under_Process/2021.05_서울시설공단 교량받침 데이터/외관손상 4종/부식/Labeling/02. annot/교차이후'
cityscapes_dataset =Bridge_Bearing_Dataset_as_Cityscapes(DATA_DIR)
TRAIN_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TRAIN_LENGTH))


# check the dataset type required!!! 
cityscapes_generator = batch_generator(cityscapes_dataset, 1, repeat= EPOCHS)


GLOBAL_BATCH_SIZE = 16
num_steps = TRAIN_LENGTH//GLOBAL_BATCH_SIZE

tf_cityscapes_generator = tf.data.Dataset.from_generator(gen, (tf.float32,  tf.uint8), 
                                                    ((680,680, 3), (680, 680, 1)))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

tf_cityscapes_generator = tf_cityscapes_generator.with_options(options)

tf_cityscapes_generator = tf_cityscapes_generator.batch(GLOBAL_BATCH_SIZE)

dist_cityscapes = mirrored_strategy.experimental_distribute_dataset(tf_cityscapes_generator)

iterator = iter(dist_cityscapes)
#path =  '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2022.01.17 CGNet_highresolution/'
for epoch in tqdm(range(1, EPOCHS + 1)):
   
    for step in range(1, num_steps+1):
        distributed_train_step(next(iterator))
        
        if step % 5 == 1 : 
            template = 'Epoch: {}/{}, steps:{}/{}, Loss: {:2f}, Accuracy: {:2f}, MeanIoU: {:2f}'

            print (template.format(epoch,
                                    EPOCHS,
                                    step,
                                    num_steps,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    train_iou.result()*100
                                    ))
    if epoch % 10 == 0 :
        model.save_weights('/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2022.02.05 bb_corrosion_tf/epoch_{}.h5'.format(epoch))
    # if epoch % 1 == 0 :
    #     tf.keras.models.save_model(model,path)
    