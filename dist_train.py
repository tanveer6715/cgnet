from re import VERBOSE
import tensorflow as tf 
import numpy as np 

from cityscapes import CityscapesDatset
from model import CGNet
from pipelines import batch_generator
import tensorflow_addons as tfa
from tqdm import tqdm 


mirrored_strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
    model = CGNet(classes = 19)
    optimizer = tf.keras.optimizers.Adam()


    loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction=tf.keras.losses.Reduction.NONE
    )

class_weight = np.load('class_weight_cityscapes.npy', 'r')
print(class_weight)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou = tf.keras.metrics.MeanIoU(num_classes=19, name='train_miou')


## TODO we need to make argument input from command line 

EPOCHS = 280

DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

cityscapes_dataset = CityscapesDatset(DATA_DIR)
TRAIN_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TRAIN_LENGTH))

# check the dataset type required!!! 
cityscapes_generator = batch_generator(cityscapes_dataset, 1)

def gen():
    for images, labels in cityscapes_generator:
        images = tf.squeeze(images)
        labels = tf.squeeze(labels, axis = 0)
        yield images, labels

GLOBAL_BATCH_SIZE = 16


tf_cityscapes_generator = tf.data.Dataset.from_generator(gen, (tf.float32,  tf.uint8), 
                                                        ((680, 680, 3), (680, 680, 1)))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
tf_cityscapes_generator = tf_cityscapes_generator.with_options(options)

tf_cityscapes_generator = tf_cityscapes_generator.batch(GLOBAL_BATCH_SIZE)
tf_cityscapes_generator = tf_cityscapes_generator.repeat(EPOCHS)

dist_cityscapes = mirrored_strategy.experimental_distribute_dataset(tf_cityscapes_generator)


def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    weight_map = tf.ones_like(per_example_loss)

    for idx in range(19):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(labels), idx)
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])
    # tf.print(weight_map)
    per_example_loss = tf.math.multiply(per_example_loss, weight_map)
    per_example_loss = tf.reduce_mean(per_example_loss)

    return per_example_loss


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
    

iterator = iter(dist_cityscapes)

num_steps = TRAIN_LENGTH//GLOBAL_BATCH_SIZE

for epoch in tqdm(range(1, EPOCHS + 1)):
    for step in range(1, num_steps+1):
        distributed_train_step(next(iterator))
        template = 'Epoch: {}/{}, Steps :{}/{}, Loss: {}, Accuracy: {}, MeanIoU: {}'

        print (template.format(epoch,
                                EPOCHS,
                                step,
                                num_steps,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                train_iou.result()*100
                                ))
    if epoch % 5 == 1 :
        model.save_weights('checkpoints/epoch_{}.h5'.format(epoch))


"""
TODO
make options for variables 

1. learning rate 
2. optimizer 
3. batch size 
4. checkpoint file name 
5. load model weights 
6. pre-training option 
.... 

"""

