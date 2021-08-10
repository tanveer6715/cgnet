from re import VERBOSE
import tensorflow as tf 
import numpy as np 

from cityscapes import CityscapesDatset
from model import CGNet
from pipeline import batch_generator
from focal_loss import SparseCategoricalFocalLoss
from tqdm import tqdm 


def compute_loss(labels, predictions, ignore_class = 255):

    idx_to_ignore = labels!= ignore_class
    labels = tf.where(idx_to_ignore, labels, 0)
    
    per_example_loss = loss_object(labels, predictions)
    weight_map = tf.ones_like(per_example_loss)

    #detach labels with 255 
    for idx in range(19):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(labels), idx)
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])

    weight_map = tf.where(tf.squeeze(idx_to_ignore), weight_map, 0)
    # tf.print(weight_map)
    
    per_example_loss = tf.math.multiply(per_example_loss, weight_map)
    per_example_loss = tf.reduce_mean(tf.boolean_mask(per_example_loss,tf.math.not_equal(per_example_loss, 0)))

    return per_example_loss


def compute_iou_loss(labels, predictions, num_classes = 19):

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


mirrored_strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
    model = CGNet(classes = 19)
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.001, 60000, 
                                                                end_learning_rate=0.0001, power=0.9,
                                                                cycle=False, name=None)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08)
    
    # optimizer = tfa.optimizers.AdamW(weight_decay = 0.0005, learning_rate=learning_rate, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08)

    loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction=tf.keras.losses.Reduction.NONE)
    

    # model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.07.24 hp/epoch_276.h5'
    # model.build((2, 680, 680, 3))
    # model.load_weights(model_weight_path)
#class_weight = np.load('class_weight_cityscapes_210711.npy', 'r')
#print(class_weight)
class_weight=[  2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
                10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
                10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
                10.396974, 10.055647   ]

# class_weight=[2.8149201869965,6.9850029945374, 3.7890393733978, 9.9428062438965, 9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606, 4.6323022842407,	
#             9.5608062744141,7.8698215484619,9.5168733596802,10.373730659485,6.6616044044495,10.260489463806,10.287888526917,10.289801597595,10.405355453491,10.138095855713]	

print(class_weight)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou = tf.keras.metrics.MeanIoU(num_classes=19, name='train_miou')


## TODO we need to make argument input from command line 

EPOCHS = 326

DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

cityscapes_dataset = CityscapesDatset(DATA_DIR)
TRAIN_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TRAIN_LENGTH))

# check the dataset type required!!! 
cityscapes_generator = batch_generator(cityscapes_dataset, 1, repeat= EPOCHS)


GLOBAL_BATCH_SIZE = 16

num_steps = TRAIN_LENGTH//GLOBAL_BATCH_SIZE

tf_cityscapes_generator = tf.data.Dataset.from_generator(gen, (tf.float32,  tf.uint8), 
                                                    ((680, 680, 3), (680, 680, 1)))
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

tf_cityscapes_generator = tf_cityscapes_generator.with_options(options)

tf_cityscapes_generator = tf_cityscapes_generator.batch(GLOBAL_BATCH_SIZE)

dist_cityscapes = mirrored_strategy.experimental_distribute_dataset(tf_cityscapes_generator)

iterator = iter(dist_cityscapes)


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
    if epoch % 5 == 1 :
        model.save_weights('/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.08.10 test/epoch_{}.h5'.format(epoch))


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

