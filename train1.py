from mode import ddrnet_23_slim
from re import VERBOSE
import tensorflow as tf
from datasets.cityscapes import CityscapesDatset
from datasets.concrete_damage_as_cityscapes import Concrete_Damage_Dataset_as_Cityscapes
#from model import CGNet
from pipeline import batch_generator
import numpy as np
from tqdm import tqdm


model = ddrnet_23_slim(num_classes = 5)
# learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.001, 60000,
#                                                             end_learning_rate=0.0001, power=0.9,
#                                                             cycle=False, name=None)

# optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-08)
loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True,
    reduction=tf.keras.losses.Reduction.AUTO
 )

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.0001, momentum=0.9, nesterov=False
)



# loss_object =tf.keras.losses.CategoricalCrossentropy(
#     from_logits = False,
#     reduction=tf.keras.losses.Reduction.NONE
# )

#class_weight = np.load('class_weight_cityscapes.npy', 'r')

# class_weight=[  2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
#                 10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
#                 10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
#                 10.396974, 10.055647   ]

class_weight = [2.5959933,8.7415504,4.5354059,4.8663225,4.690899]
print(class_weight)

#optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou = tf.keras.metrics.MeanIoU(num_classes=5, name='train_miou')

## TODO we need to make argument input from command line

EPOCHS = 3

#DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
DATA_DIR = '/home/soojin/UOS-SSaS Dropbox/05. Data/02. Training&Test/concrete_damage_autocon/as_cityscape'
cityscapes_dataset = Concrete_Damage_Dataset_as_Cityscapes(DATA_DIR)
TRAIN_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TRAIN_LENGTH))

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

#@tf.function
def compute_loss(lables, predictions):
    loss = loss_object(lables, predictions)
    weight_map = tf.ones_like(loss)

    for idx in range(5):
        # for indexing not_equal has to be used...
        class_idx_map = tf.math.not_equal(tf.squeeze(lables), idx)

        # tf.print(class_idx_map)
        # tf.print("index : {}".format(idx))
        # # tf.print(tf.squeeze(lables) == idx)

        # tf.print("Num pixels in class map")
        # tf.print(tf.math.reduce_sum(tf.cast(class_idx_map, dtype = tf.float32)))

        #     tf.print("you are here 2")
        # tf.print("Weight Value")
        # tf.print(weight)
        # tf.print(idx, class_weight[idx])
        weight_map = tf.where(class_idx_map, weight_map, class_weight[idx])
        # tf.print("Max value in weight map")
        # tf.print(tf.math.reduce_max(weight_map))
        # tf.print(weight_map)

    # tf.print("Max value in weight map")
    # tf.print(tf.math.reduce_max(weight_map))

    loss = tf.math.multiply(loss, weight_map)

    loss = tf.reduce_mean(loss)


    return loss




#@tf.function
def train_step(images, labels,):
    with tf.GradientTape() as tape:
        predictions = model(images)
        #loss = loss_object(labels, predictions)
        loss = compute_loss(labels, predictions)
    """
    TODO
    print training progress
    """
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    argmax_predictions = tf.math.argmax(predictions, 3)

    train_iou.update_state(labels, argmax_predictions)





def train():

    # model_weight_path = '/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.07.28 single_train/epoch_230.h5'

    # model.build((4, 680, 680, 3))
    # model.load_weights(model_weight_path)

    for epoch in tqdm(range(1,EPOCHS)):
        cityscapes_generator = batch_generator(cityscapes_dataset, 4)


        "TODO: add progress bar to training loop"
        for images, labels in cityscapes_generator:

            train_step(images, labels)

            template = 'Epoch: {}, Loss: {:2f}, Accuracy: {:2f}, MeanIoU: {:2f}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    train_iou.result()*100
                                    ))
        if epoch % 2 == 0 :
            model.save_weights('/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.01 tflite_testing/epoch_{}.h5'.format(epoch))
        if epoch % 2 == 0 :
            model.save('/home/soojin/UOS-SSaS Dropbox/05. Data/03. Checkpoints/#cgnet/2021.09.01 tflite_testing')#,save_traces = False)


if __name__ == "__main__" :
    train()