
from re import VERBOSE
import tensorflow as tf
from cityscapes import CityscapesDatset
from model import CGNet
from pipelines import batch_generator
import matplotlib.pyplot as plt
from tqdm import tqdm 

model = CGNet()

# loss_object =tfa.losses.SigmoidFocalCrossEntropy(gamma=0.5)
# loss_object =tf.keras.losses.SparseCategoricalCrossentropy(
#     reduction=tf.keras.losses.Reduction.NONE
# )
loss_object =tf.keras.losses.SparseCategoricalCrossentropy()


optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


## TODO we need to make argument input from command line 

EPOCHS = 50

DATA_DIR = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

cityscapes_dataset = CityscapesDatset(DATA_DIR)
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



@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(predictions,loss)
    train_accuracy(labels, predictions)
 

def train():

    # model_weight_path = 'checkpoints/epoch_10.h5'

    # model.build((2, 680, 680, 3))
    # model.load_weights(model_weight_path)

    for epoch in tqdm(range(EPOCHS)):
        cityscapes_generator = batch_generator(cityscapes_dataset, 2)

        
        "TODO: add progress bar to training loop"
        for images, labels in tqdm(cityscapes_generator):
            
            train_step(model, images, labels)
        
            template = 'Epoch: {}, Loss: {}, Accuracy: {}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100
                                    ))
        if epoch % 1 == 0 :
            model.save_weights('checkpoints/epoch_{}.h5'.format(epoch))
        
        # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        # fig.suptitle('Training Metrics')
        # axes[0].set_ylabel("Loss", fontsize=14)
        # axes[0].plot(train_loss.result())

        # axes[1].set_ylabel("Accuracy", fontsize=14)
        # axes[1].set_xlabel("Epoch", fontsize=14)
        # axes[1].plot(train_accuracy.result())
        # plt.show()


if __name__ == "__main__" : 
     train()

