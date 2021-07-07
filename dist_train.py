from re import VERBOSE
import tensorflow as tf 

from cityscapes import CityscapesDatset
from model import CGNet
from pipelines import batch_generator
from tqdm import tqdm 


mirrored_strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
    model = CGNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

## TODO we need to make argument input from command line 

EPOCHS = 280

DATA_DIR = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'

cityscapes_dataset = CityscapesDatset(DATA_DIR)
TRAIN_LENGTH = len(cityscapes_dataset)
print("Length of the dataset : {}".format(TRAIN_LENGTH))

# check the dataset type required!!! 

cityscapes_generator = batch_generator(cityscapes_dataset, 8)

tf_cityscapes_generator = tf.data.Dataset.from_generator(cityscapes_generator)

dist_cityscapes = mirrored_strategy.experimental_distribute_dataset(tf_cityscapes_generator)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()



def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


@tf.function
def train_step(model, images, labels):
    
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def distributed_train_step(dist_inputs):
  per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
    

iterator = iter(dist_cityscapes)
for _ in range(10):
  print(distributed_train_step(next(iterator)))


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





# def train():

#     #model_weight_path = 'checkpoints/epoch_10.h5'

#     model.build((1, 680, 680, 3))
#     #model.load_weights(model_weight_path)

#     for epoch in tqdm(range(11, EPOCHS)):
#         cityscapes_generator = batch_generator(cityscapes_dataset, 2)

        
#         "TODO: add progress bar to training loop"
#         for images, labels in tqdm(cityscapes_generator):
            
#             train_step(model, images, labels)
        
#             template = 'Epoch: {}, Loss: {}, Accuracy: {}'
#             print (template.format(epoch+1,
#                                     train_loss.result(),
#                                     train_accuracy.result()*100
#                                     ))
        


# if __name__ == "__main__" : 
#      train()
