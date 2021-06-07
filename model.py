import tensorflow as tf 

from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D
from tensorflow.keras import Model

from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import PReLU

from pipelines import batch_generator

from cityscapes import CityscapesDatset

class MyModel(Model):
    def __init__(self):
        """

        TODO : 
            1. add cg block 


        """
        super(MyModel, self).__init__()

        self.conv1 = Conv2D(32, 3,  padding='same', strides=(2, 2))
        self.bn1 = BatchNormalization()
        self.PReLU1 = PReLU()

        self.conv2 = Conv2D(32, 3,  padding='same')
        self.bn2 = BatchNormalization()
        self.PReLU2 = PReLU()

        self.conv3 = Conv2D(20, 3,  padding='same')
        self.bn3 = BatchNormalization()
        self.PReLU3 = PReLU()

        self.upsample = UpSampling2D()
        


    def call(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.PReLU1(x)
    
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.PReLU2(x)


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.PReLU3(x)

        x = self.upsample(x)
        
        return x




def test(): 

    "TODO: seperate trainining script"
    model = MyModel()

    "TODO : Replace kerass functions with low level tf"

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels,):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    data_dir = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    
    "TODO: add test datset mode to cityscapes"
    cityscapes_dataset = CityscapesDatset(data_dir)
    TRAIN_LENGTH = len(cityscapes_dataset)
    print("Length of the dataset : {}".format(TRAIN_LENGTH))


    for epoch in range(EPOCHS):
        cityscapes_generator = batch_generator(cityscapes_dataset, 2)
        
        "TODO: add progress bar to training loop"
        for images, labels in cityscapes_generator:
            
            train_step(images, labels)
        
        template = 'Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                ))
        


if __name__ == "__main__" : 
    test()