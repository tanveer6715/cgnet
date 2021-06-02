import tensorflow as tf 

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from pipelines import batch_generator

from cityscapes import CityscapesDatset

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv3 = Conv2D(256, 3, activation='relu', padding='same')


    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
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