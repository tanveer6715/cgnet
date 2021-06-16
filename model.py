import tensorflow as tf 

from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D
from tensorflow.keras import Model

from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Permute
from pipelines import batch_generator

from cityscapes import CityscapesDatset

"""
Code Reference : 
    https://github.com/wutianyiRosun/CGNet
"""


class ConvBNPReLU(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same'):
        """
        args:
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super(ConvBNPReLU, self).__init__()

        self.conv = Conv2D(nOut, kSize, strides=(strides, strides), padding=padding)
        self.bn = BatchNormalization(epsilon=1e-03)
        self.PReLU = PReLU()

    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.PReLU(output)
        return output


class BNPReLU(Model):
    def __init__(self, epsilon=1e-03):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = BatchNormalization(epsilon=epsilon)
        self.PReLU = PReLU()

    def call(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.PReLU(output)
        return output


class FGlo(Model):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, nOut, reduction=16):
        
        super(FGlo, self).__init__()
        self.glob_avg_pool = GlobalAveragePooling2D()#fglo
        self.FC1 = Dense(nOut // reduction, activation= 'relu')
        self.FC2 = Dense(nOut, activation= 'sigmoid') # sigmoid
    

    def call(self, input):

        output = self.glob_avg_pool(input)
        output = self.FC1(output)
        output = self.FC2(output)
        output = tf.expand_dims(output, 1)
        output = tf.expand_dims(output, 2)

        return input * output


class CGblock_down(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=True, epsilon=1e-03):
        
        super(CGblock, self).__init__()
        
        n= int(nOut/2)
        self.ConvBNPReLU = ConvBNPReLU(n, kSize, strides=1, padding='same')

        self.F_loc = SeparableConv2D(n, kSize, strides=(strides, strides), padding=padding ,activation=None) #floc
        self.F_sur = SeparableConv2D(n, kSize, strides=(strides, strides), padding=padding, dilation_rate=dilation_rate) #fsur
        self.Concatenate = Concatenate() #fjoi
        self.BNPReLU = BNPReLU(2*nOut)
        self.reduce = Conv2D(2*nOut,nOut,1,1)
        self.FGLo = FGlo(nOut, reduction=reduction)#fglo

    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.ConvBNPReLU(input)

        loc = self.F_loc(output)
        sur = self.F_sur(output)
        output = Concatenate()([loc,sur])
        output = self.BNPReLU(output)
        output = self.reduce(output)
        output = self.FGLo(output)

        return output

class CGblock(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=True, epsilon=1e-03):
        
        super(CGblock, self).__init__()
        
        n= int(nOut/2)
        self.ConvBNPReLU = ConvBNPReLU(n, kSize, strides=1, padding='same')

        self.F_loc = SeparableConv2D(n, kSize, strides=(strides, strides), padding=padding ,activation=None) #floc
        self.F_sur = SeparableConv2D(n, kSize, strides=(strides, strides), padding=padding, dilation_rate=dilation_rate) #fsur
        self.Concatenate = Concatenate() #fjoi
        self.BNPReLU = BNPReLU()
        self.FGLo = FGlo(nOut, reduction=reduction)#fglo

    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.ConvBNPReLU(input)

        loc = self.F_loc(output)
        sur = self.F_sur(output)
        output = Concatenate()([loc,sur])
        output = self.BNPReLU(output)
        output = self.FGLo(output)

        return output

class CGblock1(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=4, reduction=16, add=True):
        
        super(CGblock, self).__init__()

        self.ConvBNPReLU = ConvBNPReLU(nOut, kSize, strides=1, padding='valid')

        self.F_loc = SeparableConv2D(nOut, kSize, strides=(strides, strides), padding=padding ,activation=None) #floc
        self.F_sur = SeparableConv2D(nOut, kSize, strides=(strides, strides), padding=padding, dilation_rate=dilation_rate) #fsur
        self.Concatenate = Concatenate() #fjoi
        self.FGLo = FGlo(nOut, activation= 'relu',reduction=reduction)#fglo

    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.ConvBNPReLU(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        output = Concatenate()([loc,sur])
        output = self.FGLo(output)

        return output

class InputInjection(Model):
    def __init__(self, downsamplingRatio):
        super(InputInjection,self).__init__()
        self.pool = Model()
        for i in range(0, downsamplingRatio):
            self.pool.append(AveragePooling2D(3, stride=2, padding=1))
    def call(self, input):
        for pool in self.pool:
            input = pool(input)
        return input

class CGNet(Model):
    def __init__(self,classes=19, M= 3, N= 21, dropout_flag = False):
        """

        TODO : 
            1. add cg block 


        """
        super(CGNet, self).__init__()
        #Stage 1
        self.ConvBNPReLU1 = ConvBNPReLU(32, 3, strides=2, padding='valid')
        self.ConvBNPReLU2 = ConvBNPReLU(32, 3)
        self.ConvBNPReLU3 = ConvBNPReLU(32, 3)    
        # self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        # self.sample2 = InputInjection(2)
        self.BN1 = BNPReLU()
         # TODO : Add Input injection 
        # TODO : CG Block down
        # First CG block (M=3) dilation=2
        #Stage 2
        self.CGBlock_down=CGblock(32, 3, dilation_rate=2)
        self.CGBlock = CGblock(32, 3)
        #Stage 3
        self.CGBlock_down=CGblock(32, 3, dilation_rate=4)
        self.CGBlock1 = CGblock(64, 3)
        self.upsample = UpSampling2D()

        # self.ConvBNPReLU4 = ConvBNPReLU(64, 3, strides=2, padding='valid')

        # self.F_loc = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None) #floc
        # self.F_sur = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(2, 2), depth_multiplier=1, activation=None) #fsur
        # self.Concatenate = Concatenate() #fjoi
        # self.F_glo = GlobalAveragePooling2D()#fglo
        # self.FC1 = Dense(20, activation= 'relu')
        # self.FC2 = Dense(20, activation= 'relu')
        
        #self.upsample = UpSampling2D()


        # Second CG block (N=15) dilation=4
        # self.ConvBNPReLU5 = ConvBNPReLU(64, 1)

        # self.F_loc1 = SeparableConv2D(128, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None) #floc
        # self.F_sur1= SeparableConv2D(128, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(4, 4), depth_multiplier=1, activation=None) #fsur
        # self.Concatenate = Concatenate() #fjoi
        # self.F_glo1 = GlobalAveragePooling2D()#fglo
        # self.FC3 = Dense(20, activation= 'relu')
        # self.FC4 = Dense(20, activation= 'sigmoid')
        # self.ConvBNPReLU6 = ConvBNPReLU(19, 1)
        #self.upsample = UpSampling2D()
        #Defining CG block M
        # self.seperable1 = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None) #floc
        # self.seperable2 = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(2, 2), depth_multiplier=1, activation=None) #fsur
        # self.concatted1 = Concatenate() #fjoi
        # self.bn4 = BatchNormalization()
        # self.PReLU4 = PReLU()
        
        # self.gap = GlobalAveragePooling2D() #fglo
        # self.fc1 = Dense(19, activation = 'relu')
        # self.fc2 = Dense(19, activation = 'relu')
        # self.conv4 = Conv2D(19, 1,  padding='same')
        #self.upsample = UpSampling2D()
        
         #Defining CG block N
        #self.seperable1 = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None) #floc
        #self.seperable2 = SeparableConv2D(64, 3, strides=(1, 1), padding='same',data_format=None, dilation_rate=(4, 4), depth_multiplier=1, activation=None) #fsur
    
        #self.concatted1 = Concatenate()([self.seperable1,self.seperable2]) #fjoi
        #self.bn4 = BatchNormalization()
        #self.PReLU4 = PReLU()
        
        #self.globavg = GlobalAveragePooling2D() #fglo
        #self.dense1 = Dense(20)
        


    def call(self, input):

        output = self.ConvBNPReLU1(input)
        output = self.ConvBNPReLU2(output)
        output = self.ConvBNPReLU3(output)
        # inp1 =   self.sample1(input)
        # inp2 =   self.sample2(input)
        output = self.BN1(output)
        output= self.CGBlock_down(output)
        output = self.CGBlock(output)
        output= self.CGBlock_down(output)
        output = self.CGBlock1(output)
        output = self.upsample(output)
        # output = self.ConvBNPReLU4(output)
        # print(output.shape)

        # output1 = self.F_loc(output)
        # print(output1.shape)
        # output2 = self.F_sur(output)
        # print(output2.shape)
    
        # output3 = Concatenate()([output1, output2])
        # output4 = self.F_glo(output3)
        # output4 = self.FC1(output3)
        # output5 = self.FC2(output3)
        
        # output5 = self.ConvBNPReLU5(output5)
        # print(output.shape)

        # output6 = self.F_loc1(output5)
        # print(output1.shape)
        # output7 = self.F_sur1(output5)
        # print(output2.shape)
    
        # output8 = Concatenate()([output6, output7])
        # output8 = self.F_glo1(output8)
        # output9 = self.FC3(output8)
        # output10 = self.FC4(output8)
        # output11=self.ConvBNPReLU6(output10)
        
        # #output10 = self.upsample(output10)

        
        # print(output11.shape)
        #output4 = self.upsample(output4)


        # x1 = self.seperable1(x)
        # x2 = self.seperable2(x)
        # x = self.concatted1()[x]
        # x = self.bn4(x)
        # x = self.PReLU4(x)
        # x = self.globavg(x)
        # x = self.dense1(x)
        # x = self.conv4
       
        
        return output





def test(): 

    "TODO: seperate trainining script"
    model = CGNet()

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