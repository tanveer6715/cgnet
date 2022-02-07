from builtins import input
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv

"""
Code Reference :
    https://github.com/wutianyiRosun/CGNet
"""
__all__ = ["CGNet"]
kernel_initializer = he_normal()


def ConvBNPReLU(input_tensor,output_channels, kernal_size,strides , padding):
    x = Conv2D(output_channels, kernal_size,
        strides, use_bias=False, padding='same'
    )(input_tensor)

    x = SyncBatchNormalization(epsilon=1e-03)(x)
    x = PReLU(shared_axes=[1,2])(x)

    return x

def BNPReLU(output_channels):
    x = SyncBatchNormalization(epsilon=1e-03)(output_channels)
    x = PReLU(shared_axes=[1,2])(x)

    return x


def FGlo(input_tensor,output_channels, reduction=16):

    x = AveragePooling2D(2,data_format='channels_last')(input_tensor)
    # x = GlobalAveragePooling2D(data_format='channels_last')(input_tensor)
    x = Dense(output_channels // reduction, activation= 'relu')(x)
    x = Dense(output_channels, activation='sigmoid')(x)
    # x2 = tf.expand_dims(x, 1)
    # output = tf.expand_dims(x2, 2)
    #output1 = input_tensor * output
    #print("fglo ends with {}".format(output.shape))
    return  x


def CGblock_down(input_tensor,output_channels, kernal_size,strides , padding, dilation_rate,reduction):
    x = ConvBNPReLU(input_tensor,output_channels,kernal_size,strides , padding)
    x1 = Conv2D(output_channels,3, strides=1, use_bias=False, padding='same', groups = output_channels)(x)
    x2 = Conv2D(output_channels,3, strides=1, use_bias=False, padding='same', groups = output_channels,dilation_rate=(dilation_rate))(x)
    # x1 = Conv2D(output_channels,3, strides=1, use_bias=False, padding='same')(x)
    # x2 = Conv2D(output_channels,3, strides=1, use_bias=False, padding='same',dilation_rate=(dilation_rate))(x)
    x = concatenate([x1, x2])
    x = BNPReLU(2*x)
    x = Conv2D(output_channels,1, strides=1, use_bias= False)(x)
    x = FGlo(x,output_channels, reduction)
    return x

def CGblock(input_tensor,output_channels, kernal_size,strides , padding, dilation_rate,reduction):
    n = int(output_channels/2)
    x = ConvBNPReLU(input_tensor,n,kernal_size,strides,padding)
    x1 = Conv2D(n,3, strides=1, use_bias=False, padding='same', groups = n)(x)
    x2 = Conv2D(n,3, strides=1, use_bias=False, padding='same', groups =n,dilation_rate=(dilation_rate))(x)

    # x1 = Conv2D(n,3, strides=1, use_bias=False, padding='same')(x)
    # x2 = Conv2D(n,3, strides=1, use_bias=False, padding='same',dilation_rate=(dilation_rate))(x)
    x3 = concatenate([x1, x2])
    x3 = BNPReLU(x3)
    x4 = FGlo(x3,output_channels, reduction)
    # x = input_tensor + x4
    return x4

def InputInjection(Input_tensor):
    pool = []
    for i in range(0, 16):
        x = pool.append(AveragePooling2D(3, strides=2, padding='same'))
    for pool in pool:
        x = pool (Input_tensor)
    return x

def cgnetc(input_shape=(680,680,3), num_classes = 19, M = 3, N=21, dropout_flag = False):

    inp = tf.keras.layers.Input(input_shape)
    
    x = ConvBNPReLU(inp,32,3,2,padding ='valid')
    x = ConvBNPReLU(x,32,3,1,padding = 'same' )
    x = ConvBNPReLU(x,32,3,1,padding = 'same')
 
    inp1 = InputInjection(inp)
    x1 = BNPReLU(concatenate([x,inp1]))
    print("Stage 1 ends with {}".format(x.shape))
    x2_1 = CGblock_down(x1,64,3,1,padding  = 'same', dilation_rate=2,reduction=8)
    print("Stage 2 starts with {}".format(x2_1.shape))
    #x2_2 = CGblock(x1,64,3,1,padding  = 'same', dilation_rate=2,reduction=8)
    
    for i in tf.range (0, M-1):
        x2_2 = []
        x2_2.append(CGblock(x1,64,3,1,padding  = 'same', dilation_rate=2,reduction=8))
       
    for i, layer in tf.unstack(x2_2): 
            if i == 0 : 
                Output = layer(x2_1)
            else : 
                Output = layer(x2_2)

    inp2 = InputInjection(inp1)
    print("inp2 ends with {}".format(inp2.shape))
    x2 = BNPReLU(concatenate([x2_1,x2_2,inp2]))
    print("Stage 2 ends with {}".format(x2.shape))
    
    x3 = CGblock_down(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_1 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_2 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_3 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_4 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_5 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_6 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_7 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_8 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_9 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_10 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_11 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_12 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_13 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_14 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_16 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_17 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    # x3_19 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)
    x3_20 = CGblock(x2,128,3,1,padding  = 'same', dilation_rate=4,reduction=16)

    x4 = BNPReLU(concatenate([x3, x3_20]))
    print("Stage 3 ends with {}".format(x4.shape))

    x = Conv2D(num_classes, 1, use_bias = False)(x4)
    Output = UpSampling2D(size=(8,8), interpolation = 'bilinear')(x)
    model = Model(inp,Output)
    return model

model = cgnetc()
model.summary()

# path = '/home/soojin/cgnet'
# tf.keras.models.save_model(model,path)
