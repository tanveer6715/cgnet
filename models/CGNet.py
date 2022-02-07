from builtins import input
import tensorflow as tf 
from tensorflow.keras.layers import Layer,Input,Conv2DTranspose
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, DepthwiseConv2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
"""
# Code Reference : 
#     https://github.com/wutianyiRosun/CGNet
# """
__all__ = ["CGNet"]  
kernel_initializer = he_normal()

class ConvBNPReLU(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', kernel_initializer=kernel_initializer):
        """
        args:
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super(ConvBNPReLU, self).__init__()
        self.padding = padding
        self.kSize = kSize
        if self.padding == 'valid' and self.kSize != 1:
            self.pad = ZeroPadding2D(1)

        self.conv = Conv2D(nOut, kSize, strides=strides, 
                            padding=padding, kernel_initializer=kernel_initializer,
                            use_bias=False)
        self.bn = SyncBatchNormalization(epsilon=1e-03)
        self.PReLU = PReLU(shared_axes=[1,2])
    
    #@tf.function
    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """

        if self.padding == 'valid' and self.kSize != 1:
            input = self.pad(input)
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
        self.bn = SyncBatchNormalization(epsilon=epsilon)
        self.PReLU = PReLU(shared_axes=[1,2])

    #@tf.function
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
        
        self.glob_avg_pool = GlobalAveragePooling2D(data_format='channels_last') #fglo
        self.FC1 = Dense(nOut // reduction, activation= 'relu')
        self.FC2 = Dense(nOut, activation= 'sigmoid') # sigmoid
    
    #@tf.function
    def call(self, input):
        output = self.glob_avg_pool(input)
        output = self.FC1(output)
        output = self.FC2(output)
        output = tf.expand_dims(output, 1)
        output = tf.expand_dims(output, 2)
        output1 = input * output
        #print("fglo ends with {}".format(output1.shape))
        return output1


class CGblock_down(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=True, epsilon=1e-03, kernel_initializer=kernel_initializer):
        
        super(CGblock_down, self).__init__()
        
        self.ConvBNPReLU = ConvBNPReLU(nOut, kSize, strides=2, padding='valid',kernel_initializer= kernel_initializer)
        
        self.F_loc = DepthwiseConv2D(kSize, strides=(strides, strides), padding=padding,
                                    activation=None, kernel_initializer=kernel_initializer,
                                    use_bias = False) #floc
        self.F_sur = DepthwiseConv2D(kSize, strides=(strides, strides), padding=padding, 
                                    dilation_rate=dilation_rate, kernel_initializer=kernel_initializer,
                                    use_bias = False) #fsur
        self.BNPReLU = BNPReLU(2*nOut)
        self.reduce = Conv2D(nOut, 1, 1, use_bias = False)
        self.FGLo = FGlo(nOut,reduction=reduction) #fglo

    #@tf.function
    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.ConvBNPReLU(input)

        loc = self.F_loc(output)
        sur = self.F_sur(output)
        output = concatenate([loc,sur])
        output = self.BNPReLU(output)
        output = self.reduce(output)
        output = self.FGLo(output)
        #print("cg_down ends with {}".format(output.shape))
        return output

class CGblock(Model):
    def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=True, epsilon=1e-03, kernel_initializer=kernel_initializer):
        
        super(CGblock, self).__init__()
        
        n= int(nOut/2)
        self.ConvBNPReLU = ConvBNPReLU(n, 1, strides=1, padding='same',kernel_initializer=kernel_initializer) 

        self.F_loc = DepthwiseConv2D(kSize, strides=(strides, strides), padding=padding,
                                    activation=None, kernel_initializer=kernel_initializer,
                                    use_bias = False) #floc

        self.F_sur = DepthwiseConv2D(kSize, strides=(strides, strides), padding=padding, 
                                    dilation_rate=dilation_rate, kernel_initializer=kernel_initializer,
                                    use_bias = False) #fsur
        self.BNPReLU = BNPReLU(epsilon=epsilon)
        self.add = add
        self.FGLo = FGlo(nOut,reduction=reduction)#fglo

    #@tf.function
    def call(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.ConvBNPReLU(input)

        loc = self.F_loc(output)
        sur = self.F_sur(output)
        output = concatenate([loc,sur])
        output = self.BNPReLU(output)
        output = self.FGLo(output)
        if self.add:

            output = input + output
        #print("cg_block ends with {}".format(output.shape))
        return output


class InputInjection(Model):
    """
    args:
    """
    def __init__(self, downsamplingRatio):
        super(InputInjection, self).__init__()
        self.pool = []
        for i in range(0, downsamplingRatio):
            self.pool.append(ZeroPadding2D(1))
            self.pool.append(AveragePooling2D(3, strides=2))
    
    #@tf.function
    def call(self, input):
        for pool in self.pool:
            input = pool(input)
        return input



class CGNet(Model):
    def __init__(self, num_classes=3, M= 3, N= 21, dropout_flag = False):
        """

        """
        super(CGNet, self).__init__()
        #Stage 1
        self.dropout_flag = dropout_flag
        self.stage1_1 = ConvBNPReLU(32, 3, strides=2, padding='valid') 
        self.stage1_2 = ConvBNPReLU(32, 3,1) 
        self.stage1_3 = ConvBNPReLU(32, 3,1)

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)

        self.bn1 = BNPReLU()

        # First CG block (M=3) dilation=2
        #Stage 2
        self.stage2_1 = CGblock_down(64, 3, dilation_rate=2, reduction=8)
        self.stage2 = []

        for i in range(0, M-1): 
            self.stage2.append(CGblock(64,3, dilation_rate=2, reduction=8))
            
        self.bn2 = BNPReLU()

        #Stage 3
        self.stage3_1 = CGblock_down(128, 3, dilation_rate=4, reduction=16)
        self.stage3 = []

        for i in range(0, N-1) : 
            self.stage3.append(CGblock(128,3, dilation_rate=4, reduction=16))

        self.bn3 = BNPReLU()

        #Classifier
        if self.dropout_flag:
            print("have droput layer")
            self.dropout = tf.keras.Sequential(Dropout(0.5, False))
            self.classifier = tf.keras.Sequential(Conv2D(num_classes, 1, use_bias = False))
        else:
            self.classifier = Conv2D(num_classes,1, use_bias = False)

        self.upsample = UpSampling2D(size=(8, 8), interpolation = 'bilinear')
       
    #@tf.function
    def call(self, input):

        # Stage 1 
        #print("Training starts with {}".format(input.shape))
        #input = self.first(input)
        
        output1 = self.stage1_1(input)
        output1 = self.stage1_2(output1)
        output1 = self.stage1_3(output1)
        
        inp1 =   self.sample1(input)
        inp2 =   self.sample2(input)

        output1_cat = self.bn1(concatenate([output1, inp1]))
        
        #print("Stage 1 ends with {}".format(output1_cat.shape))

        # Stage 2
        output2_1 = self.stage2_1(output1_cat) 
        #print("Stage 2 starts with {}".format(output2_1.shape))
        for i, layer in enumerate(self.stage2): 
            if i == 0 : 
                output2 = layer(output2_1)
            else : 
                output2 = layer(output2)

        output2_cat = self.bn2(concatenate([output2, output2_1, inp2]))
        #print("Stage 2 ends with {}".format(output2_cat.shape))
        # Stage 3 
        output3_1 = self.stage3_1(output2_cat)

        for i, layer in enumerate(self.stage3): 
            if i == 0 : 
                output3 = layer(output3_1)
            else : 
                output3 = layer(output3)

        output3_cat = self.bn3(concatenate([output3, output3_1]))
        #print("Stage 3 ends with {}".format(output3_cat.shape))
        # classifier 
        if self.dropout_flag:
            output3_cat = self.dropout(output3_cat)

        classifier = self.classifier(output3_cat)

        # upsample segmenation map ---> the input image size
        #out = tf.image.resize(classifier, (input.shape[1], input.shape[2]))
        out = self.upsample(classifier)
        return out

# def get_model():
#   return CGNet()

# model = get_model()
# model = CGNet()
# model.summary
    def model(self):
        input = tf.keras.layers.Input(shape=(680,680,3))
        return Model(inputs=[input], outputs=self.call(input))
# model = CGNet()

# model_functional = model.model()

# model_functional.summary()


# from builtins import input
# import tensorflow as tf 
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D
# from tensorflow.keras import Model,Sequential
# from tensorflow.keras.initializers import he_normal
# from tensorflow.keras.layers.experimental import SyncBatchNormalization
# from tensorflow.keras.layers import PReLU
# from tensorflow.keras.layers import Concatenate
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import ZeroPadding2D
# """
# Code Reference : 
#     https://github.com/wutianyiRosun/CGNet
# """

# """
# TODO
# 1. Add a custom convolutional layer for specific padding size 
# 2. Add a custom channelwiseconv for specific padding size 
# 4. Add a custom ChannelWiseDilatedConv for specific padding size 
# """
# __all__ = ["CGNet"]  
# kernel_initializer = he_normal()

# class ConvBNPReLU(layers.Layer):
#     def __init__(self, nOut, kSize, strides=1, padding='same', kernel_initializer=kernel_initializer):
#         """
#         args:
#             nOut: number of output channels
#             kSize: kernel size
#             stride: stride rate for down-sampling. Default is 1
#         """
#         super(ConvBNPReLU, self).__init__()
#         self.padding = padding
#         self.kSize = kSize
#         if self.padding == 'valid' and self.kSize != 1:
#             self.pad = ZeroPadding2D(1)

#         self.conv = Conv2D(nOut, kSize, strides=strides, 
#                             padding=padding, kernel_initializer=kernel_initializer,
#                             use_bias=False)
#         self.bn = SyncBatchNormalization(epsilon=1e-03)
#         self.PReLU = PReLU(shared_axes=[1,2])

#     def call(self, input):
#         """
#         args:
#            input: input feature map
#            return: transformed feature map
#         """
#         if self.padding == 'valid' and self.kSize != 1:
#             input = self.pad(input)
#         output = self.conv(input)
#         output = self.bn(output)
#         output = self.PReLU(output)
#         return output


# class BNPReLU(layers.Layer):
#     def __init__(self, epsilon=1e-03):
#         """
#         args:
#            nOut: channels of output feature maps
#         """
#         super().__init__()
#         self.bn = SyncBatchNormalization(epsilon=epsilon)
#         self.PReLU = PReLU(shared_axes=[1,2])

#     def call(self, input):
#         """
#         args:
#            input: input feature map
#            return: normalized and thresholded feature map
#         """
#         output = self.bn(input)
#         output = self.PReLU(output)
#         return output


# class FGlo(layers.Layer):
#     """
#     the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
#     """
#     def __init__(self, nOut, reduction=16):
        
#         super(FGlo, self).__init__()
        
#         self.glob_avg_pool = GlobalAveragePooling2D(data_format='channels_last') #fglo
#         self.FC1 = Dense(nOut // reduction, activation= 'relu')
#         self.FC2 = Dense(nOut, activation= 'sigmoid') # sigmoid
    

#     def call(self, input):
#         output = self.glob_avg_pool(input)
#         output = self.FC1(output)
#         output = self.FC2(output)
#         output = tf.expand_dims(output, 1)
#         output = tf.expand_dims(output, 2)
        
#         return input * output


# class CGblock_down(layers.Layer):
#     def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=True, epsilon=1e-03, kernel_initializer=kernel_initializer):
        
#         super(CGblock_down, self).__init__()
        
#         self.ConvBNPReLU = ConvBNPReLU(nOut, kSize, strides=2, padding='valid',kernel_initializer= kernel_initializer)
#         self.F_loc = Conv2D(nOut, kSize, strides=(strides, strides), padding=padding,
#                                     activation=None, kernel_initializer=kernel_initializer, groups = nOut,
#                                     use_bias = False) #floc
#         self.F_sur = Conv2D(nOut,kSize, strides=(strides, strides), padding=padding, 
#                                     dilation_rate=dilation_rate, kernel_initializer=kernel_initializer, groups = nOut,
#                                     use_bias = False) #fsur
#         self.Concatenate = Concatenate() #fjoi
#         self.BNPReLU = BNPReLU(2*nOut)
#         self.reduce = Conv2D(nOut, 1, 1, use_bias = False)
#         self.FGLo = FGlo(nOut,reduction=reduction) #fglo

#     def call(self, input):
#         """
#         args:
#            input: input feature map
#            return: transformed feature map
#         """
#         output = self.ConvBNPReLU(input)

#         loc = self.F_loc(output)
#         sur = self.F_sur(output)
#         output = Concatenate()([loc,sur])
#         output = self.BNPReLU(output)
#         output = self.reduce(output)
#         output = self.FGLo(output)

#         return output

# class CGblock(layers.Layer):
#     def __init__(self, nOut, kSize, strides=1, padding='same', dilation_rate=2, reduction=16, add=False, epsilon=1e-03, kernel_initializer=kernel_initializer):
        
#         super(CGblock, self).__init__()
        
#         n= int(nOut/2)
#         self.ConvBNPReLU = ConvBNPReLU(n, 1, strides=1, padding='same',kernel_initializer=kernel_initializer) 
#         self.F_loc = Conv2D(n, 3, strides=(strides, strides), padding=padding,
#                                     activation=None,kernel_initializer=kernel_initializer, groups=n,
#                                     use_bias = False) #floc
#         self.F_sur = Conv2D(n, 3, strides=(strides, strides), padding=padding, 
#                                     dilation_rate=dilation_rate,kernel_initializer=kernel_initializer, groups=n,
#                                     use_bias = False) #fsur
#         self.Concatenate = Concatenate() #fjoi
#         self.BNPReLU = BNPReLU(epsilon=epsilon)
#         self.add = add
#         self.FGLo = FGlo(nOut,reduction=reduction)#fglo

#     def call(self, input):
#         """
#         args:
#            input: input feature map
#            return: transformed feature map
#         """
#         output = self.ConvBNPReLU(input)

#         loc = self.F_loc(output)
#         sur = self.F_sur(output)
#         output = Concatenate()([loc,sur])
#         output = self.BNPReLU(output)
#         output = self.FGLo(output)
#         if self.add:

#             output = input + output

#         return output


# class InputInjection(layers.Layer):
#     """
#     args:
#     """
#     def __init__(self, downsamplingRatio):
#         super(InputInjection, self).__init__()
#         self.pool = []
#         for i in range(0, downsamplingRatio):
#             self.pool.append(ZeroPadding2D(1))
#             self.pool.append(AveragePooling2D(3, strides=2))
#     def call(self, input):
#         for pool in self.pool:
#             input = pool(input)
#         return input



# class CGNet(Model):
#     def __init__(self, input_shape = (680,680,3), num_classes=19, M= 3, N= 21, dropout_flag = False):
#         """
#         """
#         super(CGNet, self).__init__()
#         #Stage 1
        
#         self.dropout_flag = dropout_flag
#         self.stage1_1 = ConvBNPReLU(32,3,2,kernel_initializer=kernel_initializer) 
#         self.stage1_2 = ConvBNPReLU(32, 3,1,kernel_initializer=kernel_initializer) 
#         self.stage1_3 = ConvBNPReLU(32, 3,1,kernel_initializer=kernel_initializer)

#         self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
#         self.sample2 = InputInjection(2)

#         self.bn1 = BNPReLU()

#         # First CG block (M=3) dilation=2
#         #Stage 2
#         self.stage2_1 = CGblock_down(64, 3, dilation_rate=2, reduction=8)
#         self.stage2 = []

#         for i in range(0, M-1): 
#             self.stage2.append(CGblock(64,3, dilation_rate=2, reduction=8))
            
#         self.bn2 = BNPReLU()

#         #Stage 3
#         self.stage3_1 = CGblock_down(128, 3, dilation_rate=4, reduction=16)
#         self.stage3 = []

#         for i in range(0, N-1) : 
#             self.stage3.append(CGblock(128,3, dilation_rate=4, reduction=16))

#         self.bn3 = BNPReLU()

#         #Classifier
#         if self.dropout_flag:
#             print("have droput layer")
#             self.dropout = tf.keras.Sequential(Dropout(0.4, False))
#             self.classifier = tf.keras.Sequential(Conv2D(num_classes, 1, use_bias = False))
#         else:
#             self.classifier = Conv2D(num_classes,1, use_bias = False)

#         self.upsample = UpSampling2D(size=(8, 8), interpolation = 'bilinear')
#         """
#         TODO 
#         1. add an initialization 
#         """
       
            
#     def call(self, input):


#         # Stage 1 
#         print("Training starts with {}".format(input.shape))
#         output1 = self.stage1_1(input)
#         output1 = self.stage1_2(output1)
#         output1 = self.stage1_3(output1)
        

#         inp1 =   self.sample1(input)
#         inp2 =   self.sample2(input)

#         output1_cat = self.bn1(Concatenate()([output1, inp1]))
#         print("Stage 1 ends with {}".format(output1_cat.shape))

#         # Stage 2
#         output2_1 = self.stage2_1(output1_cat) 
#         print("Stage 2 starts with {}".format(output2_1.shape))

#         for i, layer in enumerate(self.stage2): 
#             if i == 0 : 
#                 output2 = layer(output2_1)
#             else : 
#                 output2 = layer(output2)

#         output2_cat = self.bn2(Concatenate()([output2, output2_1, inp2]))
        
#         # Stage 3 
#         output3_1 = self.stage3_1(output2_cat)

#         for i, layer in enumerate(self.stage3): 
#             if i == 0 : 
#                 output3 = layer(output3_1)
#             else : 
#                 output3 = layer(output3)

#         output3_cat = self.bn3(Concatenate()([output3, output3_1]))
#         print("Stage 3 ends with {}".format(output3_cat.shape))
#         # classifier 
#         if self.dropout_flag:
#             output3_cat = self.dropout(output3_cat)

#         classifier = self.classifier(output3_cat)

#         # upsample segmenation map ---> the input image size
#         # out = tf.image.resize(classifier, (input.shape[1], input.shape[2]))
#         out = self.upsample(classifier)
#         return out 
    
#     def model(self):
#         input = tf.keras.layers.Input(shape=(680,680,3))
#         return Model(inputs=[input], outputs=self.call(input))

# model = CGNet()
# model_functional = model.model()
# #model.build((None, 32,32,3))
# model_functional.summary()