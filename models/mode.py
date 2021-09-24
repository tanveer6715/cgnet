# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.layers import add
# import tensorflow.keras.backend as K

# classes = 19

# def resdiual_block(in_tensor,filters,stage,block):
#     conv = "block2_conv"
#     bnorm =  "block2_bn"
#     filter1,filter2,filter3 = filters
#     x = Conv2D(filter1,(1,1),name=conv+stage+block+"a",padding="same")(in_tensor)
#     x = BatchNormalization(axis=3,name=bnorm+stage+block+"a")(x)
#     x = Activation("relu")(x)
#     x = Conv2D(filter2,(3,3),name=conv+stage+block+"b",padding="same")(x)
#     x = BatchNormalization(axis=3,name=bnorm+stage+block+"b")(x)
#     x = Activation("relu")(x)
#     x = Conv2D(filter3,(1,1),name=conv+stage+block+"c",strides=(2,2),padding="same")(x)
#     x = BatchNormalization(axis=3,name=bnorm+stage+block+"c")(x)
#     x = Activation("relu")(x)
#     shortcut = Conv2D(filter3,(1,1),padding="same",strides=(2,2),name="shortcut"+conv+stage+block)(in_tensor)
#     shortcut = BatchNormalization(axis=3,name="shortcut_bn"+stage+block)(shortcut)
#     shortcut = add([x,shortcut])
#     shortcut = Activation("relu")(shortcut)
#     return shortcut


# def downsampling(in_tensor,filter_,block):
#     tname = "downsample_layer"+str(block)
#     c1 = Conv2D(filter_,3,strides=(2,2),padding="same",name=tname+"conv_3x3")(in_tensor)
#     c1 = BatchNormalization(axis=-1)(c1)
#     c1 = Activation("relu")(c1)
#     return c1


# def resizer(x,size,dataformat = "channels_last"):
#     res = tf.image.resize(x,size)
#     return res

# def resizer_block(tensor,size):
#     layer = Lambda(lambda x : resizer(x,size))(tensor)
#     return layer


# def channel_split(x):
#     split1,split2 = tf.split(x,num_or_size_splits=2,axis=-1)
#     return split1,split2

# def splitter(tensor):
#     layer = Lambda(lambda x : channel_split(x))(tensor)
#     return layer


# def nnbt_conv(in_tensor1,in_tensor2,block,filter_,dilation_rate=None):
#     tname = "NNBT_channel_id_"+str(block)
#     c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv1")(in_tensor1)
#     c1 = Activation("relu")(c1)
#     c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv2")(c1)
#     c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn1")(c1)
#     c1 = Activation("relu")(c1)
#     c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv3")(c1)
#     c1 = Activation("relu")(c1)
#     c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv4")(c1)
#     c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn2")(c1)
#     c1 = Activation("relu")(c1)
#     c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv1")(in_tensor2)
#     c2 = Activation("relu")(c2)
#     c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv2")(c2)
#     c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn1")(c2)
#     c2 = Activation("relu")(c2)
#     c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv3")(c2)
#     c2 = Activation("relu")(c2)
#     c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv4")(c2)
#     c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn2")(c2)
#     c2 = Activation("relu")(c2)
    
#     if dilation_rate:
#         c1 = Conv2D(filter_,(3,1),padding="same",name=tname+"path1_conv1")(in_tensor1)
#         c1 = Activation("relu")(c1)
#         c1 = Conv2D(filter_,(1,3),padding="same",name=tname+"path1_conv2")(c1)
#         c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn1")(c1)
#         c1 = Activation("relu")(c1)
#         c1 = Conv2D(filter_,(3,1),padding="same",dilation_rate=dilation_rate,name=tname+"path1_dil_conv3")(c1)
#         c1 = Activation("relu")(c1)
#         c1 = Conv2D(filter_,(1,3),padding="same",dilation_rate=dilation_rate,name=tname+"path1_dil_conv4")(c1)
#         c1 = BatchNormalization(axis=-1,name = tname+"bn_"+"path1_bn2")(c1)
#         c1 = Activation("relu")(c1)
#         c2 = Conv2D(filter_,(1,3),padding="same",name=tname+"path2_conv1")(in_tensor2)
#         c2 = Activation("relu")(c2)
#         c2 = Conv2D(filter_,(3,1),padding="same",name=tname+"path2_conv2")(c2)
#         c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn1")(c2)
#         c2 = Activation("relu")(c2)
#         c2 = Conv2D(filter_,(1,3),padding="same",dilation_rate=dilation_rate,name=tname+"path2_dil_conv3")(c2)
#         c2 = Activation("relu")(c2)
#         c2 = Conv2D(filter_,(3,1),padding="same",dilation_rate=dilation_rate,name=tname+"path2_dil_conv4")(c2)
#         c2 = BatchNormalization(axis=-1,name = tname+"bn_"+"path2_bn2")(c2)
#         c2 = Activation("relu")(c2)

#     concat_layer = concatenate([c1,c2],axis=-1)
#     concat_layer = Conv2D(256,1,padding="same")(concat_layer)
#     concat_layer = BatchNormalization(axis=-1)(concat_layer)
#     concat_layer = Activation("relu")(concat_layer)
#     return concat_layer
    

# def nnbt(in_tensor,block,filt_val,dilation_rate):
#     ch0,ch1 = splitter(in_tensor)
#     nconv = nnbt_conv(ch0,ch1,block,filt_val,dilation_rate)
#     skip = Conv2D(K.int_shape(nconv)[3],1,padding="same")(in_tensor)
#     skip_layer = add([skip,nconv])
#     skip_layer = Conv2D(256,1,padding="same")(skip_layer)
#     skip_layer = Activation("relu")(skip_layer)
#     skip_layer = BatchNormalization(axis=-1)(skip_layer)
#     return skip_layer

# def encoder(inpu):
#     ds1 = downsampling(inpu,32,1)
#     nb1 = nnbt(ds1,1,16,dilation_rate=None)
#     nb2 = nnbt(nb1,2,16,dilation_rate=None)
#     nb3 = nnbt(nb2,3,16,dilation_rate=None)
#     ds2 = downsampling(nb3,64,2)
#     nb4 = nnbt(ds2,4,32,dilation_rate=None)
#     nb5 = nnbt(nb4,5,32,dilation_rate=None)
#     ds3 = downsampling(nb5,128,3)
#     nb6 = nnbt(ds3,6,64,dilation_rate=1)
#     nb7 = nnbt(nb6,7,64,dilation_rate=2)
#     nb8 = nnbt(nb7,8,64,dilation_rate=5)
#     nb9 = nnbt(nb8,9,64,dilation_rate=9)
#     nb10 = nnbt(nb9,10,64,dilation_rate=2)
#     nb11 = nnbt(nb10,11,64,dilation_rate=5)
#     nb12 = nnbt(nb11,12,64,dilation_rate=9)
#     nb13 = nnbt(nb12,13,64,dilation_rate=17)
#     return nb13

# def reshape(tensor,shape):
#     la = Lambda(lambda x : K.reshape(x,shape))(tensor)
#     return la


# def expand(tensor,ax):
#     la = Lambda(lambda x : K.expand_dims(x,axis=ax))(tensor)
#     return la

# def decoder(inpu_tensor,classes):
    
    
#     mtensor_1,mtensor_2 = K.int_shape(inpu_tensor)[1],K.int_shape(inpu_tensor)[2]
#     #print(mtensor_1,mtensor_2)
#     pooled_enco = GlobalAveragePooling2D()(inpu_tensor)
#     pooled_enco = expand(pooled_enco,ax=1)
#     pooled_enco = expand(pooled_enco,ax=1)
#     cc = resizer_block(pooled_enco,(mtensor_1,mtensor_2))
#     cc = Conv2D(classes,3,padding="same")(cc)
#     cc = BatchNormalization(axis=-1)(cc)
#     cc = Activation("relu")(cc)

    
#     base_patch= Conv2D(classes,3,padding="same",name="base_patch_conv")(inpu_tensor)
#     base_patch = Activation("relu")(base_patch)
    
#     c3 = Conv2D(128,3,strides=(2,2),padding="same",name="feature_conv3x3")(inpu_tensor)
#     c3 = BatchNormalization(axis=-1)(c3)
#     c3 = Activation("relu")(c3)
#     class_contextc3 = Conv2D(classes,1,padding="same",name="feature3_convclass1x1")(c3)
#     class_contextc3 = BatchNormalization(axis=-1)(class_contextc3)
#     class_contextc3 = Activation("relu")(class_contextc3)
    
    
#     c5 = Conv2D(128,5,strides=(2,2),padding="same",name="feature_conv5x5")(c3)
#     c5 = BatchNormalization(axis=-1)(c5)
#     c5 = Activation("relu")(c5)
#     class_contextc5 = Conv2D(classes,1,padding="same",name="feature5_convclass1x1")(c5)
#     class_contextc5 = BatchNormalization(axis=-1)(class_contextc5)
#     class_contextc5 = Activation("relu")(class_contextc5)
    
    
#     c7 = Conv2D(128,7,strides=(2,2),padding="same",name="feature_conv7x7")(c5)
#     c7 = BatchNormalization(axis=-1)(c7)
#     c7 = Activation("relu")(c7)
#     class_contextc7 = Conv2D(classes,1,padding="same",name="feature7_convclass1x1")(c7)
#     class_contextc7 = BatchNormalization(axis=-1)(class_contextc7)
#     class_contextc7 = Activation("relu")(class_contextc7)
    
#     up_dim1,up_dim2 = 2*K.int_shape(class_contextc7)[1],2*K.int_shape(class_contextc7)[2]
    
#     upcon_7 = resizer_block(class_contextc7,(up_dim1,up_dim2))
#     up_sum = add([upcon_7,class_contextc5])
#     up1_dim1,up2_dim2 = 2*K.int_shape(up_sum)[1],2*K.int_shape(up_sum)[2]
#     upcon_8 = resizer_block(up_sum,(up1_dim1,up2_dim2))
#     up1_sum = add([upcon_8,class_contextc3])
#     up2_dim1,up2_dim2 = 2*K.int_shape(up1_sum)[1],2*K.int_shape(up1_sum)[2]
#     upcon_9 = resizer_block(up1_sum,(up2_dim1,up2_dim2))
    
#     patch1_merger = multiply([upcon_9,base_patch])
    
#     final_merge = add([patch1_merger,cc])
    
#     final_merge = resizer_block(final_merge,(680,680))
#     print("Stage 1 ends with {}".format(final_merge.shape))
#     final_merge = Conv2D(classes,1,padding="same")(final_merge)
#     final_merge = Activation("softmax")(final_merge)
    
#     return final_merge
# def compute_output_shape(self, input_shape):
#     return (input_shape[1], self.out_units)


# def LEDnet(classes):
#     inpu = Input(shape=(680,680,3))
#     encode = encoder(inpu)
#     dec = decoder(encode,classes)
#     comp = Model(inputs=inpu,outputs=dec)
#     return comp


# model = LEDnet(classes)
# model.build((680,680,3))
# model_functional = model.model()
# model_functional.summary()

# input = tf.keras.layers.Input(shape=(680,680,3))
#         return Model(inputs=[input], outputs=self.call(input))



import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.layers.experimental import SyncBatchNormalization
"""
creates a 3*3 conv with given filters and stride
ref: https://github.com/hamidriasat/DDRNets/blob/9872cf900c8f0d1e14f83ea2cd11946948643abe/ddrnet_23_slim.py#L181
"""
def conv3x3(out_planes, stride=1):
    return layers.Conv2D(kernel_size=(3,3), filters=out_planes, strides=stride, padding="same",
                       use_bias=False)

"""
Creates a residual block with two 3*3 conv's
in paper it's represented by RB block
"""
basicblock_expansion = 1
def basic_block(x_in, planes, stride=1, downsample=None, no_relu=False):
    residual = x_in

    x = conv3x3(planes, stride)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = conv3x3(planes,)(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return x

"""
creates a bottleneck block of 1*1 -> 3*3 -> 1*1
"""
bottleneck_expansion = 2
def bottleneck_block(x_in, planes, stride=1, downsample=None, no_relu=True):
    residual = x_in

    x = layers.Conv2D(filters=planes, kernel_size=(1,1), use_bias=False)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes, kernel_size=(3,3), strides=stride, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=planes* bottleneck_expansion, kernel_size=(1,1), use_bias=False)(x)
    x= layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample

    # x += residual
    x = layers.Add()([x, residual])

    if not no_relu:
        x = layers.Activation("relu")(x)

    return  x

# Deep Aggregation Pyramid Pooling Module
def DAPPPM(x_in, branch_planes, outplanes):
    input_shape = tf.keras.backend.int_shape(x_in)
    height = input_shape[1]
    width = input_shape[2]
    # Average pooling kernel size
    kernal_sizes_height = [5, 9, 17, height]
    kernal_sizes_width =  [5, 9, 17, width]
    # Average pooling strides size
    stride_sizes_height = [2, 4, 8, height]
    stride_sizes_width =  [2, 4, 8, width]
    x_list = []

    # y1
    scale0 = layers.BatchNormalization()(x_in)
    scale0 = layers.Activation("relu")(scale0)
    scale0 = layers.Conv2D(branch_planes, kernel_size=(1,1), use_bias=False, )(scale0)
    x_list.append(scale0)

    for i in range( len(kernal_sizes_height)):
        # first apply average pooling
        temp = layers.AveragePooling2D(pool_size=(kernal_sizes_height[i],kernal_sizes_width[i]),
                                       strides=(stride_sizes_height[i],stride_sizes_width[i]),
                                       padding="same")(x_in)
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # then apply 1*1 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(1, 1), use_bias=False, )(temp)
        # then resize using bilinear
        temp = tf.image.resize(temp, size=(height,width), )
        # add current and previous layer output
        temp = layers.Add()([temp, x_list[i]])
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("relu")(temp)
        # at the end apply 3*3 conv
        temp = layers.Conv2D(branch_planes, kernel_size=(3, 3), use_bias=False, padding="same")(temp)
        # y[i+1]
        x_list.append(temp)

    # concatenate all
    combined = layers.concatenate(x_list, axis=-1)

    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation("relu")(combined)
    combined = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(combined)

    shortcut = layers.BatchNormalization()(x_in)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=False, )(shortcut)

    # final = combined + shortcut
    final = layers.Add()([combined, shortcut])

    return final

"""
Segmentation head 
3*3 -> 1*1 -> rescale
"""
def segmentation_head(x_in, interplanes, outplanes, scale_factor=None):
    x = layers.BatchNormalization()(x_in)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(interplanes, kernel_size=(3, 3), use_bias=False, padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(outplanes, kernel_size=(1, 1), use_bias=range, padding="valid")(x)

    if scale_factor is not None:
        input_shape = tf.keras.backend.int_shape(x)
        height2 = input_shape[1] * scale_factor
        width2 = input_shape[2] * scale_factor
        x = tf.image.resize(x, size =(height2, width2))

    return x

"""
apply multiple RB or RBB blocks.
x_in: input tensor
block: block to apply it can be RB or RBB
inplanes: input tensor channes
planes: output tensor channels
blocks_num: number of time block to applied
stride: stride
expansion: expand last dimension
"""
def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D(((planes * expansion)), kernel_size=(1, 1),strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization()(downsample)
        downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x


"""
ddrnet 23 slim
input_shape : shape of input data
layers_arg : how many times each Rb block is repeated
num_classes: output classes
planes: filter size kept throughout model
spp_planes: DAPPM block output dimensions
head_planes: segmentation head dimensions
scale_factor: scale output factor
augment: whether auxiliary loss is added or not
"""
def ddrnet_23_slim(input_shape=[680,680,3], layers_arg=[2, 2, 2, 2], num_classes=5, planes=32, spp_planes=128,
                   head_planes=64, scale_factor=8,augment=False):

    x_in = layers.Input(input_shape)

    highres_planes = planes * 2
    input_shape = tf.keras.backend.int_shape(x_in)
    height_output = input_shape[1] // 8
    width_output = input_shape[2] // 8

    layers_inside = []

    # 1 -> 1/2 first conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3),strides=2, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1/2 -> 1/4 second conv layer
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # layer 1
    # 1/4 -> 1/4 first basic residual block not mentioned in the image
    x = make_layer(x, basic_block, planes, planes, layers_arg[0], expansion=basicblock_expansion)
    layers_inside.append(x)

    # layer 2
    # 2 High :: 1/4 -> 1/8 storing results at index:1
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes, planes*2, layers_arg[1], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)

    """
    For next layers 
    x:  low branch
    x_: high branch
    """

    # layer 3
    # 3 Low :: 1/8 -> 1/16 storing results at index:2
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes*2, planes*4, layers_arg[2], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)
    # 3 High :: 1/8 -> 1/8 retrieving from index:1
    x_ = layers.Activation("relu")(layers_inside[1])
    x_ = make_layer(x_, basic_block, planes*2, highres_planes, 2, expansion=basicblock_expansion)

    # Fusion 1
    # x -> 1/16 to 1/8, x_ -> 1/8 to 1/16
    # High to Low
    x_temp = layers.Activation("relu")(x_)
    x_temp =  layers.Conv2D(planes*4, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[2])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1,1), use_bias=False)(x_temp)
    x_temp =layers.BatchNormalization()(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output)) # 1/16 -> 1/8
    x_ = layers.Add()([x_, x_temp]) # next high branch input, 1/8

    if augment:
        temp_output = x_  # Auxiliary loss from high branch

    # layer 4
    # 4 Low :: 1/16 -> 1/32 storing results at index:3
    x = layers.Activation("relu")(x)
    x = make_layer(x, basic_block, planes * 4, planes * 8, layers_arg[3], stride=2, expansion=basicblock_expansion)
    layers_inside.append(x)
    # 4 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, basic_block, highres_planes, highres_planes, 2, expansion=basicblock_expansion)

    # Fusion 2 :: x_ -> 1/32 to 1/8, x -> 1/8 to 1/32 using two conv's
    # High to low
    x_temp = layers.Activation("relu")(x_)
    x_temp = layers.Conv2D(planes * 4, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = layers.Activation("relu")(x_temp)
    x_temp = layers.Conv2D(planes * 8, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x = layers.Add()([x, x_temp])
    # Low to High
    x_temp = layers.Activation("relu")(layers_inside[3])
    x_temp = layers.Conv2D(highres_planes, kernel_size=(1, 1), use_bias=False)(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = tf.image.resize(x_temp, (height_output, width_output))
    x_ = layers.Add()([x_, x_temp])

    # layer 5
    # 5 High :: 1/8 -> 1/8
    x_ = layers.Activation("relu")(x_)
    x_ = make_layer(x_, bottleneck_block, highres_planes, highres_planes, 1, expansion=bottleneck_expansion)
    x = layers.Activation("relu")(x)
    # 5 Low :: 1/32 -> 1/64
    x = make_layer(x, bottleneck_block,  planes * 8, planes * 8, 1, stride=2, expansion=bottleneck_expansion)

    # Deep Aggregation Pyramid Pooling Module
    x = DAPPPM(x, spp_planes, planes * 4)

    # resize from 1/64 to 1/8
    x = tf.image.resize(x, (height_output, width_output))

    x_ = layers.Add()([x, x_])

    x_ = segmentation_head( (x_), head_planes, num_classes, scale_factor)

    # apply softmax at the output layer
    x_ = tf.nn.softmax(x_)

    if augment:
        x_extra = segmentation_head( temp_output, head_planes, num_classes, scale_factor) # without scaling
        x_extra = tf.nn.softmax(x_extra)
        model_output = [x_, x_extra]
    else:
        model_output = x_

    model = models.Model(inputs=[x_in], outputs=[model_output])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model

model = ddrnet_23_slim()
model.summary()





# ###FASTNET

# # import tensorflow as tf 

# # #ref https://github.com/ayushmankumar7/Fast-Segementation/tree/master/src/models

# # def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding="same", relu = True):

# #     if (conv_type == 'ds'):
        
# #         x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)
    
# #     else:
        
# #         x = tf.keras.layers.Conv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)

# #     x = tf.keras.layers.BatchNormalization()(x)

# #     if(relu):
# #         x = tf.keras.activations.relu(x)

# #     return x 



# # def _res_bottleneck(inputs, filters, kernel, t, s, r = False):

# #     tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

# #     x = conv_block(inputs, "conv", tchannel, (1,1), strides =(1,1) )

# #     x = tf.keras.layers.DepthwiseConv2D(kernel, strides = (s,s), depth_multiplier = 1, padding = 'same')(x)
# #     x = tf.keras.layers.BatchNormalization()(x)
# #     x = tf.keras.activations.relu(x)

# #     x = conv_block(x, 'conv', filters, (1,1), strides =(1,1), padding = 'same', relu = False)

# #     if r:
# #         x = tf.keras.layers.add([x, inputs])

# #     return x


# # def bottleneck_block(inputs, filters, kernel, t, strides, n):

# #     x = _res_bottleneck(inputs, filters, kernel, t, strides)

# #     for i in range(1, n):
# #         x = _res_bottleneck(x, filters, kernel, t, 1, True)

# #     return x 


# # def pyramid_pooling_block(input_tensor, bin_sizes):

# #     concat_list = [input_tensor]
# #     w = 64 
# #     h = 32

# #     for bin_size in bin_sizes:
# #         x = tf.keras.layers.AveragePooling2D(pool_size = (w//bin_size, h//bin_size), strides = (w//bin_size, h//bin_size))(input_tensor)
# #         x = tf.keras.layers.Conv2D(128, 3, 2, padding = 'same')(x)
# #         x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

# #         concat_list.append(x) 

# #     return tf.keras.layers.concatenate(concat_list)


# # def Fastnet():

# #     input_layer = tf.keras.layers.Input(shape = (2048, 1024, 3), name = "input_layer")

# #     #Learning to Down Sample 

# #     lds_layer = conv_block(input_layer, 'conv', 32,(3,3), strides = (2,2) )
# #     lds_layer = conv_block(lds_layer, 'ds', 48,(3,3), strides = (2,2) )
# #     lds_layer = conv_block(lds_layer,'ds', 64,(3,3), strides = (2,2) )

# #     #Global Feature Extractor

# #     gfe_layer = bottleneck_block(lds_layer, 64, (3,3), t = 6, strides = 2, n =3)
# #     gfe_layer = bottleneck_block(gfe_layer, 96, (3,3), t = 6, strides = 2, n=3)
# #     gfe_layer = bottleneck_block(gfe_layer, 128,(3,3), t = 6, strides = 1, n=3)


# #     # PPM 

# #     gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8])

# #     # Feature Fusion 

# #     ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides = (1,1), relu=False )

# #     ff_layer2 = tf.keras.layers.UpSampling2D((4,4))(gfe_layer)
# #     ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides = (1,1), depth_multiplier=1, padding='same')(ff_layer2)
# #     ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
# #     ff_layer2 = tf.keras.activations.relu(ff_layer2)
# #     ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation = None)(ff_layer2)


# #     ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
# #     ff_final = tf.keras.layers.BatchNormalization()(ff_final)
# #     ff_final = tf.keras.activations.relu(ff_final)

# #     #Classifier 

# #     classifier = tf.keras.layers.SeparableConv2D(128, (3,3), padding = 'same', strides = (1,1), name = 'DSConv1_classifier')(ff_final)
# #     classifier = tf.keras.layers.BatchNormalization()(classifier)
# #     classifier = tf.keras.activations.relu(classifier)

# #     classifier = tf.keras.layers.SeparableConv2D(128, (3,3), padding = 'same', strides = (1,1), name = 'DSConv2_classifier')(classifier)
# #     classifier = tf.keras.layers.BatchNormalization()(classifier)
# #     classifier = tf.keras.activations.relu(classifier)


# #     classifier = conv_block(classifier, 'conv', 19, (1,1), strides = (1,1), padding='same', relu=True)

# #     classifier = tf.keras.layers.Dropout(0.3)(classifier)

# #     classifier = tf.keras.layers.UpSampling2D((8,8))(classifier)
# #     classifier = tf.keras.activations.softmax(classifier)

# #     fast_scnn = tf.keras.Model(inputs = input_layer, outputs = classifier, name = 'Fast_SCNN')

# #     return fast_scnn

# # model = Fastnet()

# # model.summary()



import tensorflow as tf
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras.layers import SpatialDropout2D,Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,AveragePooling2D, UpSampling2D, BatchNormalization, Activation, add,Dropout,Permute,ZeroPadding2D,Add, Reshape
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, PReLU
from tensorflow.keras import backend as K 
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from tensorflow.keras import applications, optimizers, callbacks
import matplotlib
from tensorflow.keras.layers import *

#code reference =https://github.com/AngeLouCN/CFPNet-Medicine/blob/main/network/ESPNet.py

# # def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
# #     '''
# #     2D Convolutional layers
    
# #     Arguments:
# #         x {keras layer} -- input layer 
# #         filters {int} -- number of filters
# #         num_row {int} -- number of rows in filters
# #         num_col {int} -- number of columns in filters
    
# #     Keyword Arguments:
# #         padding {str} -- mode of padding (default: {'same'})
# #         strides {tuple} -- stride of convolution operation (default: {(1, 1)})
# #         activation {str} -- activation function (default: {'relu'})
# #         name {str} -- name of the layer (default: {None})
    
# #     Returns:
# #         [keras layer] -- [output layer]
# #     '''

# #     x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
# #     x = BatchNormalization(axis=3, scale=False)(x)

# #     if(activation == None):
# #         return x

# #     x = Activation(activation, name=name)(x)

# #     return x


# # def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
# #     '''
# #     2D Transposed Convolutional layers
    
# #     Arguments:
# #         x {keras layer} -- input layer 
# #         filters {int} -- number of filters
# #         num_row {int} -- number of rows in filters
# #         num_col {int} -- number of columns in filters
    
# #     Keyword Arguments:
# #         padding {str} -- mode of padding (default: {'same'})
# #         strides {tuple} -- stride of convolution operation (default: {(2, 2)})
# #         name {str} -- name of the layer (default: {None})
    
# #     Returns:
# #         [keras layer] -- [output layer]
# #     '''

# #     x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
# #     x = BatchNormalization(axis=3, scale=False)(x)
    
# #     return x


# def DCBlock(U, inp, alpha = 1.67):
#     '''
#     DC Block
    
#     Arguments:
#         U {int} -- Number of filters in a corrsponding UNet stage
#         inp {keras layer} -- input layer 
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     W = alpha * U

#     #shortcut = inp

#     #shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
#    #                      int(W*0.5), 1, 1, activation=None, padding='same')

#     conv3x3_1 = conv2d_bn(inp, int(W*0.167), 3, 3,
#                         activation='relu', padding='same')

#     conv5x5_1 = conv2d_bn(conv3x3_1, int(W*0.333), 3, 3,
#                         activation='relu', padding='same')

#     conv7x7_1 = conv2d_bn(conv5x5_1, int(W*0.5), 3, 3,
#                         activation='relu', padding='same')

#     out1 = concatenate([conv3x3_1, conv5x5_1, conv7x7_1], axis=3)
#     out1 = BatchNormalization(axis=3)(out1)
    
#     conv3x3_2 = conv2d_bn(inp, int(W*0.167), 3, 3,
#                         activation='relu', padding='same')

#     conv5x5_2 = conv2d_bn(conv3x3_2, int(W*0.333), 3, 3,
#                         activation='relu', padding='same')

#     conv7x7_2 = conv2d_bn(conv5x5_2, int(W*0.5), 3, 3,
#                         activation='relu', padding='same')
#     out2 = concatenate([conv3x3_2, conv5x5_2, conv7x7_2], axis=3)
#     out2 = BatchNormalization(axis=3)(out2)

#     out = add([out1, out2])
#     out = Activation('relu')(out)
#     out = BatchNormalization(axis=3)(out)

#     return out

# # def ResPath(filters, length, inp):
# #     '''
# #     ResPath
    
# #     Arguments:
# #         filters {int} -- [description]
# #         length {int} -- length of ResPath
# #         inp {keras layer} -- input layer 
    
# #     Returns:
# #         [keras layer] -- [output layer]
# #     '''

# #     shortcut = inp
# #     shortcut = conv2d_bn(shortcut, filters, 1, 1,
# #                          activation=None, padding='same')

# #     out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

# #     out = add([shortcut, out])
# #     out = Activation('relu')(out)
# #     out = BatchNormalization(axis=3)(out)

# #     for i in range(length-1):

# #         shortcut = out
# #         shortcut = conv2d_bn(shortcut, filters, 1, 1,
# #                              activation=None, padding='same')

# #         out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

# #         out = add([shortcut, out])
# #         out = Activation('relu')(out)
# #         out = BatchNormalization(axis=3)(out)

# #     return out

# def DCUNet(height, width, channels):
#     '''
#     DC-UNet
    
#     Arguments:
#         height {int} -- height of image 
#         width {int} -- width of image 
#         n_channels {int} -- number of channels in image
    
#     Returns:
#         [keras model] -- MultiResUNet model
#     '''

#     inputs = Input((height, width, channels))

#     dcblock1 = DCBlock(32, inputs)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(dcblock1)
#     dcblock1 = ResPath(32, 4, dcblock1)

#     dcblock2 = DCBlock(32*2, pool1)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(dcblock2)
#     dcblock2 = ResPath(32*2, 3, dcblock2)

#     dcblock3 = DCBlock(32*4, pool2)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(dcblock3)
#     dcblock3 = ResPath(32*4, 2, dcblock3)

#     dcblock4 = DCBlock(32*8, pool3)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(dcblock4)
#     dcblock4 = ResPath(32*8, 1, dcblock4)

#     dcblock5 = DCBlock(32*16, pool4)

#     up6 = concatenate([Conv2DTranspose(
#         32*8, (2, 2), strides=(2, 2), padding='same')(dcblock5), dcblock4], axis=3)
#     dcblock6 = DCBlock(32*8, up6)

#     up7 = concatenate([Conv2DTranspose(
#         32*4, (2, 2), strides=(2, 2), padding='same')(dcblock6), dcblock3], axis=3)
#     dcblock7 = DCBlock(32*4, up7)

#     up8 = concatenate([Conv2DTranspose(
#         32*2, (2, 2), strides=(2, 2), padding='same')(dcblock7), dcblock2], axis=3)
#     dcblock8 = DCBlock(32*2, up8)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
#         2, 2), padding='same')(dcblock8), dcblock1], axis=3)
#     dcblock9 = DCBlock(32, up9)

#     conv10 = conv2d_bn(dcblock9, 1, 1, 1, activation='sigmoid')
    
#     model = Model(inputs=[inputs], outputs=[conv10])
    
#     return model


# def conv2d_bn(x, filters, ksize, d_rate, strides,padding='same', activation='relu', groups=1, name=None):
#     '''
#     2D Convolutional layers
    
#     Arguments:
#         x {keras layer} -- input layer 
#         filters {int} -- number of filters
#         num_row {int} -- number of rows in filters
#         num_col {int} -- number of columns in filters
    
#     Keyword Arguments:
#         padding {str} -- mode of padding (default: {'same'})
#         strides {tuple} -- stride of convolution operation (default: {(1, 1)})
#         activation {str} -- activation function (default: {'relu'})
#         name {str} -- name of the layer (default: {None})
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''

#     x = Conv2D(filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate = d_rate, use_bias=False)(x)
#     x = BatchNormalization(axis=3, scale=False)(x)

#     if(activation == None):
#         return x

#     x = Activation(activation, name=name)(x)

#     return x

# def CFPModule(inp, filters, d_size):
#     '''
#     MultiRes Block
    
#     Arguments:
#         U {int} -- Number of filters in a corrsponding UNet stage
#         inp {keras layer} -- input layer 
    
#     Returns:
#         [keras layer] -- [output layer]
#     '''
#     x_inp = conv2d_bn(inp, filters//4, ksize=1, d_rate=1, strides=1)
    
#     x_1_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
#     x_1_2 = conv2d_bn(x_1_1, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
#     x_1_3 = conv2d_bn(x_1_2, filters//8, ksize=3, d_rate=1, strides=1,groups=filters//8)
    
#     x_2_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
#     x_2_2 = conv2d_bn(x_2_1, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
#     x_2_3 = conv2d_bn(x_2_2, filters//8, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//8)

#     x_3_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
#     x_3_2 = conv2d_bn(x_3_1, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
#     x_3_3 = conv2d_bn(x_3_2, filters//8, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//8)
    
#     x_4_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
#     x_4_2 = conv2d_bn(x_4_1, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
#     x_4_3 = conv2d_bn(x_4_2, filters//8, ksize=3, d_rate=d_size+1, strides=1, groups=filters//8)
    
#     o_1 = concatenate([x_1_1,x_1_2,x_1_3], axis=3)
#     o_2 = concatenate([x_2_1,x_2_2,x_2_3], axis=3)
#     o_3 = concatenate([x_1_1,x_3_2,x_3_3], axis=3)
#     o_4 = concatenate([x_1_1,x_4_2,x_4_3], axis=3)
    
#     o_1 = BatchNormalization(axis=3)(o_1)
#     o_2 = BatchNormalization(axis=3)(o_2)
#     o_3 = BatchNormalization(axis=3)(o_3)
#     o_4 = BatchNormalization(axis=3)(o_4)
    
#     ad1 = o_1
#     ad2 = add([ad1,o_2])
#     ad3 = add([ad2,o_3])
#     ad4 = add([ad3,o_4])
#     output = concatenate([ad1,ad2,ad3,ad4],axis=3)
#     #output = add([ad1,ad2,ad3,ad4])
#     output = BatchNormalization(axis=3)(output)
#     #output = Activation('relu')(output)
#     output = conv2d_bn(output, filters, ksize=1, d_rate=1, strides=1,padding='valid')
#     output = add([output, inp])

#     return output


# def CFPNet(height, width, channels):


#     inputs = Input(shape=(height, width, channels))
    
#     conv1=conv2d_bn(inputs, 32, 3, 1, 2)
#     conv2 = conv2d_bn(conv1, 32, 3, 1, 1)
#     conv3 = conv2d_bn(conv2, 32, 3, 1, 1)
    
#     injection_1 = AveragePooling2D()(inputs)
#     injection_1 = BatchNormalization(axis=3)(injection_1)
#     injection_1 = Activation('relu')(injection_1)
#     opt_cat_1 = concatenate([conv3,injection_1], axis=3)
    
#     #CFP block 1
#     opt_cat_1_0 = conv2d_bn(opt_cat_1, 64, 3, 1, 2)
#     cfp_1 = CFPModule(opt_cat_1_0, 64, 2)
#     cfp_2 = CFPModule(cfp_1, 64, 2)
    
#     injection_2 = AveragePooling2D()(injection_1)
#     injection_2 = BatchNormalization(axis=3)(injection_2)
#     injection_2 = Activation('relu')(injection_2)
#     opt_cat_2 = concatenate([cfp_2,opt_cat_1_0,injection_2], axis=3)
    
#     #CFP block 2
#     opt_cat_2_0 = conv2d_bn(opt_cat_2, 128, 3, 1, 2)
#     cfp_3 = CFPModule(opt_cat_2_0, 128, 4)
#     cfp_4 = CFPModule(cfp_3, 128, 4)
#     cfp_5 = CFPModule(cfp_4, 128, 8)
#     cfp_6 = CFPModule(cfp_5, 128, 8)
#     cfp_7 = CFPModule(cfp_6, 128, 16)
#     cfp_8 = CFPModule(cfp_7, 128, 16)
    
#     injection_3 = AveragePooling2D()(injection_2)
#     injection_3 = BatchNormalization(axis=3)(injection_3)
#     injection_3 = Activation('relu')(injection_3)
#     opt_cat_3 = concatenate([cfp_8,opt_cat_2_0,injection_3], axis=3)
    
    
#     conv4 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same',activation='relu')(opt_cat_3)
#     up_1 = concatenate([conv4,opt_cat_2])
    
#     conv5 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same',activation='relu')(up_1)
#     up_2 = concatenate([conv5, opt_cat_1],axis=3)    
    
#     conv6 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same',activation='relu')(up_2)
    
#     conv7 = conv2d_bn(conv6, 1, 1, 1, 1, activation='sigmoid', padding='valid')

#     # output = UpSampling2D(size=(8,8),interpolation='bilinear')(conv7)
    
#     model = Model(inputs=inputs, outputs=conv7)
    
#     return model

dropout = 0.2
def unet(num_classes = 5):
    inputs = Input(shape = (688,688,3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
   
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)
    
    # up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='PReLU', padding='same',
    #                       kernel_initializer='he_normal')(drop5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='PReLU', padding='same',
    #                       kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='PReLU', padding='same',
    #                       kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='PReLU', padding='same',
    #                       kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1,activation ='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    return model
# model = unet()
# model.summary()
def ICNet(num_classes=5):
    inp = Input(shape=(680,680,3))
    x = Lambda(lambda x: x/1.0)(inp)

    # (1/2)
    y = Lambda(lambda x: tf.image.resize(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='data_sub2')(x)
    y = Conv2D(32, 3, strides=2, padding='same', activation='PReLU', name='conv1_1_3x3_s2')(y)
    y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
    y = Conv2D(32, 3, padding='same', activation='PReLU', name='conv1_2_3x3')(y)
    y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
    y = Conv2D(64, 3, padding='same', activation='PReLU', name='conv1_3_3x3')(y)
    y = BatchNormalization(name='conv1_3_3x3_bn')(y)
    y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)
    
    y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)
    y_ = Conv2D(32, 1, activation='PReLU', name='conv2_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(name='padding1')(y_)
    y_ = Conv2D(32, 3, activation='PReLU', name='conv2_1_3x3')(y_)
    y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
    y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
    y = Add(name='conv2_1')([y,y_])
    y_ = Activation('PReLU', name='conv2_1/PReLU')(y)

    y = Conv2D(32, 1, activation='PReLU', name='conv2_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding2')(y)
    y = Conv2D(32, 3, activation='PReLU', name='conv2_2_3x3')(y)
    y = BatchNormalization(name='conv2_2_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
    y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
    y = Add(name='conv2_2')([y,y_])
    y_ = Activation('PReLU', name='conv2_2/PReLU')(y)

    y = Conv2D(32, 1, activation='PReLU', name='conv2_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding3')(y)
    y = Conv2D(32, 3, activation='PReLU', name='conv2_3_3x3')(y)
    y = BatchNormalization(name='conv2_3_3x3_bn')(y)
    y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
    y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
    y = Add(name='conv2_3')([y,y_])
    y_ = Activation('PReLU', name='conv2_3/PReLU')(y)

    y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
    y_ = Conv2D(64, 1, strides=2, activation='PReLU', name='conv3_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_) 
    y_ = ZeroPadding2D(name='padding4')(y_)
    y_ = Conv2D(64, 3, activation='PReLU', name='conv3_1_3x3')(y_)
    y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
    y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
    y = Add(name='conv3_1')([y,y_])
    z = Activation('PReLU', name='conv3_1/PReLU')(y)

    # (1/4)
    y_ = Lambda(lambda x: tf.image.resize(x, size=(int(x.shape[1])//2, int(x.shape[2])//2)), name='conv3_1_sub4')(z)
    y = Conv2D(64, 1, activation='PReLU', name='conv3_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding5')(y)
    y = Conv2D(64, 3, activation='PReLU', name='conv3_2_3x3')(y)
    y = BatchNormalization(name='conv3_2_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
    y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
    y = Add(name='conv3_2')([y,y_])
    y_ = Activation('PReLU', name='conv3_2/PReLU')(y)

    y = Conv2D(64, 1, activation='PReLU', name='conv3_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding6')(y)
    y = Conv2D(64, 3, activation='PReLU', name='conv3_3_3x3')(y)
    y = BatchNormalization(name='conv3_3_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
    y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
    y = Add(name='conv3_3')([y,y_])
    y_ = Activation('PReLU', name='conv3_3/PReLU')(y)

    y = Conv2D(64, 1, activation='PReLU', name='conv3_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(name='padding7')(y)
    y = Conv2D(64, 3, activation='PReLU', name='conv3_4_3x3')(y)
    y = BatchNormalization(name='conv3_4_3x3_bn')(y)
    y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
    y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
    y = Add(name='conv3_4')([y,y_])
    y_ = Activation('PReLU', name='conv3_4/PReLU')(y)

    y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
    y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
    y_ = Conv2D(128, 1, activation='PReLU', name='conv4_1_1x1_reduce')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
    y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
    y_ = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_1_3x3')(y_)
    y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
    y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
    y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
    y = Add(name='conv4_1')([y,y_])
    y_ = Activation('PReLU', name='conv4_1/PReLU')(y)

    y = Conv2D(128, 1, activation='PReLU', name='conv4_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding9')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_2_3x3')(y)
    y = BatchNormalization(name='conv4_2_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
    y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
    y = Add(name='conv4_2')([y,y_])
    y_ = Activation('PReLU', name='conv4_2/PReLU')(y)

    y = Conv2D(128, 1, activation='PReLU', name='conv4_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding10')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_3_3x3')(y)
    y = BatchNormalization(name='conv4_3_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
    y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
    y = Add(name='conv4_3')([y,y_])
    y_ = Activation('PReLU', name='conv4_3/PReLU')(y)

    y = Conv2D(128, 1, activation='PReLU', name='conv4_4_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding11')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_4_3x3')(y)
    y = BatchNormalization(name='conv4_4_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
    y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
    y = Add(name='conv4_4')([y,y_])
    y_ = Activation('PReLU', name='conv4_4/PReLU')(y)

    y = Conv2D(128, 1, activation='PReLU', name='conv4_5_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding12')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_5_3x3')(y)
    y = BatchNormalization(name='conv4_5_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
    y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
    y = Add(name='conv4_5')([y,y_])
    y_ = Activation('PReLU', name='conv4_5/PReLU')(y)

    y = Conv2D(128, 1, activation='PReLU', name='conv4_6_1x1_reduce')(y_)
    y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=2, name='padding13')(y)
    y = Conv2D(128, 3, dilation_rate=2, activation='PReLU', name='conv4_6_3x3')(y)
    y = BatchNormalization(name='conv4_6_3x3_bn')(y)
    y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
    y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
    y = Add(name='conv4_6')([y,y_])
    y = Activation('PReLU', name='conv4_6/PReLU')(y)

    y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
    y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
    y = Conv2D(256, 1, activation='PReLU', name='conv5_1_1x1_reduce')(y)
    y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding14')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='PReLU', name='conv5_1_3x3')(y)
    y = BatchNormalization(name='conv5_1_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
    y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
    y = Add(name='conv5_1')([y,y_])
    y_ = Activation('PReLU', name='conv5_1/PReLU')(y)

    y = Conv2D(256, 1, activation='PReLU', name='conv5_2_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding15')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='PReLU', name='conv5_2_3x3')(y)
    y = BatchNormalization(name='conv5_2_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
    y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
    y = Add(name='conv5_2')([y,y_])
    y_ = Activation('PReLU', name='conv5_2/PReLU')(y)

    y = Conv2D(256, 1, activation='PReLU', name='conv5_3_1x1_reduce')(y_)
    y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
    y = ZeroPadding2D(padding=4, name='padding16')(y)
    y = Conv2D(256, 3, dilation_rate=4, activation='PReLU', name='conv5_3_3x3')(y)
    y = BatchNormalization(name='conv5_3_3x3_bn')(y)
    y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
    y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
    y = Add(name='conv5_3')([y,y_])
    y = Activation('PReLU', name='conv5_3/PReLU')(y)

    h, w = y.shape[1:3].as_list()
    pool1 = AveragePooling2D(pool_size=(h,w), strides=(h,w), name='conv5_3_pool1')(y)
    pool1 = Lambda(lambda x: tf.image.resize(x, size=(h,w)), name='conv5_3_pool1_interp')(pool1)
    pool2 = AveragePooling2D(pool_size=(h/2,w/2), strides=(h//2,w//2), name='conv5_3_pool2')(y)
    pool2 = Lambda(lambda x: tf.image.resize(x, size=(h,w)), name='conv5_3_pool2_interp')(pool2)
    pool3 = AveragePooling2D(pool_size=(h/3,w/3), strides=(h//3,w//3), name='conv5_3_pool3')(y)
    pool3 = Lambda(lambda x: tf.image.resize(x, size=(h,w)), name='conv5_3_pool3_interp')(pool3)
    pool6 = AveragePooling2D(pool_size=(h/4,w/4), strides=(h//4,w//4), name='conv5_3_pool6')(y)
    pool6 = Lambda(lambda x: tf.image.resize(x, size=(h,w)), name='conv5_3_pool6_interp')(pool6)

    y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
    y = Conv2D(256, 1, activation='PReLU', name='conv5_4_k1')(y)
    y = BatchNormalization(name='conv5_4_k1_bn')(y)
    aux_1 = Lambda(lambda x: tf.image.resize(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='conv5_4_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
    y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
    y = BatchNormalization(name='conv_sub4_bn')(y)
    y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
    y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)
    y = Add(name='sub24_sum')([y,y_])
    y = Activation('PReLU', name='sub24_sum/PReLU')(y)

    aux_2 = Lambda(lambda x: tf.image.resize(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub24_sum_interp')(y)
    y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
    y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
    y_ = BatchNormalization(name='conv_sub2_bn')(y_)

    # (1)
    y = Conv2D(32, 3, strides=2, padding='same', activation='PReLU', name='conv1_sub1')(x)
    y = BatchNormalization(name='conv1_sub1_bn')(y)
    y = Conv2D(32, 3, strides=2, padding='same', activation='PReLU', name='conv2_sub1')(y)
    y = BatchNormalization(name='conv2_sub1_bn')(y)
    y = Conv2D(64, 3, strides=2, padding='same', activation='PReLU', name='conv3_sub1')(y)
    y = BatchNormalization(name='conv3_sub1_bn')(y)
    y = Conv2D(128, 1, name='conv3_sub1_proj')(y)
    y = BatchNormalization(name='conv3_sub1_proj_bn')(y)

    y = Add(name='sub12_sum')([y,y_])
    y = Activation('PReLU', name='sub12_sum/PReLU')(y)
    y = Lambda(lambda x: tf.image.resize(x, size=(int(x.shape[1])*2, int(x.shape[2])*2)), name='sub12_sum_interp')(y)
    y = UpSampling2D(size=(4,4))(y)
    out = Conv2D(num_classes, 1, activation='sigmoid', name='conv6_cls')(y)


    model = Model(inputs=inp, outputs=out)

    return model

# model =ICNet()
# model.summary()


def initial_block(tensor):

    conv = Conv2D(filters=13, kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_conv', kernel_initializer='he_normal')(tensor)

    pool = MaxPooling2D(pool_size=(2, 2), name='initial_block_pool')(tensor)

    concat = concatenate([conv, pool], axis=-1, name='initial_block_concat')

    return concat

def bottleneck_encoder(tensor, nfilters, downsampling=False, dilated=False, asymmetric=False, normal=False, drate=0.1, name=''):

    y = tensor

    skip = tensor

    stride = 1

    ksize = 1

    if downsampling:

        stride = 2

        ksize = 2

        skip = MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{name}')(skip)

        skip = Permute((1,3,2), name=f'permute_1_{name}')(skip)       #(B, H, W, C) -> (B, H, C, W)

        ch_pad = nfilters - K.int_shape(tensor)[-1]

        skip = ZeroPadding2D(padding=((0,0),(0,ch_pad)), name=f'zeropadding_{name}')(skip)

        skip = Permute((1,3,2), name=f'permute_2_{name}')(skip)       #(B, H, C, W) -> (B, H, W, C)        

    

    y = Conv2D(filters=nfilters//4, kernel_size=(ksize, ksize), kernel_initializer='he_normal', strides=(stride, stride), padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)

    y = PReLU(name=f'PReLU_1x1_{name}')(y)

    

    if normal:

        y = Conv2D(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same', name=f'3x3_conv_{name}')(y)

    elif asymmetric:

        y = Conv2D(filters=nfilters//4, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same', use_bias=False, name=f'5x1_conv_{name}')(y)

        y = Conv2D(filters=nfilters//4, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same', name=f'1x5_conv_{name}')(y)

    elif dilated:

        y = Conv2D(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', dilation_rate=(dilated, dilated), padding='same', name=f'dilated_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)

    y = PReLU(name=f'pPReLU_{name}')(y)

    

    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False, name=f'final_1x1_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)

    y = SpatialDropout2D(rate=drate, name=f'spatial_dropout_final_{name}')(y)

    

    y = Add(name=f'add_{name}')([y, skip])

    y = PReLU(name=f'pPReLU_out_{name}')(y)

    

    return y

def bottleneck_decoder(tensor, nfilters, upsampling=False, normal=False, name=''):

    y = tensor

    skip = tensor

    if upsampling:

        skip = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1), padding='same', use_bias=False, name=f'1x1_conv_skip_{name}')(skip)

        skip = UpSampling2D(size=(2, 2), name=f'upsample_skip_{name}')(skip)

    

    y = Conv2D(filters=nfilters//4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1), padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)

    y = PReLU(name=f'pPReLU_1x1_{name}')(y)

    

    if upsampling:

        y = Conv2DTranspose(filters=nfilters//4, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', name=f'3x3_deconv_{name}')(y)

    elif normal:

        Conv2D(filters=nfilters//4, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same', name=f'3x3_conv_{name}')(y)    

    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)

    y = PReLU(name=f'pPReLU_{name}')(y)

    

    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False, name=f'final_1x1_{name}')(y)

    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)



    y = Add(name=f'add_{name}')([y, skip])

    y = PReLU(name=f'PReLU_out_{name}')(y)

    

    return y

def ENet(num_classes=5):

    print('. . . . .Building ENet. . . . .')

    img_input = Input(shape=(680,680,3), name='image_input')



    x = initial_block(img_input)



    x = bottleneck_encoder(x, 64, downsampling=True, normal=True, name='1.0', drate=0.01)

    for _ in range(1,5):

        x = bottleneck_encoder(x, 64, normal=True, name=f'1.{_}', drate=0.01)



    x = bottleneck_encoder(x, 128, downsampling=True, normal=True, name=f'2.0')

    x = bottleneck_encoder(x, 128, normal=True, name=f'2.1')

    x = bottleneck_encoder(x, 128, dilated=2, name=f'2.2')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.3')

    x = bottleneck_encoder(x, 128, dilated=4, name=f'2.4')

    x = bottleneck_encoder(x, 128, normal=True, name=f'2.5')

    x = bottleneck_encoder(x, 128, dilated=8, name=f'2.6')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.7')

    x = bottleneck_encoder(x, 128, dilated=16, name=f'2.8')



    x = bottleneck_encoder(x, 128, normal=True, name=f'3.0')

    x = bottleneck_encoder(x, 128, dilated=2, name=f'3.1')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.2')

    x = bottleneck_encoder(x, 128, dilated=4, name=f'3.3')

    x = bottleneck_encoder(x, 128, normal=True, name=f'3.4')

    x = bottleneck_encoder(x, 128, dilated=8, name=f'3.5')

    x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.6')

    x = bottleneck_encoder(x, 128, dilated=16, name=f'3.7')



    x = bottleneck_decoder(x, 64, upsampling=True, name='4.0')

    x = bottleneck_decoder(x, 64, normal=True, name='4.1')

    x = bottleneck_decoder(x, 64, normal=True, name='4.2')



    x = bottleneck_decoder(x, 16, upsampling=True, name='5.0')

    x = bottleneck_decoder(x, 16, normal=True, name='5.1')



    img_output = Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same', name='image_output')(x)

    #img_output = Activation('softmax')(img_output)


    model = Model(inputs=img_input, outputs=img_output, name='ENET')

    print('. . . . .Build Compeleted. . . . .')

    return model

# model =ENet()
# model.build((1,680,680,3))
# model.summary()

def conv_layer(inp,number_of_filters, kernel, stride):
    
    network = Conv2D(filters=number_of_filters, kernel_size=kernel, 
                      strides=stride, padding = 'same', activation='PReLU',
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def dilation_conv_layer(inp, number_of_filters, kernel, stride, dilation_rate):
    
    network = Conv2D(filters=number_of_filters, kernel_size=kernel, activation='PReLU',
                      strides=stride, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def BN_PReLU(out):
    
    bacth_conv = BatchNormalization(axis=3)(out)
    PReLU_batch_norm = PReLU()(bacth_conv)
    return PReLU_batch_norm

def conv_one_cross_one(inp, number_of_classes):
    number_of_classes = 5
    network = Conv2D(filters=number_of_classes, kernel_size=1, 
                      strides=1, padding = 'valid', activation='PReLU',
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02))(inp)
    return network

def esp(inp, n_out):
    
    number_of_branches = 5
    n = int(n_out/number_of_branches)
    n1 = n_out - (number_of_branches - 1) * n
    
    # Reduce
    output1 = conv_layer(inp, number_of_filters=n, kernel=3, stride=2)
    
    # Split and Transform
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=3, stride=1, dilation_rate=1)
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=2)
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=4)
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=8)
    d16 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=16)
    
    add1 = dilated_conv2
    add2 = add([add1,dilated_conv4])
    add3 = add([add2,dilated_conv8])
    add4 = add([add3,d16])
    
    # Merge
    concat = concatenate([dilated_conv1,add1,add2,add3,add4], axis=3)
    concat = BN_PReLU(concat)
    return concat

def esp_alpha(inp,n_out):
    number_of_branches = 5
    if n_out == 2:
        n = n1 = 2
    else:
        n = int(n_out/number_of_branches)
        n1 = n_out - (number_of_branches - 1) * n
    
    # Reduce
    output1 = conv_layer(inp, number_of_filters=n, kernel=3, stride=1)
    
    # Split and Transform
    dilated_conv1 = dilation_conv_layer(output1, number_of_filters=n1, kernel=3, stride=1, dilation_rate=1)
    dilated_conv2 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=2)
    dilated_conv4 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=4)
    dilated_conv8 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=8)
    dilated_conv16 = dilation_conv_layer(output1, number_of_filters=n, kernel=3, stride=1, dilation_rate=16)
    
    add1 = dilated_conv2
    add2 = add([add1,dilated_conv4])
    add3 = add([add2,dilated_conv8])
    add4 = add([add3,dilated_conv16])
    
    # Merge
    concat = concatenate([dilated_conv1,add1,add2,add3,add4], axis=3)
    concat = BN_PReLU(concat)
    return concat

def espnet(num_classes = 5):
    inputs = Input(shape=(680,680,3))
    conv_output = conv_layer(inputs, number_of_filters=16, kernel=3, stride=2)
    PReLU_ = BN_PReLU(conv_output)
    
    avg_pooling1 = conv_output
    avg_pooling2 = AveragePooling2D()(avg_pooling1)
    avg_pooling2 = BN_PReLU(avg_pooling2)
    
    concat1 = concatenate([avg_pooling1,PReLU_], axis=3)
    concat1 = BN_PReLU(concat1)
    esp_1 = esp(concat1,64)
    esp_1 = BN_PReLU(esp_1)

    esp_alpha_1 = esp_1
    
    esp_alpha_1 = esp_alpha(esp_alpha_1, 64)
    esp_alpha_1 = esp_alpha(esp_alpha_1, 64)
    concat2 = concatenate([esp_alpha_1,esp_1,avg_pooling2], axis=3)
    
    esp_2 = esp(concat2,128)
    esp_alpha_2 = esp_2

    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)
    esp_alpha_2 = esp_alpha(esp_alpha_2, 128)

    
    concat3 = concatenate([esp_alpha_2,esp_2],axis = 3)
    pred = conv_one_cross_one(concat3, 16)
    
    deconv1 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='PReLU')(pred)
    conv_1 = conv_one_cross_one(concat2, 16)
    concat4 = concatenate([deconv1,conv_1], axis=3)
    esp_3 = esp_alpha(concat4, 16)
    deconv2 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='PReLU')(esp_3)
    conv_2 = conv_one_cross_one(concat1, 16)
    concat5 = concatenate([deconv2,conv_2], axis=3)
    conv_3 = conv_one_cross_one(concat5, 16)
    deconv3 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same',activation='PReLU')(conv_3)
    deconv3 = conv_one_cross_one(deconv3, 1)
    deconv3 = Activation('softmax')(deconv3)
    
    model = Model(inputs=inputs,outputs = deconv3)
    return model

model =espnet()
model.build((1,680,680,3))
model.summary()


