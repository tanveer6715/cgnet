import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D , BatchNormalization
from tensorflow.keras.layers import concatenate, Activation, UpSampling2D

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def unpool_with_argmax(pool, ind, name = None, ksize=[1,2,2,1]):
    #refer from : https://github.com/mathildor/TF-SegNet/blob/master/AirNet/layers.py
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
    
    return ret
    

def max_pool_with_argmax(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool_with_argmax')[0]

def convBnRelu(x, filters=64):
    
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    
    return x

 

def SegNet(nb_classes=5, input_shape=(680,680,3)):
    inputs = Input((input_shape))
    
    #Encoders
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
        
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    #Decoder
#     x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nb_classes, (1,1), activation='softmax', padding='same', name='output')(x)

    model = Model(inputs=inputs, outputs=x)
    return model 

model = SegNet()
model.summary()
