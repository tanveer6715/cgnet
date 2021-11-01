import tensorflow as tf
from tensorflow.keras.layers.experimental import SyncBatchNormalization
#code reference: https://github.com/soumik12345/ESNet/blob/master/src/model.py
def DownsamplingBlock(input_tensor, input_channels, output_channels):
    '''Downsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        input_channels  -> Number of channels in the input tensor
        output_channels -> Number of output channels
    '''
    x1 = tf.keras.layers.Conv2D(
        output_channels - input_channels, (3, 3),
        strides=(2, 2), use_bias=True, padding='same'
    )(input_tensor)
    x2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(input_tensor)
    x = tf.keras.layers.concatenate([x1, x2])
    x = SyncBatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def FCU(input_tensor, output_channels, K=3, dropout_prob=0.03):
    '''Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        K -> Size of Kernel
    '''
    x = tf.keras.layers.Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = SyncBatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (K, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, K),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    x = SyncBatchNormalization()(x)
    x = tf.keras.layers.Add()([input_tensor, x])
    x = tf.keras.layers.Dropout(dropout_prob)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def PFCU(input_tensor, output_channels):
    '''Parallel Factorized Convolutional Unit
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = tf.keras.layers.Conv2D(
        output_channels, (3, 1),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        output_channels, (1, 3),
        strides=(1, 1), use_bias=True, padding='same'
    )(input_tensor)
    x = SyncBatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Branch 1
    branch_1 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_1 = tf.keras.layers.ReLU()(branch_1)
    branch_1 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (2, 2),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_1)
    branch_1 = SyncBatchNormalization()(branch_1)
    # Branch 2
    branch_2 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_2 = tf.keras.layers.ReLU()(branch_2)
    branch_2 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (5, 5),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_2)
    branch_2 = SyncBatchNormalization()(branch_2)
    # Branch 3
    branch_3 = tf.keras.layers.Conv2D(
        output_channels, (3, 1), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(x)
    branch_3 = tf.keras.layers.ReLU()(branch_3)
    branch_3 = tf.keras.layers.Conv2D(
        output_channels, (1, 3), dilation_rate = (9, 9),
        strides=(1, 1), use_bias=True, padding='same'
    )(branch_3)
    branch_3 = SyncBatchNormalization()(branch_3)
    x = tf.keras.layers.Add()([input_tensor, branch_1, branch_2, branch_3])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def UpsamplingBlock(input_tensor, output_channels):
    '''Upsampling Block
    Reference: https://arxiv.org/pdf/1906.09826v1.pdf
    Params:
        input_tensor    -> Input Tensor
        output_channels -> Number of output channels
    '''
    x = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, padding='same',
        strides=(2, 2), use_bias=True
    )(input_tensor)
    x = SyncBatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def ESNet(input_shape=(680,680, 3), output_channels=5):
    inp = tf.keras.layers.Input(input_shape)
    ##### Encoder #####
    # Block 1
    x = DownsamplingBlock(inp, 3, 16)
    x = FCU(x, 16, K=3)
    x = FCU(x, 16, K=3)
    x = FCU(x, 16, K=3)
    # Block 2
    x = DownsamplingBlock(x, 16, 64)
    x = FCU(x, 64, K=5)
    x = FCU(x, 64, K=5)
    # Block 3
    x = DownsamplingBlock(x, 64, 128)
    x = PFCU(x, 128)
    x = PFCU(x, 128)
    x = PFCU(x, 128)
    ##### Decoder #####
    # Block 4
    x = UpsamplingBlock(x, 64)
    x = FCU(x, 64, K=5, dropout_prob=0.0)
    x = FCU(x, 64, K=5, dropout_prob=0.0)
    # Block 5
    x = UpsamplingBlock(x, 16)
    x = FCU(x, 16, K=3, dropout_prob=0.0)
    x = FCU(x, 16, K=3, dropout_prob=0.0)
    output = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, padding='same',
        strides=(2, 2), use_bias=True
    )(x)
    return tf.keras.models.Model(inp, output)
# model = ESNet()
# model.summary()

