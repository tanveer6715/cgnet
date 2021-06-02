from cityscapes import CityscapesDatset

import tensorflow as tf 



def normalize(input_image, input_mask):
    """
    

    """

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

@tf.function
def load_image_train(datapoint, size=(128, 128)):

    """

    """
    input_image = tf.image.resize(datapoint['image'], size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], size)

    #TODO: add more augmentation 
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint, size=(128, 128)):
    """

    """

    input_image = tf.image.resize(datapoint['image'], size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], size)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def test(): 
    data_dir = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    cityscapes_dataset = CityscapesDatset(data_dir)
    TRAIN_LENGTH = len(cityscapes_dataset)
    print("Length of the dataset : {}".format(TRAIN_LENGTH))


if __name__ == "__main__" : 
    test() 