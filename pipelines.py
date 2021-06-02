from cityscapes import CityscapesDatset

import tensorflow as tf 



def normalize(input_image, input_mask):
    """
    

    """

    input_image = input_image / 255.0

    return input_image, input_mask

@tf.function
def load_image_train(datapoint, size=(128, 128)):

    """

    """

    input_image = tf.cast(datapoint['image'], tf.float32)
    input_mask = tf.cast(datapoint['segmentation_mask'], tf.uint8)
    input_mask = input_mask[..., tf.newaxis]

    input_image = tf.image.resize(datapoint['image'], size)
    input_mask = tf.image.resize(input_mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #TODO: add more augmentation 
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint, size=(128, 128)):
    """

    """
    input_image = tf.cast(datapoint['image'], tf.float32)
    input_mask = tf.cast(datapoint['segmentation_mask'], tf.uint8)
    input_mask = input_mask[..., tf.newaxis]
    

    input_image = tf.image.resize(datapoint['image'], size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def batch_generator(Dataset, batch_size, shuffle=True, ignore_class = 255):
    """
    Args : 
        Dataset : dataset class defined in cityscapes.py. 
    
    Return : 
        
    """
    idx_dataset = [range(len(Dataset))]

    if shuffle :
        from random import shuffle
        shuffle(idx_dataset)

    for idx in range(len(Dataset)//batch_size):
        
        imgs_to_stack = []
        labels_to_stack = []

        for data_idx in range(idx, idx+batch_size):
            image, label = load_image_train(Dataset[data_idx])
            imgs_to_stack.append(image)
            labels_to_stack.append(label)
        
        images = tf.stack(imgs_to_stack)
        labels = tf.stack(labels_to_stack)

        # if ignore_class : 
        #     idx_to_ignore = labels==255
        #     print(idx_to_ignore.dtype)
        #     labels[idx_to_ignore] = 0.0

        yield images, labels
        
    


def test(): 
    data_dir = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    cityscapes_dataset = CityscapesDatset(data_dir)
    TRAIN_LENGTH = len(cityscapes_dataset)
    print("Length of the dataset : {}".format(TRAIN_LENGTH))

    input_image, input_mask = load_image_train(cityscapes_dataset[1])
    # print("Shape of input_image : {}".format(input_image))
    # print("Shape of input_mask : {}".format(input_mask))
    print(tf.reduce_max(input_image))
    print(tf.reduce_max(input_mask))


if __name__ == "__main__" : 
    test() 