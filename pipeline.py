
from random import shuffle
import tensorflow as tf 
from datasets.cityscapes import CityscapesDatset
import numpy as np

def normalize(image, label):
    "TODO : add better normalization strategy"
    """

    Args

    """
    image = tf.math.subtract(image, [72.39239876, 82.90891754, 73.15835921])
    
    return image, label

#@tf.function
def load_image_train(datapoint, size=(680,680)):

    """
    Load training data 

    Args : 

    Returns : 
    
    """

    image = tf.cast(datapoint['image'], tf.float32)
    label = tf.cast(datapoint['segmentation_mask'], tf.uint8)
    label = label[..., tf.newaxis]

    image = tf.image.resize(datapoint['image'], size)
    label = tf.image.resize(label, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # #TODO: add more augmentation 
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    
        

    image, label = normalize(image, label)

    return image, label

#@tf.function
def load_image_test(datapoint, size=(680,680), is_normalize = True):

    """
    Load test images 

    Args : 
        datapoint
        size 
        is_normalize 

    Returns :
        image
        label 

    """
    image = tf.cast(datapoint['image'], tf.float32)
    label = tf.cast(datapoint['segmentation_mask'], tf.uint8)
    label = label[..., tf.newaxis]
    

    image = tf.image.resize(datapoint['image'], size)
    label = tf.image.resize(label, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if is_normalize :
        image, label = normalize(image, label)


    return image, label

def batch_generator(Dataset, batch_size, shuffle=True, repeat = 1, ignore_class = 255):
    """    
    TODO 
        2. add test mode 
        3. add a function to handle ignore class 
            Currently, the ignore_class is regared as a background. 

    """

    """
    Args : 
        Dataset (class) : dataset class defined in cityscapes.py. 
        batch_size (int) : batch size 
        shuffle (bool) : shuffle dataset order 
        ignore_class (int) : class number to be ignored 

    Return : 
        images (np.array) : images 
        labels (np.array) : labels array in 2d 
        
    """
    
    idx_dataset = list(range(len(Dataset)))
    idx_dataset = idx_dataset*repeat
    

    if shuffle :
        from random import shuffle
        shuffle(idx_dataset)

    for idx in range(len(idx_dataset)//batch_size):
        
        imgs_to_stack = []
        labels_to_stack = []

        for _data_idx in range(idx*batch_size, (idx+1)*batch_size):
            data_idx = idx_dataset[_data_idx]
            image, label = load_image_train(Dataset[data_idx])
            imgs_to_stack.append(image)
            labels_to_stack.append(label)
        
        images = tf.stack(imgs_to_stack)
        labels = tf.stack(labels_to_stack)

        if ignore_class : 
            idx_to_ignore = labels!=ignore_class
            labels = tf.where(idx_to_ignore, labels, 0)

        yield (images, labels)

class _batch_generator: 

    def __init__(self, dataset, batch_size, shuffle=True, ignore_class = 255):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx_dataset = list(range(len(dataset)))
        self.ignore_class = ignore_class
        self.idx = 0
        
        if shuffle :
            from random import shuffle
            shuffle(self.idx_dataset)

    def __iter__(self):
        return self


    def __next__(self):
        self.idx += self.batch_size
        imgs_to_stack = []
        labels_to_stack = []

        for _data_idx in range(self.idx, self.idx+self.batch_size):
            data_idx = self.idx_dataset[_data_idx]
            image, label = load_image_train(self.dataset[data_idx])
            imgs_to_stack.append(image)
            labels_to_stack.append(label)
        
        images = tf.stack(imgs_to_stack)
        labels = tf.stack(labels_to_stack)

        if self.ignore_class : 
            idx_to_ignore = labels!= self.ignore_class
            labels = tf.where(idx_to_ignore, labels, 0)
        
        return images, labels

def test(): 
    data_dir = '/home/sss/UOS-SSaS Dropbox/05. Data/00. Benchmarks/01. cityscapes'
    cityscapes_dataset = CityscapesDatset(data_dir)
    TRAIN_LENGTH = len(cityscapes_dataset)
    print("Length of the dataset : {}".format(TRAIN_LENGTH))

    image, label = load_image_train(cityscapes_dataset[1])
    # print("Shape of image : {}".format(image))
    # print("Shape of label : {}".format(label))
    print(tf.reduce_max(image))
    print(tf.reduce_max(label))


if __name__ == "__main__" : 
    test() 