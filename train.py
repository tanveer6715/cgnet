import argparse
import os 

import tensorflow as tf
import numpy as np

from cityscapes import CityscapesDatset
from model import CGNet
from pipeline import batch_generator
from utils import load_config
from loss import compute_loss
from optim import load_optimizer

# @tf.function
# def train_step(model, images, labels, optimizer, class_weight, train_loss, train_accuracy, train_iou) :
#     """ Training Step

#     Args : 
#         model (tf.model)
#         images (tf.Tensor)
#         lables (tf.Tensor)
#         optimizer (tf.optimizer)
#         train_loss (tf.loss)
        
    
#     """
#     with tf.GradientTape() as tape:
#         predictions = model(images)
#         loss = compute_loss(labels, predictions, class_weight)
    
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_accuracy(labels, predictions)

#     argmax_predictions = tf.math.argmax(predictions, 3)
    
#     train_iou.update_state(labels, argmax_predictions)


def train(config_path):

    config = load_config(config_path)


    data_dir = config.get('DATA_DIR', None)
    resume_from = config.get('RESUME_FROM', None)
    model_save_to = config.get('MODEL_SAVE_TO', None)

    dataset = config.get('DATASET', None)
    img_width = config.get('IMG_WIDTH', None)
    img_height = config.get('IMG_HEIGHT', None)
    num_classes = config.get('NUM_CLASSES', None)
    ignore_calss = config.get('IGNORE_CLASS', None)

    batch_size = config.get('BATCH_SIZE', None)
    epochs = config.get('EPOCHS', None)
    num_gpu = config.get('NUM_GPU', None)

    num_m_blocks = config.get('NUM_M_BLOCKS', None)
    num_n_blocks = config.get('NUM_N_BLOCKS', None)

    init_learn_rate = config.get('INIT_LEARN_RATE', None)
    end_learn_rate = config.get('END_LEARN_RATE',  None)
    power = config.get('POWER', None)

    class_weight = config.get('CLASS_WEIGHT', None)

    

    model = CGNet(num_classes = num_classes, M= num_m_blocks, N=num_n_blocks)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_iou = tf.keras.metrics.MeanIoU(num_classes= num_classes, name='train_miou')


    if dataset == 'Cityscapes':
        train_dataset = CityscapesDatset(data_dir)

    num_steps = len(train_dataset)//batch_size

    optimizer = load_optimizer(init_learn_rate, end_learn_rate, power)


    for epoch in range(1, epochs):

        train_dataset_generator = batch_generator(train_dataset, batch_size)

        for step, batch in enumerate(train_dataset_generator):
            
            # train_step(model, images, labels, optimizer, class_weight, train_loss, train_accuracy, train_iou)
            images, labels =  batch
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = compute_loss(labels, predictions, class_weight)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

            argmax_predictions = tf.math.argmax(predictions, 3)
            
            train_iou.update_state(labels, argmax_predictions)
        
            template = 'Epoch: {}/{}, steps:{}/{}, Loss: {:2f}, Accuracy: {:2f}, MeanIoU: {:2f}'
            print (template.format(epoch,
                                    epochs,
                                    step,
                                    num_steps,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    train_iou.result()*100
                                    ))
        if epoch % 5 == 0 :
            model.save_weights(os.path.join(model_save_to, 'epoch_{}.h5'.format(epoch)))


if __name__ == "__main__" : 

    
    parser = argparse.ArgumentParser("Please Set Training Configuration File")
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    train(args.config)


